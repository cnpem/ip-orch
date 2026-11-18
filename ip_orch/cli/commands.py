import argparse
import importlib.util
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import resources
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from ..config.config_store import (
    add_model,
    load_config,
    remove_model,
    save_config,
    set_model_status,
)
from ..core.reference_energies import (
    LEGACY_REFERENCE_WARNING,
    check_reference_elements,
    is_reference_energy_computable,
    load_precomputed_reference_energies,
    supported_reference_elements,
)
from .env_utils import (
    _current_conda_env,
    _discover_conda_envs,
    _discover_envs_from_dir,
    _guess_envs_dir,
    _is_path_like_env,
    _match_known_token,
    _python_for_env,
)
from .helpers import (
    DEFAULT_MODELS_BY_ENV,
    PACKAGE_VARIANTS,
    _alias_match_token,
    _canonical_alias,
    _clean_env,
    _dedup_pairs,
    _get_aliases,
    _group_pairs,
)
from .repo_map import repo_url_for_alias

console = Console()


@dataclass
class _RunJob:
    env_name: str
    model_name: str
    device: str
    cmd: list[str]
    cmd_display: str
    extra: str = ""


@dataclass
class _RunResult:
    env_name: str
    model_name: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    preflight_failed: bool = False


def _package_parent_path() -> str:
    try:
        spec = importlib.util.find_spec("ip_orch")
        if spec and spec.submodule_search_locations:
            package_path = os.path.abspath(next(iter(spec.submodule_search_locations)))
            return os.path.dirname(package_path)
    except Exception:
        pass
    return ""


def _worker_resource_path() -> str:
    return str(resources.files("ip_orch.core").joinpath("worker.py"))


def _format_worker_command(cmd: list[str], worker_path: str) -> str:
    cmd = list(cmd)
    try:
        worker_idx = cmd.index(worker_path)
        tail = cmd[worker_idx + 1 :]
        if len(tail) >= 11:
            if tail[8]:
                tail[8] = _summarize_elements_arg(tail[8])
            if tail[9]:
                tail[9] = _summarize_json_arg(tail[9])
            cmd = cmd[: worker_idx + 1] + tail
    except ValueError:
        pass

    cmd_display = " ".join(cmd)
    try:
        worker_idx = cmd.index(worker_path)
        if worker_idx + 1 < len(cmd):
            cmd_display = " ".join(cmd[: worker_idx + 1]) + "\n" + " ".join(cmd[worker_idx + 1 :])
    except ValueError:
        pass
    return cmd_display


def _summarize_elements_arg(elements_arg: str) -> str:
    elements = _parse_elements(elements_arg)
    if len(elements) <= 8:
        return elements_arg
    return f"<{len(elements)} elements>"


def _summarize_json_arg(json_arg: str) -> str:
    try:
        parsed = json.loads(json_arg)
    except Exception:
        return "<reference-energy-json>"
    if isinstance(parsed, dict):
        return f"<{len(parsed)} reference energies>"
    return "<reference-energy-json>"


def _reference_energy_summary(element_energies) -> str:
    if not element_energies:
        return ""
    if len(element_energies) > 8:
        return "\nreference energy correction: precomputed values loaded"
    parts = []
    for k, v in sorted(element_energies.items()):
        try:
            parts.append(f"{k}={float(v):.2f} eV")
        except Exception:
            parts.append(f"{k}={v} eV")
    return "\nreference energy correction: " + " | ".join(parts)


def _parse_elements(elements) -> list[str]:
    if isinstance(elements, (list, tuple)):
        return [str(e).strip() for e in elements if str(e).strip()]
    if not elements:
        return []
    return [e.strip() for e in str(elements).split(",") if e.strip()]


def _periodic_elements() -> list[str]:
    try:
        from ase.data import chemical_symbols

        return [symbol for symbol in chemical_symbols[1:] if symbol]
    except Exception:
        return []


def _element_status_table(supported: set[str], requested: Optional[set[str]] = None, *, groups: int = 3) -> Table:
    elements = _periodic_elements()
    if not elements:
        elements = sorted(supported | (requested or set()))

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    for idx in range(groups):
        table.add_column("element", style="cyan")
        table.add_column("status")
        if idx < groups - 1:
            table.add_column("", no_wrap=True)

    chunk_size = (len(elements) + groups - 1) // groups
    chunks = [elements[idx * chunk_size : (idx + 1) * chunk_size] for idx in range(groups)]
    for row_idx in range(chunk_size):
        row = []
        for group_idx, chunk in enumerate(chunks):
            if row_idx < len(chunk):
                element = chunk[row_idx]
                if element in supported:
                    status = Text("supported", style="green")
                else:
                    status = Text("missing", style="red")
                if requested is not None and element in requested:
                    element_text = Text(element, style="bold cyan")
                else:
                    element_text = Text(element, style="cyan")
                row.extend([element_text, status])
            else:
                row.extend(["", ""])
            if group_idx < groups - 1:
                row.append("")
        table.add_row(*row)
    return table


def _run_subprocess(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=capture_output, env=_subprocess_env())


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    backend = env.get("MPLBACKEND", "")
    if not backend or "matplotlib_inline" in backend or "backend_inline" in backend:
        env["MPLBACKEND"] = "Agg"
    return env


def _supported_alias(model: str) -> Optional[str]:
    norm = _alias_match_token(model)
    for alias in _get_aliases().keys():
        if _alias_match_token(alias) == norm:
            return alias
    return None


def cmd_add(args: argparse.Namespace) -> int:
    model = _supported_alias(args.model)
    if model is None:
        console.print(f"[yellow]Unknown model alias: {args.model!r}.")
        console.print("Use 'ip-orch --supported-models' or IPOrch().supported_models() to list supported aliases.")
        return 1
    add_model(args.env, model)
    console.print(f"Added: env='{args.env}' model='{model}'")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    changed = remove_model(args.env, args.model)
    if changed:
        console.print("Removed entry.")
        return 0
    console.print("No matching entry found.")
    return 1


def cmd_models(args: argparse.Namespace) -> int:
    aliases = _get_aliases()
    cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))
    configured_aliases = {_canonical_alias(m) for _e, m in pairs}

    rows = [(alias, target) for alias, target in aliases.items()]

    if args.contains:
        sub = args.contains.lower()
        rows = [
            r
            for r in rows
            if sub in r[0].lower() or sub in (r[1] or "").lower() or sub in repo_url_for_alias(r[0]).lower()
        ]

    # Show configured aliases first, then the rest; keep alphabetical within groups.
    rows = sorted(rows, key=lambda r: (0 if r[0] in configured_aliases else 1, r[0]))

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("alias", style="cyan")
    table.add_column("registry", style="magenta")
    table.add_column("repo", style="blue")
    table.add_column("configured")
    for alias, target in rows:
        url = repo_url_for_alias(alias)
        configured_mark = Text("✓", style="green") if alias in configured_aliases else Text("×", style="red")
        table.add_row(alias, target, url, configured_mark)
    console.print(table)
    return 0


def cmd_check_elements(args: argparse.Namespace) -> int:
    elements = _parse_elements(getattr(args, "elements", None))
    model = getattr(args, "model", "")
    supported = supported_reference_elements(model)
    title = f"{model} | supported elements for reference energy"
    requested = set(elements) if elements else None

    if not supported:
        table = _element_status_table(set(), requested=requested)
        notes = [f"No precomputed reference energies are available for model {model!r}."]
        if not is_reference_energy_computable(model):
            notes.append("This model family also cannot compute isolated-atom reference energies.")
        else:
            notes.append("It may still work with --reference-energy-source computed.")
        console.print(Panel(table, title=title, subtitle="\n".join(notes), border_style="red"))
        return 1

    table = _element_status_table(set(supported), requested=requested)
    _, missing = check_reference_elements(model, elements) if elements else ([], [])
    border_style = "red" if missing else "green"
    console.print(Panel(table, title=title, subtitle=LEGACY_REFERENCE_WARNING, border_style=border_style))
    return 1 if missing else 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))
    models_path = getattr(args, "models_path", None) or cfg.get("models_path", "")
    envs_base_dir = cfg.get("envs_base_dir", "")
    # Selection: either --envs or --models must be provided by caller
    selected_pairs = []
    if getattr(args, "envs", None):
        envs_sel = {_clean_env(x.strip()) for x in args.envs.split(",") if x.strip()}
        selected_pairs = [p for p in pairs if _clean_env(p[0]) in envs_sel]
    elif getattr(args, "models", None):
        models_sel = {_canonical_alias(x.strip()) for x in args.models.split(",") if x.strip()}
        selected_pairs = [p for p in pairs if _canonical_alias(p[1]) in models_sel]
    else:
        console.print("[red]For --run, provide either --envs or --models.")
        return 2

    if not selected_pairs:
        console.print("[red]No matching (env, model) pairs found for the selection.")
        console.print("Use 'ip-orch --configure' or 'ip-orch --add ENV MODEL' to set them.")
        return 1

    parallel = max(1, int(getattr(args, "parallel", 1) or 1))
    summary_lines = [f"{_clean_env(e)} → {m}" for e, m in selected_pairs]
    title = "IP-Orch: starting execution"
    if parallel > 1:
        title = f"{title} | running up to {parallel} models in parallel"
    panel_lines = [title]
    summary = "\n".join(summary_lines) if summary_lines else "(no models)"
    panel_lines.append(summary)
    console.print(Panel("\n".join(panel_lines), border_style="blue"))

    # Optional linear correction parameters (can come from JSON config and/or args)
    linear_a = getattr(args, "energy_linear_a", None)
    linear_b = getattr(args, "energy_linear_b", None)
    linear_mode = getattr(args, "energy_linear_mode", None) or "total_energy"
    elements = getattr(args, "correction_elements", None)
    no_energy_corr = bool(getattr(args, "no_energy_correction", False))
    reference_energy_source = getattr(args, "reference_energy_source", None) or "computed"
    linear_cfg_path = getattr(args, "energy_linear_config", None)
    if linear_cfg_path:
        try:
            with open(linear_cfg_path, encoding="utf-8") as f:
                linear_cfg = json.load(f) or {}
            # Only fill missing values from config.
            if linear_a is None:
                linear_a = linear_cfg.get("a", None)
            if linear_b is None:
                linear_b = linear_cfg.get("b", None)
            if getattr(args, "energy_linear_mode", None) in (None, ""):
                linear_mode = linear_cfg.get("mode", linear_mode) or linear_mode
            if elements is None and not no_energy_corr:
                elements = linear_cfg.get("correction_elements", None)
            reference_energy_source = linear_cfg.get("reference_energy_source", reference_energy_source)
        except Exception as exc:
            console.print(f"[red]Failed to read --energy-linear-config: {exc}")
            return 2

    if no_energy_corr:
        linear_a = None
        linear_b = None
        elements = None
        reference_energy_source = "computed"

    if (linear_a is None) ^ (linear_b is None):
        console.print("[red]Provide both --energy-linear-a and --energy-linear-b (or neither).")
        return 2
    element_list = _parse_elements(elements)
    elements = ",".join(element_list)
    reference_enabled = bool(elements) or reference_energy_source == "precomputed"
    linear_enabled = linear_a is not None and linear_b is not None
    if linear_enabled and linear_mode not in ("total_energy", "per_atom"):
        console.print("[red]--energy-linear-mode must be 'total_energy' or 'per_atom'.")
        return 2
    if reference_energy_source not in ("computed", "precomputed"):
        console.print("[red]--reference-energy-source must be 'computed' or 'precomputed'.")
        return 2

    repo_root = _package_parent_path()
    worker_path = _worker_resource_path()

    def build_base_command(env_name: str, model_name: str) -> list[str]:
        python_bin = _python_for_env(env_name, envs_base_dir)
        if python_bin:
            return [
                python_bin,
                worker_path,
                args.script,
                env_name,
                model_name,
                models_path,
                repo_root,
            ]
        if _is_path_like_env(env_name):
            raise FileNotFoundError(
                f"Could not find a Python executable for environment path {env_name!r}. "
                "Expected bin/python or Scripts/python.exe."
            )
        return [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            worker_path,
            args.script,
            env_name,
            model_name,
            models_path,
            repo_root,
        ]

    def build_worker_mode_command(base_cmd: list[str], mode: str) -> list[str]:
        cmd = list(base_cmd)
        cmd.extend(["", "", "total_energy", "", "", mode])
        return cmd

    def resolve_job_device(base_cmd: list[str]) -> tuple[Optional[str], Optional[_RunResult]]:
        try:
            res = _run_subprocess(build_worker_mode_command(base_cmd, "device"), capture_output=True)
        except Exception as exc:
            return None, _RunResult("", "", 1, stderr=str(exc), preflight_failed=True)
        if res.returncode != 0:
            return None, _RunResult(
                "",
                "",
                res.returncode,
                stdout=res.stdout or "",
                stderr=res.stderr or "",
                preflight_failed=True,
            )
        for line in (res.stdout or "").splitlines():
            if line.startswith("IPORCH_DEVICE="):
                return line.split("=", 1)[1].strip() or "unknown", None
        return "unknown", None

    def prepare_job(env_name: str, model_name: str) -> tuple[Optional[_RunJob], Optional[_RunResult]]:
        try:
            cmd = build_base_command(env_name, model_name)
        except Exception as exc:
            return None, _RunResult(
                env_name=env_name,
                model_name=model_name,
                returncode=1,
                stderr=str(exc),
                preflight_failed=True,
            )
        element_energies = None
        job_element_list = list(element_list)
        worker_element_list = list(element_list)
        if reference_enabled and reference_energy_source == "precomputed" and not job_element_list:
            job_element_list = supported_reference_elements(model_name)
            if not job_element_list:
                return None, _RunResult(
                    env_name=env_name,
                    model_name=model_name,
                    returncode=1,
                    stderr=f"No precomputed reference energies are available for model {model_name!r}.",
                    preflight_failed=True,
                )
        job_elements = ",".join(job_element_list)
        worker_elements = ",".join(worker_element_list)

        if reference_enabled:
            if reference_energy_source == "precomputed":
                try:
                    element_energies = load_precomputed_reference_energies(model_name, job_element_list)
                except Exception as exc:
                    return None, _RunResult(
                        env_name=env_name,
                        model_name=model_name,
                        returncode=1,
                        stderr=str(exc),
                        preflight_failed=True,
                    )
            elif not is_reference_energy_computable(model_name):
                return None, _RunResult(
                    env_name=env_name,
                    model_name=model_name,
                    returncode=1,
                    stderr=(
                        f"Reference energies cannot be computed for {model_name!r}. "
                        "Use a model with isolated-atom support; no precomputed values are available for "
                        "GRACE, MatRIS, or M3GNet."
                    ),
                    preflight_failed=True,
                )
            else:
                device, device_error = resolve_job_device(cmd)
                if device_error:
                    device_error.env_name = env_name
                    device_error.model_name = model_name
                    return None, device_error
                # Preflight to compute per-element reference energies inside the MLIP env,
                # so we can show the correction term in the panel.
                preflight_cmd = list(cmd)
                preflight_cmd.extend(
                    [
                        "" if linear_a is None else str(linear_a),
                        "" if linear_b is None else str(linear_b),
                        str(linear_mode),
                        str(job_elements),
                        "",  # element_energies_json (computed in preflight)
                        "preflight",
                    ]
                )
                try:
                    pre = _run_subprocess(preflight_cmd, capture_output=True)
                except Exception as exc:
                    return None, _RunResult(
                        env_name=env_name,
                        model_name=model_name,
                        returncode=1,
                        stderr=str(exc),
                        preflight_failed=True,
                    )
                if pre.returncode != 0:
                    return None, _RunResult(
                        env_name=env_name,
                        model_name=model_name,
                        returncode=pre.returncode,
                        stdout=pre.stdout or "",
                        stderr=pre.stderr or "",
                        preflight_failed=True,
                    )
                for line in (pre.stdout or "").splitlines():
                    if line.startswith("IPORCH_ELEMENT_ENERGIES="):
                        try:
                            element_energies = json.loads(line.split("=", 1)[1])
                        except Exception:
                            element_energies = None
                        break
        if not (reference_enabled and reference_energy_source == "computed"):
            device, device_error = resolve_job_device(cmd)
            if device_error:
                device_error.env_name = env_name
                device_error.model_name = model_name
                return None, device_error

        if linear_enabled or reference_enabled:
            cmd.extend(
                [
                    "" if linear_a is None else str(linear_a),
                    "" if linear_b is None else str(linear_b),
                    str(linear_mode),
                    "" if not worker_elements else str(worker_elements),
                    "" if not element_energies else json.dumps(element_energies),
                    "",  # preflight flag (empty = normal run)
                ]
            )
        header = f"{_clean_env(env_name).upper()} → {model_name} | device = {device}"
        cmd_display = _format_worker_command(cmd, worker_path)
        extra = _reference_energy_summary(element_energies)
        return _RunJob(
            env_name=env_name,
            model_name=model_name,
            device=device or "unknown",
            cmd=cmd,
            cmd_display=f"{header}\n{cmd_display}",
            extra=extra,
        ), None

    def persist_status(model_name: str, returncode: int) -> None:
        alias_key = _canonical_alias(model_name)
        try:
            set_model_status(alias_key, "ok" if returncode == 0 else "broken")
        except Exception:
            pass

    def run_job(job: _RunJob, *, capture_output: bool) -> _RunResult:
        try:
            res = _run_subprocess(job.cmd, capture_output=capture_output)
        except Exception as exc:
            return _RunResult(
                env_name=job.env_name,
                model_name=job.model_name,
                returncode=1,
                stderr=str(exc),
            )
        return _RunResult(
            env_name=job.env_name,
            model_name=job.model_name,
            returncode=res.returncode,
            stdout=getattr(res, "stdout", "") or "",
            stderr=getattr(res, "stderr", "") or "",
        )

    if parallel == 1:
        failed = False
        preflight_failed = False
        for env_name, model_name in selected_pairs:
            job, preflight_error = prepare_job(env_name, model_name)
            if preflight_error:
                console.print(f"[red]Failed to prepare reference energies for {env_name} ({model_name}).")
                if preflight_error.stdout:
                    console.print(preflight_error.stdout)
                if preflight_error.stderr:
                    console.print(preflight_error.stderr)
                persist_status(model_name, 1)
                failed = True
                preflight_failed = True
                continue
            if job is None:
                failed = True
                preflight_failed = True
                continue

            console.print(Panel(f"{job.cmd_display}{job.extra}", border_style="blue"))
            result = run_job(job, capture_output=False)
            persist_status(model_name, result.returncode)
            if result.returncode != 0:
                failed = True
                console.print(f"[red]Failed for {env_name} ({model_name})")
            else:
                console.print(f"[green]Success: {model_name}")
        return 2 if preflight_failed else (1 if failed else 0)

    jobs = []
    had_preflight_error = False
    for env_name, model_name in selected_pairs:
        job, preflight_error = prepare_job(env_name, model_name)
        if preflight_error:
            console.print(f"[red]Failed to prepare reference energies for {env_name} ({model_name}).")
            if preflight_error.stdout:
                console.print(preflight_error.stdout)
            if preflight_error.stderr:
                console.print(preflight_error.stderr)
            persist_status(model_name, 1)
            had_preflight_error = True
        elif job is not None:
            jobs.append(job)

    failed = had_preflight_error
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(run_job, job, capture_output=True): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            console.print(Panel(f"{job.cmd_display}{job.extra}", border_style="blue"))
            if result.stdout:
                console.print(result.stdout.rstrip())
            if result.stderr:
                console.print(result.stderr.rstrip())
            persist_status(job.model_name, result.returncode)
            if result.returncode != 0:
                failed = True
                console.print(f"[red]Failed for {job.env_name} ({job.model_name})")
            else:
                console.print(f"[green]Success: {job.model_name}")
    if failed:
        return 2
    return 0


def _interactive_edit_pairs(pairs: list[list[str]]):
    while True:
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
        table.add_column("#", style="dim")
        table.add_column("env", style="cyan")
        table.add_column("model", style="magenta")
        for i, (e, m) in enumerate(pairs, start=1):
            table.add_row(str(i), e, m)
        console.print(table)

        choice = Prompt.ask("(a)dd / (r)emove / (m)odify / (d)one", choices=["a", "r", "m", "d"], default="d")
        if choice == "d":
            return pairs
        if choice == "a":
            env = _clean_env(Prompt.ask("Environment name"))
            model = Prompt.ask("Model identifier")
            if [env, model] not in pairs:
                pairs.append([env, model])
        elif choice == "m":
            if not pairs:
                console.print("No entries to edit.")
                continue
            idx = int(Prompt.ask("Index to edit", default="1")) - 1
            if 0 <= idx < len(pairs):
                env = _clean_env(Prompt.ask("Environment name", default=pairs[idx][0]))
                model = Prompt.ask("Model identifier", default=pairs[idx][1])
                pairs[idx] = [env, model]
        elif choice == "r":
            if not pairs:
                console.print("No entries to remove.")
                continue
            raw = Prompt.ask("Index(es) to remove (e.g. 1 3,5)", default="1")
            tokens = [t for t in re.split(r"[\s,]+", (raw or "").strip()) if t]
            try:
                idxs = sorted({int(t) - 1 for t in tokens}, reverse=True)
            except ValueError:
                console.print("[red]Invalid indices.[/red]")
                continue
            removed = 0
            for idx in idxs:
                if 0 <= idx < len(pairs):
                    pairs.pop(idx)
                    removed += 1
            if removed == 0:
                console.print("No valid indices removed.")
            else:
                console.print(f"Removed {removed} entr{'y' if removed == 1 else 'ies'}.")


def cmd_configure(args: argparse.Namespace) -> int:
    console.print(
        Panel(
            "[bold]Interactive configuration[/bold]\nConfigure base env, scan directories and select models.",
            border_style="blue",
        )
    )
    cfg = load_config()

    base_env = Prompt.ask(
        "Base IP-Orch environment (e.g. mlip):", default=cfg.get("base_env", os.environ.get("CONDA_DEFAULT_ENV", ""))
    )

    default_envs_dir = _guess_envs_dir()
    envs_base_dir = Prompt.ask(
        "Base MLIPs environments directory (e.g. ~/miniconda3/envs)", default=cfg.get("envs_base_dir", default_envs_dir)
    )

    discovered = _discover_envs_from_dir(envs_base_dir)
    current_env = _current_conda_env()
    if current_env:
        discovered.add(current_env)
    discovered |= set(_discover_conda_envs())
    discovered_list = sorted(discovered)
    if discovered_list:
        table_envs = Table(box=box.SIMPLE_HEAVY, header_style="bold")
        table_envs.add_column("env", style="cyan")
        table_envs.add_column("match", style="magenta")
        for e in discovered_list:
            token = _match_known_token(e)
            table_envs.add_row(e, token or "-")
        console.print(table_envs)
    else:
        console.print("No environments found under base directory.")

    proposed = []
    for env in discovered_list:
        match_token = _match_known_token(env)
        if match_token:
            variants = PACKAGE_VARIANTS.get(match_token)
            if variants:
                for model_alias in variants:
                    proposed.append([_clean_env(env), _canonical_alias(model_alias)])
            else:
                proposed.append([_clean_env(env), _canonical_alias(DEFAULT_MODELS_BY_ENV[match_token])])

    if proposed:
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
        table.add_column("env", style="cyan")
        table.add_column("suggested model", style="magenta")
        for e, m in proposed:
            table.add_row(e, m)
        console.print(table)

        if Confirm.ask("Add all suggested pairs?", default=True):
            pairs = proposed[:]
        else:
            pairs = []
    else:
        pairs = []

    existing = cfg.get("full_models", [])
    for env, model in existing:
        alias = _canonical_alias(model)
        clean_env = _clean_env(env)
        if [clean_env, alias] not in pairs:
            pairs.append([clean_env, alias])

    pairs = _interactive_edit_pairs(pairs)
    pairs = _dedup_pairs(pairs)

    cfg["base_env"] = base_env
    cfg["full_models"] = pairs
    cfg["envs_base_dir"] = envs_base_dir
    save_config(cfg)
    console.print("Configuration saved")

    grouped = _group_pairs(cfg["full_models"]) if cfg.get("full_models") else {}
    final = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    final.add_column("#", style="dim")
    final.add_column("env", style="cyan")
    final.add_column("model", style="magenta")
    for idx, (env, models) in enumerate(sorted(grouped.items()), start=1):
        final.add_row(str(idx), env, ",".join(models))
    console.print(final)
    return 0
