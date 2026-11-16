import argparse
import os
import subprocess
from unittest.mock import patch

from ip_orch.cli.commands import (
    _package_parent_path,
    _run_subprocess,
    _subprocess_env,
    _worker_resource_path,
    cmd_add,
    cmd_check_elements,
    cmd_models,
    cmd_run,
)
from ip_orch.cli.main import main


@patch("ip_orch.cli.commands.add_model")
def test_cmd_add(mock_add_model):
    """Route --add data to the config layer and return success."""

    args = argparse.Namespace(env="myenv", model="mace-mp-0")
    assert cmd_add(args) == 0
    mock_add_model.assert_called_once_with("myenv", "mace-mp-0")


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.add_model")
def test_cmd_add_rejects_unknown_model_alias(mock_add_model, mock_print):
    """Warn and refuse to register unsupported model aliases."""

    args = argparse.Namespace(env="mace", model="mace-mpa")

    assert cmd_add(args) == 1

    mock_add_model.assert_not_called()
    assert "Unknown model alias" in str(mock_print.call_args_list[0].args[0])


def test_subprocess_env_replaces_notebook_matplotlib_backend(monkeypatch):
    """Do not leak Jupyter's inline backend into worker subprocesses."""

    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    assert _subprocess_env()["MPLBACKEND"] == "Agg"


def test_subprocess_env_preserves_explicit_non_notebook_matplotlib_backend(monkeypatch):
    """Respect explicit Matplotlib backends that can be used outside notebooks."""

    monkeypatch.setenv("MPLBACKEND", "svg")

    assert _subprocess_env()["MPLBACKEND"] == "svg"


@patch("ip_orch.cli.commands.subprocess.run")
def test_run_subprocess_passes_sanitized_environment(mock_run, monkeypatch):
    """Pass the cleaned subprocess env to worker commands."""

    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

    _run_subprocess(["python", "worker.py"], capture_output=True)

    assert mock_run.call_args.kwargs["env"]["MPLBACKEND"] == "Agg"


@patch("ip_orch.cli.commands.console.print")
def test_cmd_models(mock_print):
    """Render the supported-models table using configured aliases first."""

    args = argparse.Namespace(contains=None)
    with patch("ip_orch.cli.commands.load_config", return_value={"full_models": [["env1", "orb-v3"]]}):
        assert cmd_models(args) == 0
        # Should have built the table and printed
        assert mock_print.called


@patch("ip_orch.cli.main.cmd_add", return_value=0)
def test_main_routing(mock_cmd_add):
    """Dispatch the top-level --add command to cmd_add."""

    # Main should route correctly
    assert main(["--add", "myenv", "mod"]) == 0
    mock_cmd_add.assert_called_once()


@patch("ip_orch.cli.main.cmd_run", return_value=0)
def test_main_forwards_parallel(mock_cmd_run):
    """Parse --parallel and forward it to cmd_run."""

    assert (
        main(
            [
                "--run",
                "script.py",
                "--models",
                "orb-v3",
                "--parallel",
                "3",
                "--models-path",
                "/home/p.zanineli/pretrained",
            ]
        )
        == 0
    )

    args = mock_cmd_run.call_args.args[0]
    assert args.parallel == 3
    assert args.models_path == "/home/p.zanineli/pretrained"


@patch("ip_orch.cli.main.cmd_check_elements", return_value=0)
def test_main_routes_check_elements(mock_cmd_check_elements):
    """Dispatch --check-elements to the element support command."""

    assert main(["--check-elements", "mace-mp", "C,Cu"]) == 0

    args = mock_cmd_check_elements.call_args.args[0]
    assert args.model == "mace-mp"
    assert args.elements == "C,Cu"


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._python_for_env", return_value="/env/python")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_parallel_executes_all_models(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_python_for_env,
    mock_set_model_status,
    mock_print,
):
    """Run all selected models with captured output when --parallel is greater than one."""

    mock_load_config.return_value = {
        "full_models": [["env1", "orb-v3"], ["env2", "mace-mp"]],
        "models_path": "/models",
        "envs_base_dir": "/envs",
    }
    mock_run_subprocess.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")
    args = argparse.Namespace(
        script="script.py",
        envs=None,
        models="orb-v3,mace-mp",
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements=None,
        no_energy_correction=False,
        parallel=2,
    )

    assert cmd_run(args) == 0

    assert mock_run_subprocess.call_count == 4
    assert mock_set_model_status.call_count == 2


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._python_for_env", return_value="/env/python")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_sequential_uses_streaming_subprocess_output(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_python_for_env,
    mock_set_model_status,
    mock_print,
):
    """Run one selected model at a time with live subprocess output by default."""

    mock_load_config.return_value = {
        "full_models": [["env1", "orb-v3"]],
        "models_path": "/models",
        "envs_base_dir": "/envs",
    }
    mock_run_subprocess.side_effect = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout="IPORCH_DEVICE=cpu\n", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0),
    ]
    args = argparse.Namespace(
        script="script.py",
        envs="env1",
        models=None,
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements=None,
        no_energy_correction=False,
        parallel=1,
    )

    assert cmd_run(args) == 0

    assert mock_run_subprocess.call_count == 2
    assert mock_run_subprocess.call_args.kwargs["capture_output"] is False
    run_panel = mock_print.call_args_list[1].args[0]
    assert "ENV1 → orb-v3 | device = cpu" in run_panel.renderable
    mock_set_model_status.assert_called_once_with("orb-v3", "ok")


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_uses_python_from_path_like_virtualenv(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_set_model_status,
    mock_print,
    tmp_path,
    monkeypatch,
):
    """Treat configured env paths like ./mace as virtualenv paths, not Conda env names."""

    python_bin = tmp_path / "mace" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    python_bin.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    mock_load_config.return_value = {
        "full_models": [["./mace", "mace-mp"]],
        "models_path": "/models",
        "envs_base_dir": "",
    }
    mock_run_subprocess.side_effect = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout="IPORCH_DEVICE=cpu\n", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0),
    ]
    args = argparse.Namespace(
        script="script.py",
        envs="./mace",
        models=None,
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements=None,
        no_energy_correction=False,
        reference_energy_source="computed",
        parallel=1,
    )

    assert cmd_run(args) == 0

    first_cmd = mock_run_subprocess.call_args_list[0].args[0]
    assert first_cmd[0] == str(python_bin)
    assert "conda" not in first_cmd


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._python_for_env", return_value="/env/python")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_precomputed_reference_energies_skip_preflight(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_python_for_env,
    mock_set_model_status,
    mock_print,
):
    """Use the bundled CSV reference energies instead of computing isolated atoms."""

    mock_load_config.return_value = {
        "full_models": [["env1", "mace-mp"]],
        "models_path": "/models",
        "envs_base_dir": "/envs",
    }
    mock_run_subprocess.side_effect = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout="IPORCH_DEVICE=cpu\n", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0),
    ]
    args = argparse.Namespace(
        script="script.py",
        envs="env1",
        models=None,
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements="C",
        no_energy_correction=False,
        reference_energy_source="precomputed",
        parallel=1,
    )

    assert cmd_run(args) == 0

    assert mock_run_subprocess.call_count == 2
    cmd = mock_run_subprocess.call_args.args[0]
    assert '"C": -1.6918' in cmd[11]


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._python_for_env", return_value="/env/python")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_precomputed_reference_energies_without_elements_enables_all_supported(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_python_for_env,
    mock_set_model_status,
    mock_print,
):
    """Treat --reference-energy-source precomputed as enabling reference correction."""

    mock_load_config.return_value = {
        "full_models": [["env1", "mace-mp"]],
        "models_path": "/models",
        "envs_base_dir": "/envs",
    }
    mock_run_subprocess.side_effect = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout="IPORCH_DEVICE=cpu\n", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0),
    ]
    args = argparse.Namespace(
        script="script.py",
        envs="env1",
        models=None,
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements=None,
        no_energy_correction=False,
        reference_energy_source="precomputed",
        parallel=1,
    )

    assert cmd_run(args) == 0

    cmd = mock_run_subprocess.call_args.args[0]
    assert cmd[10] == ""
    assert '"Cu":' in cmd[11]
    run_panel = mock_print.call_args_list[1].args[0]
    assert "<89 reference energies>" in run_panel.renderable
    assert '"Cu":' not in run_panel.renderable


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._python_for_env", return_value="/env/python")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_rejects_computed_reference_energies_for_grace(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_python_for_env,
    mock_set_model_status,
    mock_print,
):
    """Do not try to compute isolated atom references for unsupported calculator families."""

    mock_load_config.return_value = {
        "full_models": [["env1", "grace-2l-oam"]],
        "models_path": "/models",
        "envs_base_dir": "/envs",
    }
    args = argparse.Namespace(
        script="script.py",
        envs="env1",
        models=None,
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements="C",
        no_energy_correction=False,
        reference_energy_source="computed",
        parallel=1,
    )

    assert cmd_run(args) == 2

    mock_run_subprocess.assert_not_called()
    mock_set_model_status.assert_called_once_with("grace-2l-oam", "broken")


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._python_for_env", return_value="/env/python")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_preflight_failure_marks_model_broken(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_python_for_env,
    mock_set_model_status,
    mock_print,
):
    """Stop execution and mark a model broken when element-energy preflight fails."""

    mock_load_config.return_value = {
        "full_models": [["env1", "orb-v3"]],
        "models_path": "/models",
        "envs_base_dir": "/envs",
    }
    mock_run_subprocess.side_effect = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout="IPORCH_DEVICE=cpu\n", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=9, stdout="", stderr="preflight failed"),
    ]
    args = argparse.Namespace(
        script="script.py",
        envs="env1",
        models=None,
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements="C",
        no_energy_correction=False,
        parallel=1,
    )

    assert cmd_run(args) == 2

    mock_set_model_status.assert_called_once_with("orb-v3", "broken")


@patch("ip_orch.cli.commands.console.print")
@patch("ip_orch.cli.commands.set_model_status")
@patch("ip_orch.cli.commands._python_for_env", return_value="/env/python")
@patch("ip_orch.cli.commands._worker_resource_path", return_value="/pkg/worker.py")
@patch("ip_orch.cli.commands._package_parent_path", return_value="/pkg")
@patch("ip_orch.cli.commands.load_config")
@patch("ip_orch.cli.commands._run_subprocess")
def test_cmd_run_parallel_summary_includes_parallel_notice(
    mock_run_subprocess,
    mock_load_config,
    mock_package_parent_path,
    mock_worker_resource_path,
    mock_python_for_env,
    mock_set_model_status,
    mock_print,
):
    """Show the parallel execution notice in the initial model summary panel."""

    mock_load_config.return_value = {
        "full_models": [["env1", "orb-v3"], ["env2", "mace-mp"]],
        "models_path": "/models",
        "envs_base_dir": "/envs",
    }
    mock_run_subprocess.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    args = argparse.Namespace(
        script="script.py",
        envs=None,
        models="orb-v3,mace-mp",
        energy_linear_config=None,
        energy_linear_a=None,
        energy_linear_b=None,
        energy_linear_mode=None,
        correction_elements=None,
        no_energy_correction=False,
        parallel=2,
    )

    assert cmd_run(args) == 0

    first_panel = mock_print.call_args_list[0].args[0]
    assert "IP-Orch: starting execution | running up to 2 models in parallel" in first_panel.renderable


@patch("ip_orch.cli.commands.console.print")
def test_cmd_check_elements_reports_missing_elements(mock_print):
    """Check element support using the precomputed reference table."""

    assert cmd_check_elements(argparse.Namespace(model="mace-mp", elements="C,Og")) == 1

    panel = mock_print.call_args.args[0]
    assert panel.title == "mace-mp | supported elements for reference energy"
    assert str(panel.border_style) == "red"


def test_worker_resource_path():
    """Resolve package resource paths used to invoke the worker script."""

    assert _worker_resource_path().endswith(os.path.join("ip_orch", "core", "worker.py"))
    assert _package_parent_path()


def test_sequential_worker_failure_returns_nonzero(monkeypatch):
    monkeypatch.setattr("ip_orch.cli.commands.load_config", lambda: {"full_models": [["env", "orb-v3"]]})
    monkeypatch.setattr("ip_orch.cli.commands._python_for_env", lambda *a: "/env/python")
    monkeypatch.setattr("ip_orch.cli.commands.set_model_status", lambda *a: None)
    results = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="IPORCH_DEVICE=cpu\n"),
            subprocess.CompletedProcess([], 7),
        ]
    )
    monkeypatch.setattr("ip_orch.cli.commands._run_subprocess", lambda *a, **kw: next(results))
    args = argparse.Namespace(script="script.py", models="orb-v3", envs=None, parallel=1)
    assert cmd_run(args) != 0


def test_parallel_mixed_success_and_failure(monkeypatch):
    monkeypatch.setattr(
        "ip_orch.cli.commands.load_config", lambda: {"full_models": [["env1", "orb-v3"], ["env2", "mace-mp"]]}
    )
    monkeypatch.setattr("ip_orch.cli.commands._python_for_env", lambda *a: "/env/python")
    states = {}
    monkeypatch.setattr("ip_orch.cli.commands.set_model_status", lambda k, v: states.update({k: v}))

    def execute(cmd, **kwargs):
        if cmd[-1] == "device":
            return subprocess.CompletedProcess(cmd, 0, stdout="IPORCH_DEVICE=cpu\n", stderr="")
        return subprocess.CompletedProcess(cmd, 9 if "orb-v3" in cmd else 0, stdout="", stderr="failure")

    monkeypatch.setattr("ip_orch.cli.commands._run_subprocess", execute)
    args = argparse.Namespace(script="script.py", models="orb-v3,mace-mp", envs=None, parallel=2)
    assert cmd_run(args) != 0
    assert states == {"orb-v3": "broken", "mace-mp": "ok"}
