# API and environments

Install with `python -m pip install .`; develop with `python -m pip install -e
'.[dev]'`. Install IP-Orch in each worker environment as well. Python 3.9–3.13
is the orchestrator target; individual MLIPs impose additional restrictions.
CPU is sufficient for the minimal example. GPU runs require a backend-compatible
PyTorch/CUDA installation; IP-Orch does not install drivers or allocate GPUs.

```python
from ip_orch import IPOrch, RunOptions

orch = IPOrch()
orch.add_model("/absolute/path/to/venv", "mace-mp-0")
orch.run("examples/benchmark/run.py", models=["mace-mp-0"], no_energy_correction=True, raise_on_error=True)
```

A script must define `main(calculator_name, ase_calculator)`. `run` also accepts
a callable with that signature. Callable extraction is for Python logic, not
arbitrary serialization of live notebook state; use a script for reproducibility.

`RunOptions` is an immutable dataclass accepted by `run_with_options`:

| Parameter | Format/default | Meaning |
| --- | --- | --- |
| script | path or callable; required | Scientific logic |
| envs, models | comma-separated string or iterable; None | Select one of these |
| parallel | integer, 1 | Maximum simultaneous workers |
| models_path | path, None | Checkpoint root; default `~/.ip-orch/models` |
| energy_linear_config | JSON path, None | Coefficients `a`, `b`, `mode` |
| energy_linear_a, energy_linear_b | float, None | Supply both or neither |
| energy_linear_mode | total_energy or per_atom | Offset per system or atom |
| correction_elements | string or iterable, None | Elements for reference shift |
| no_energy_correction | bool, False | Disable all corrections |
| reference_energy_source | computed or precomputed | Reference source |

`IPOrch.run` and `run_with_options` return None and accept `raise_on_error`.
The module-level equivalents return integer exit status; nonzero status raises
RuntimeError when requested. Prefer `raise_on_error=True` for automated work.

Other public methods: `add_model(env, model)` registers a pair;
`remove_model(env, model=None)` removes a pair or environment;
`supported_models(contains=None)` prints aliases;
`check_elements(model, elements=None)` checks reference coverage (not model
accuracy); `configure()` starts interactive environment discovery.

Environments can be absolute virtualenv paths or Conda names. Configuration is
stored by `ip_orch.config.config_store`; use `ip-orch --configure` to discover
and register environments. Check `ip-orch --help` for equivalent CLI flags.

## Errors and parallel execution

Missing Python executable: pass the environment directory containing bin/python
(or Scripts/python.exe), not a stale machine-specific path. Missing module:
install the backend inside that worker environment. Missing weights: provide
`models_path` and the filename in the model matrix. CUDA out of memory: use
sequential execution or limit devices externally with CUDA_VISIBLE_DEVICES.
There is no GPU memory scheduler; parallel workers each load their own model.

The parallel executor captures per-model stdout/stderr and returns nonzero if
any worker fails. Each scientific script must synchronize its own shared output
files. Prefer the benchmark's independent UUID JSON records. The bilayer CSV example uses a process lock and checks its schema. Other
scripts (including historical notebooks) must implement their own locking.

Real MACE/ORB parallel CUDA validation is archived in examples/benchmark/results/gpu.
Only this small silicon task was measured; the orchestrator still does not reserve
GPU memory. Conda installation was validated using Micromamba with
PYTHONNOUSERSITE=1 to exclude unrelated user-site packages.

## Tested Conda installation procedure

```sh
micromamba create -y -p /tmp/iporch-conda -c conda-forge python=3.11 pip
PYTHONNOUSERSITE=1 /tmp/iporch-conda/bin/python -m pip install .
PYTHONNOUSERSITE=1 /tmp/iporch-conda/bin/python -m pip check
cd /tmp
PYTHONNOUSERSITE=1 /tmp/iporch-conda/bin/ip-orch --help
```

`PYTHONNOUSERSITE=1` prevents unrelated packages in the user site directory from
contaminating dependency inspection. Worker environments need their own backend
installation; installing the orchestrator does not install MLIP dependencies.
