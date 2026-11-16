# Contributing

Use a separate environment for development:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
```

CI covers Python 3.9–3.13 for the orchestrator. Backend support depends on
its own Python, PyTorch and accelerator requirements; passing unit tests does
not validate a real MLIP installation.

To add a model, implement a builder following `ModelBuildContext` and
`ModelBuilder` in `ip_orch/core/model_builders.py`. Register it with
`ModelFactory.register(key, builder, *aliases)` or the
`ip_orch.model_builders` entry point group. Keep backend imports inside the
builder so missing optional dependencies do not prevent other models loading.
Add alias and failure tests, document checkpoint requirements, and report a real
energy/force calculation with package versions before claiming validation.

Open a specific issue at https://github.com/cnpem/ip-orch/issues with the
expected behavior, a minimal reproducer, actual output, OS, Python and backend
versions. Remove credentials and private paths from logs. For scientific
results include structures, units, checkpoint identity and calculation settings.

Pull requests should explain the problem, behavior change and validation.
Include regression tests for behavior fixes and update documentation and the
Unreleased changelog. Request review from another collaborator; report checks
that could not be run. Never replace missing scientific evidence with estimates.
