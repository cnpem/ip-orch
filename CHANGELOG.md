# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Audit corrections
- Made linear potential transformations consistent for energy, free energy, forces and stress.
- Fixed nonzero exit propagation for sequential worker failures.
- Added locked CSV append with schema checks and concurrent-process regression coverage.
- Added portable silicon benchmark, opt-in integration tests, and research evidence extracted from the saved surface notebook.
- Validated MACE 0.3.14 and ORB 0.5.5 in isolated CPU environments and a parallel two-model orchestration run.
- Corrected project MIT attribution while preserving the historical Bootstrap notice in THIRD_PARTY_NOTICES.md.
- Pinned Docker MACE source to the verified v0.3.14 commit.
- Renamed the manuscript to paper/paper.md and added required sections with explicit author evidence gaps.
- Added API, model, reference provenance and release documentation.
- Extended CI coverage to Python 3.9 and 3.13, with independent matrix failures.
- Fixed Ruff import ordering and formatting in model builders.
- Updated repository links and replaced the static build badge with the CI badge.
- Added contribution and support guidelines and an explicit evidence-gap ledger.
- Local Python 3.9–3.13 validation: 78 tests passed and one opt-in integration
  test skipped in each environment; real backend validation is reported separately.

### Changes recorded 2026-04-26

### Added
- Added a programmatic Python API through `IPOrch`, including direct callable execution for `run(...)` and class methods for CLI-equivalent actions.
- Added notebook-friendly callable execution by materializing Python logic functions into temporary worker scripts.
- Added `ase` to package dependencies.
- Added a registry-based `ModelFactory.register(...)` API and `ip_orch.model_builders` entry point support.
- Added `--parallel N` for running selected model benchmarks concurrently.
- Added CSV append output to the graphene bilayer example, configurable with `IPORCH_RESULTS_CSV`.
- Added documented pytest coverage for CLI orchestration, config recovery, helper normalization, environment discovery, plugin registration, repository mapping, examples, and energy correction wrappers.

### Changed
- Refactored built-in model creation logic into `ip_orch/core/model_builders.py`, replacing the monolithic `ModelFactory` if-chain with a registry.
- Resolved the worker script path via package resources so installed packages no longer depend on a source checkout layout.
- Updated `requires-python` and Ruff target from Python 3.8 to Python 3.9 to match the package's type syntax.
- Replaced worker/example `print()` calls with structured `logging`.
- Tightened repository URL alias matching to exact/prefix checks to avoid accidental cross-family matches.
- Moved the parallel execution notice into the initial run summary panel.
- Updated the graphene bilayer example to use all-axis periodic boundary conditions for ORB compatibility.
- Added pytest collection metadata so `pytest -vv` displays each test's docstring summary alongside the test id.
- Ignored the default graphene bilayer CSV output in `.gitignore`.
- Fixed CLI alias canonicalization so underscore and dash variants map to the same supported model alias.
- Validated model aliases during `add_model` so unsupported aliases are rejected with a warning.
- Preserved path-like virtualenv names such as `./mace` and resolved them to their Python executables instead of passing them to Conda as environment names.
- Sanitized notebook Matplotlib inline backends in worker subprocesses by defaulting to `MPLBACKEND=Agg`.

### Changes recorded 2026-04-21

### Added
- Created `ip_orch/core/worker.py` to decouple execution logic from `commands.py`.
- Added `pytest`, `pytest-mock`, and `ruff` to `[project.optional-dependencies]` in `pyproject.toml`.
- Added new automated tests to the `tests/` directory covering:
  - Configuration parsing (`config_store.py`)
  - CLI operations (`main.py` & `commands.py`)
  - Model Factory resolution logic (`model_factory.py`)

### Changed
- Refactored `cmd_run` inside `cli/commands.py` to execute the pre-built `worker.py` module rather than generating string-based execution scripts at runtime.
- Modified `ModelFactory` mapping to construct directories dynamically via the `--models-path` CLI parameter (or falling back to `~/.ip-orch/models`) instead of relying on heavily localized, hardcoded paths.
- Applied standard Type Hinting (`typing.Optional`, `typing.Dict`, `typing.Any`) and `Callable` definitions effectively across internal components.
- Adjusted broad `except Exception:` clauses to throw exceptions more specifically to improve overall API debugging capabilities.
- Enforced complete formatting and code quality checks through `ruff` across the `ip_orch` package.
