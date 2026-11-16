# Portable silicon integration benchmark

This is an integration calculation, not a DFT accuracy or speed ranking.
`silicon.extxyz` is a two-atom diamond Si primitive cell at a=5.43 Å with one
atom displaced by 0.03 Å. Periodic boundary conditions apply in all directions.
Energy is in eV, forces in eV/Å and stress in eV/Å³.

Create separate environments and install the same checkout in each:

```sh
python3.11 -m venv /tmp/mace-env
/tmp/mace-env/bin/pip install . mace-torch==0.3.14
python3.11 -m venv /tmp/orb-env
/tmp/orb-env/bin/pip install . orb-models==0.5.5
MLIP_DEVICE=cpu /tmp/mace-env/bin/python examples/benchmark/run.py mace-mp-0
MLIP_DEVICE=cpu /tmp/orb-env/bin/python examples/benchmark/run.py orb-v3
```

The first execution downloads upstream checkpoints and needs internet access.
Record checkpoint hashes and retain pip freeze output with results. Backend
packages may download large PyTorch dependencies; use suitable CPU/GPU wheels.

To exercise orchestration, register each environment and reuse the script:

```sh
ip-orch --add /tmp/mace-env mace-mp-0
ip-orch --add /tmp/orb-env orb-v3
ip-orch --run examples/benchmark/run.py --models mace-mp-0,orb-v3 --no-energy-correction --parallel 2
```

Each run creates an exclusive UUID JSON file under `benchmark-results` (override
with IPORCH_BENCHMARK_OUTPUT). Files include versions, input hash, energy,
forces, stress and a central finite-difference force check with h=0.001 Å.
Check finite values, force shape (2,3), identical input hashes and conservative
force difference below 0.05 eV/Å. This tolerance is an integration criterion,
not an accuracy claim. ORB-v2 is nonconservative and is not subject to that
criterion. Sequential execution is preferable on memory-limited hardware.

Historical surface notebook evidence is separately preserved in
`notebook_evidence.json`; its numbers are not references for this silicon task.

## Recorded execution

`results/` contains logs, JSON and environment metadata from this audit.
MACE-MP-0 (mace-torch 0.3.14, CPU) produced energy -10.72087669 eV and a
finite-difference force discrepancy of 0.000129 eV/Å. Inspect the JSON for full
precision and input hash. Checkpoint identity is recorded separately. Numerical
outputs can differ across library versions and hardware; they are integration
references, not DFT ground truth.

ORB-v3 (orb-models 0.5.5, CPU) produced -10.82204437 eV and a
force discrepancy of 0.002614 eV/Å on the same input.
