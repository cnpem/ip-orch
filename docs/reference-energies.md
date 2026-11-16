# Reference-energy provenance

Both bundled CSV files are legacy data. Model versions, checkpoint hashes,
precision, backend settings and original generating environments are not
recorded. Their presence does not make them validated physical references.
Do not use them for publication without regenerating and archiving provenance.

`reference_energies.csv` has one row per element and columns for model aliases;
values are energies in eV. Missing cells are unavailable, not zero.
`isolated_atom_energies.csv` records calculator, atomic number, symbol, energy
in eV and status/error. Error rows must not be interpreted as energies.
Use `ip-orch --check-elements MODEL` to inspect current reference coverage.

The regeneration script is `ip_orch/scripts/isolated_atom_energies.py`. It
places one atom at the center of a 20 Å cubic nonperiodic cell and attempts
all elements. Run it using a configured backend, for example:

```sh
IPORCH_ATOM_RESULTS_CSV=./new-isolated.csv ip-orch --run \
  ip_orch/scripts/isolated_atom_energies.py --models mace-mp-0 --no-energy-correction
```

Use sequential execution and a fresh output path. Archive `pip freeze`, Python,
hardware, device, checkpoint hash, dtype and all calculation settings alongside
the generated table. A successful isolated-atom calculation does not establish
physical validity outside a potential's training domain. Periodic-only models
or models requiring neighbors may fail for isolated atoms. Reference shifts
change absolute energies but do not improve interatomic forces or prove
formation-energy accuracy.


## Auditable regenerated subset

`examples/benchmark/reference_atoms.py` regenerates H, C and Si references for
one CPU backend. Run, for example:

```sh
python examples/benchmark/reference_atoms.py mace-mp-0 ./reference-results
```

Repeat in the ORB environment with orb-v3. Archived `reference-*.json` files in
examples/benchmark/results include positions, cell, periodicity, units, versions,
model and errors. Checkpoint hashes and environment records are alongside them.
These are a new explicit subset and do not establish provenance for the legacy
CSV values or demonstrate accuracy for isolated atoms outside training domains.
