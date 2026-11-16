"""Single point and numerical-force validation on periodic silicon."""

import hashlib
import json
import os
import platform
import sys
import uuid
from importlib import metadata
from pathlib import Path

import numpy as np
from ase.io import read

from ip_orch.core.model_factory import ModelFactory


def main(calculator_name, ase_calculator):
    source = Path(__file__).with_name("silicon.extxyz")
    atoms = read(source)
    atoms.calc = ase_calculator
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces().copy()
    stress = atoms.get_stress().copy()
    h = 1e-3
    atoms.positions[0, 0] += h
    plus = atoms.get_potential_energy()
    atoms.positions[0, 0] -= 2 * h
    minus = atoms.get_potential_energy()
    numerical_force = float(-(plus - minus) / (2 * h))
    assert np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()
    assert forces.shape == (len(atoms), 3)
    # Conservative models should satisfy this; nonconservative models are recorded.
    versions = {}
    for package in ("ip-orch", "ase", "numpy", "torch", "mace-torch", "orb-models"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    result = dict(
        model=calculator_name,
        python=sys.version,
        platform=platform.platform(),
        versions=versions,
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        energy_eV=energy,
        forces_eV_per_A=forces.tolist(),
        stress_eV_per_A3=stress.tolist(),
        numerical_force_eV_per_A=numerical_force,
        force_difference_eV_per_A=float(abs(forces[0, 0] - numerical_force)),
        finite_difference_step_A=h,
        device=os.environ.get("MLIP_DEVICE", "cpu"),
        run_id=str(uuid.uuid4()),
    )
    target = Path(os.environ.get("IPORCH_BENCHMARK_OUTPUT", "benchmark-results"))
    target.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids shared CSV writes and accidental overwriting.
    with (target / f"{result['run_id']}.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    alias = sys.argv[1]
    main(alias, ModelFactory.create(alias, device=os.environ.get("MLIP_DEVICE", "cpu")))
