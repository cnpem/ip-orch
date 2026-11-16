"""Regenerate selected isolated-atom references with explicit provenance."""

import hashlib
import json
import platform
import sys
import uuid
from importlib import metadata
from pathlib import Path

from ase import Atoms

from ip_orch.core.model_factory import ModelFactory


def main(alias, output):
    calc = ModelFactory.create(alias, device="cpu")
    rows = []
    for symbol in ("H", "C", "Si"):
        atoms = Atoms(symbol, positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=False)
        atoms.calc = calc
        row = {"element": symbol}
        try:
            row.update(energy_eV=float(atoms.get_potential_energy()), status="ok")
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    versions = {}
    for package in ("ip-orch", "ase", "torch", "mace-torch", "orb-models"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    import torch

    result = dict(
        calculator_class=f"{type(calc).__module__}.{type(calc).__name__}",
        dtype=str(torch.get_default_dtype()),
        matmul_precision=torch.get_float32_matmul_precision(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=alias,
        device="cpu",
        python=sys.version,
        platform=platform.platform(),
        versions=versions,
        cell_A=[20, 20, 20],
        position_A=[10, 10, 10],
        pbc=False,
        units="eV",
        correction=None,
        rows=rows,
    )
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"reference-{alias}-{uuid.uuid4()}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
