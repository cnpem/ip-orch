"""Check silicon result records; pass only explicitly selected JSON paths."""

import json
import math
import sys
from pathlib import Path


def validate(path):
    result = json.loads(Path(path).read_text())
    assert result["model"] in {"mace-mp-0", "orb-v3"}
    assert len(result["input_sha256"]) == 64
    assert math.isfinite(result["energy_eV"])
    assert len(result["forces_eV_per_A"]) == 2
    assert all(len(row) == 3 and all(map(math.isfinite, row)) for row in result["forces_eV_per_A"])
    assert all(map(math.isfinite, result["stress_eV_per_A3"]))
    assert result["force_difference_eV_per_A"] < 0.05, "force consistency exceeds 0.05 eV/Å"
    return result["input_sha256"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python validate_results.py RESULT.json [RESULT.json ...]")
    hashes = {validate(path) for path in sys.argv[1:]}
    assert len(hashes) == 1, "Input structures differ"
    print(f"Validated {len(sys.argv) - 1} records with identical input hashes")
