import logging
import os
from pathlib import Path

from ase import Atoms
from ase.data import chemical_symbols

from ip_orch.core.results import append_csv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "calculator_name",
    "atomic_number",
    "symbol",
    "energy_ev",
    "status",
    "error",
]

CELL_ANGSTROM = 20.0
RESULTS_CSV = Path(__file__).resolve().parent / "isolated_atom_energies.csv"


def _isolated_atom(symbol):
    return Atoms(
        symbol,
        positions=[(CELL_ANGSTROM / 2, CELL_ANGSTROM / 2, CELL_ANGSTROM / 2)],
        cell=(CELL_ANGSTROM, CELL_ANGSTROM, CELL_ANGSTROM),
        pbc=False,
    )


def _csv_path():
    return Path(os.environ.get("IPORCH_ATOM_RESULTS_CSV", RESULTS_CSV)).expanduser()


def main(calculator_name, ase_calculator):
    csv_path = _csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    for atomic_number, symbol in enumerate(chemical_symbols[1:], start=1):
        row = {
            "calculator_name": calculator_name,
            "atomic_number": atomic_number,
            "symbol": symbol,
            "energy_ev": "",
            "status": "ok",
            "error": "",
        }

        try:
            atoms = _isolated_atom(symbol)
            atoms.calc = ase_calculator
            row["energy_ev"] = float(atoms.get_potential_energy())
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"

        append_csv(csv_path, row)

    logger.info("Appended isolated-atom energies for %s to %s", calculator_name, csv_path)
