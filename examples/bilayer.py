import logging
import os
from pathlib import Path

from ase.build import graphene

from ip_orch.core.results import append_csv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main(calculator_name, ase_calculator):
    layer1 = graphene(formula="C2", a=2.46, size=(2, 2, 1), vacuum=15.0)
    layer2 = layer1.copy()

    distance = 3.35
    layer2.positions[:, 2] += distance

    bilayer = layer1 + layer2
    for atoms in (layer1, layer2, bilayer):
        atoms.set_cell(layer1.cell)
        atoms.set_pbc(True)

    bilayer.calc = ase_calculator
    e_bilayer = bilayer.get_potential_energy()

    layer1.calc = ase_calculator
    e_layer1 = layer1.get_potential_energy()

    layer2.calc = ase_calculator
    e_layer2 = layer2.get_potential_energy()

    e_binding = e_bilayer - (e_layer1 + e_layer2)
    e_binding_per_atom = e_binding / len(bilayer)

    csv_path = Path(os.environ.get("IPORCH_RESULTS_CSV", "graphene_bilayer_results.csv"))
    row = {
        "calculator_name": calculator_name,
        "n_atoms": len(bilayer),
        "distance_angstrom": distance,
        "bilayer_energy_ev": e_bilayer,
        "layer1_energy_ev": e_layer1,
        "layer2_energy_ev": e_layer2,
        "binding_energy_ev": e_binding,
        "binding_energy_ev_per_atom": e_binding_per_atom,
    }

    append_csv(csv_path, row)

    logger.info("Appended graphene bilayer result for %s to %s", calculator_name, csv_path)
