"""Opt-in integration tests: IPORCH_REAL_MODEL=mace-mp-0 or orb-v3."""

import os

import numpy as np
import pytest
from ase.build import bulk

from ip_orch.core.model_factory import ModelFactory


@pytest.mark.integration
def test_real_calculator_energy_forces():
    alias = os.environ.get("IPORCH_REAL_MODEL")
    if not alias:
        pytest.skip("Set IPORCH_REAL_MODEL in an environment with the real backend")
    calc = ModelFactory.create(alias, device="cpu")
    assert calc is not None
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[0, 0] += 0.03
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (2, 3)
    assert np.isfinite(atoms.get_forces()).all()
