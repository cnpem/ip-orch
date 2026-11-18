import pytest

from ip_orch.core.reference_energies import load_precomputed_reference_energies, supported_reference_elements


def test_legacy_reference_energy_warns():
    """Retain compatibility without silently treating legacy values as validated."""
    with pytest.warns(UserWarning, match="legacy values with incomplete provenance"):
        values = load_precomputed_reference_energies("mace-mp", ["C"])
    assert values["C"] == pytest.approx(-1.6918)


def test_reference_coverage_is_numeric_availability():
    """Missing values are not reported as available."""
    assert "C" in supported_reference_elements("mace-mp")
    assert "Og" not in supported_reference_elements("mace-mp")
