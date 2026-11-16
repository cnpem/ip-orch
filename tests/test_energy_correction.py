import pytest

from ip_orch.core.energy_correction import (
    apply_linear_correction,
    compute_element_reference_energy_shift,
    wrap_linear_energy_correction,
    wrap_reference_energy_correction,
)


def test_apply_linear_correction_optional_noop():
    """Leave energy unchanged unless both linear coefficients are provided."""

    assert apply_linear_correction(energy=10.0, natoms=5, a=None, b=1.0) == 10.0
    assert apply_linear_correction(energy=10.0, natoms=5, a=2.0, b=None) == 10.0


def test_apply_linear_correction_total_energy():
    """Apply E_corr = a * E + b in total-energy mode."""

    assert apply_linear_correction(energy=10.0, natoms=5, a=2.0, b=1.0, mode="total_energy") == 21.0


def test_apply_linear_correction_per_atom():
    """Apply the linear fit to per-atom energy and scale back to total energy."""

    # E=10, N=5 => eps=2; eps_corr=2*2+1=5 => E_corr=25
    assert apply_linear_correction(energy=10.0, natoms=5, a=2.0, b=1.0, mode="per_atom") == 25.0


def test_apply_linear_correction_per_atom_requires_positive_natoms():
    """Reject per-atom correction when there are no atoms."""

    with pytest.raises(ValueError):
        apply_linear_correction(energy=10.0, natoms=0, a=1.0, b=0.0, mode="per_atom")


def test_apply_linear_correction_rejects_unknown_mode():
    """Reject unsupported correction modes instead of silently changing energy."""

    with pytest.raises(ValueError):
        apply_linear_correction(energy=10.0, natoms=5, a=1.0, b=0.0, mode="bad")  # type: ignore[arg-type]


def test_compute_element_reference_energy_shift():
    """Sum per-element reference energies according to atom counts."""

    shift = compute_element_reference_energy_shift(["Cu", "Cu", "C"], element_energies={"Cu": -3.0, "C": -1.0})
    assert shift == pytest.approx(-7.0)


def test_compute_element_reference_energy_shift_requires_all_elements():
    """Raise a clear error when a required element reference is missing."""

    with pytest.raises(KeyError):
        compute_element_reference_energy_shift(["Cu", "Fe"], element_energies={"Cu": -3.0})


def test_wrap_linear_energy_correction_preserves_raw_energy_and_corrects_result():
    """Wrap an ASE calculator and expose corrected energy plus original MLIP energy."""

    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes

    class ConstantEnergyCalculator(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 10.0}

    atoms = ase.Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)])
    atoms.calc = wrap_linear_energy_correction(
        ConstantEnergyCalculator(),
        a=2.0,
        b=1.0,
        mode="per_atom",
    )

    assert atoms.get_potential_energy() == pytest.approx(22.0)
    assert atoms.calc.results["energy_mlip"] == pytest.approx(10.0)


def test_wrap_reference_energy_correction_subtracts_element_baseline():
    """Wrap an ASE calculator and subtract per-element reference energies."""

    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes

    class ConstantEnergyCalculator(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 5.0}

    atoms = ase.Atoms("H2O", positions=[(0, 0, 0), (0, 0, 1), (0, 1, 0)])
    atoms.calc = wrap_reference_energy_correction(
        ConstantEnergyCalculator(),
        element_energies={"H": 1.0, "O": 2.0},
    )

    assert atoms.get_potential_energy() == pytest.approx(1.0)
    assert atoms.calc.results["energy_mlip"] == pytest.approx(5.0)


@pytest.mark.parametrize("mode", ["total_energy", "per_atom"])
def test_linear_potential_has_consistent_derivatives(mode):
    import numpy as np
    from ase.build import bulk
    from ase.calculators.emt import EMT

    atoms = bulk("Cu", "fcc", a=3.7).repeat((2, 1, 1))
    atoms.positions[0, 0] += 0.05
    atoms.calc = EMT()
    raw_forces = atoms.get_forces()
    raw_stress = atoms.get_stress()
    raw_free = atoms.get_potential_energy(force_consistent=True)
    atoms.calc = wrap_linear_energy_correction(EMT(), a=2.0, b=1.0, mode=mode)
    np.testing.assert_allclose(atoms.get_forces(), 2 * raw_forces)
    np.testing.assert_allclose(atoms.get_stress(), 2 * raw_stress)
    expected = 2 * raw_free + (len(atoms) if mode == "per_atom" else 1)
    assert atoms.get_potential_energy(force_consistent=True) == pytest.approx(expected)
    h = 1e-5
    atoms.positions[0, 0] += h
    plus = atoms.get_potential_energy()
    atoms.positions[0, 0] -= 2 * h
    minus = atoms.get_potential_energy()
    assert -(plus - minus) / (2 * h) == pytest.approx(2 * raw_forces[0, 0], abs=1e-6)
