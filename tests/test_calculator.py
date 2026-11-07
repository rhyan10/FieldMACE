"""ASE calculator for field models."""
import ase
import numpy as np
import pytest
import torch

from mace.calculators import MACECalculator
from mace.tools.testing import build_field_model, make_batch, methane_in_charges


@pytest.fixture
def two_models(tmp_path):
    paths = []
    for seed in (0, 1):
        p = tmp_path / f"m{seed}.model"
        torch.save(build_field_model("PerAtomFieldEMACE", seed=seed, n_energies=2), p)
        paths.append(str(p))
    return paths


def _atoms(n_mm=8):
    sym, qm, mm, q = methane_in_charges(n_mm, seed=3)
    a = ase.Atoms(sym, positions=qm)
    if n_mm:
        a.info["mm_positions"], a.info["mm_charges"] = mm, q
    return a, (sym, qm, mm, q)


def test_calculator_matches_model_and_selects_state(two_models):
    a, cfg = _atoms()
    m = torch.load(two_models[0], weights_only=False)
    ref = m(make_batch(m, [cfg]).to_dict())
    for state in (0, 1):
        calc = MACECalculator(two_models[0], device="cpu", default_dtype="float64", state=state)
        a.calc = calc
        assert a.get_potential_energy() == pytest.approx(ref["energy"][0, state].item(), abs=1e-10)
        assert np.allclose(a.get_forces(), ref["forces"][:, state].detach().numpy(), atol=1e-10)
        assert calc.results["energies"].shape == (2,)
        assert calc.results["mm_forces"].shape == (8, 2, 3)


def test_calculator_gas_phase_and_committee(two_models):
    a, _ = _atoms(n_mm=0)
    calc = MACECalculator(two_models, device="cpu", default_dtype="float64")
    a.calc = calc
    e = a.get_potential_energy()
    assert np.isfinite(e)
    assert calc.results["energy_var"].shape == (2,) and calc.results["forces_var"].shape == (5, 2, 3)
    assert calc.results["mm_forces"].shape == (0, 2, 3)


def test_calculator_bad_state(two_models):
    a, _ = _atoms()
    a.calc = MACECalculator(two_models[0], device="cpu", default_dtype="float64", state=5)
    with pytest.raises(ValueError):
        a.get_potential_energy()
