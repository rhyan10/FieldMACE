# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Diabatic models: mace.tools.diabatic, the model outputs, the ASE calculator, training, OpenMM."""
import os
import subprocess
import sys

import ase
import ase.io
import numpy as np
import pytest
import torch

from mace.tools.diabatic import (adiabatic_energies_torch, adiabatize, align_columns, labels_from_adiabatic,
                                 n_outputs, n_states_from_outputs, pack, state_pairs, unpack)
from mace.tools.testing import build_field_model, make_batch, methane_in_charges

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# mace.tools.diabatic
# --------------------------------------------------------------------------- #
def model_hamiltonian(x, n=3):
    """A smooth symmetric H(x) for x in R^(N*3), with an avoided crossing near x = 0."""
    x = np.asarray(x).ravel()
    rng = np.random.default_rng(7)
    A = rng.normal(size=(n, n, x.size))
    B = rng.normal(size=(n, n, x.size)) * 0.1
    H = np.diag(np.linspace(0.0, 0.3, n)) + np.einsum("ija,a->ij", A, x) + np.einsum("ija,a->ij", B, x ** 2)
    return 0.5 * (H + H.T)


def num_dH(x, n=3, h=1e-6):
    dH = np.zeros((n, n, x.size))
    for a in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[a] += h
        xm[a] -= h
        dH[:, :, a] = (model_hamiltonian(xp, n) - model_hamiltonian(xm, n)) / (2 * h)
    return dH.reshape(n, n, -1, 3)


def test_pack_unpack_and_counts():
    assert [n_outputs(s) for s in (1, 2, 3, 4)] == [1, 3, 6, 10]
    assert [n_states_from_outputs(k) for k in (1, 3, 6, 10)] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        n_states_from_outputs(7)
    H = model_hamiltonian(np.full(6, 0.05), 4)
    v = pack(H)
    assert v.shape == (10,) and np.allclose(v[:4], np.diag(H)) and np.isclose(v[4], H[0, 1]) and np.isclose(v[9], H[2, 3])
    assert np.allclose(unpack(v, 4), H)
    vt = torch.tensor(np.stack([v, 2 * v]))
    Ht = unpack(vt, 4)
    assert Ht.shape == (2, 4, 4) and np.allclose(Ht[1].numpy(), 2 * H)
    assert np.allclose(adiabatic_energies_torch(vt, 4)[0].numpy(), np.linalg.eigvalsh(H))
    assert state_pairs(3) == [[0, 1], [0, 2], [1, 2]]


def test_adiabatize_matches_finite_differences():
    n = 3
    x = np.array([0.03, -0.02, 0.01, 0.05, 0.0, -0.04])
    H = model_hamiltonian(x, n)
    ad = adiabatize(H, num_dH(x, n))
    E, U = ad["energies"], ad["U"]
    assert np.allclose(U @ np.diag(E) @ U.T, H)
    h = 1e-6
    for a in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[a] += h
        xm[a] -= h
        ap = adiabatize(model_hamiltonian(xp, n), U_ref=U)
        am = adiabatize(model_hamiltonian(xm, n), U_ref=U)
        grad_fd = (ap["energies"] - am["energies"]) / (2 * h)
        assert np.allclose(ad["grad"].reshape(n, -1)[:, a], grad_fd, atol=1e-6)
        dU = (ap["U"] - am["U"]) / (2 * h)  # d_kl = <k|d/dR|l> = (U^T dU/dR)_kl
        assert np.allclose(ad["nac"].reshape(n, n, -1)[:, :, a], (U.T @ dU), atol=1e-5)
    assert np.allclose(ad["nac"], -np.transpose(ad["nac"], (1, 0, 2, 3)))


def test_labels_from_adiabatic_roundtrip_and_blocks():
    n = 4
    x = np.linspace(-0.05, 0.05, 9)
    H = model_hamiltonian(x, n)
    blocks = [[0, 1], [2, 3]]
    H[np.ix_([0, 1], [2, 3])] = H[np.ix_([2, 3], [0, 1])] = 0.0  # no coupling between the blocks
    dH = num_dH(x, n)
    dH[np.ix_([0, 1], [2, 3])] = dH[np.ix_([2, 3], [0, 1])] = 0.0
    dH_pc = dH[:, :, :2] * 0.3
    ad = adiabatize(H, dH, dH_pc, blocks=blocks)
    assert np.allclose(ad["U"][np.ix_([0, 1], [2, 3])], 0) and np.allclose(ad["nac"][0, 2], 0)
    lab = labels_from_adiabatic(ad["energies"], ad["U"], ad["grad"], ad["nac"], ad["grad_pc"], ad["nac_pc"])
    assert np.allclose(lab["H"], H) and np.allclose(lab["dH"], dH) and np.allclose(lab["dH_pc"], dH_pc)
    with pytest.raises(ValueError):
        adiabatize(H, blocks=[[0, 1], [1, 2, 3]])


def test_align_columns():
    U = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    flipped = U * np.array([1, -1, -1])[None, :]
    assert np.allclose(align_columns(flipped, U), U)
    V = align_columns(U, None)
    assert np.all(V[np.argmax(np.abs(V), axis=0), range(3)] > 0)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dia_model():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    m = build_field_model("PerAtomFieldEMACE", seed=5, n_energies=3, diabatic=True, compute_dipoles=True)
    yield m
    torch.set_default_dtype(old)


def test_model_outputs_and_e0_on_diagonal_only(dia_model):
    assert dia_model.n_states == 3 and dia_model.n_energies == 6 and dia_model.diabatic
    cfg = [methane_in_charges(12, seed=1)]
    out = dia_model(make_batch(dia_model, cfg).to_dict())
    assert out["energy"].shape == (1, 6) and out["forces"].shape == (5, 6, 3) and out["mm_forces"].shape == (12, 6, 3)
    assert out["dipoles"].shape[-2:] == (6, 3)
    # shifting the atomic reference energies changes only the diagonal
    m2 = build_field_model("PerAtomFieldEMACE", seed=5, n_energies=3, diabatic=True, compute_dipoles=True,
                           atomic_energies=np.linspace(-1.0, -4.0, 4) - 3.0)
    out2 = m2(make_batch(m2, cfg).to_dict())
    diff = (out2["energy"] - out["energy"]).detach().numpy().ravel()
    assert np.all(np.abs(diff[:3]) > 1.0) and np.allclose(diff[3:], 0)
    with pytest.raises(ValueError):
        build_field_model("PerAtomFieldEMACE", n_energies=3, diabatic=True, compute_nacs=True)
    with pytest.raises(ValueError):
        build_field_model("PerAtomFieldEMACE", n_energies=1, diabatic=True)


def test_couplings_can_be_parity_odd(dia_model):
    """Mirroring the whole system (molecule and charges) leaves the diagonal of H unchanged; the couplings get a
    pseudoscalar part (states of opposite mirror symmetry) - without it, A1-A2 couplings could not be learned."""
    from e3nn import o3

    sym, qm, mm, q = methane_in_charges(12, seed=4)
    qm = qm + np.random.default_rng(0).normal(scale=0.1, size=qm.shape)
    out = {}
    for name, M in (("id", np.eye(3)), ("rot", o3.rand_matrix().numpy()), ("mirror", np.diag([-1.0, 1.0, 1.0]))):
        o = dia_model(make_batch(dia_model, [(sym, qm @ M.T, mm @ M.T, q)]).to_dict())
        out[name] = (o["energy"].detach().numpy().ravel(), o["dipoles"].detach().numpy().reshape(-1, 3), M)
    e, d, _ = out["id"]
    er, dr, R = out["rot"]
    assert np.allclose(er, e, atol=1e-9) and np.allclose(dr, d @ R.T, atol=1e-9)  # rotations: invariant/covariant
    em, dm, P = out["mirror"]
    assert np.allclose(em[:3], e[:3], atol=1e-9)  # diagonal: true scalars
    odd = 0.5 * (e[3:] - em[3:])
    assert np.abs(odd).max() > 1e-6  # couplings: pseudoscalar part present
    assert np.allclose(dm[:3], d[:3] @ P.T, atol=1e-9)  # permanent dipoles: vectors
    assert not np.allclose(dm[3:], d[3:] @ P.T, atol=1e-6)  # transition dipoles: pseudovector part present


def test_model_compiles(dia_model):
    from e3nn.util import jit

    m = build_field_model("PerAtomFieldEMACE", seed=5, n_energies=3, diabatic=True, compute_dipoles=True)
    ref = dia_model(make_batch(dia_model, [methane_in_charges(5, seed=2)]).to_dict())["energy"]
    cm = jit.compile(m)  # compiles the submodules in place: never the shared fixture
    assert torch.allclose(cm(make_batch(m, [methane_in_charges(5, seed=2)]).to_dict())["energy"], ref)


def test_old_models_unpickle_as_adiabatic():
    m = build_field_model("PerAtomFieldEMACE", n_energies=2)
    state = m.__dict__.copy()
    for k in ("diabatic", "n_states"):
        state.pop(k, None)
    m2 = m.__class__.__new__(m.__class__)
    m2.__setstate__(state)
    assert m2.diabatic is False and m2.n_states == 2 and torch.all(m2.e0_mask == 1)


# --------------------------------------------------------------------------- #
# ASE calculator: adiabatic energies, forces, NACs on atoms and point charges
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dia_calc(dia_model, tmp_path_factory):
    from mace.calculators import MACECalculator

    p = tmp_path_factory.mktemp("dia") / "dia.model"
    torch.save(dia_model, p)
    return MACECalculator(str(p), device="cpu", default_dtype="float64")


def atoms_with_charges(seed=3, n_mm=10):
    sym, qm, mm, q = methane_in_charges(n_mm, seed=seed)
    at = ase.Atoms(sym, positions=qm)
    at.info.update(mm_positions=mm, mm_charges=q)
    return at


def test_calculator_adiabatic_results(dia_calc):
    at = atoms_with_charges()
    dia_calc._diabatic_U = None
    dia_calc.calculate(at)
    r = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in dia_calc.results.items()}
    H = r["diabatic_hamiltonian"]
    assert np.allclose(r["energies"], np.linalg.eigvalsh(H))
    assert r["forces_all"].shape == (5, 3, 3) and r["nacs"].shape == (5, 3, 3)
    assert r["mm_forces"].shape == (10, 3, 3) and r["mm_nacs"].shape == (10, 3, 3)
    assert r["dipoles"].shape == (6, 3) and "diabatic_dipoles" in r
    h = 1e-5
    U0 = r["U"]
    for a, k in ((0, 0), (1, 2)):
        e, Us = [], []
        for s in (1, -1):
            x = at.copy()
            x.info = dict(at.info)
            x.positions[a, k] += s * h
            dia_calc._diabatic_U = U0
            dia_calc.calculate(x)
            e.append(dia_calc.results["energies"].copy())
            Us.append(dia_calc.results["U"].copy())
        assert np.allclose(-(e[0] - e[1]) / (2 * h), r["forces_all"][a, :, k], atol=1e-6)
        nac_fd = U0.T @ (Us[0] - Us[1]) / (2 * h)
        assert np.allclose([nac_fd[i, j] for i, j in state_pairs(3)], r["nacs"][a, :, k], atol=1e-5)
    for j, k in ((0, 1), (7, 2)):
        e = []
        for s in (1, -1):
            x = at.copy()
            x.info = dict(at.info)
            mm = at.info["mm_positions"].copy()
            mm[j, k] += s * h
            x.info["mm_positions"] = mm
            dia_calc._diabatic_U = U0
            dia_calc.calculate(x)
            e.append(dia_calc.results["energies"].copy())
        assert np.allclose(-(e[0] - e[1]) / (2 * h), r["mm_forces"][j, :, k], atol=1e-6)


def test_calculator_blocks(dia_model, tmp_path):
    from mace.calculators import MACECalculator

    p = tmp_path / "dia.model"
    torch.save(dia_model, p)
    calc = MACECalculator(str(p), device="cpu", default_dtype="float64", diabatic_blocks=[[0], [1, 2]])
    calc.calculate(atoms_with_charges(seed=4))
    r = calc.results
    H = r["diabatic_hamiltonian"]
    assert np.isclose(r["energies"][0], H[0, 0])
    assert np.allclose(r["energies"][1:], np.linalg.eigvalsh(H[1:, 1:]))
    assert np.allclose(r["nacs"][:, :2], 0)  # pairs (0, 1), (0, 2) are in different blocks


# --------------------------------------------------------------------------- #
# training from the command line
# --------------------------------------------------------------------------- #
def test_train_diabatic(tmp_path):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    teacher = build_field_model("PerAtomFieldEMACE", seed=11, n_energies=2, diabatic=True, compute_dipoles=True)
    frames = []
    for i in range(20):
        n_mm = [0, 6, 15][i % 3]
        sym, qm, mm, q = methane_in_charges(n_mm, seed=i)
        qm = qm + np.random.default_rng(i).normal(scale=0.05, size=qm.shape)
        out = teacher(make_batch(teacher, [(sym, qm, mm, q)]).to_dict())
        a = ase.Atoms(sym, positions=qm)
        if n_mm:
            a.info.update(mm_positions=mm, mm_charges=q, REF_mm_forces=out["mm_forces"].detach().numpy())
        a.info["REF_energy"] = out["energy"].detach().numpy()
        a.info["REF_forces"] = out["forces"].detach().numpy()
        a.info["REF_dipoles"] = out["dipoles"].detach().numpy().reshape(3, 3)
        frames.append(a)
    torch.set_default_dtype(old)
    ase.io.write(tmp_path / "train.xyz", frames, format="extxyz")
    cmd = [
        sys.executable, os.path.join(REPO, "scripts", "run_train.py"),
        "--name=t", f"--train_file={tmp_path / 'train.xyz'}", "--valid_fraction=0.25", "--seed=1", "--E0s=average",
        "--model=PerAtomFieldEMACE", "--diabatic=True", "--n_energies=2", "--compute_dipoles=True",
        "--dipoles_key=REF_dipoles", "--r_max=5.0", "--batch_size=4", "--lr=0.01", "--correlation=2",
        "--max_num_epochs=2", "--default_dtype=float64", "--device=cpu", "--hidden_irreps=8x0e + 8x1o",
        "--MLP_irreps=8x0e", "--num_radial_basis=6", "--num_interactions=2", "--multipole_max_ell=1",
        "--error_table=EnergyNacsDipoleMAE", f"--work_dir={tmp_path}",
    ]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    m = torch.load(tmp_path / "t.model", weights_only=False)
    assert m.diabatic and m.n_states == 2 and m.n_energies == 3
    # E0s fitted to the diagonal; couplings have no atomic offset
    gas = [f for f in frames if "mm_charges" not in f.info][0]
    out = m(make_batch(m, [(gas.get_chemical_symbols(), gas.positions, np.zeros((0, 3)), np.zeros(0))]).to_dict())
    assert np.isfinite(out["energy"].detach().numpy()).all()


# --------------------------------------------------------------------------- #
# OpenMM: the ML/MM force uses the adiabatic energy of the chosen state
# --------------------------------------------------------------------------- #
def test_openmm_force_uses_adiabatic_energy(tmp_path):
    pytest.importorskip("openmm")
    T = pytest.importorskip("openmmtools.testsystems")
    from openmm import unit

    from mace.mlmm import create_fieldmace_mm_system

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        p = tmp_path / "dia.model"
        torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2, diabatic=True), p)
        ala = T.AlanineDipeptideExplicit()
        pos = ala.positions.value_in_unit(unit.angstrom)
        box = np.array([v.value_in_unit(unit.angstrom) for v in ala.system.getDefaultPeriodicBoxVectors()])
        es = []
        for st in (0, 1):
            _, _, comp = create_fieldmace_mm_system(ala.system, list(range(22)), str(p), topology=ala.topology,
                                                   model_kwargs=dict(device="cpu", state=st))
            sub = np.asarray(pos)[comp.frame.particles]
            e, g = comp.evaluate(sub, box)
            H = unpack(comp.last["diabatic_hamiltonian"], 2)
            es.append(e)
            assert e == pytest.approx(np.linalg.eigvalsh(H)[st], abs=1e-8)
            h = 1e-5
            for a, k in ((0, 0), (25, 1)):
                xp, xm = sub.copy(), sub.copy()
                xp[a, k] += h
                xm[a, k] -= h
                fd = (comp.evaluate(xp, box)[0] - comp.evaluate(xm, box)[0]) / (2 * h)
                assert fd == pytest.approx(g[a, k], abs=1e-5)
        assert es[0] < es[1]
    finally:
        torch.set_default_dtype(old)
