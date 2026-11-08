# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Physics and edge-case tests for the field models (random weights, float64)."""
import math
import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mace.tools.testing import FIELD_MODELS
from mace import modules
from mace.tools.testing import build_field_model, make_batch, methane_in_charges


def run(model, configs, **kw):
    b = make_batch(model, configs)
    return model(b.to_dict(), training=False, **kw), b


def energy(model, cfg):
    return run(model, [cfg])[0]["energy"].detach().numpy().ravel()


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", FIELD_MODELS)
@pytest.mark.parametrize("n_e", [1, 3])
def test_shapes_variable_number_of_charges(cls, n_e):
    m = build_field_model(cls, n_energies=n_e)
    cfgs = [methane_in_charges(25, seed=1), methane_in_charges(10, seed=2), methane_in_charges(0, seed=3)]
    out, b = run(m, cfgs)
    assert out["energy"].shape == (3, n_e)
    assert out["forces"].shape == (15, n_e, 3)
    assert out["mm_forces"].shape == (35, n_e, 3)
    assert b["num_mm"].tolist() == [25, 10, 0]
    for k in ("energy", "forces", "mm_forces"):
        assert torch.isfinite(out[k]).all()


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_batched_equals_single(cls):
    m = build_field_model(cls, n_energies=2)
    cfgs = [methane_in_charges(25, seed=1), methane_in_charges(10, seed=2), methane_in_charges(0, seed=3)]
    out, _ = run(m, cfgs)
    fm_all = out["mm_forces"].detach().numpy()
    start = 0
    for g, c in enumerate(cfgs):
        o1, _ = run(m, [c])
        assert np.allclose(o1["energy"].detach().numpy(), out["energy"][g].detach().numpy(), atol=1e-10)
        assert np.allclose(o1["forces"].detach().numpy(), out["forces"][5 * g:5 * g + 5].detach().numpy(), atol=1e-10)
        n = len(c[3])
        assert np.allclose(o1["mm_forces"].detach().numpy(), fm_all[start:start + n], atol=1e-10)
        start += n


@pytest.mark.parametrize("cls", FIELD_MODELS)
@pytest.mark.parametrize("lmax", [0, 1, 2])
def test_rotation_invariance(cls, lmax):
    m = build_field_model(cls, multipole_max_ell=lmax)
    sym, qm, mm, q = methane_in_charges(30, seed=4, shift=(0.3, -0.2, 0.5))
    R = Rotation.random(random_state=7).as_matrix()
    o1, _ = run(m, [(sym, qm, mm, q)])
    o2, _ = run(m, [(sym, qm @ R.T, mm @ R.T, q)])
    assert abs(o1["energy"].item() - o2["energy"].item()) < 1e-9
    assert np.allclose(o1["forces"][:, 0].detach().numpy() @ R.T, o2["forces"][:, 0].detach().numpy(), atol=1e-9)
    assert np.allclose(o1["mm_forces"][:, 0].detach().numpy() @ R.T, o2["mm_forces"][:, 0].detach().numpy(), atol=1e-9)


def test_legacy_embedding_breaks_rotation_invariance():
    """Documents why the default changed: the original embedding is not invariant for l >= 1."""
    m = build_field_model("PerAtomFieldEMACE", multipole_embedding="legacy")
    sym, qm, mm, q = methane_in_charges(30, seed=4)
    R = Rotation.random(random_state=7).as_matrix()
    e1 = energy(m, (sym, qm, mm, q))[0]
    e2 = energy(m, (sym, qm @ R.T, mm @ R.T, q))[0]
    assert abs(e1 - e2) > 1e-4


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_translation_invariance_and_momentum(cls):
    m = build_field_model(cls)
    sym, qm, mm, q = methane_in_charges(30, seed=5)
    t = np.array([13.0, -7.5, 2.25])
    o1, _ = run(m, [(sym, qm, mm, q)])
    o2, _ = run(m, [(sym, qm + t, mm + t, q)])
    assert abs(o1["energy"].item() - o2["energy"].item()) < 1e-9
    total = o1["forces"][:, 0].sum(0) + o1["mm_forces"][:, 0].sum(0)
    assert total.abs().max().item() < 1e-9


def test_fieldemace_origin_mode_is_not_translation_invariant():
    m = build_field_model("FieldEMACE", expansion_centre="origin")
    sym, qm, mm, q = methane_in_charges(30, seed=5)
    t = np.array([13.0, -7.5, 2.25])
    assert abs(energy(m, (sym, qm, mm, q))[0] - energy(m, (sym, qm + t, mm + t, q))[0]) > 1e-6


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_permutation_invariance(cls):
    m = build_field_model(cls)
    sym, qm, mm, q = methane_in_charges(20, seed=6)
    perm_mm = np.random.default_rng(0).permutation(len(q))
    perm_qm = [0, 3, 1, 4, 2]  # permute the hydrogens
    o1, _ = run(m, [(sym, qm, mm, q)])
    o2, _ = run(m, [(sym, qm[perm_qm], mm[perm_mm], q[perm_mm])])
    assert abs(o1["energy"].item() - o2["energy"].item()) < 1e-10
    assert np.allclose(o1["mm_forces"].detach().numpy()[perm_mm], o2["mm_forces"].detach().numpy(), atol=1e-10)
    assert np.allclose(o1["forces"].detach().numpy()[perm_qm], o2["forces"].detach().numpy(), atol=1e-10)


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_forces_are_negative_gradient(cls):
    m = build_field_model(cls, n_energies=2)
    sym, qm, mm, q = methane_in_charges(12, seed=8)
    out, _ = run(m, [(sym, qm, mm, q)])
    h = 1e-5
    for state in (0, 1):
        for which, arr, idx in (("qm", qm, [0, 2]), ("mm", mm, [0, 5, 11])):
            for i in idx:
                for k in range(3):
                    e = []
                    for s in (1, -1):
                        a = arr.copy()
                        a[i, k] += s * h
                        cfg = (sym, a, mm, q) if which == "qm" else (sym, qm, a, q)
                        e.append(energy(m, cfg)[state])
                    fd = -(e[0] - e[1]) / (2 * h)
                    f = out["forces" if which == "qm" else "mm_forces"][i, state, k].item()
                    assert fd == pytest.approx(f, abs=1e-6), (which, i, k, state)


@pytest.mark.parametrize("cls", ["PerAtomFieldEMACE", "FieldEMACE"])
def test_zero_charges_are_inert(cls):
    """No charges == zero charges == zero-charge padding far away (energies and forces)."""
    m = build_field_model(cls)
    sym, qm, mm, q = methane_in_charges(15, seed=9)
    e_none = energy(m, (sym, qm, np.zeros((0, 3)), np.zeros(0)))
    e_zero = energy(m, (sym, qm, mm, np.zeros(15)))
    assert np.allclose(e_none, e_zero, atol=1e-12)
    pad_mm = np.vstack([mm, np.full((7, 3), 1e3)])
    pad_q = np.concatenate([q, np.zeros(7)])
    o1, _ = run(m, [(sym, qm, mm, q)])
    o2, _ = run(m, [(sym, qm, pad_mm, pad_q)])
    assert abs(o1["energy"].item() - o2["energy"].item()) < 1e-12
    assert np.allclose(o1["forces"].detach().numpy(), o2["forces"].detach().numpy(), atol=1e-12)
    assert np.allclose(o2["mm_forces"][15:].detach().numpy(), 0.0)


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_charge_on_top_of_atom_or_centre_is_finite(cls):
    m = build_field_model(cls)
    sym, qm, mm, q = methane_in_charges(5, seed=10)
    mm = mm.copy()
    mm[0] = qm[1]  # exactly on a hydrogen
    mm[1] = qm.mean(0)  # exactly on the expansion centre of FieldEMACE
    out, _ = run(m, [(sym, qm, mm, q)])
    for k in ("energy", "forces", "mm_forces"):
        assert torch.isfinite(out[k]).all(), k


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_single_atom_qm_region(cls):
    m = build_field_model(cls)
    out, b = run(m, [("O", np.zeros((1, 3)), np.array([[3.0, 0, 0], [0, -4.0, 0]]), np.array([0.5, -0.4]))])
    assert b["edge_index"].shape[1] == 0
    assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["mm_forces"]).all()
    # an isolated atom in a field still feels a force
    assert out["forces"].abs().sum() > 0


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_padded_input_layout_still_accepted(cls):
    m = build_field_model(cls)
    sym, qm, mm, q = methane_in_charges(12, seed=11)
    b = make_batch(m, [(sym, qm, mm, q)])
    d = b.to_dict()
    flat = m(dict(d), training=False)
    d2 = dict(d)
    d2["mm_positions"] = d["mm_positions"].reshape(1, -1, 3)
    d2["mm_charges"] = d["mm_charges"].reshape(1, -1)
    del d2["num_mm"]
    padded = m(d2, training=False)
    assert torch.allclose(flat["energy"], padded["energy"], atol=1e-12)
    assert padded["mm_forces"].shape == (1, 12, 1, 3)


def test_compute_flags():
    m = build_field_model("PerAtomFieldEMACE")
    b = make_batch(m, [methane_in_charges(5)])
    out = m(b.to_dict(), compute_force=False)
    assert out["forces"] is None and out["mm_forces"] is None
    out = m(make_batch(m, [methane_in_charges(5)]).to_dict(), compute_pc_grads=False)
    assert out["forces"] is not None and out["mm_forces"] is None


def test_hessian():
    m = build_field_model("PerAtomFieldEMACE")
    out = m(make_batch(m, [methane_in_charges(5)]).to_dict(), compute_hessian=True)
    h = out["hessian"]
    assert h.shape == (15, 5, 3)
    hm = h.reshape(15, 15).detach().numpy()
    assert np.allclose(hm, hm.T, atol=1e-8)


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_float32(cls):
    m64 = build_field_model(cls)
    torch.set_default_dtype(torch.float32)
    m32 = build_field_model(cls).float()
    m32.load_state_dict({k: v.float() for k, v in m64.state_dict().items()})
    cfg = methane_in_charges(20, seed=12)
    e32 = energy(m32, cfg)
    torch.set_default_dtype(torch.float64)
    e64 = energy(m64, cfg)
    assert np.allclose(e32, e64, atol=1e-3)


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_torchscript(cls):
    from e3nn.util import jit

    import copy

    m = build_field_model(cls, n_energies=2)
    sm = jit.compile(copy.deepcopy(m))  # what run_train does before torch.jit.save
    assert isinstance(sm, torch.jit.ScriptModule)
    cfgs = [methane_in_charges(20, seed=1), methane_in_charges(0, seed=2)]
    o1 = m(make_batch(m, cfgs).to_dict())
    o2 = sm(make_batch(m, cfgs).to_dict())
    for k in ("energy", "forces", "mm_forces"):
        assert torch.allclose(o1[k], o2[k], atol=1e-10), k


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_all_pairs_graph_equals_cutoff_graph(cls):
    """MD fast path: a graph with every atom pair gives the same result as the cutoff graph."""
    m = build_field_model(cls)
    sym, qm, mm, q = methane_in_charges(10, seed=13)
    qm = np.vstack([qm, qm + np.array([6.0, 0, 0])])  # two methanes, some pairs beyond r_max
    sym = sym * 2
    b = make_batch(m, [(sym, qm, mm, q)])
    d = b.to_dict()
    ref = m(dict(d))
    n = len(qm)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = ii != jj
    d["edge_index"] = torch.tensor(np.stack([ii[mask], jj[mask]]), dtype=torch.long)
    d["shifts"] = torch.zeros(d["edge_index"].shape[1], 3)
    assert d["edge_index"].shape[1] > b["edge_index"].shape[1]
    allp = m(d)
    for k in ("energy", "forces", "mm_forces"):
        assert torch.allclose(ref[k], allp[k], atol=1e-11), k


def test_save_load_roundtrip(tmp_path):
    m = build_field_model("FieldEMACE")
    torch.save(m, tmp_path / "m.model")
    m2 = torch.load(tmp_path / "m.model", weights_only=False)
    from mace.modules.field import FIELD_MODEL_VERSION

    assert m2.legacy is False and m2.field_model_version == FIELD_MODEL_VERSION and m2.centre_on_qm
    cfg = methane_in_charges(10)
    assert np.allclose(energy(m, cfg), energy(m2, cfg), atol=1e-12)


def test_unsupported_options_fail_clearly():
    with pytest.raises(ValueError):
        build_field_model("PerAtomFieldEMACE", compute_nacs=True, n_energies=1)
    with pytest.raises(ValueError):
        build_field_model("PerAtomFieldEMACE", compute_socs=True, n_energies=1)
    assert not hasattr(modules, "AttentionFieldEMACE") and not hasattr(modules, "AttentionPerAtomFieldEMACE")
    with pytest.raises(ValueError):
        build_field_model("PerAtomFieldEMACE", multipole_embedding="nonsense")


# --------------------------------------------------------------------------- #
# legacy mode == original formulas
# --------------------------------------------------------------------------- #
def _legacy_phi(qm, mm, q, lmax, per_atom):
    """Original FieldMACE multipole features, written out with numpy + e3nn spherical harmonics."""
    from e3nn import o3

    sh = o3.SphericalHarmonics(o3.Irreps.spherical_harmonics(lmax), normalize=True, normalization="component")
    ls = [l for l in range(lmax + 1) for _ in range(2 * l + 1)]
    pre = np.array([math.sqrt(4 * math.pi / (2 * l + 1)) for l in ls])
    if per_atom:
        out = []
        for r in qm:
            v = mm - r
            Y = sh(torch.tensor(v)).numpy()
            d = np.linalg.norm(v, axis=1)
            out.append(sum(pre * q[j] / d[j] ** (np.array(ls) + 1) * Y[j] for j in range(len(q))))
        return np.array(out)
    Y = sh(torch.tensor(mm)).numpy()
    d = np.linalg.norm(mm, axis=1)
    mom = sum(pre * q[j] / d[j] ** (np.array(ls) + 1) * Y[j] for j in range(len(q)))
    R = np.linalg.norm(qm, axis=1)[:, None]
    return pre * mom * R ** np.array(ls)


@pytest.mark.parametrize("cls,per_atom", [("PerAtomFieldEMACE", True), ("FieldEMACE", False)])
def test_legacy_features_match_original_formula(cls, per_atom):
    m = build_field_model(cls, multipole_embedding="legacy", expansion_centre="origin", multipole_max_ell=2)
    sym, qm, mm, q = methane_in_charges(9, seed=14, shift=(0.4, 0.1, -0.3))
    b = make_batch(m, [(sym, qm, mm, q)])
    phi = m._field_features(
        b["positions"], b["batch"], b["ptr"], b["mm_positions"], b["mm_charges"], b["num_mm"],
        torch.zeros(9, dtype=torch.long), 1,
    )
    assert np.allclose(phi.detach().numpy(), _legacy_phi(qm, mm, q, 2, per_atom), atol=1e-10)


@pytest.mark.skipif("FIELDMACE_LEGACY_DIR" not in os.environ,
                    reason="set FIELDMACE_LEGACY_DIR to the output of tests/legacy/make_legacy_reference.py")
def test_original_pickles_reproduce_original_numbers():
    import ase

    from mace import data
    from mace.tools import torch_geometric, utils

    d = os.environ["FIELDMACE_LEGACY_DIR"]
    r = np.load(os.path.join(d, "legacy_ref.npz"))
    atoms = ase.Atoms("CHHHH", positions=r["qm"])
    atoms.info = {"mm_positions": r["mm"], "mm_charges": r["q"]}
    for key in ["FieldEMACE_1", "PerAtomFieldEMACE_1", "PerAtomFieldEMACE_3"]:
        m = torch.load(os.path.join(d, f"{key}.model"), weights_only=False)
        assert m.legacy
        z = utils.AtomicNumberTable([int(x) for x in m.atomic_numbers])
        ad = data.AtomicData.from_config(data.config_from_atoms(atoms), z_table=z, cutoff=5.0)
        b = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1)))
        out = m(b.to_dict())
        assert np.allclose(out["energy"].detach().numpy(), r[key + "_E"], atol=1e-10), key
        assert np.allclose(out["forces"].detach().numpy(), r[key + "_F"], atol=1e-10), key
        if r[key + "_Fmm"].size:
            assert np.allclose(out["mm_forces"].detach().numpy().ravel(), r[key + "_Fmm"].ravel(), atol=1e-10), key
