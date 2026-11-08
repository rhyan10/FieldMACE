# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""NACs, dipoles and SOCs: model outputs, labels, phase-free losses, training and calculators."""
import copy
import os
import subprocess
import sys

import ase
import ase.io
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mace import data, modules
from mace.modules.loss import phase_aligned_error
from mace.tools import torch_geometric, utils
from mace.tools.testing import FIELD_MODELS, build_field_model, make_batch, methane_in_charges

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = dict(compute_nacs=True, compute_dipoles=True, compute_socs=True)
Z = utils.AtomicNumberTable([1, 6, 7, 8])


def prop_model(cls="PerAtomFieldEMACE", n_e=3, seed=0, **kw):
    return build_field_model(cls, n_energies=n_e, seed=seed, **{**ALL, **kw})


def out_for(m, cfgs):
    return m(make_batch(m, cfgs).to_dict(), training=False)


# --------------------------------------------------------------------------- #
# model outputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", FIELD_MODELS)
@pytest.mark.parametrize("n_e", [2, 3])
def test_property_shapes(cls, n_e):
    m = prop_model(cls, n_e)
    o = out_for(m, [methane_in_charges(20, seed=1), methane_in_charges(0, seed=2)])
    p = n_e * (n_e - 1) // 2
    assert o["nacs"].shape == (10, p, 3)
    assert o["dipoles"].shape == (2, n_e + p, 3)
    assert o["socs"].shape == (2, p, 2)
    for k in ("nacs", "dipoles", "socs"):
        assert torch.isfinite(o[k]).all() and o[k].abs().sum() > 0


def test_dipoles_only_single_state():
    m = build_field_model("PerAtomFieldEMACE", n_energies=1, compute_dipoles=True)
    o = out_for(m, [methane_in_charges(5)])
    assert o["dipoles"].shape == (1, 1, 3) and o["nacs"] is None and o["socs"] is None


def test_properties_absent_by_default():
    o = out_for(build_field_model("PerAtomFieldEMACE", n_energies=2), [methane_in_charges(5)])
    assert o["nacs"] is None and o["dipoles"] is None and o["socs"] is None


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_property_equivariance(cls):
    m = prop_model(cls)
    sym, qm, mm, q = methane_in_charges(25, seed=3, shift=(0.4, -0.3, 0.2))
    R = Rotation.random(random_state=11).as_matrix()
    Rt = torch.tensor(R)
    t = np.array([5.0, -2.0, 7.5])
    o1 = out_for(m, [(sym, qm, mm, q)])
    o2 = out_for(m, [(sym, qm @ R.T + t, mm @ R.T + t, q)])
    assert torch.allclose(o1["nacs"] @ Rt.T, o2["nacs"], atol=1e-9)  # vectors rotate
    assert torch.allclose(o1["dipoles"] @ Rt.T, o2["dipoles"], atol=1e-9)
    assert torch.allclose(o1["socs"], o2["socs"], atol=1e-9)  # scalars invariant
    # inversion: NACs and dipoles are odd (1o)
    o3 = out_for(m, [(sym, -qm, -mm, q)])
    assert torch.allclose(o1["nacs"], -o3["nacs"], atol=1e-9)
    assert torch.allclose(o1["dipoles"], -o3["dipoles"], atol=1e-9)


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_property_permutation_and_batching(cls):
    m = prop_model(cls)
    sym, qm, mm, q = methane_in_charges(15, seed=4)
    perm = [0, 3, 1, 4, 2]
    o1 = out_for(m, [(sym, qm, mm, q)])
    o2 = out_for(m, [(sym, qm[perm], mm, q)])
    assert torch.allclose(o1["nacs"][perm], o2["nacs"], atol=1e-10)
    assert torch.allclose(o1["dipoles"], o2["dipoles"], atol=1e-10)
    ob = out_for(m, [methane_in_charges(7, seed=9), (sym, qm, mm, q)])
    assert torch.allclose(ob["nacs"][5:], o1["nacs"], atol=1e-10)
    assert torch.allclose(ob["socs"][1], o1["socs"][0], atol=1e-10)


def test_properties_respond_to_the_field():
    m = prop_model()
    sym, qm, mm, q = methane_in_charges(15, seed=5)
    o1 = out_for(m, [(sym, qm, mm, q)])
    o2 = out_for(m, [(sym, qm, mm, 2 * q)])
    assert not torch.allclose(o1["dipoles"], o2["dipoles"])


@pytest.mark.parametrize("cls", FIELD_MODELS)
def test_property_torchscript(cls):
    from e3nn.util import jit

    m = prop_model(cls)
    sm = jit.compile(copy.deepcopy(m))
    cfgs = [methane_in_charges(12, seed=1), methane_in_charges(0, seed=2)]
    o1, o2 = out_for(m, cfgs), sm(make_batch(m, cfgs).to_dict())
    for k in ("energy", "nacs", "dipoles", "socs"):
        assert torch.allclose(o1[k], o2[k], atol=1e-10), k


def test_old_models_still_load(tmp_path):
    """A 2.0 model pickled before the property heads existed loads with properties off."""
    m = build_field_model("PerAtomFieldEMACE")
    del m._modules["property_readouts"]
    for a in ("compute_socs", "n_prop_out"):
        delattr(m, a)
    torch.save(m, tmp_path / "old.model")
    m2 = torch.load(tmp_path / "old.model", weights_only=False)
    o = out_for(m2, [methane_in_charges(5)])
    assert o["nacs"] is None and m2.n_prop_out == 0


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def labelled_atoms(n_e=3, n_mm=6, seed=0, nacs_layout="NP3", socs_kind="re_im", **skip):
    rng = np.random.default_rng(seed)
    sym, qm, mm, q = methane_in_charges(n_mm, seed=seed)
    p = n_e * (n_e - 1) // 2
    a = ase.Atoms(sym, positions=qm)
    a.info.update(mm_positions=mm, mm_charges=q, REF_energy=rng.normal(size=n_e), REF_forces=rng.normal(size=(5, n_e, 3)))
    nacs = rng.normal(size=(5, p, 3))
    if "nacs" not in skip:
        a.info["REF_nacs"] = nacs if nacs_layout == "NP3" else np.transpose(nacs, (1, 0, 2))
    if "dipoles" not in skip:
        a.info["REF_dipoles"] = rng.normal(size=(n_e + p, 3))
    socs = rng.normal(size=(p, 2))
    if "socs" not in skip:
        a.info["REF_socs"] = {"re_im": socs, "real": socs[:, 0], "complex": socs[:, 0] + 1j * socs[:, 1]}[socs_kind]
    return a, nacs, socs


def cfg(a):
    return data.config_from_atoms(a, nacs_key="REF_nacs", dipoles_key="REF_dipoles", socs_key="REF_socs")


@pytest.mark.parametrize("layout", ["NP3", "PN3"])
def test_nac_label_layouts(layout):
    a, nacs, _ = labelled_atoms(nacs_layout=layout)
    c = cfg(a)
    assert np.allclose(c.nacs, nacs) and c.nacs_weight == 1.0


@pytest.mark.parametrize("kind", ["re_im", "real", "complex"])
def test_soc_label_formats(kind):
    a, _, socs = labelled_atoms(socs_kind=kind)
    c = cfg(a)
    assert c.socs.shape == (3, 2) and np.allclose(c.socs[:, 0], socs[:, 0])
    if kind != "real":
        assert np.allclose(c.socs[:, 1], socs[:, 1])


def test_missing_property_labels_have_zero_weight():
    a, _, _ = labelled_atoms(nacs=1, dipoles=1, socs=1)
    c = cfg(a)
    assert (c.nacs_weight, c.dipoles_weight, c.socs_weight) == (0.0, 0.0, 0.0)
    ad = data.AtomicData.from_config(c, z_table=Z, cutoff=5.0)
    assert ad.nacs.shape == (5, 3, 3) and ad.dipoles.shape == (1, 6, 3) and ad.socs.shape == (1, 6)  # SOCs stored flat (P pairs x re/im)


@pytest.mark.parametrize("bad", ["nacs", "dipoles", "socs"])
def test_wrong_property_shapes_raise(bad):
    a, _, _ = labelled_atoms()
    a.info["REF_" + bad] = np.ones(7)
    with pytest.raises(ValueError):
        cfg(a)


def test_property_batching_and_hdf5(tmp_path):
    import h5py

    configs = [cfg(labelled_atoms(seed=s, n_mm=3 + s)[0]) for s in range(3)]
    configs.append(cfg(labelled_atoms(seed=9, nacs=1, socs=1)[0]))  # partly labelled
    ds = [data.AtomicData.from_config(c, z_table=Z, cutoff=5.0) for c in configs]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=4, shuffle=False)))
    assert b["nacs"].shape == (20, 3, 3) and b["dipoles"].shape == (4, 6, 3) and b["socs"].shape == (4, 6)  # flat SOCs
    assert b["socs_weight"].tolist() == [1.0, 1.0, 1.0, 0.0] and b["nacs_weight"].tolist() == [1.0, 1.0, 1.0, 0.0]
    with h5py.File(tmp_path / "d.h5", "w") as f:
        data.utils.save_configurations_as_HDF5(configs, 0, f)
        f.attrs["batch_size"] = len(configs)
    h5 = data.HDF5Dataset(str(tmp_path / "d.h5"), r_max=5.0, z_table=Z)
    for i in range(4):
        assert torch.allclose(h5[i].nacs, ds[i].nacs) and torch.allclose(h5[i].socs, ds[i].socs)
        assert float(h5[i].socs_weight) == float(ds[i].socs_weight)


# --------------------------------------------------------------------------- #
# phase-free loss
# --------------------------------------------------------------------------- #
def _batch_and_pred(seed=0):
    configs = [cfg(labelled_atoms(seed=s, n_mm=4)[0]) for s in range(2)]
    ds = [data.AtomicData.from_config(c, z_table=Z, cutoff=5.0) for c in configs]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=2, shuffle=False)))
    m = prop_model()
    return b, m(b.to_dict(), training=True), m


def _only(**w):
    base = dict(energy_weight=0.0, forces_weight=0.0, mm_forces_weight=0.0, nacs_weight=0.0, dipoles_weight=0.0,
                socs_weight=0.0)
    base.update(w)
    return modules.WeightedEnergyForcesNacsDipoleLoss(**base)


def test_phase_aligned_error_picks_best_sign():
    ref = torch.tensor([[[1.0, 2.0, 3.0]], [[1.0, 0.0, 0.0]]])
    pred = -ref
    err = phase_aligned_error(ref, pred, torch.tensor([0, 1]), 2)
    assert torch.allclose(err, torch.zeros_like(err))


@pytest.mark.parametrize("key,weight", [("nacs", "nacs_weight"), ("socs", "socs_weight"), ("dipoles", "dipoles_weight")])
def test_loss_invariant_to_coupling_phase(key, weight):
    b, out, _ = _batch_and_pred()
    loss_fn = _only(**{weight: 1.0})
    l0 = loss_fn(b, out).item()
    b2 = b.clone()
    if key == "nacs":  # flip pair 1 of configuration 0 only
        b2["nacs"][:5, 1] *= -1
    elif key == "socs":
        b2["socs"][0, 2:4] *= -1  # flat (re, im) per pair: pair 1 of configuration 0
    else:  # transition dipole of pair 0 in configuration 1
        b2["dipoles"][1, 3] *= -1
    assert loss_fn(b2, out).item() == pytest.approx(l0, rel=1e-12)
    if key == "dipoles":  # a permanent dipole has no phase freedom
        b3 = b.clone()
        b3["dipoles"][1, 0] *= -1
        assert loss_fn(b3, out).item() != pytest.approx(l0, rel=1e-6)


def test_loss_zero_for_sign_flipped_perfect_prediction():
    b, out, _ = _batch_and_pred()
    out2 = dict(out)
    out2["nacs"] = -b["nacs"].reshape(out["nacs"].shape)
    out2["socs"] = -b["socs"].reshape(out["socs"].shape)
    d = b["dipoles"].reshape(out["dipoles"].shape).clone()
    d[:, 3:] *= -1
    out2["dipoles"] = d
    assert _only(nacs_weight=1.0, socs_weight=1.0, dipoles_weight=1.0)(b, out2).item() == pytest.approx(0.0, abs=1e-20)


def test_property_loss_reaches_property_heads():
    b, out, m = _batch_and_pred()
    _only(nacs_weight=1.0, socs_weight=1.0, dipoles_weight=1.0)(b, out).backward()
    for name, p in m.named_parameters():
        if name.startswith("property_readouts"):
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_unlabelled_frames_do_not_contribute():
    configs = [cfg(labelled_atoms(seed=0)[0]), cfg(labelled_atoms(seed=1, nacs=1, dipoles=1, socs=1)[0])]
    ds = [data.AtomicData.from_config(c, z_table=Z, cutoff=5.0) for c in configs]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=2, shuffle=False)))
    m = prop_model()
    out = m(b.to_dict(), training=True)
    loss_fn = _only(nacs_weight=1.0, socs_weight=1.0, dipoles_weight=1.0)
    l0 = loss_fn(b, out).item()
    b2 = b.clone()
    b2["nacs"][5:] += 100.0
    b2["socs"][1] += 100.0
    b2["dipoles"][1] += 100.0
    assert loss_fn(b2, out).item() == pytest.approx(l0, rel=1e-12)


# --------------------------------------------------------------------------- #
# training through the CLI with random phases in the labels
# --------------------------------------------------------------------------- #
def make_property_dataset(path, n=24, n_e=3):
    teacher = prop_model(n_e=n_e, seed=42)
    rng = np.random.default_rng(1)
    frames = []
    for i in range(n):
        sym, qm, mm, q = methane_in_charges([0, 6, 15][i % 3], seed=i)
        qm = qm + rng.normal(scale=0.05, size=qm.shape)
        o = out_for(teacher, [(sym, qm, mm, q)])
        a = ase.Atoms(sym, positions=qm)
        if len(q):
            a.info.update(mm_positions=mm, mm_charges=q, REF_mm_forces=o["mm_forces"].detach().numpy())
        signs = rng.choice([-1.0, 1.0], size=n_e * (n_e - 1) // 2)  # arbitrary QM phases
        dip = o["dipoles"][0].detach().numpy().copy()
        dip[n_e:] *= signs[:, None]
        a.info.update(
            REF_energy=o["energy"].detach().numpy(), REF_forces=o["forces"].detach().numpy(),
            smooth_nacs=o["nacs"].detach().numpy() * signs[None, :, None], REF_dipoles=dip,
            REF_socs=o["socs"][0].detach().numpy() * signs[:, None],
        )
        frames.append(a)
    ase.io.write(path, frames, format="extxyz")


def test_cli_training_with_all_properties(tmp_path):
    make_property_dataset(tmp_path / "train.xyz")
    cmd = [
        sys.executable, os.path.join(REPO, "scripts", "run_train.py"), "--name=p", "--train_file=train.xyz",
        "--valid_fraction=0.25", "--seed=1", "--E0s=average", "--model=PerAtomFieldEMACE", "--batch_size=4",
        "--lr=0.01", "--n_energies=3", "--max_num_epochs=6", "--default_dtype=float64", "--device=cpu",
        "--hidden_irreps=8x0e + 8x1o", "--MLP_irreps=8x0e", "--num_interactions=2", "--correlation=2",
        "--error_table=EnergyNacsDipoleMAE", "--compute_nacs=True", "--compute_dipoles=True", "--compute_socs=True",
        "--nacs_weight=10", "--dipoles_weight=10", "--socs_weight=10",
    ]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=900)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert "MAE_nacs" in log and "MAE_dipoles" in log and "MAE_socs" in log
    init = float(log.split("Initial: loss=")[1].split(",")[0])
    last = float(log.split("Epoch 5: loss=")[1].split(",")[0])
    assert last < init
    m = torch.load(tmp_path / "p.model", weights_only=False)
    assert m.compute_nacs and m.compute_dipoles and m.compute_socs
    assert torch.jit.load(str(tmp_path / "p_compiled.model")) is not None
    # eval_configs writes the properties
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "eval_configs.py"), "--configs=train.xyz",
         "--model=p.model", "--output=pred.xyz"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    f = ase.io.read(tmp_path / "pred.xyz", "1")
    assert f.info["MACE_nacs"].shape == (5, 3, 3) and f.info["MACE_dipoles"].shape == (6, 3)
    assert f.info["MACE_socs"].shape == (3, 2)


# --------------------------------------------------------------------------- #
# calculators
# --------------------------------------------------------------------------- #
@pytest.fixture
def prop_model_file(tmp_path):
    p = tmp_path / "props.model"
    torch.save(prop_model(), p)
    return str(p)


def test_ase_calculator_properties(prop_model_file):
    from mace.calculators import MACECalculator

    m = torch.load(prop_model_file, weights_only=False)
    sym, qm, mm, q = methane_in_charges(8, seed=3)
    a = ase.Atoms(sym, positions=qm)
    a.info.update(mm_positions=mm, mm_charges=q)
    calc = MACECalculator(prop_model_file, device="cpu", default_dtype="float64", state=1)
    a.calc = calc
    a.get_potential_energy()
    ref = out_for(m, [(sym, qm, mm, q)])
    r = calc.results
    assert np.allclose(r["nacs"], ref["nacs"].detach().numpy(), atol=1e-10)
    assert np.allclose(r["dipoles"], ref["dipoles"][0].detach().numpy(), atol=1e-10)
    assert np.allclose(r["dipole"], ref["dipoles"][0, 1].detach().numpy(), atol=1e-10)
    s = ref["socs"][0].detach().numpy()
    assert np.allclose(r["socs"], s[:, 0] + 1j * s[:, 1], atol=1e-10)
    assert np.allclose(a.get_dipole_moment(), r["dipole"])


def test_committee_phase_alignment():
    from mace.calculators.mace import _phase_aligned_mean

    a = torch.tensor([[[1.0, 0, 0], [0, 2.0, 0]]])  # [1 atom, 2 pairs, 3]
    b = a.clone()
    b[:, 1] *= -1  # second model has the opposite sign for pair 1
    assert torch.allclose(_phase_aligned_mean([a, b], pair_axis=1), a)


def test_sharc_calculator(prop_model_file):
    from mace.calculators.mace import MACECalculator
    from mace.calculators.sharc_calculator import SharcCalculator

    sym, qm, mm, q = methane_in_charges(8, seed=3)
    bohr = 0.529177210903
    # three singlets with SOCs: unphysical (SOCs between singlets vanish), but it tests the plumbing;
    # the selection rules (default) would zero them
    assert np.allclose(np.triu(np.array(SharcCalculator(sym, prop_model_file, device="cpu", properties=[
        "energy", "socs"]).calculate(qm / bohr, mm, q)["h"]), 1), 0)
    sc = SharcCalculator(sym, prop_model_file, device="cpu", soc_selection_rules=False,
                         properties=["energy", "forces", "nacs", "dipoles", "socs"])
    out = sc.calculate(qm / bohr, mm, q)
    S = 3
    h = np.array(out["h"])
    assert h.shape == (S, S) and np.allclose(h, h.conj().T)
    nac = np.array(out["nacdr"])
    assert nac.shape == (S, S, 5, 3) and np.allclose(nac, -np.transpose(nac, (1, 0, 2, 3)))
    dm = np.array(out["dm"])
    assert dm.shape == (3, S, S) and np.allclose(dm, np.transpose(dm, (0, 2, 1)))
    assert np.array(out["grad"]).shape == (S, 5, 3) and np.array(out["grad_pc"]).shape == (S, 8, 3)

    # unit conversions against the ASE calculator (eV, Angstrom)
    a = ase.Atoms(sym, positions=qm)
    a.info.update(mm_positions=mm, mm_charges=q)
    calc = MACECalculator(prop_model_file, device="cpu", default_dtype="float64")
    calc.calculate(a)
    r = calc.results
    assert np.allclose(np.diag(h).real, r["energies"] / 27.211386245988)
    assert np.allclose(np.array(out["grad"])[1], -r["forces_all"][:, 1] / 27.211386245988 * bohr)
    assert np.allclose(nac[0, 1], r["nacs"][:, 0] * bohr)
    assert np.allclose(dm[:, 0, 2], r["dipoles"][3 + 1] / bohr)
    assert np.allclose(h[1, 2], r["socs"][2] / 27.211386245988)

    # smooth NACs: the model output (E_j - E_i) * NAC_ij is divided by the model's own gap
    sm = SharcCalculator(sym, prop_model_file, device="cpu", soc_selection_rules=False, nac_type="smooth",
                         properties=["energy", "forces", "nacs"])
    nac_s = np.array(sm.calculate(qm / bohr, mm, q)["nacdr"])
    e = r["energies"]
    assert np.allclose(nac_s[0, 1], r["nacs"][:, 0] * bohr / (e[1] - e[0]))
    assert np.allclose(nac_s[1, 2], r["nacs"][:, 2] * bohr / (e[2] - e[1]))
    assert np.allclose(nac_s, -np.transpose(nac_s, (1, 0, 2, 3)))
    with pytest.raises(ValueError):
        SharcCalculator(sym, prop_model_file, device="cpu", nac_type="smoothed")

    # phase tracking: a previous step with flipped couplings flips the new output
    prev = {k: v.copy() for k, v in sc.last_prediction.items()}
    prev["nacs"][:, 0] *= -1
    prev["socs"][1] *= -1
    prev["dipoles"][3 + 2] *= -1
    sc.last_prediction = prev
    out2 = sc.calculate(qm / bohr, mm, q)
    assert np.allclose(np.array(out2["nacdr"])[0, 1], -nac[0, 1])
    assert np.allclose(np.array(out2["h"])[0, 2], -h[0, 2])
    assert np.allclose(np.array(out2["dm"])[:, 1, 2], -dm[:, 1, 2])
    assert np.allclose(np.array(out2["nacdr"])[0, 2], nac[0, 2])  # untouched pairs keep their sign


def test_sharc_calculator_returns_double_precision(prop_model_file, tmp_path):
    """pysharc reads raw buffers as float64: a float32 model must still give float64/complex128 arrays."""
    from mace.calculators.sharc_calculator import SharcCalculator

    m = torch.load(prop_model_file, weights_only=False).float()
    p = tmp_path / "f32.model"
    torch.save(m, p)
    sym, qm, mm, q = methane_in_charges(8, seed=3)
    sc = SharcCalculator(sym, str(p), device="cpu", properties=["energy", "forces", "nacs", "dipoles", "socs"])
    out = sc.get_qm(sc.predict(qm / 0.529177210903, mm, q), as_arrays=True)
    assert out["h"].dtype == np.complex128
    for k in ("grad", "grad_pc", "nacdr", "dm"):
        assert out[k].dtype == np.float64 and out[k].flags["C_CONTIGUOUS"], k
    assert np.abs(out["grad"]).max() > 0


def test_sharc_soc_selection_rules(tmp_path):
    """Singlets + triplets: SOCs vanish between singlets, within a multiplet and for |dM_S| > 1."""
    from mace.calculators.sharc_calculator import SharcCalculator, sharc_mstates
    from mace.tools.testing import build_field_model

    m = build_field_model("PerAtomFieldEMACE", seed=5, atomic_numbers=[1, 6], channels=8, n_energies=4,
                          compute_socs=True, soc_num=56)
    p = tmp_path / "st.model"
    torch.save(m, p)
    sym, qm, mm, q = methane_in_charges(8, seed=3)
    ms = sharc_mstates([2, 0, 2])
    on = np.array(SharcCalculator(sym, str(p), states=[2, 0, 2], properties=["energy", "socs"]).calculate(
        qm / 0.529177210903, mm, q)["h"])
    off = np.array(SharcCalculator(sym, str(p), states=[2, 0, 2], properties=["energy", "socs"],
                                   soc_selection_rules=False).calculate(qm / 0.529177210903, mm, q)["h"])
    assert np.allclose(on, on.conj().T)
    for a, (ma, ja, sa) in enumerate(ms):
        for b, (mb, jb, sb) in enumerate(ms):
            if a == b:
                continue
            forbidden = (ma == mb == 1) or (ma == mb and ja == jb) or abs(sa - sb) > 1
            assert (on[a, b] == 0) == forbidden, (a, b)
            if not forbidden:
                assert on[a, b] == off[a, b]
    assert np.abs(off[0, 1]) > 0  # the raw model output is not zero there


def test_sharc_calculator_rejects_missing_property(tmp_path):
    from mace.calculators.sharc_calculator import SharcCalculator

    p = tmp_path / "plain.model"
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2), p)
    with pytest.raises(ValueError, match="compute_nacs"):
        SharcCalculator("CH4", str(p), device="cpu", properties=["energy", "nacs"])


def test_mlmm_exposes_properties(prop_model_file):
    pytest.importorskip("openmm")
    openmmtools = pytest.importorskip("openmmtools")
    from openmm import unit

    from mace.mlmm import create_fieldmace_mm_system

    ts = openmmtools.testsystems.AlanineDipeptideExplicit()
    _, _, comp = create_fieldmace_mm_system(ts.system, range(22), prop_model_file, topology=ts.topology,
                                            model_kwargs=dict(device="cpu"))
    box = np.array([v.value_in_unit(unit.angstrom) for v in ts.system.getDefaultPeriodicBoxVectors()])
    comp.evaluate(np.asarray(ts.positions.value_in_unit(unit.angstrom))[comp.frame.particles], box)
    assert comp.last["dipoles"].shape == (6, 3) and comp.last["socs"].shape == (3, 2)
    assert comp.last["nacs"].shape == (22, 3, 3)
