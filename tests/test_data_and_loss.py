# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Data parsing, batching, HDF5 and loss edge cases."""
import ase
import ase.io
import h5py
import numpy as np
import pytest
import torch

from mace import data, modules
from mace.tools import torch_geometric, utils
from mace.tools.testing import build_field_model, methane_in_charges

Z = utils.AtomicNumberTable([1, 6, 7, 8])


def atoms_with(n_mm=6, seed=0, energy=-1.0, forces=None, mm_forces=None, where="info", **info):
    sym, qm, mm, q = methane_in_charges(n_mm or 0, seed=seed)
    a = ase.Atoms(sym, positions=qm)
    if n_mm is not None:
        a.info["mm_positions"] = mm
        a.info["mm_charges"] = q
    if energy is not None:
        a.info["REF_energy"] = energy
    if forces is not None:
        (a.info if where == "info" else a.arrays)["REF_forces"] = forces
    if mm_forces is not None:
        a.info["REF_mm_forces"] = mm_forces
    a.info.update(info)
    return a


def to_batch(atoms_list):
    ds = [data.AtomicData.from_config(data.config_from_atoms(a), z_table=Z, cutoff=5.0) for a in atoms_list]
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds), shuffle=False)))


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("where", ["info", "arrays"])
def test_forces_from_info_or_arrays(where):
    f = np.random.default_rng(0).normal(size=(5, 3))
    c = data.config_from_atoms(atoms_with(forces=f, where=where))
    assert c.forces_weight == 1.0 and np.allclose(c.forces[:, 0], f)


def test_label_layouts_single_and_multi_state():
    f = np.ones((5, 3))
    c1 = data.config_from_atoms(atoms_with(energy=-2.0, forces=f, mm_forces=np.ones((6, 3))))
    assert c1.energy.shape == (1,) and c1.forces.shape == (5, 1, 3) and c1.mm_forces.shape == (6, 1, 3)
    c3 = data.config_from_atoms(atoms_with(energy=np.array([[-1.0, -0.5, 0.1]]), forces=np.ones((5, 3, 3)),
                                           mm_forces=np.ones((6, 3, 3))))
    assert c3.energy.shape == (3,) and c3.forces.shape == (5, 3, 3)
    ad = data.AtomicData.from_config(c3, z_table=Z, cutoff=5.0)
    assert ad.energy.shape == (1, 3) and ad.mm_forces.shape == (6, 3, 3) and int(ad.num_mm) == 6


def test_missing_labels_get_zero_weight_and_right_shape():
    c = data.config_from_atoms(atoms_with(energy=None, forces=np.ones((5, 2, 3))))
    assert c.energy_weight == 0.0 and c.energy.shape == (2,)
    assert c.mm_forces_weight == 0.0 and c.mm_forces.shape == (6, 2, 3)
    c = data.config_from_atoms(atoms_with(energy=[-1.0, -2.0], forces=None))
    assert c.forces_weight == 0.0 and c.forces.shape == (5, 2, 3)


def test_gas_phase_configuration_has_no_charges():
    c = data.config_from_atoms(atoms_with(n_mm=None, forces=np.ones((5, 3))))
    ad = data.AtomicData.from_config(c, z_table=Z, cutoff=5.0)
    assert ad.mm_positions.shape == (0, 3) and ad.mm_charges.shape == (0,) and int(ad.num_mm) == 0


@pytest.mark.parametrize("bad", ["mismatch", "half", "nan", "mm_forces_shape", "states"])
def test_invalid_inputs_raise(bad):
    a = atoms_with(forces=np.ones((5, 3)))
    if bad == "mismatch":
        a.info["mm_charges"] = a.info["mm_charges"][:-1]
    elif bad == "half":
        del a.info["mm_charges"]
    elif bad == "nan":
        a.info["mm_positions"][0, 0] = np.nan
    elif bad == "mm_forces_shape":
        a.info["REF_mm_forces"] = np.ones((3, 3))
    elif bad == "states":
        a.info["REF_energy"] = [-1.0, -2.0]
    with pytest.raises(ValueError):
        data.config_from_atoms(a)


def test_batching_mixed_charge_counts():
    b = to_batch([atoms_with(10, seed=1, forces=np.ones((5, 3))), atoms_with(None, forces=np.ones((5, 3))),
                  atoms_with(3, seed=2, forces=np.ones((5, 3)), mm_forces=np.ones((3, 3)))])
    assert b["num_mm"].tolist() == [10, 0, 3]
    assert b["mm_positions"].shape == (13, 3) and b["mm_forces"].shape == (13, 1, 3)
    assert b["mm_forces_weight"].tolist() == [0.0, 0.0, 1.0]
    assert b["energy"].shape == (3, 1)


def test_extxyz_roundtrip(tmp_path):
    a = atoms_with(7, forces=np.ones((5, 2, 3)), energy=np.array([[-1.0, -2.0]]), mm_forces=np.ones((7, 2, 3)))
    ase.io.write(tmp_path / "d.xyz", [a, atoms_with(None, forces=np.ones((5, 2, 3)), energy=[-1.0, -2.0])])
    _, configs = data.load_from_xyz(str(tmp_path / "d.xyz"), config_type_weights={"Default": 1.0})
    assert configs[0].mm_forces.shape == (7, 2, 3) and len(configs[1].mm_charges) == 0


def test_hdf5_roundtrip(tmp_path):
    configs = [data.config_from_atoms(atoms_with(n, seed=n, forces=np.ones((5, 3)), mm_forces=np.ones((n, 3))))
               for n in (4, 9)]
    with h5py.File(tmp_path / "d.h5", "w") as f:
        data.utils.save_configurations_as_HDF5(configs, 0, f)
        f.attrs["batch_size"] = len(configs)
    ds = data.HDF5Dataset(str(tmp_path / "d.h5"), r_max=5.0, z_table=Z)
    for i, c in enumerate(configs):
        ad = ds[i]
        assert int(ad.num_mm) == len(c.mm_charges)
        assert np.allclose(ad.mm_positions.numpy(), c.mm_positions)
        assert float(ad.mm_forces_weight) == 1.0


def test_average_e0s_ignores_unlabelled_configs():
    labelled = [data.config_from_atoms(atoms_with(energy=-10.0 - i, forces=np.ones((5, 3)))) for i in range(3)]
    unlabelled = data.config_from_atoms(atoms_with(energy=None, forces=np.ones((5, 3))))
    e0 = data.compute_average_E0s(labelled + [unlabelled], utils.AtomicNumberTable([1, 6]))
    e0_ref = data.compute_average_E0s(labelled, utils.AtomicNumberTable([1, 6]))
    assert e0 == pytest.approx(e0_ref)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def _model_and_batch(n_e=1):
    m = build_field_model("PerAtomFieldEMACE", n_energies=n_e)
    f = np.random.default_rng(1).normal(size=(5, n_e, 3))
    b = to_batch([
        atoms_with(6, seed=1, energy=np.full((1, n_e), -1.0), forces=f, mm_forces=np.ones((6, n_e, 3))),
        atoms_with(4, seed=2, energy=None, forces=f),  # energy and MM forces missing
    ])
    out = m(b.to_dict(), training=True)
    return m, b, out


def test_loss_matches_manual_computation():
    _, b, out = _model_and_batch()
    loss_fn = modules.WeightedEnergyForcesNacsDipoleLoss(energy_weight=2.0, forces_weight=3.0, mm_forces_weight=5.0)
    w_e = torch.tensor([1.0, 0.0]).unsqueeze(-1)
    e = 2.0 * torch.mean(w_e * (b["energy"] - out["energy"]) ** 2)
    f = 3.0 * torch.mean((b["forces"] - out["forces"]) ** 2)
    w_m = torch.tensor([1.0] * 6 + [0.0] * 4).view(-1, 1, 1)
    fm = 5.0 * torch.mean(w_m * (b["mm_forces"] - out["mm_forces"]) ** 2)
    assert loss_fn(b, out).item() == pytest.approx((e + f + fm).item(), rel=1e-12)


def test_loss_mm_weight_zero_and_none_outputs():
    _, b, out = _model_and_batch()
    l0 = modules.WeightedEnergyForcesNacsDipoleLoss(1.0, 1.0, mm_forces_weight=0.0)(b, out)
    out2 = dict(out)
    out2["mm_forces"] = None
    assert l0.item() == pytest.approx(modules.WeightedEnergyForcesNacsDipoleLoss(1.0, 1.0)(b, out2).item())


def test_loss_backprop_reaches_every_parameter():
    m, b, out = _model_and_batch(n_e=2)
    loss = modules.WeightedEnergyForcesNacsDipoleLoss(1.0, 1.0)(b, out)
    loss.backward()
    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    assert not no_grad, no_grad


def test_loss_state_mismatch_raises():
    m = build_field_model("PerAtomFieldEMACE", n_energies=2)
    b = to_batch([atoms_with(6, energy=-1.0, forces=np.ones((5, 3)))])
    out = m(b.to_dict(), training=True)
    with pytest.raises(ValueError, match="n_energies"):
        modules.WeightedEnergyForcesNacsDipoleLoss()(b, out)
