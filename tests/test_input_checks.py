# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Wrong inputs give clear errors (or warnings) instead of silent nonsense."""
import logging
import os
import subprocess
import sys
import types

import ase
import ase.io
import numpy as np
import pytest
import torch

from mace import data
from mace.tools.data_checks import DataError, InputError, check_arguments, check_datasets
from mace.tools.testing import build_field_model, make_batch, methane_in_charges

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def labelled_frames(n=8, n_states=1, mm=(0, 5)):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    t = build_field_model("PerAtomFieldEMACE", seed=3, n_energies=n_states)
    out = []
    for i in range(n):
        sym, qm, mmp, q = methane_in_charges(mm[i % len(mm)], seed=i)
        o = t(make_batch(t, [(sym, qm, mmp, q)]).to_dict())
        a = ase.Atoms(sym, positions=qm)
        if len(q):
            a.info.update(mm_positions=mmp, mm_charges=q, REF_mm_forces=o["mm_forces"].detach().numpy())
        a.info["REF_energy"] = o["energy"].detach().numpy()
        a.info["REF_forces"] = o["forces"].detach().numpy()
        out.append(a)
    torch.set_default_dtype(old)
    return out


def load(tmp_path, frames, name="d.xyz", **kw):
    p = tmp_path / name
    ase.io.write(p, frames, format="extxyz")
    return data.load_from_xyz(str(p), config_type_weights={}, **kw)[1]


def test_wrong_number_of_states_suggests_flag(tmp_path):
    cfg = load(tmp_path, labelled_frames(n_states=2))
    with pytest.raises(DataError, match="--n_energies=2"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=1)
    with pytest.raises(DataError, match="not a diabatic layout"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=2, diabatic=True)


def test_nan_labels_are_reported_with_file_and_frame(tmp_path):
    fr = labelled_frames()
    fr[3].info["REF_forces"][0, 0, 1] = np.nan
    cfg = load(tmp_path, fr, name="nan.xyz")
    with pytest.raises(DataError, match=r"nan\.xyz, frame 3 .*forces"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=1)


def test_bad_force_shape_names_the_frame(tmp_path):
    fr = labelled_frames()
    fr[2].info["REF_forces"] = fr[2].info["REF_forces"][:3]
    with pytest.raises(DataError, match=r"frame 2: 'REF_forces' has shape"):
        load(tmp_path, fr)


def test_missing_labels_and_wrong_keys(tmp_path):
    fr = labelled_frames()
    for a in fr:
        a.info.pop("REF_energy")
    cfg = load(tmp_path, fr)
    with pytest.raises(DataError, match="no training frame has energies"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=1, first_frame_info=["REF_forces"])
    cfg = load(tmp_path, labelled_frames(n_states=2))
    with pytest.raises(DataError, match="--compute_dipoles=True but no training frame"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=2, compute_dipoles=True,
                       keys={"dipoles": "dipole"})


def test_unknown_element_in_test_set_and_empty_validation(tmp_path):
    cfg = load(tmp_path, labelled_frames())
    g = labelled_frames(n=1)[0]
    g.symbols[1] = "N"
    test = load(tmp_path, [g], name="t.xyz")
    with pytest.raises(DataError, match="Z=7, which is not in the training set"):
        check_datasets({"train": cfg, "valid": cfg, "test": test}, n_energies=1, z_table_zs=[1, 6])
    with pytest.raises(DataError, match="validation set is empty"):
        check_datasets({"train": cfg, "valid": []}, n_energies=1)


def test_close_contacts_and_periodic_frames_warn(tmp_path, caplog):
    fr = labelled_frames()
    mp = fr[1].info["mm_positions"].copy()
    mp[0] = fr[1].positions[2]
    fr[1].info["mm_positions"] = mp
    fr[4].pbc = True
    fr[4].cell = [12, 12, 12]
    cfg = load(tmp_path, fr)
    with caplog.at_level(logging.WARNING):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=1)
    assert "within 0.5 A of a QM atom" in caplog.text and "are periodic" in caplog.text


def test_argument_checks(tmp_path):
    ok = tmp_path / "x.extxyz"
    ok.write_text("")
    base = dict(train_file=str(ok), valid_file=None, test_file=None, statistics_file=None, n_energies=1,
                diabatic=False, compute_nacs=False, compute_socs=False, soc_num=None, multipole_max_ell=1)
    check_arguments(types.SimpleNamespace(**base))  # .extxyz is accepted
    for change, msg in ((dict(train_file="missing.xyz"), "file not found"),
                        (dict(compute_nacs=True), "needs --n_energies >= 2"),
                        (dict(diabatic=True), "needs --n_energies >= 2"),
                        (dict(diabatic=True, n_energies=2, compute_nacs=True), "no NAC outputs"),
                        (dict(compute_socs=True), "--soc_num")):
        with pytest.raises(InputError, match=msg):
            check_arguments(types.SimpleNamespace(**{**base, **change}))


def test_cli_reports_data_errors_without_traceback(tmp_path):
    fr = labelled_frames(n=8, n_states=2)
    ase.io.write(tmp_path / "d.xyz", fr, format="extxyz")
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "run_train.py"), "--name=t",
                        f"--train_file={tmp_path / 'd.xyz'}", "--valid_fraction=0.25", "--E0s=average",
                        "--model=PerAtomFieldEMACE", "--n_energies=1", "--max_num_epochs=1", "--device=cpu",
                        "--hidden_irreps=8x0e + 8x1o", "--MLP_irreps=8x0e", f"--work_dir={tmp_path}"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "Traceback" not in out
    assert "use --n_energies=2" in out


# --------------------------------------------------------------------------- #
# ASE calculator
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def model_paths(tmp_path_factory):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    d = tmp_path_factory.mktemp("m")
    p1, p2 = str(d / "m1.model"), str(d / "m2.model")
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=1), p1)
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2), p2)
    torch.set_default_dtype(old)
    return p1, p2


def test_calculator_recomputes_when_only_the_charges_change(model_paths):
    from mace.calculators import MACECalculator

    sym, qm, mm, q = methane_in_charges(6)
    calc = MACECalculator(model_paths[0], device="cpu", default_dtype="float64")
    es = []
    for scale in (0.0, 1.0, 2.0):
        a = ase.Atoms(sym, positions=qm)
        if scale:
            a.info.update(mm_positions=mm, mm_charges=scale * q)
        a.calc = calc
        es.append(a.get_potential_energy())
    assert len(set(np.round(es, 10))) == 3  # ASE's cache must not return the old field's energy


def test_calculator_errors(model_paths):
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths[0], device="cpu", default_dtype="float64")
    with pytest.raises(ValueError, match="Z=\\[16\\] are not in the model"):
        calc.calculate(ase.Atoms("SH2", positions=[[0, 0, 0], [1.3, 0, 0], [0, 1.3, 0]]))
    with pytest.raises(ValueError, match="committee models differ"):
        MACECalculator(list(model_paths), device="cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="no CUDA GPU"):
            MACECalculator(model_paths[0], device="cuda")
    sym, qm, mm, q = methane_in_charges(4)
    a = ase.Atoms(sym, positions=qm)
    a.info.update(mm_positions=np.r_[mm[:-1], qm[:1]], mm_charges=q)
    with pytest.warns(UserWarning, match="is a QM atom also listed"):
        calc.calculate(a)


# --------------------------------------------------------------------------- #
# OpenMM interface
# --------------------------------------------------------------------------- #
def test_openmm_boundary_and_embedding_checks(model_paths):
    pytest.importorskip("openmm")
    from mace.mlmm import EmbeddingSettings, create_fieldmace_mm_system, prepare_qmmm_system
    from mace.mlmm.io import load_system

    s = load_system(pdb=os.path.join(REPO, "tutorials", "01_data_and_training", "inputs", "aladip_solvated.pdb"),
                    forcefield=["amber14-all.xml", "amber14/tip3p.xml"])
    side = [8, 9, 10, 11, 12, 13]
    with pytest.raises(ValueError, match="to a hydrogen"):
        prepare_qmmm_system(s.system, [8], topology=s.topology)  # CA alone: cuts C-H bonds
    with pytest.raises(ValueError, match="Z0, Z1"):
        prepare_qmmm_system(s.system, side, topology=s.topology, link_charge_scheme="Z9")
    with pytest.warns(UserWarning, match="sp3 carbon"):
        prepare_qmmm_system(s.system, list(range(14, 22)), topology=s.topology)  # amide C on the boundary
    with pytest.warns(UserWarning, match="Z0 keeps"):
        prepare_qmmm_system(s.system, side, topology=s.topology, link_charge_scheme="Z0")
    with pytest.raises(ValueError, match="switch_width"):
        EmbeddingSettings(cutoff=2.0, switch_width=3.0)
    with pytest.raises(ValueError, match="half the box"):
        create_fieldmace_mm_system(s.system, side, model_paths[0], topology=s.topology,
                                   settings=EmbeddingSettings(cutoff=20.0), model_kwargs=dict(device="cpu"))


def test_active_learning_config_errors(tmp_path):
    pytest.importorskip("yaml")
    from mace.mlmm.active_learning import ALConfig

    (tmp_path / "d.xyz").write_text("")
    (tmp_path / "a.yaml").write_text("work_dir: w\nsetup: nope.json\ninitial_data: d.xyz\n")
    with pytest.raises(ValueError, match="setup = .*nope.json does not exist"):
        ALConfig.from_file(str(tmp_path / "a.yaml"))
    (tmp_path / "b.yaml").write_text("work_dir: w\nsetup: d.xyz\ninitial_data: d.xyz\nsampling: {temperature: 300}\n")
    with pytest.raises(ValueError, match="unknown key.*temperature"):
        ALConfig.from_file(str(tmp_path / "b.yaml"))
