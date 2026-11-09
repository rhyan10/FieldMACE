# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Regression tests for the bugs fixed in FieldMACE 2.4.2 (numbers refer to the CHANGELOG list)."""
import json
import os
import subprocess
import sys
import types

import ase
import ase.io
import numpy as np
import pytest
import torch
from e3nn import o3

from mace.tools.testing import build_field_model, make_batch, methane_in_charges

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def frames(n=12, n_states=1, dipoles=False, mm=(0, 5, 12), seed=99, zs=(1, 6, 7, 8)):
    t = build_field_model("PerAtomFieldEMACE", seed=seed, n_energies=n_states, compute_dipoles=dipoles,
                          atomic_numbers=list(zs))
    out = []
    for i in range(n):
        sym, qm, mmp, q = methane_in_charges(mm[i % len(mm)], seed=i)
        o = t(make_batch(t, [(sym, qm, mmp, q)]).to_dict())
        a = ase.Atoms(sym, positions=qm)
        if len(q):
            a.info.update(mm_positions=mmp, mm_charges=q, REF_mm_forces=o["mm_forces"].detach().numpy())
        a.info["REF_energy"] = o["energy"].detach().numpy()
        a.info["REF_forces"] = o["forces"].detach().numpy()
        if dipoles:
            a.info["REF_dipoles"] = o["dipoles"].detach().numpy().reshape(-1, 3)
        out.append(a)
    return out


def run_train(tmp_path, train, *extra, n_states=1, epochs=2, ok=True):
    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_train.py"), "--name=t", f"--train_file={train}",
           "--valid_fraction=0.25", "--seed=1", "--E0s=average", "--model=PerAtomFieldEMACE", "--r_max=5.0",
           "--batch_size=4", "--lr=0.01", f"--n_energies={n_states}", "--correlation=2",
           f"--max_num_epochs={epochs}", "--default_dtype=float64", "--device=cpu", "--hidden_irreps=8x0e + 8x1o",
           "--MLP_irreps=8x0e", "--num_radial_basis=6", "--num_interactions=2", f"--work_dir={tmp_path}", *extra]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=900)
    log = r.stdout + r.stderr
    if ok:
        assert r.returncode == 0, log[-3000:]
    return r.returncode, log


# --------------------------------------------------------------------------- #
# training command line
# --------------------------------------------------------------------------- #
def test_stage_two_patience_tables_and_unlabelled_test_set(tmp_path):
    """1, 2, 4, 13, 26, 33, 34, 40, 44: early stopping and Stage Two finish, the tables work, an unlabelled
    test set of a multi-state model is evaluated, the last epoch is evaluated and saved."""
    fr = frames(n=12, n_states=2, dipoles=True)
    ase.io.write(tmp_path / "d.xyz", fr, format="extxyz")
    unl = [ase.Atoms(a.symbols, positions=a.positions, info={k: a.info[k] for k in ("mm_positions", "mm_charges")
                                                               if k in a.info}) for a in fr[:3]]
    ase.io.write(tmp_path / "u.xyz", unl, format="extxyz")
    _, log = run_train(tmp_path, tmp_path / "d.xyz", "--swa", "--patience=1", "--lr=0.5", "--eval_interval=3",
                       "--radial_type=gaussian", "--error_table=DipoleRMSE", f"--test_file={tmp_path / 'u.xyz'}",
                       "--compute_dipoles=True", "--valid_batch_size=2", n_states=2, epochs=4)
    assert os.path.exists(tmp_path / "t.model") and os.path.exists(tmp_path / "t_stagetwo.model")
    assert "Traceback" not in log and "failed" not in log and "RMSE MU" in log


def test_e0s_json_and_partial_isolated_atoms(tmp_path):
    """16, 17, 19: JSON E0s (string keys), IsolatedAtom frames for some elements, multi-state isolated atoms."""
    (tmp_path / "e0.json").write_text(json.dumps({"1": -1.0, "6": -5.0}))
    iso = ase.Atoms("H", positions=[[0, 0, 0]], info={"config_type": "IsolatedAtom", "REF_energy": np.array([-0.5, 0.1])})
    ase.io.write(tmp_path / "d.xyz", [iso] + frames(n=12, n_states=2), format="extxyz")
    _, log = run_train(tmp_path, tmp_path / "d.xyz", f"--E0s={tmp_path / 'e0.json'}", n_states=2, epochs=1)
    assert "using --E0s=" in log and "1: -1.0" in log


def test_input_errors_are_messages(tmp_path):
    """14, 25, 29, 35, 27: clear messages instead of tracebacks."""
    from mace.tools.data_checks import InputError, check_arguments

    ok = tmp_path / "x.xyz"
    ok.write_text("")
    h5 = tmp_path / "t.h5"
    h5.write_text("")
    base = dict(train_file=str(ok), valid_file=None, test_file=None, statistics_file=None, n_energies=1,
                diabatic=False, compute_nacs=False, compute_socs=False, soc_num=None, multipole_max_ell=1,
                valid_fraction=0.1, device="cpu", E0s="average")
    check_arguments(types.SimpleNamespace(**base))
    cases = [(dict(valid_fraction=1.0), "valid_fraction"), (dict(train_file=str(h5)), "--valid_file"),
             (dict(train_file=str(h5), valid_file=str(h5)), "statistics_file")]
    if not torch.cuda.is_available():
        cases.append((dict(device="cuda"), "no CUDA GPU"))
    for change, msg in cases:
        with pytest.raises(InputError, match=msg):
            check_arguments(types.SimpleNamespace(**{**base, **change}))
    ase.io.write(tmp_path / "d.xyz", frames(n=8), format="extxyz")
    code, log = run_train(tmp_path, tmp_path / "d.xyz", "--E0s={1: oops}", ok=False)
    assert code == 1 and "Traceback" not in log and "is not valid" in log, log[-2000:]


def test_loss_and_scaling_options():
    """3, 12, 15: --loss accepts only the field-model loss; --interaction_first defaults to what was used."""
    from mace.tools import build_default_arg_parser

    p = build_default_arg_parser()
    assert p.parse_args(["--name=x", "--train_file=x.xyz"]).interaction_first == "RealAgnosticInteractionBlock"
    with pytest.raises(SystemExit):
        p.parse_args(["--name=x", "--train_file=x.xyz", "--loss=forces_only"])
    src = open(os.path.join(REPO, "mace", "cli", "run_train.py")).read()
    assert "mace-torch[schedulefree]" not in src


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_model_irreps_checks():
    """27, 28, 36."""
    m = build_field_model("PerAtomFieldEMACE", hidden_irreps=o3.Irreps("8x1o + 8x0e"))
    assert str(m.products[-1].linear.irreps_out) == "8x0e"
    with pytest.raises(ValueError, match="no vectors"):
        build_field_model("PerAtomFieldEMACE", n_energies=2, compute_dipoles=True, hidden_irreps=o3.Irreps("8x0e"))
    with pytest.raises(ValueError, match="multipole_max_ell=4"):
        build_field_model("PerAtomFieldEMACE", multipole_max_ell=4)
    build_field_model("PerAtomFieldEMACE", multipole_max_ell=4,
                      field_irreps=o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o + 4x4e"))


def test_foundation_transfer_of_a_field_model():
    """18: fine-tuning from a FieldMACE model copies every layer (identical predictions), any correlation."""
    from mace.tools.finetuning_utils import load_foundations
    from mace.tools.utils import AtomicNumberTable

    for corr in (2, 3):
        src = build_field_model("PerAtomFieldEMACE", seed=1, n_energies=2, correlation=corr)
        dst = build_field_model("PerAtomFieldEMACE", seed=2, n_energies=2, correlation=corr)
        dst = load_foundations(dst, src, AtomicNumberTable([int(z) for z in src.atomic_numbers]))
        b = make_batch(src, [methane_in_charges(7)])
        assert torch.allclose(src(b.to_dict())["energy"], dst(b.to_dict())["energy"], atol=1e-10)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def test_periodic_without_cell_and_inference_ignores_labels():
    """8, 30."""
    from mace import data

    a = ase.Atoms("CHHHH", positions=methane_in_charges(0)[1], pbc=True)
    with pytest.raises(ValueError, match="lattice vector 0"):
        data.config_for_inference(a)
    with pytest.raises(ValueError, match="lattice vector 0"):
        data.config_from_atoms(a)
    a.pbc = False
    a.info.update(REF_energy=np.zeros(3), dipoles=np.zeros((3, 3)), nacs=np.zeros(7))  # any layout
    c = data.config_for_inference(a)
    assert c.energy_weight == 0 and len(c.mm_charges) == 0


def test_normalised_atomic_energies():
    from mace.tools.scripts_utils import normalise_atomic_energies

    assert normalise_atomic_energies({"1": -1, 6: np.array([[-5.0, -4.0]])}) == {1: -1.0, 6: -5.0}


def test_preprocess_more_processes_than_frames_and_stale_shards(tmp_path):
    """31, 32, 37, 41: no crash, no stale shards, HDF5 test sets are found, mixed file/folder HDF5 inputs."""
    ase.io.write(tmp_path / "d.xyz", frames(n=12), format="extxyz")
    ase.io.write(tmp_path / "t.xyz", frames(n=3), format="extxyz")
    os.makedirs(tmp_path / "h5" / "train")
    (tmp_path / "h5" / "train" / "train_99.h5").write_text("stale")
    pre = [sys.executable, "-m", "mace.cli.preprocess_data", f"--train_file={tmp_path / 'd.xyz'}",
           f"--test_file={tmp_path / 't.xyz'}", "--valid_fraction=0.25", f"--h5_prefix={tmp_path / 'h5'}/",
           "--compute_statistics", "--E0s=average", "--r_max=5.0", "--num_process=32", "--seed=1"]
    r = subprocess.run(pre, cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert not (tmp_path / "h5" / "train" / "train_99.h5").exists()
    stats = json.loads((tmp_path / "h5" / "statistics.json").read_text())
    assert np.isfinite(stats["avg_num_neighbors"])
    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_train.py"), "--name=t",
           f"--train_file={tmp_path / 'h5' / 'train'}", f"--valid_file={tmp_path / 'h5' / 'val' / 'val_0.h5'}",
           f"--test_dir={tmp_path / 'h5' / 'test'}", f"--statistics_file={tmp_path / 'h5' / 'statistics.json'}",
           "--model=PerAtomFieldEMACE", "--batch_size=4", "--max_num_epochs=1", "--device=cpu",
           "--hidden_irreps=8x0e + 8x1o", "--MLP_irreps=8x0e", f"--work_dir={tmp_path}"]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=900)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert "Error-table on TEST" in log and "Default" in log.split("Error-table on TEST")[1]


# --------------------------------------------------------------------------- #
# calculators and command-line tools
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    p32 = str(d / "f32.model")
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2, atomic_numbers=[1, 6]), p32)
    torch.set_default_dtype(torch.float64)
    pd = str(d / "dia.model")
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2, diabatic=True, atomic_numbers=[1, 6]), pd)
    pd2 = str(d / "dia2.model")
    torch.save(build_field_model("PerAtomFieldEMACE", seed=5, n_energies=2, diabatic=True, atomic_numbers=[1, 6]), pd2)
    torch.set_default_dtype(old)
    return {"f32": p32, "dia": pd, "dia2": pd2, "dir": d}


def test_hessian_committee_and_labels(models):
    """7, 8, 9: Hessian of the chosen (adiabatic) state, committee spread of the adiabatic results."""
    from mace.calculators import MACECalculator

    sym, qm, mm, q = methane_in_charges(5, seed=3)
    for path in (models["f32"], models["dia"]):
        c = MACECalculator(path, device="cpu", default_dtype="float64", state=1)
        a = ase.Atoms(sym, positions=qm, info=dict(mm_positions=mm, mm_charges=q, REF_energy=np.zeros(5)))
        a.calc = c
        h = c.get_hessian(a).reshape(15, 15)
        fd = np.zeros((15, 15))
        for k in range(15):
            for sgn in (1, -1):
                b = a.copy()
                b.positions.reshape(-1)[k] += sgn * 1e-4
                b.calc = c
                fd[k] -= sgn * b.get_forces().reshape(-1) / 2e-4
        assert np.abs(h - fd).max() < 1e-5
    cc = MACECalculator([models["dia"], models["dia2"]], device="cpu", default_dtype="float64")
    a = ase.Atoms(sym, positions=qm, info=dict(mm_positions=mm, mm_charges=q))
    a.calc = cc
    a.get_potential_energy()
    assert cc.results["energy_var"].shape == (2,) and cc.results["forces_var"].shape == (5, 2, 3)
    assert cc.results["forces_comm"].shape == (2, 5, 3)


def test_eval_configs_dtype_and_diabatic(models, tmp_path):
    """5, 6."""
    from mace.calculators import MACECalculator
    from mace.cli.eval_configs import run

    fr = frames(n=3, zs=(1, 6))
    for a in fr:
        a.info["REF_energy"] = np.zeros(7)  # labels in the file must not matter
    ase.io.write(tmp_path / "in.xyz", fr, format="extxyz")
    for key in ("f32", "dia"):
        out = tmp_path / f"{key}.xyz"
        run(types.SimpleNamespace(configs=str(tmp_path / "in.xyz"), model=models[key], output=str(out), device="cpu",
                                  default_dtype=None, batch_size=2, compute_stress=False, return_contributions=False,
                                  info_prefix="MACE_"))
    res = ase.io.read(tmp_path / "dia.xyz", index=1)
    b = fr[1].copy()
    b.calc = MACECalculator(models["dia"], device="cpu", default_dtype="float64")
    b.get_potential_energy()
    assert np.allclose(res.info["MACE_energies"], b.calc.results["energies"])
    assert res.info["MACE_diabatic_hamiltonian"].shape == (2, 2)


def test_lammps_and_al_md_tools(models, tmp_path):
    """20, 21."""
    r = subprocess.run([sys.executable, "-m", "mace.cli.create_lammps_model", models["dia"]], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode != 0 and "LAMMPS" in r.stderr and "Traceback" not in r.stderr
    ase.io.write(tmp_path / "s.xyz", frames(n=1, zs=(1, 6)), format="extxyz")
    for model in (models["dia"], str(models["dir"] / "dia*.model")):
        out = tmp_path / "md.xyz"
        if out.exists():
            out.unlink()
        r = subprocess.run([sys.executable, "-m", "mace.cli.active_learning_md", "--config", str(tmp_path / "s.xyz"),
                            "--model", model, "--output", str(out), "--nsteps", "10", "--nsave", "5",
                            "--error_threshold", "1e6"], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]


def test_checkpoint_names_stage_two_from_its_first_epoch(tmp_path):
    """34."""
    from mace.tools.checkpoint import CheckpointIO

    io = CheckpointIO(str(tmp_path), "t", swa_start=3)
    assert io._get_checkpoint_filename(3, 3).endswith("_swa.pt")
    assert not io._get_checkpoint_filename(2, 3).endswith("_swa.pt")
    assert io._get_latest_checkpoint_path(swa=True) is None


# --------------------------------------------------------------------------- #
# active learning and labellers
# --------------------------------------------------------------------------- #
def test_al_flags_paths_and_committee(tmp_path):
    """22."""
    pytest.importorskip("yaml")
    from mace.mlmm.active_learning import ALConfig, train_flags

    assert train_flags({"ema": True, "amsgrad": False, "foundation_model_readout": False, "lr": 0.1}) == \
        ["--ema", "--foundation_model_readout", "--lr=0.1"]
    (tmp_path / "d.xyz").write_text("")
    (tmp_path / "e0.json").write_text("{}")
    sub = tmp_path / "cfg"
    sub.mkdir()
    (sub / "a.yaml").write_text("work_dir: w\nsetup: ../d.xyz\ninitial_data: ../d.xyz\nsystem: {pdb: ../d.xyz}\n"
                                "train_args: {E0s: ../e0.json, valid_file: ../d.xyz, model: PerAtomFieldEMACE}\n")
    cfg = ALConfig.from_file(str(sub / "a.yaml"))
    assert cfg.train_args["E0s"] == str(tmp_path / "e0.json") and cfg.train_args["valid_file"] == str(tmp_path / "d.xyz")
    (sub / "b.yaml").write_text("work_dir: w\nsetup: ../d.xyz\ninitial_data: ../d.xyz\nsystem: {pdb: ../d.xyz}\ncommittee_size: 1\n"
                                "initial_models: [../d.xyz, ../d.xyz]\n")
    with pytest.raises(ValueError, match="committee_size"):
        ALConfig.from_file(str(sub / "b.yaml"))


def test_al_does_not_call_an_unstable_model_converged(tmp_path, monkeypatch):
    """10."""
    pytest.importorskip("yaml")
    from mace.mlmm import active_learning as al

    loop = al.ActiveLearningLoop.__new__(al.ActiveLearningLoop)
    loop.c = types.SimpleNamespace(selection={}, device="cpu", converged_fraction=0.02, work_dir=str(tmp_path))
    loop.log = lambda m: None
    loop.state = {"round": 0, "done": [], "history": [], "finished": False}
    os.makedirs(tmp_path / "round_00")
    al.write_extxyz(str(tmp_path / "round_00" / "frames.xyz"), frames(n=4, zs=(1, 6)))
    monkeypatch.setattr(al, "committee_disagreement",
                        lambda m, f, d: {"energy_std": np.zeros(len(f)), "force_std": np.full(len(f), 5.0)})
    loop.committee = lambda r: ["a", "b"]
    loop.rdir = lambda r=None: str(tmp_path / "round_00")
    with pytest.raises(RuntimeError, match="leaves its training domain"):
        loop._select(0)
    assert not loop.state["finished"]


def test_pyscf_open_shell_point_charge_forces():
    """42."""
    pytest.importorskip("pyscf")
    from mace.mlmm.labelers import pyscf_qmmm

    qm = np.array([[0, 0, 0], [0.97, 0, 0]])
    mm = np.array([[3.0, 0.5, 0.2], [-2.5, 1.0, 0.0]])
    q = np.array([0.4, -0.3])
    _, _, fm = pyscf_qmmm(["O", "H"], qm, mm, q, basis="sto-3g", spin=1)
    es = []
    for d in (1e-4, -1e-4):
        m2 = mm.copy()
        m2[0, 0] += d
        es.append(pyscf_qmmm(["O", "H"], qm, m2, q, basis="sto-3g", spin=1)[0])
    assert abs(fm[0, 0] + (es[0] - es[1]) / 2e-4) < 1e-3


def test_calculator_length_units(models):
    """43: a model in bohr sees bohr: same numbers as an Angstrom model on the scaled geometry."""
    from mace.calculators import MACECalculator

    sym, qm, mm, q = methane_in_charges(5, seed=3)
    L = 0.529177210903
    res = []
    for scale in (1.0, L):
        c = MACECalculator(models["dia"], device="cpu", default_dtype="float64", length_units_to_A=scale)
        a = ase.Atoms(sym, positions=qm * scale, info=dict(mm_positions=mm * scale, mm_charges=q))
        a.calc = c
        res.append((a.get_potential_energy(), a.get_forces() * scale, c.results["nacs"] * scale))
    for x, y in zip(*res):
        assert np.allclose(x, y, atol=1e-10)


def test_writing_atoms_with_the_calculator(models, tmp_path):
    """45: ASE writers must not take the state energies for per-atom energies."""
    from mace.calculators import MACECalculator

    sym, qm, mm, q = methane_in_charges(5, seed=3)
    a = ase.Atoms(sym, positions=qm, info=dict(mm_positions=mm, mm_charges=q))
    a.calc = MACECalculator(models["dia"], device="cpu", default_dtype="float64")
    e = a.get_potential_energy()
    assert a.calc.results["energies"].shape == (2,) and "energies" in a.calc.results
    ase.io.write(tmp_path / "w.xyz", a)
    b = ase.io.read(tmp_path / "w.xyz")
    assert "energies" not in b.arrays and abs(b.get_potential_energy() - e) < 1e-8


def test_invariant_descriptors_of_vector_models():
    """46."""
    from scipy.spatial.transform import Rotation

    from mace.calculators import MACECalculator

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.model")
        torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2, compute_dipoles=True), p)
        c = MACECalculator(p, device="cpu", default_dtype="float64")
        sym, qm, mm, q = methane_in_charges(5, seed=3)
        R = Rotation.random(random_state=1).as_matrix()
        d1 = c.get_descriptors(ase.Atoms(sym, positions=qm, info=dict(mm_positions=mm, mm_charges=q)))
        d2 = c.get_descriptors(ase.Atoms(sym, positions=qm @ R.T, info=dict(mm_positions=mm @ R.T, mm_charges=q)))
        assert d1.shape == (5, 32) and np.abs(d1 - d2).max() < 1e-10


def test_validation_only_element_is_reported(tmp_path):
    """47."""
    from mace import data
    from mace.tools.data_checks import DataError, check_datasets

    fr = frames(n=4)
    ase.io.write(tmp_path / "t.xyz", fr, format="extxyz")
    g = fr[0].copy()
    g.symbols[1] = "N"
    ase.io.write(tmp_path / "v.xyz", [g], format="extxyz")
    tr = data.load_from_xyz(str(tmp_path / "t.xyz"), config_type_weights={})[1]
    va = data.load_from_xyz(str(tmp_path / "v.xyz"), config_type_weights={})[1]
    with pytest.raises(DataError, match="Z=7, which is not in the training set"):
        check_datasets({"train": tr, "valid": va}, n_energies=1, z_table_zs=[1, 6, 7])


def test_openmm_set_up_mistakes():
    """48, 49, 50."""
    pytest.importorskip("openmm")
    from openmm import app, unit

    from mace.mlmm import EmbeddingFrame, EmbeddingSettings, prepare_qmmm_system
    from mace.mlmm.dataset import frame_to_atoms
    from mace.mlmm.io import select_atoms

    pdb = app.PDBFile(os.path.join(REPO, "tutorials", "01_data_and_training", "inputs", "aladip_solvated.pdb"))
    system = app.ForceField("amber14-all.xml", "amber14/tip3p.xml").createSystem(
        pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer)
    with pytest.warns(UserWarning, match="dangling bonds"):
        _, spec = prepare_qmmm_system(system, [8, 9, 10, 11, 12, 13], topology=pdb.topology, link_atoms=False)
    with pytest.raises(ValueError, match="periodic: pass the box"):
        frame_to_atoms(EmbeddingFrame(spec, EmbeddingSettings(cutoff=9.0)), pdb.positions)
    for sel, msg in (("resid 2 and", "incomplete"), ("resname ACE or index 30-20", "reversed")):
        with pytest.raises(ValueError, match=msg):
            select_atoms(pdb.topology, sel)
    assert select_atoms(pdb.topology, "resid 2 and name CA CB") == [8, 10]
