# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Regression tests for the bugs fixed in FieldMACE 2.4.3 (numbers refer to the CHANGELOG list)."""
import json
import os
import subprocess
import sys
import types
import warnings

import ase
import ase.io
import numpy as np
import pytest
import torch

from mace.tools.testing import build_field_model, make_batch, methane_in_charges

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP_MEDIUM = os.path.join(REPO, "mace", "calculators", "foundations_models", "2023-12-03-mace-mp.model")
PDB = os.path.join(REPO, "tutorials", "01_data_and_training", "inputs", "aladip_solvated.pdb")


def frames(n=12, n_states=1, mm=(0, 5, 12), zs=(1, 6, 7, 8)):
    t = build_field_model("PerAtomFieldEMACE", seed=99, n_energies=n_states, atomic_numbers=list(zs))
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
    return out


def run_train(tmp_path, train, *extra, ok=True):
    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_train.py"), "--name=t", f"--train_file={train}",
           "--valid_fraction=0.25", "--seed=1", "--E0s=average", "--model=PerAtomFieldEMACE", "--batch_size=4",
           "--max_num_epochs=1", "--device=cpu", "--hidden_irreps=8x0e + 8x1o", "--MLP_irreps=8x0e",
           "--num_radial_basis=6", "--correlation=2", f"--work_dir={tmp_path}", *extra]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=1200)
    log = r.stdout + r.stderr
    if ok:
        assert r.returncode == 0, log[-3000:]
    return r.returncode, log


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rank", [0, 1])
def test_nan_validation_loss_is_not_saved_and_all_ranks_stop(monkeypatch, rank):
    """1, 2: a NaN loss is never 'best'; early stopping happens on every rank."""
    import importlib

    tr = importlib.import_module("mace.tools.train")

    losses = iter([1.0, float("nan"), float("nan"), float("nan"), 0.5, 0.4])
    monkeypatch.setattr(tr, "evaluate", lambda **kw: (next(losses), {"mae_e": 0.1}))
    monkeypatch.setattr(tr, "train_one_epoch", lambda **kw: None)
    saved = []
    handler = types.SimpleNamespace(save=lambda state, epochs, keep_last: saved.append(epochs))
    sched = types.SimpleNamespace(step=lambda **kw: None)
    model = torch.nn.Linear(1, 1)
    tr.train(model=model, loss_fn=None, train_loader=None, valid_loader=None, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
             lr_scheduler=sched, start_epoch=0, max_num_epochs=5, patience=2, checkpoint_handler=handler,
             logger=types.SimpleNamespace(log=lambda d: None), eval_interval=1, output_args={}, device="cpu",
             log_errors="", rank=rank)
    # epoch 0 (initial loss 1.0 is the reference), epochs 0 and 1 NaN -> patience 2 reached, stop before 0.5
    assert not saved  # NaN epochs never saved, and the loop stopped (rank 1 too) before the 0.5 epoch


def test_distributed_without_slurm_fails(tmp_path):
    """3."""
    ase.io.write(tmp_path / "d.xyz", frames(), format="extxyz")
    code, log = run_train(tmp_path, tmp_path / "d.xyz", "--distributed", ok=False)
    assert code == 1 and "needs a SLURM job" in log


@pytest.mark.skipif(not os.path.exists(MP_MEDIUM), reason="shipped MACE-MP medium model")
def test_fine_tuning_from_mace_mp(tmp_path):
    """5, 12, 30: first block matched (skip connection transferred), E0s from the foundation, float32."""
    ase.io.write(tmp_path / "d.xyz", frames(zs=(1, 6)), format="extxyz")
    _, log = run_train(tmp_path, tmp_path / "d.xyz", f"--foundation_model={MP_MEDIUM}", "--E0s=foundation",
                       "--default_dtype=float32")
    assert "--interaction_first=RealAgnosticInteractionBlock replaced by RealAgnosticResidualInteractionBlock" in log
    assert "not transferred" not in log and "Using Atomic Energies from foundation model" in log


def test_input_checks(tmp_path):
    """6, 26, 41, 42, 45, 49."""
    from mace.tools import build_default_arg_parser
    from mace.tools.arg_parser_tools import check_args
    from mace.tools.data_checks import InputError, check_arguments

    p = build_default_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--name=x", "--train_file=x.xyz", "--scheduler=ExponentialLr"])
    args = p.parse_args(["--name=x", "--train_file=x.xyz", "--hidden_irreps=32x0e + 16x1o"])
    with pytest.raises(InputError, match="same number of channels"):
        check_args(args)
    ok = tmp_path / "x.xyz"
    ok.write_text("")
    h5 = tmp_path / "v.h5"
    h5.write_text("")
    base = vars(p.parse_args(["--name=x", f"--train_file={ok}", "--E0s=average"]))
    check_arguments(types.SimpleNamespace(**base))
    for change, msg in ((dict(E0s="missing.json"), "file not found"), (dict(radial_MLP="[64, 64"), "cannot be read"),
                        (dict(num_interactions=0), "at least 1"), (dict(r_max=-1.0), "positive"),
                        (dict(valid_file=str(h5)), "same format"), (dict(foundation_model="nope.model"), "file not found")):
        with pytest.raises(InputError, match=msg):
            check_arguments(types.SimpleNamespace(**{**base, **change}))


# --------------------------------------------------------------------------- #
# models and data
# --------------------------------------------------------------------------- #
def test_model_argument_checks():
    """44, 45."""
    for kw, msg in ((dict(max_ell=0), "max_ell=0"), (dict(num_interactions=0), "num_interactions"),
                    (dict(r_max=0.0), "r_max"), (dict(correlation=0), "correlation")):
        with pytest.raises(ValueError, match=msg):
            build_field_model("PerAtomFieldEMACE", **kw)


def test_neighbour_list_far_from_origin():
    """18: same edges, and no memory blow-up, for QM coordinates far from the origin."""
    import resource
    import time

    from mace.data.neighborhood import get_neighborhood

    pos = np.random.default_rng(0).normal(size=(20, 3)) * 2
    ref = get_neighborhood(pos, cutoff=5.0)[0]
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.time()
    e, s, _ = get_neighborhood(pos + 400.0, cutoff=5.0)
    assert time.time() - t < 0.5 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before < 500_000
    assert sorted(map(tuple, e.T)) == sorted(map(tuple, ref.T)) and not s.any()


def test_label_layout_checks(tmp_path):
    """19, 39, 48, 25, 50."""
    from mace import data
    from mace.tools.data_checks import DataError, check_datasets

    fr = frames(n=6, n_states=2)
    fr[3].info.pop("REF_energy")
    fr[3].info["REF_forces"] = fr[3].info["REF_forces"][:, :1]
    ase.io.write(tmp_path / "a.xyz", fr, format="extxyz")
    with pytest.raises(DataError, match="'REF_forces' \\(no energy in this frame\\) has 1 state"):
        data.load_from_xyz(str(tmp_path / "a.xyz"), config_type_weights={}, n_outputs=2)
    fr = frames(n=6)
    fr[1].positions[0, 0] = np.nan
    ase.io.write(tmp_path / "b.xyz", fr, format="extxyz")
    with pytest.raises(DataError, match="non-finite"):
        data.load_from_xyz(str(tmp_path / "b.xyz"), config_type_weights={})
    fr = frames(n=6)
    fr[2].positions[2] = fr[2].positions[1]
    ase.io.write(tmp_path / "c.xyz", fr, format="extxyz")
    cfg = data.load_from_xyz(str(tmp_path / "c.xyz"), config_type_weights={})[1]
    with pytest.raises(DataError, match="two QM atoms 0.000 A apart"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=1)
    cfg = cfg[:2]
    with pytest.raises(DataError, match="not in --atomic_numbers"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=1, z_table_zs=[6], atomic_numbers_given=True)
    for c in cfg:
        c.energy_weight = 0.0
    with pytest.raises(DataError, match="--energy_key=energy --forces_key=forces"):
        check_datasets({"train": cfg, "valid": cfg}, n_energies=1, first_frame_info=["energy", "forces"])


def test_write_extxyz_frames_without_charges(tmp_path):
    """47."""
    pytest.importorskip("openmm")
    from mace import data
    from mace.mlmm.dataset import write_extxyz

    a = ase.Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    a.info.update(mm_positions=np.zeros((0, 3)), mm_charges=np.zeros(0), mm_indices=np.zeros(0, int), REF_energy=-1.0)
    write_extxyz(str(tmp_path / "e.xyz"), [a, a])
    c = data.load_from_xyz(str(tmp_path / "e.xyz"), config_type_weights={})[1][0]
    assert float(c.energy[0]) == -1.0 and c.energy_weight == 1.0 and len(c.mm_charges) == 0


def test_preprocess_and_hdf5_checks(tmp_path):
    """35, 46."""
    ase.io.write(tmp_path / "d.xyz", frames(n=12, zs=(1, 6)), format="extxyz")
    xe = frames(n=2, zs=(1, 6))
    xe[0].symbols[1] = "Xe"
    ase.io.write(tmp_path / "t.xyz", xe, format="extxyz")
    pre = [sys.executable, "-m", "mace.cli.preprocess_data", f"--train_file={tmp_path / 'd.xyz'}", "--valid_fraction=0.25",
           f"--h5_prefix={tmp_path / 'h5'}/", "--compute_statistics", "--E0s=average", "--r_max=5.0", "--num_process=2"]
    r = subprocess.run(pre + [f"--test_file={tmp_path / 't.xyz'}"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "Z=[54]" in r.stderr
    r = subprocess.run(pre, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_train.py"), "--name=t",
           f"--train_file={tmp_path / 'h5' / 'train'}", f"--valid_file={tmp_path / 'h5' / 'val'}",
           f"--statistics_file={tmp_path / 'h5' / 'statistics.json'}", "--model=PerAtomFieldEMACE", "--n_energies=2",
           "--max_num_epochs=1", "--device=cpu", "--hidden_irreps=8x0e + 8x1o", f"--work_dir={tmp_path}"]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 1 and "HDF5 frames have 1 energies" in r.stdout + r.stderr


def test_setup_declares_pyyaml():
    """43."""
    assert "    pyyaml\n" in open(os.path.join(REPO, "setup.cfg")).read().split("[options.extras_require]")[0]


# --------------------------------------------------------------------------- #
# calculators and tools
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    p32 = str(d / "a32.model")
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2, seed=1), p32)
    torch.set_default_dtype(torch.float64)
    p64 = str(d / "a64.model")
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2, seed=2), p64)
    torch.set_default_dtype(old)
    return {"f32": p32, "f64": p64, "dir": d}


def test_calculator_state_properties_and_committee(models):
    """10, 17, 34, 39."""
    from mace.calculators import MACECalculator

    sym, qm, mm, q = methane_in_charges(5, seed=3)
    c = MACECalculator(models["f64"], device="cpu", default_dtype="float64")
    assert "stress" not in c.implemented_properties and "node_energy" in c.implemented_properties
    a = ase.Atoms(sym, positions=qm, info=dict(mm_positions=mm, mm_charges=q))
    a.calc = c
    e = []
    for s in (0, 1):
        c.state = s
        e.append(a.get_potential_energy())
    assert np.allclose(e, c.results["energies"])
    assert abs(c.get_property("node_energy", a).sum() - e[1]) < 1e-8
    cc = MACECalculator([models["f32"], models["f64"]], device="cpu")
    b = a.copy()
    b.calc = cc
    assert np.isfinite(b.get_potential_energy())
    d = a.copy()
    d.positions[2] = d.positions[1]
    with pytest.raises(ValueError, match="apart"):
        c.calculate(d)


def test_upstream_models_in_the_calculator():
    """23, 24."""
    from mace.calculators import mace_anicc, mace_mp

    ch4 = ase.Atoms("CH4", positions=[[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89],
                                      [-0.36, -0.51, -0.89]])
    calcs = [mace_anicc(device="cpu")]
    if os.path.exists(MP_MEDIUM):
        calcs.append(mace_mp(model="medium", device="cpu", default_dtype="float64"))
    for c in calcs:
        a = ch4.copy()
        a.calc = c
        assert np.isfinite(a.get_potential_energy()) and np.abs(a.get_forces().sum(0)).max() < 1e-6


def test_download_is_atomic(tmp_path):
    """13: a web page is not cached as a model."""
    from mace.calculators.foundations_models import _download

    page = tmp_path / "p.html"
    page.write_text("<html>not a model</html>")
    with pytest.raises(RuntimeError, match="web page"):
        _download(page.as_uri(), str(tmp_path / "cache" / "m.model"))
    assert not os.listdir(tmp_path / "cache")


def test_plot_train(tmp_path):
    """21."""
    pytest.importorskip("matplotlib")
    rows = [dict(loss=1.0 / (e + 1), mae_e=0.1, mae_f=0.2, mae_mm_f=0.01, mode=m, epoch=e) for e in range(3)
            for m in ("eval", "opt")]
    (tmp_path / "t_run-1_train.txt").write_text("\n".join(json.dumps(r) for r in rows))
    r = subprocess.run([sys.executable, "-m", "mace.cli.plot_train", "--path", str(tmp_path)], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0 and (tmp_path / "t.pdf").exists(), r.stderr[-1000:]


def test_eval_configs_errors(tmp_path, models):
    """29."""
    r = subprocess.run([sys.executable, "-m", "mace.cli.eval_configs", "--configs", "nope.xyz", "--model", models["f64"],
                        "--output", "x.xyz"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "ERROR:" in r.stderr and "Traceback" not in r.stderr


# --------------------------------------------------------------------------- #
# OpenMM
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def solvated():
    pytest.importorskip("openmm")
    from openmm import app, unit

    pdb = app.PDBFile(PDB)
    system = app.ForceField("amber14-all.xml", "amber14/tip3p.xml").createSystem(
        pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer)
    return pdb, system


def test_openmm_set_up_checks(solvated, models, tmp_path):
    """8, 9, 15, 36, 38."""
    from openmm import app, unit

    from mace.mlmm import EmbeddingFrame, EmbeddingSettings, create_fieldmace_mm_system, prepare_qmmm_system
    from mace.mlmm.io import load_system

    pdb, system = solvated
    with pytest.raises(ValueError, match="needs inpcrd"):
        load_system(prmtop="x.prmtop")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.deleteWater()
    mod.topology.setPeriodicBoxVectors(None)
    with open(tmp_path / "dry.pdb", "w") as fh:
        app.PDBFile.writeFile(mod.topology, mod.positions, fh)
    with pytest.warns(UserWarning, match="no box vectors"):
        load_system(pdb=str(tmp_path / "dry.pdb"), forcefield=["amber14-all.xml"])
    top = app.Topology()  # no bonds
    for ch in pdb.topology.chains():
        c = top.addChain()
        for r in ch.residues():
            rr = top.addResidue(r.name, c)
            for at in r.atoms():
                top.addAtom(at.name, at.element, rr)
    with pytest.warns(UserWarning, match="topology has no such bond"):
        prepare_qmmm_system(system, [8, 9, 10, 11, 12, 13], topology=top)
    _, spec = prepare_qmmm_system(system, list(range(22)), topology=pdb.topology)
    box = np.array([v.value_in_unit(unit.angstrom) for v in pdb.topology.getPeriodicBoxVectors()])
    pos = np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    fr = EmbeddingFrame(spec, EmbeddingSettings(cutoff=0.5 * box.diagonal().min() - 0.5, cutoff_reference="nearest"))
    with pytest.warns(UserWarning, match="two periodic images"):
        fr.numpy(pos, box)
    torch.save(build_field_model("PerAtomFieldEMACE", r_max=3.0), tmp_path / "hcno.model")
    _, spec, comp = create_fieldmace_mm_system(system, list(range(22)) + list(range(22, 70)), str(tmp_path / "hcno.model"),
                                               topology=pdb.topology, model_kwargs=dict(device="cpu"))
    assert not comp.static_graph


def test_serialized_system_loads_from_another_directory(solvated, tmp_path):
    """14 (before the fix: a segfault)."""
    script = tmp_path / "s.py"
    torch.save(build_field_model("PerAtomFieldEMACE"), tmp_path / "m.model")
    script.write_text(f"""
import os, openmm as mm
from openmm import app
from mace.mlmm import create_fieldmace_mm_system
pdb = app.PDBFile({PDB!r})
s = app.ForceField("amber14-all.xml", "amber14/tip3p.xml").createSystem(pdb.topology, nonbondedMethod=app.PME)
s2, _, _ = create_fieldmace_mm_system(s, [8, 9, 10, 11, 12, 13], "m.model", topology=pdb.topology,
                                      model_kwargs=dict(device="cpu"))
xml = mm.XmlSerializer.serialize(s2)
os.chdir("/")
print("forces", mm.XmlSerializer.deserialize(xml).getNumForces())
""")
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "forces" in r.stdout, r.stderr[-2000:]


# --------------------------------------------------------------------------- #
# active learning
# --------------------------------------------------------------------------- #
def test_al_config_checks(tmp_path):
    """27, 28, 31."""
    pytest.importorskip("yaml")
    from mace.mlmm.active_learning import ALConfig
    from mace.mlmm.labelers import pyscf_labeller

    (tmp_path / "d.xyz").write_text("")
    for extra, msg in (("", "'system' needs the MM system"), ("system: {pbd: d.xyz}\n", "unknown key"),
                       ("system: {pdb: d.xyz}\ntrain_args: {lr0: 1}\n", "unknown mace_run_train option"),
                       ("system: {pdb: d.xyz}\ntrain_args: {train_file: d.xyz}\n", "set by the loop")):
        (tmp_path / "a.yaml").write_text("work_dir: w\nsetup: d.xyz\ninitial_data: d.xyz\n" + extra)
        with pytest.raises(ValueError, match=msg):
            ALConfig.from_file(str(tmp_path / "a.yaml"))
    with pytest.raises(ValueError, match="basis_set"):
        pyscf_labeller([], basis_set="def2-svp")


def test_al_resumes_after_raising_rounds_and_reports_errors(tmp_path, monkeypatch):
    """22, 40."""
    pytest.importorskip("yaml")
    from mace.cli import active_learning as cli
    from mace.mlmm import active_learning as al

    loop = al.ActiveLearningLoop.__new__(al.ActiveLearningLoop)
    loop.c = types.SimpleNamespace(rounds=3, work_dir=str(tmp_path))
    loop.log = lambda m: None
    loop.state_file = str(tmp_path / "state.json")
    loop.state = {"round": 2, "done": [], "history": [], "finished": True, "reason": "rounds"}
    ran = []
    for ph in al.PHASES:
        monkeypatch.setattr(loop, f"_{ph}", lambda r, ph=ph: ran.append(ph) or True, raising=False)
    loop.run()
    assert ran == al.PHASES and loop.state["round"] == 3

    class Boom:
        def __init__(self, *a, **k):
            pass

        def run(self, max_rounds=None):
            raise RuntimeError("training failed, see x.log")

        def status(self):
            return "status"

    monkeypatch.setattr(al, "ALConfig", types.SimpleNamespace(from_file=lambda p: None))
    monkeypatch.setattr(al, "ActiveLearningLoop", Boom)
    assert cli.main(["--config", "x.yaml"]) == 1
