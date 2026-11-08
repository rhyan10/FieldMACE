# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Automated active learning: a random FieldMACE 'teacher' plays the QM method."""
import json
import os
import shutil

import ase.io
import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mace.mlmm.active_learning import ActiveLearningLoop, ALConfig, select_frames
from mace.mlmm.labelers import get_labeller
from mace.tools.testing import build_field_model

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T01 = os.path.join(REPO, "tutorials", "01_data_and_training")


def test_select_frames():
    fs = np.array([0.01, 0.5, 0.45, 2.0, 0.3, 0.2, 0.06])
    traj = np.array([0, 0, 0, 0, 1, 1, 1])
    assert select_frames(fs, traj, 3, 0.05, 1.0, min_separation=2) == [1, 4, 6]  # 2 is next to 1, 3 is too wild
    assert select_frames(fs, traj, 10, 0.05, 1.0, min_separation=1) == [1, 2, 4, 5, 6]


@pytest.fixture(scope="module")
def al_inputs(tmp_path_factory):
    """setup.json of tutorial 01 (alanine side chain + 2 link atoms), 12 labelled classical frames."""
    from mace.mlmm import EmbeddingSettings, create_fieldmace_mm_system
    from mace.mlmm.dataset import frame_to_atoms, load_setup, write_extxyz
    from mace.mlmm.labelers import model_labeller

    d = tmp_path_factory.mktemp("al")
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.save(build_field_model("PerAtomFieldEMACE", seed=7), d / "teacher.model")
    torch.set_default_dtype(old)
    shutil.copy(os.path.join(T01, "data", "setup.json"), d / "setup.json")
    pdb = app.PDBFile(os.path.join(T01, "inputs", "aladip_solvated.pdb"))
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds)
    qm, settings, prep = load_setup(str(d / "setup.json"))
    _, spec, comp = create_fieldmace_mm_system(system, qm, str(d / "teacher.model"), topology=pdb.topology,
                                               settings=settings, model_kwargs=dict(device="cpu"), **prep)
    sim = app.Simulation(pdb.topology, system, mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                                            1 * unit.femtosecond))
    sim.context.setPositions(pdb.positions)
    frames = []
    for k in range(12):
        sim.step(5)
        st = sim.context.getState(getPositions=True)
        frames.append(frame_to_atoms(comp.frame, st.getPositions(asNumpy=True), st.getPeriodicBoxVectors(asNumpy=True)))
    write_extxyz(str(d / "train.xyz"), model_labeller(frames, model=str(d / "teacher.model")))
    return d


def write_config(d, labeller, rounds=2):
    cfg = f"""
work_dir: run_{labeller.split(':')[0]}
system: {{pdb: {os.path.join(T01, 'inputs', 'aladip_solvated.pdb')}, forcefield: [amber14-all.xml, amber14/tip3p.xml], cutoff_nm: 0.9}}
setup: setup.json
initial_data: train.xyz
committee_size: 2
train_args: {{model: PerAtomFieldEMACE, E0s: average, r_max: 4.0, num_interactions: 2, correlation: 2,
  hidden_irreps: 8x0e + 8x1o, MLP_irreps: 8x0e, num_radial_basis: 6, multipole_max_ell: 1, valid_fraction: 0.2,
  batch_size: 4, max_num_epochs: 2, default_dtype: float64, error_table: EnergyNacsDipoleMAE}}
sampling: {{temperatures: [300], n_frames: 6, every: 2, dt_fs: 0.5, minimize: false}}
selection: {{n_select: 3, force_std_min: 0.0, force_std_max: 1000.0, min_separation: 1}}
labeller: {labeller}
rounds: {rounds}
converged_fraction: 0.0
device: cpu
"""
    path = d / f"al_{labeller.split(':')[0]}.yaml"
    path.write_text(cfg)
    return str(path)


def test_loop_with_model_labeller(al_inputs):
    d = al_inputs
    cfg = ALConfig.from_file(write_config(d, "model:teacher.model"))
    loop = ActiveLearningLoop(cfg, log=lambda m: None)
    st = loop.run()
    assert st["finished"] and st["round"] == 2 and len(st["history"]) == 2
    n0 = len(ase.io.read(d / "train.xyz", index=":"))
    for r, n in ((0, n0 + 3), (1, n0 + 6)):
        rd = d / cfg.work_dir / f"round_{r:02d}"
        assert len(ase.io.read(rd / "train.xyz", index=":")) == n
        assert (rd / "models" / "model_1.model").exists()
        fr = ase.io.read(rd / "frames.xyz", index=":")
        assert len(fr) == 6 and "committee_force_std" in fr[0].info
        lab = ase.io.read(rd / "labelled.xyz", index=":")
        assert np.shape(lab[0].info["REF_mm_forces"])[0] == len(lab[0].info["mm_charges"])
    assert loop.train_file(1).endswith(os.path.join("round_00", "train.xyz"))  # round 1 trains on the extended set
    with open(d / cfg.work_dir / "state.json") as fh:
        assert json.load(fh)["finished"]


def test_loop_external_labelling_and_resume(al_inputs):
    d = al_inputs
    path = write_config(d, "external", rounds=1)
    cfg = ALConfig.from_file(path)
    st = ActiveLearningLoop(cfg, log=lambda m: None).run()
    assert not st["finished"] and st["done"] == ["train", "sample", "select"]  # waiting for labels
    rd = d / cfg.work_dir / "round_00"
    cand = ase.io.read(rd / "candidates.xyz", index=":")
    assert len(cand) == 3
    lab = get_labeller("model:" + str(d / "teacher.model"))
    ase.io.write(rd / "labelled.xyz", lab(cand), format="extxyz")
    st = ActiveLearningLoop(ALConfig.from_file(path), log=lambda m: None).run()  # resumes, no retraining
    assert st["finished"] and st["round"] == 1
    assert len(ase.io.read(rd / "train.xyz", index=":")) == len(ase.io.read(d / "train.xyz", index=":")) + 3


def test_cli_example_parses(tmp_path):
    import yaml

    from mace.cli.active_learning import EXAMPLE

    d = yaml.safe_load(EXAMPLE)
    assert set(d) <= set(ALConfig.__dataclass_fields__)
