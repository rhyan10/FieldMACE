# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""TorchScript backend: the scripted ML/MM module and openmmtorch.TorchForce agree with the PythonForce."""
import os

import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mace.mlmm import EmbeddingSettings, create_fieldmace_mm_system
from mace.mlmm.torchforce import build_torch_module, export_torchscript
from mace.tools.testing import build_field_model

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDB = os.path.join(REPO, "tutorials", "01_data_and_training", "inputs", "aladip_solvated.pdb")
SIDECHAIN = [8, 9, 10, 11, 12, 13]
KJ_PER_EV = 96.48533212331002


@pytest.fixture(scope="module")
def solvated():
    pdb = app.PDBFile(PDB)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=None, rigidWater=True)
    return system, pdb


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    d = tmp_path_factory.mktemp("tf")
    paths = {"adiabatic": str(d / "a.model"), "diabatic": str(d / "d.model")}
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2), paths["adiabatic"])
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=2, diabatic=True), paths["diabatic"])
    torch.set_default_dtype(old)
    return paths


CASES = [
    ("adiabatic", SIDECHAIN, EmbeddingSettings(cutoff=8.0)),
    ("adiabatic", list(range(22)), EmbeddingSettings()),
    ("diabatic", SIDECHAIN, EmbeddingSettings(origin="com", cutoff=7.0, cutoff_reference="nearest")),
]


@pytest.mark.parametrize("kind,qm,settings", CASES)
def test_module_matches_python_force(solvated, model_files, kind, qm, settings):
    system, pdb = solvated
    _, spec, comp = create_fieldmace_mm_system(system, qm, model_files[kind], topology=pdb.topology, settings=settings,
                                               model_kwargs=dict(device="cpu", state=1))
    mod = build_torch_module(model_files[kind], spec, settings, state=1)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    box_nm = np.array([v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()])
    for state in (0, 1):
        e_ref, g_ref = comp.evaluate(pos_nm[comp.frame.particles] * 10, box_nm * 10, state=state)
        p = torch.tensor(pos_nm, requires_grad=True)
        e = mod(p, torch.tensor(box_nm), torch.tensor(float(state)))
        e.backward()
        assert float(e) / KJ_PER_EV == pytest.approx(e_ref, abs=1e-9)
        g = p.grad.numpy() / 10 / KJ_PER_EV  # kJ/mol/nm -> eV/A
        assert np.allclose(g[comp.frame.particles], g_ref, atol=1e-9)
        others = np.setdiff1d(np.arange(len(pos_nm)), comp.frame.particles)
        assert np.all(g[others] == 0)


def test_export_and_load(solvated, model_files, tmp_path):
    system, pdb = solvated
    _, spec, comp = create_fieldmace_mm_system(system, SIDECHAIN, model_files["adiabatic"], topology=pdb.topology,
                                               model_kwargs=dict(device="cpu"))
    path = export_torchscript(model_files["adiabatic"], spec, tmp_path / "fm.pt")
    mod = torch.jit.load(path)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    box_nm = np.array([v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()])
    e_ref, _ = comp.evaluate(pos_nm[comp.frame.particles] * 10, box_nm * 10)
    assert float(mod(torch.tensor(pos_nm), torch.tensor(box_nm))) / KJ_PER_EV == pytest.approx(e_ref, abs=1e-9)


def _context(system, positions, platform="Reference"):
    integ = mm.VerletIntegrator(0.5 * unit.femtosecond)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    ctx.setPositions(positions)
    return ctx, integ


@pytest.mark.parametrize("kind", ["adiabatic", "diabatic"])
def test_torchforce_matches_python_force_in_openmm(solvated, model_files, kind):
    pytest.importorskip("openmmtorch")
    system, pdb = solvated
    settings = EmbeddingSettings(cutoff=8.0)
    res = {}
    for backend in ("python", "torch"):
        s, spec, _ = create_fieldmace_mm_system(system, SIDECHAIN, model_files[kind], topology=pdb.topology,
                                                settings=settings, backend=backend, force_group=31,
                                                model_kwargs=dict(device="cpu", state=0))
        ctx, integ = _context(s, pdb.positions)
        out = []
        for state in (0, 1):
            ctx.setParameter("fieldmace_state", state)
            st = ctx.getState(getEnergy=True, getForces=True, groups={31})
            out.append((st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                        st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))
        res[backend] = out
        del ctx, integ
    for state in (0, 1):
        assert res["torch"][state][0] == pytest.approx(res["python"][state][0], abs=1e-5)
        assert np.allclose(res["torch"][state][1], res["python"][state][1], atol=1e-4)
    assert res["torch"][0][0] != pytest.approx(res["torch"][1][0])


def test_torchforce_md_runs(solvated, model_files):
    pytest.importorskip("openmmtorch")
    system, pdb = solvated
    s, _, force = create_fieldmace_mm_system(system, SIDECHAIN, model_files["adiabatic"], topology=pdb.topology,
                                             settings=EmbeddingSettings(cutoff=8.0), backend="torch")
    assert force.getName() == "FieldMACE"
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtosecond)
    sim = app.Simulation(pdb.topology, s, integ, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(pdb.positions)
    sim.step(5)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert np.isfinite(e)


def test_dynamic_graph(solvated, model_files):
    system, pdb = solvated
    settings = EmbeddingSettings(cutoff=8.0)
    _, spec, comp = create_fieldmace_mm_system(system, list(range(22)), model_files["adiabatic"], topology=pdb.topology,
                                               settings=settings, model_kwargs=dict(device="cpu"))
    mod = build_torch_module(model_files["adiabatic"], spec, settings, static_graph=False)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    box_nm = np.array([v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()])
    e_ref, _ = comp.evaluate(pos_nm[comp.frame.particles] * 10, box_nm * 10)
    assert float(mod(torch.tensor(pos_nm), torch.tensor(box_nm))) / KJ_PER_EV == pytest.approx(e_ref, abs=1e-9)
