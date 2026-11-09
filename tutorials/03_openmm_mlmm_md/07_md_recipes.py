#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Part 7: practical ML/MM MD recipes - everything a production run needs, on tutorial 01's system.

  1. platform: the fastest available OpenMM platform (CUDA > HIP > OpenCL > CPU) and the model device
  2. time step and constraints: X-H constraints everywhere except in the QM region
  3. NVT, then NPT with a Monte Carlo barostat
  4. reporters: thermodynamics, trajectory, FieldMACE energies, and energy components by force group
  5. restart: save the state, continue later (also on another machine / platform)
  6. PythonForce or TorchForce, and what the embedding cutoff costs

Outputs (results/): recipes_thermo.csv, recipes.dcd, recipes_components.csv, recipes_components.png,
recipes_state.xml (restart file), recipes_fieldmace.csv
"""
import argparse
import importlib.util
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openmm as mm  # noqa: E402
from openmm import app, unit  # noqa: E402

from mace.mlmm import EmbeddingSettings, FieldMACEStateReporter, create_fieldmace_mm_system  # noqa: E402
from mace.mlmm.dataset import load_setup  # noqa: E402
from mace.mlmm.io import constrain_hbonds_except, load_system  # noqa: E402
from common import default_pdb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T01 = os.path.join(HERE, "..", "01_data_and_training")
GROUPS = {"bonded": 0, "nonbonded (MM)": 1, "FieldMACE": 2, "other": 3}


def fastest_platform(name=None):
    """The fastest OpenMM platform, with mixed precision on GPUs."""
    if name:
        plat = mm.Platform.getPlatformByName(name)
    else:
        plat = max((mm.Platform.getPlatform(k) for k in range(mm.Platform.getNumPlatforms())),
                   key=lambda p: p.getSpeed())
    props = {"Precision": "mixed"} if plat.getName() in ("CUDA", "HIP", "OpenCL") else {}
    return plat, props


def assign_groups(system):
    """Force groups so that the energy can be split into components."""
    for f in system.getForces():
        if f.getName() == "FieldMACE":
            f.setForceGroup(GROUPS["FieldMACE"])
        elif isinstance(f, (mm.HarmonicBondForce, mm.HarmonicAngleForce, mm.PeriodicTorsionForce,
                            mm.RBTorsionForce, mm.CMAPTorsionForce)):
            f.setForceGroup(GROUPS["bonded"])
        elif isinstance(f, (mm.NonbondedForce, mm.CustomNonbondedForce)):
            f.setForceGroup(GROUPS["nonbonded (MM)"])
        else:
            f.setForceGroup(GROUPS["other"])


class ComponentReporter:
    """OpenMM reporter: potential energy of every force group (kJ/mol) to a CSV file."""

    def __init__(self, path, interval):
        self.fh = open(path, "w")
        self.interval = interval
        self.fh.write("step,time_ps," + ",".join(GROUPS) + "\n")

    def describeNextReport(self, simulation):
        return {"steps": self.interval - simulation.currentStep % self.interval, "periodic": None, "include": []}

    def report(self, simulation, state):
        es = [simulation.context.getState(getEnergy=True, groups={g}).getPotentialEnergy()
              .value_in_unit(unit.kilojoule_per_mole) for g in GROUPS.values()]
        t = simulation.context.getState().getTime().value_in_unit(unit.picosecond)
        self.fh.write(f"{simulation.currentStep},{t:.4f}," + ",".join(f"{e:.4f}" for e in es) + "\n")
        self.fh.flush()


def ms_per_step(system, s, plat, props, steps):
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtosecond)
    sim = app.Simulation(s.topology, system, integ, plat, props)
    sim.context.setPositions(s.positions)
    sim.context.setPeriodicBoxVectors(*s.box_vectors)
    sim.step(2)
    t0 = time.time()
    sim.step(steps)
    return 1000 * (time.time() - t0) / steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(T01, "models", "aladip_PerAtomFieldEMACE.model"))
    p.add_argument("--setup", default=os.path.join(T01, "data", "setup.json"))
    p.add_argument("--pdb", default=None, help="start structure (default: tutorial 01's equilibrated.pdb)")
    p.add_argument("--nvt-ps", type=float, default=0.5)
    p.add_argument("--npt-ps", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--platform", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--bench-steps", type=int, default=20)
    a = p.parse_args()
    a.pdb = a.pdb or default_pdb()
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)

    # 1. platform
    plat, props = fastest_platform(a.platform)
    print(f"1. OpenMM platform {plat.getName()} {props}; model device: {a.device or 'cuda if available'}")

    # 2. constraints: built without constraints, then X-H constraints for the MM region only
    s = load_system(pdb=a.pdb, forcefield=["amber14-all.xml", "amber14/tip3p.xml"], constraints=None, cutoff_nm=0.9)
    qm, settings, prep_kw = load_setup(a.setup)
    n_c = constrain_hbonds_except(s.system, s.topology, qm)
    print(f"2. {n_c} X-H constraints added in the MM peptide (waters are rigid anyway), none in the QM region:")
    print("   its C-H bonds vibrate, so dt <= 1 fs (0.5 fs here); FieldMACE's C-H frequencies come from the model")
    system, spec, comp = create_fieldmace_mm_system(s.system, qm, a.model, topology=s.topology, settings=settings,
                                                    model_kwargs=dict(device=a.device), **prep_kw)
    assign_groups(system)

    # 3. NVT, then NPT
    dt = a.dt_fs * unit.femtosecond
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, dt)
    sim = app.Simulation(s.topology, system, integ, plat, props)
    sim.context.setPositions(s.positions)
    sim.context.setPeriodicBoxVectors(*s.box_vectors)
    sim.minimizeEnergy(maxIterations=100)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)

    # 4. reporters
    every = max(1, int(10 / a.dt_fs))  # every 10 fs
    sim.reporters += [
        app.StateDataReporter(os.path.join(out, "recipes_thermo.csv"), every, step=True, time=True,
                              potentialEnergy=True, temperature=True, volume=True, density=True, speed=True),
        app.DCDReporter(os.path.join(out, "recipes.dcd"), every * 5),
        ComponentReporter(os.path.join(out, "recipes_components.csv"), every),
        FieldMACEStateReporter(os.path.join(out, "recipes_fieldmace.csv"), every, comp),
    ]
    n_nvt = int(a.nvt_ps * 1000 / a.dt_fs)
    t0 = time.time()
    sim.step(n_nvt)
    print(f"3. NVT {a.nvt_ps} ps: {1000 * (time.time() - t0) / max(n_nvt, 1):.1f} ms/step")
    baro = mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25)
    baro.setForceGroup(GROUPS["other"])
    system.addForce(baro)
    sim.context.reinitialize(preserveState=True)  # the barostat is part of the System from now on
    v0 = sim.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
    n_npt = int(a.npt_ps * 1000 / a.dt_fs)
    sim.step(n_npt)
    v1 = sim.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
    print(f"   NPT {a.npt_ps} ps (MonteCarloBarostat, 1 bar, every 25 steps): box volume {v0:.2f} -> {v1:.2f} nm^3")
    print("   (the barostat scales molecule centres; the QM region moves with its molecule and FieldMACE is")
    print("   simply evaluated at the new positions)")
    print(f"4. wrote results/recipes_thermo.csv, recipes.dcd, recipes_components.csv, recipes_fieldmace.csv")

    # 5. restart
    state_file = os.path.join(out, "recipes_state.xml")
    saved = sim.context.getState(getPositions=True, getVelocities=True)
    sim.saveState(state_file)  # positions, velocities, box, parameters (fieldmace_state), time - portable XML
    integ2 = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, dt)
    sim2 = app.Simulation(s.topology, system, integ2, plat, props)
    sim2.loadState(state_file)
    back = sim2.context.getState(getPositions=True, getVelocities=True)
    nm, nm_ps = unit.nanometer, unit.nanometer / unit.picosecond
    dx = np.abs(back.getPositions(asNumpy=True).value_in_unit(nm) - saved.getPositions(asNumpy=True).value_in_unit(nm)).max()
    dv = np.abs(back.getVelocities(asNumpy=True).value_in_unit(nm_ps)
                - saved.getVelocities(asNumpy=True).value_in_unit(nm_ps)).max()
    sim2.step(20)
    print(f"5. saveState -> loadState in a new Simulation: t = {back.getTime().value_in_unit(unit.picosecond):.3f} ps, "
          f"positions/velocities restored to {dx:.0e} nm / {dv:.0e} nm/ps; it continues normally")
    print("   (saveCheckpoint/loadCheckpoint also keeps the integrator's random state, but only on the same platform)")

    # 6. backends and cutoff cost
    print("6. cost per step (CPU numbers are dominated by the model; on a GPU everything is faster):")
    for cut in (6.0, settings.cutoff, None):
        st = EmbeddingSettings(origin=settings.origin, cutoff=cut, switch_width=settings.switch_width,
                               cutoff_reference=settings.cutoff_reference)
        sysc, _, c = create_fieldmace_mm_system(s.system, qm, a.model, topology=s.topology, settings=st,
                                                model_kwargs=dict(device=a.device), **prep_kw)
        ms = ms_per_step(sysc, s, plat, props, a.bench_steps)
        note = "  <- setup.json (what the model was trained with)" if cut == settings.cutoff else ""
        print(f"   PythonForce, cutoff {str(cut):>5s} A: {c.last['n_mm_charges']:5d} point charges, {ms:6.1f} ms/step{note}")
    if importlib.util.find_spec("openmmtorch"):
        syst, _, _ = create_fieldmace_mm_system(s.system, qm, a.model, topology=s.topology, settings=settings,
                                                backend="torch", **prep_kw)
        print(f"   TorchForce,  cutoff {settings.cutoff} A: {ms_per_step(syst, s, plat, props, a.bench_steps):6.1f} ms/step")
    else:
        print("   TorchForce: openmm-torch not installed (conda-forge); see README part 1")
    print("   Only the cutoff the model was trained with is physically right; the others show the cost.")

    # energy components plot
    d = np.genfromtxt(os.path.join(out, "recipes_components.csv"), delimiter=",", names=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for x, key, label in zip(ax, ("bonded", "nonbonded_MM", "FieldMACE"), ("bonded (MM)", "nonbonded (MM)", "FieldMACE")):
        x.plot(d["time_ps"], d[key] - d[key][0])
        x.set(title=label, xlabel="time (ps)", ylabel="E - E(0) (kJ/mol)")
        x.axvline(a.nvt_ps, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "recipes_components.png"), dpi=120)
    print("wrote results/recipes_components.png (dotted line: NVT -> NPT)")


if __name__ == "__main__":
    main()
