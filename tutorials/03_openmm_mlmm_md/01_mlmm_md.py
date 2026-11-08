#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Part 1: ML/MM molecular dynamics in OpenMM with the ground-state model from tutorial 01.

The QM region (alanine side chain, 2 H link atoms) is described by FieldMACE, electrostatically
embedded in the point charges of the rest of the system; everything else is the force field
(Amber14 + TIP3P here; any force field OpenMM reads works the same way).

What happens in create_fieldmace_mm_system:
  * bonded terms with only QM atoms are removed, QM charges are zeroed (QM-QM and QM-MM
    Coulomb come from FieldMACE), QM-QM Lennard-Jones is excluded, QM-MM Lennard-Jones is kept
  * link atoms are placed on the QM-MM bonds; forces on them are passed to the real atoms
  * a PythonForce runs the model every step (GPU if available), forces on QM atoms, link
    atoms AND point charges come from one backward pass

The embedding (link atoms, cutoff, switching, origin) is read from the setup.json written with the
training data, so the model is evaluated exactly as it was trained.

Runs: minimisation -> NVT (Langevin) -> NVE (energy conservation check).
Outputs: results/mlmm_nvt.dcd, results/mlmm_nvt.csv, results/mlmm_nve.csv
"""
import argparse
import os
import time

import numpy as np
import openmm as mm
from openmm import app, unit

from mace.mlmm import create_fieldmace_mm_system
from mace.mlmm.dataset import load_setup
from mace.mlmm.io import constrain_hbonds_except, load_system

HERE = os.path.dirname(os.path.abspath(__file__))
T01 = os.path.join(HERE, "..", "01_data_and_training")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(T01, "models", "aladip_PerAtomFieldEMACE.model"))
    p.add_argument("--setup", default=os.path.join(T01, "data", "setup.json"))
    p.add_argument("--pdb", default=os.path.join(T01, "inputs", "aladip_solvated.pdb"))
    p.add_argument("--forcefield", nargs="*", default=["amber14-all.xml", "amber14/tip3p.xml"])
    p.add_argument("--nvt-ps", type=float, default=2.0)
    p.add_argument("--nve-ps", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=0.5, help="QM X-H bonds are flexible: 0.5-1 fs")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--device", default=None, help="model device (default: cuda if available)")
    p.add_argument("--platform", default=None, help="OpenMM platform (default: fastest available)")
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    # 1. the classical system, built exactly like the one used for the training data
    s = load_system(pdb=a.pdb, forcefield=a.forcefield, constraints=None, cutoff_nm=0.9)
    qm_atoms, settings, prep_kw = load_setup(a.setup)
    constrain_hbonds_except(s.system, s.topology, qm_atoms)
    print(f"QM atoms {qm_atoms}, embedding {settings}, {prep_kw}")

    # 2. ML/MM system: one call
    system, spec, comp = create_fieldmace_mm_system(
        s.system, qm_atoms, a.model, topology=s.topology, settings=settings,
        model_kwargs=dict(device=a.device), **prep_kw)
    print(f"FieldMACE on {comp.device}, {len(spec.model_atomic_numbers)} atoms incl. {len(spec.link_atoms)} link "
          f"atoms, model dtype {comp.dtype}")

    # 3. NVT
    dt = a.dt_fs * unit.femtosecond
    integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 1.0 / unit.picosecond, dt)
    plat = [mm.Platform.getPlatformByName(a.platform)] if a.platform else []
    sim = app.Simulation(s.topology, system, integ, *plat)
    sim.context.setPositions(s.positions)
    if s.box_vectors is not None:
        sim.context.setPeriodicBoxVectors(*s.box_vectors)
    sim.minimizeEnergy(maxIterations=200)
    sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin, 1)
    n_nvt = int(a.nvt_ps * 1000 / a.dt_fs)
    every = max(1, n_nvt // 20)
    sim.reporters.append(app.DCDReporter(os.path.join(HERE, "results", "mlmm_nvt.dcd"), every))
    sim.reporters.append(app.StateDataReporter(os.path.join(HERE, "results", "mlmm_nvt.csv"), every, step=True,
                                               time=True, potentialEnergy=True, temperature=True, speed=True))
    t0 = time.time()
    sim.step(n_nvt)
    ms = 1000 * (time.time() - t0) / n_nvt
    T = sim.context.getState(getEnergy=True).getKineticEnergy() / (0.5 * unit.MOLAR_GAS_CONSTANT_R * (
        3 * system.getNumParticles() - system.getNumConstraints() - 3))
    print(f"NVT {a.nvt_ps} ps on {sim.context.getPlatform().getName()}: {ms:.1f} ms/step, "
          f"final T = {T.value_in_unit(unit.kelvin):.0f} K; last FieldMACE energy {comp.last['energies'][0]:.4f} eV, "
          f"{comp.last['n_mm_charges']} point charges")

    # 4. NVE: total energy should be conserved (drift / fluctuation tell you dt is fine and the
    #    forces are consistent with the energy - including the link-atom and switching terms)
    state = sim.context.getState(getPositions=True, getVelocities=True)
    nve = mm.VerletIntegrator(dt)
    sim2 = app.Simulation(s.topology, system, nve, *plat)
    sim2.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    sim2.context.setPositions(state.getPositions())
    sim2.context.setVelocities(state.getVelocities())
    n_nve = int(a.nve_ps * 1000 / a.dt_fs)
    rows = []
    for k in range(20):
        sim2.step(max(1, n_nve // 20))
        st = sim2.context.getState(getEnergy=True)
        rows.append([sim2.currentStep * a.dt_fs / 1000, st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                     st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)])
    rows = np.array(rows)
    etot = rows[:, 1] + rows[:, 2]
    np.savetxt(os.path.join(HERE, "results", "mlmm_nve.csv"), np.c_[rows, etot], delimiter=",",
               header="time_ps,E_pot_kJmol,E_kin_kJmol,E_tot_kJmol")
    slope = np.polyfit(rows[:, 0], etot, 1)[0]
    print(f"NVE {a.nve_ps} ps: total energy std {etot.std():.3f} kJ/mol, drift {slope:.2f} kJ/mol/ps "
          f"(kinetic energy fluctuates by {rows[:, 2].std():.1f} kJ/mol)")


if __name__ == "__main__":
    main()
