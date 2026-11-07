#!/usr/bin/env python
"""
Step 3: FieldMACE/MM molecular dynamics with the trained model and the same force field.

    python 03_run_mlmm_md.py --prmtop complex.prmtop --inpcrd complex.inpcrd \
        --model my_fieldmace.model --setup data/setup.json --steps 100000
"""
import argparse
import sys

import openmm as mm
from openmm import app, unit

from mace.mlmm import create_fieldmace_mm_system
from mace.mlmm.dataset import load_setup
from mace.mlmm.io import load_system

ap = argparse.ArgumentParser()
ap.add_argument("--pdb"); ap.add_argument("--forcefield", nargs="*", default=[])
ap.add_argument("--prmtop"); ap.add_argument("--inpcrd")
ap.add_argument("--top"); ap.add_argument("--gro"); ap.add_argument("--gmx-include")
ap.add_argument("--model", required=True)
ap.add_argument("--setup", required=True, help="setup.json written by 01_sample_and_export.py")
ap.add_argument("--state", type=int, default=0, help="electronic state to propagate (n_energies > 1)")
ap.add_argument("--device", default=None)
ap.add_argument("--dt-fs", type=float, default=0.5)
ap.add_argument("--temperature", type=float, default=300.0)
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--report", type=int, default=100)
ap.add_argument("--out", default="mlmm")
a = ap.parse_args()

ls = load_system(pdb=a.pdb, forcefield=a.forcefield, prmtop=a.prmtop, inpcrd=a.inpcrd, top=a.top,
                 gro=a.gro, gmx_include=a.gmx_include)
qm, settings, prep_kw = load_setup(a.setup)
system, spec, comp = create_fieldmace_mm_system(
    ls.system, qm, a.model, topology=ls.topology, settings=settings,
    model_kwargs=dict(state=a.state, device=a.device), **prep_kw)
print(f"QM atoms: {spec.n_qm}, link atoms: {len(spec.link_atoms)}, "
      f"embedding charges: {len(spec.mm_atoms)}, settings: {settings}")

integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 1 / unit.picosecond, a.dt_fs * unit.femtosecond)
sim = app.Simulation(ls.topology, system, integ)
sim.context.setPositions(ls.positions)
if ls.box_vectors is not None:
    sim.context.setPeriodicBoxVectors(*ls.box_vectors)
sim.minimizeEnergy(maxIterations=200)
sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin)
sim.reporters += [
    app.DCDReporter(f"{a.out}.dcd", a.report),
    app.StateDataReporter(sys.stdout, a.report, step=True, potentialEnergy=True, temperature=True, speed=True),
    app.StateDataReporter(f"{a.out}.csv", a.report, step=True, potentialEnergy=True, totalEnergy=True, temperature=True),
]
sim.step(a.steps)
