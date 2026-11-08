#!/usr/bin/env python
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 1: sample QM/MM configurations with a classical force field and export them
for QM labelling + FieldMACE training.

Examples
  # Amber
  python 01_sample_and_export.py --prmtop complex.prmtop --inpcrd complex.inpcrd --qm "resname LIG" -o data
  # OPLS-AA through GROMACS files (e.g. from LigParGen)
  python 01_sample_and_export.py --top topol.top --gro conf.gro --qm "resname UNK" -o data
  # any OpenMM ffxml
  python 01_sample_and_export.py --pdb solvated.pdb --forcefield amber14-all.xml amber14/tip3pfb.xml --qm "resid 42" -o data

Output (in -o):
  frames.xyz          unlabelled frames in FieldMACE extxyz convention
  qm/frame_XXXXX/     ORCA input + point-charge file per frame (if --orca)
  setup.json          QM selection + embedding settings (reuse for MD!)
"""
import argparse
import os

import openmm as mm
from openmm import app, unit

from mace.mlmm import EmbeddingFrame, EmbeddingSettings, prepare_qmmm_system
from mace.mlmm.dataset import (frames_from_simulation, save_setup, write_extxyz,
                                      write_orca_input, write_orca_pointcharges)
from mace.mlmm.io import constrain_hbonds_except, load_system, select_atoms

ap = argparse.ArgumentParser()
ap.add_argument("--pdb"); ap.add_argument("--forcefield", nargs="*", default=[])
ap.add_argument("--prmtop"); ap.add_argument("--inpcrd")
ap.add_argument("--top"); ap.add_argument("--gro"); ap.add_argument("--gmx-include")
ap.add_argument("--psf"); ap.add_argument("--crd"); ap.add_argument("--charmm-params", nargs="*", default=[])
ap.add_argument("--qm", required=True, help="e.g. 'resname LIG' or 'index 0-21'")
ap.add_argument("--origin", default="centroid", choices=["centroid", "com", "none"])
ap.add_argument("--cutoff", type=float, default=None, help="point-charge cutoff in Angstrom (default: all)")
ap.add_argument("--switch-width", type=float, default=1.0)
ap.add_argument("--link-scheme", default="Z1")
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--stride", type=int, default=500, help="MD steps between frames")
ap.add_argument("--equil-steps", type=int, default=5000)
ap.add_argument("--temperature", type=float, default=300.0)
ap.add_argument("--dt-fs", type=float, default=1.0)
ap.add_argument("--keep-qm-constraints", action="store_true",
                help="keep X-H constraints inside the QM region while sampling (default: removed, "
                     "so the training data contain bond vibrations the ML/MM MD will visit)")
ap.add_argument("--orca", action="store_true", help="also write ORCA inputs")
ap.add_argument("--orca-method", default="B3LYP D3BJ def2-SVP")
ap.add_argument("--qm-charge", type=int, default=0); ap.add_argument("--qm-mult", type=int, default=1)
ap.add_argument("-o", "--out", default="data")
a = ap.parse_args()

ls = load_system(pdb=a.pdb, forcefield=a.forcefield, prmtop=a.prmtop, inpcrd=a.inpcrd, top=a.top, gro=a.gro,
                 gmx_include=a.gmx_include, psf=a.psf, crd=a.crd, charmm_params=a.charmm_params,
                 constraints=app.HBonds if a.keep_qm_constraints else None)
qm = select_atoms(ls.topology, a.qm)
if not a.keep_qm_constraints:  # flexible X-H bonds in the QM region only
    constrain_hbonds_except(ls.system, ls.topology, qm)
print(f"{len(qm)} QM atoms")

# the spec defines what FieldMACE sees; the MM system returned here is not needed for sampling
_, spec = prepare_qmmm_system(ls.system, qm, topology=ls.topology, link_charge_scheme=a.link_scheme)
settings = EmbeddingSettings(origin=a.origin, cutoff=a.cutoff, switch_width=a.switch_width)
frame = EmbeddingFrame(spec, settings)
os.makedirs(a.out, exist_ok=True)
save_setup(os.path.join(a.out, "setup.json"), spec, settings, link_charge_scheme=a.link_scheme)

# classical sampling with the full force field
integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 1 / unit.picosecond, a.dt_fs * unit.femtosecond)
sim = app.Simulation(ls.topology, ls.system, integ)
sim.context.setPositions(ls.positions)
if ls.box_vectors is not None:
    sim.context.setPeriodicBoxVectors(*ls.box_vectors)
sim.minimizeEnergy()
sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin)
sim.step(a.equil_steps)


def write_qm(atoms, k):
    if a.orca:
        d = os.path.join(a.out, "qm", f"frame_{k:05d}")
        os.makedirs(d, exist_ok=True)
        write_orca_pointcharges(os.path.join(d, "mm.pc"), atoms)
        write_orca_input(os.path.join(d, "qm.inp"), atoms, "mm.pc", a.orca_method, a.qm_charge, a.qm_mult)
    print(f"frame {k}: {len(atoms)} QM(+link) atoms, {len(atoms.info['mm_charges'])} point charges")


frames = frames_from_simulation(sim, frame, a.frames, a.stride, callback=write_qm)
write_extxyz(os.path.join(a.out, "frames.xyz"), frames)
print("done ->", a.out)
