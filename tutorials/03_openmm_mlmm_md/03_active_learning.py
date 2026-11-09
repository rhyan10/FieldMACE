#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Part 3: active learning - find configurations from ML/MM MD that the model is unsure about.

The first model was trained on classical-MD geometries. ML/MM MD with that model visits its own
distribution, and there it can be wrong. A committee (several models trained with different seeds,
or different architectures) disagrees most where the data are missing:

  1. run ML/MM MD with model A, store a frame every --every steps (exactly in the training format,
     with frame_to_atoms and the same embedding)
  2. evaluate all committee members on these frames; disagreement = std of the energies (after
     removing each model's mean offset) and of the forces
  3. write the most uncertain frames to work/al_candidates.xyz

Then label and retrain:
  python ../01_data_and_training/03_label_with_pyscf.py --frames work/al_candidates.xyz \
         --partial work/al_labelled.xyz --out work/al_data --test-fraction 0
  cat ../01_data_and_training/data/train.xyz work/al_data/train.xyz > work/train_round2.xyz
  ... and train again on work/train_round2.xyz (04_train.sh with train_file changed).

A committee from different seeds is better: SEED=2 MODELS=PerAtomFieldEMACE bash 04_train.sh (rename the output).
"""
import argparse
import os

import ase.io
import numpy as np
import openmm as mm
from openmm import app, unit

from mace.calculators import MACECalculator
from mace.mlmm import create_fieldmace_mm_system
from mace.mlmm.dataset import frame_to_atoms, load_setup, write_extxyz
from mace.mlmm.io import constrain_hbonds_except, load_system
from common import default_pdb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T01 = os.path.join(HERE, "..", "01_data_and_training")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--committee", nargs="*",
                   default=[f for f in (os.path.join(T01, "models", "aladip_PerAtomFieldEMACE.model"),
                                        os.path.join(T01, "models", "aladip_FieldEMACE.model")) if os.path.exists(f)],
                   help="model files; the first one drives the MD")
    p.add_argument("--setup", default=os.path.join(T01, "data", "setup.json"))
    p.add_argument("--pdb", default=None, help="start structure (default: tutorial 01's equilibrated.pdb)")
    p.add_argument("--forcefield", nargs="*", default=["amber14-all.xml", "amber14/tip3p.xml"])
    p.add_argument("--n-frames", type=int, default=40)
    p.add_argument("--every", type=int, default=100, help="MD steps between stored frames")
    p.add_argument("--n-select", type=int, default=10)
    p.add_argument("--temperature", type=float, default=400.0, help="a bit hotter than the data, to find gaps")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    a.pdb = a.pdb or default_pdb()
    if len(a.committee) < 2:
        raise SystemExit("need at least two models (tutorial 01 trains PerAtomFieldEMACE and FieldEMACE)")
    os.makedirs(os.path.join(HERE, "work"), exist_ok=True)

    s = load_system(pdb=a.pdb, forcefield=a.forcefield, constraints=None, cutoff_nm=0.9)
    qm_atoms, settings, prep_kw = load_setup(a.setup)
    constrain_hbonds_except(s.system, s.topology, qm_atoms)
    system, spec, comp = create_fieldmace_mm_system(s.system, qm_atoms, a.committee[0], topology=s.topology,
                                                    settings=settings, model_kwargs=dict(device=a.device), **prep_kw)
    integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 1.0 / unit.picosecond, 0.5 * unit.femtosecond)
    sim = app.Simulation(s.topology, system, integ)
    sim.context.setPositions(s.positions)
    sim.context.setPeriodicBoxVectors(*s.box_vectors)
    sim.minimizeEnergy(maxIterations=200)
    sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin, 2)

    frames = []
    for k in range(a.n_frames):
        sim.step(a.every)
        st = sim.context.getState(getPositions=True)
        frames.append(frame_to_atoms(comp.frame, st.getPositions(asNumpy=True), st.getPeriodicBoxVectors(asNumpy=True),
                                     extra_info={"md_step": sim.currentStep, "source": "mlmm_md"}))
    print(f"ML/MM MD at {a.temperature} K with {os.path.basename(a.committee[0])}: {len(frames)} frames")

    E, F = [], []
    for path in a.committee:
        calc = MACECalculator(path, device=a.device or "cpu")
        e, f = [], []
        for at in frames:
            x = at.copy()
            x.info = {"mm_positions": at.info["mm_positions"], "mm_charges": at.info["mm_charges"]}
            calc.calculate(x)
            e.append(calc.results["energy"])
            f.append(calc.results["forces"])
        E.append(np.array(e) - np.mean(e))
        F.append(np.array(f))
    E, F = np.array(E), np.array(F)  # [models, frames], [models, frames, atoms, 3]
    e_std = E.std(0)
    f_std = np.sqrt(((F - F.mean(0)) ** 2).sum(-1).mean(0)).max(-1)  # max over atoms of the force std
    score = f_std / np.median(f_std) + e_std / np.median(e_std)
    pick = np.argsort(score)[::-1][: a.n_select]
    for i in pick:
        frames[i].info["committee_energy_std"] = float(e_std[i])
        frames[i].info["committee_force_std"] = float(f_std[i])
    out = os.path.join(HERE, "work", "al_candidates.xyz")
    write_extxyz(out, [frames[i] for i in sorted(pick)])
    print(f"committee energy std: median {1000 * np.median(e_std):.1f} meV, max {1000 * e_std.max():.1f} meV; "
          f"force std: median {1000 * np.median(f_std):.0f}, max {1000 * f_std.max():.0f} meV/A")
    print(f"wrote the {len(pick)} most uncertain frames to {out} - label them with "
          f"01_data_and_training/03_label_with_pyscf.py --frames {out}")


if __name__ == "__main__":
    main()
