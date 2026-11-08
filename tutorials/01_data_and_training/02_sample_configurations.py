#!/usr/bin/env python3
"""
Step 2: sample configurations with classical MD and write them in FieldMACE's format.

Every frame holds the QM atoms plus H link atoms (positions in Angstrom, centred on the QM
centroid) and, in atoms.info, the MM point charges FieldMACE will see: positions
("mm_positions") and charges ("mm_charges", already scaled by the cutoff switching function).
This is exactly the frame the OpenMM ML/MM force uses later (same setup.json), so training
data and simulation can never disagree about link atoms, cutoffs or charge schemes.

The number of point charges differs from frame to frame; FieldMACE >= 2 does not need padding.

Classical MD samples the force field's distribution. That is fine to start with; later add
frames from ML/MM MD with your first model (active learning, see tutorial 03).

Output: work/frames.xyz (unlabelled)
"""
import argparse
import os

import numpy as np
import openmm as mm
from openmm import app, unit

from mace.mlmm import EmbeddingFrame, prepare_qmmm_system
from mace.mlmm.dataset import frames_from_simulation, load_setup, write_extxyz

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    p.add_argument("--n-frames", type=int, default=300)
    p.add_argument("--stride-ps", type=float, default=0.5, help="time between frames (ps)")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--platform", default=None)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    with open(os.path.join(a.work, "system_classical.xml")) as fh:
        system = mm.XmlSerializer.deserialize(fh.read())
    pdb = app.PDBFile(os.path.join(a.work, "equilibrated.pdb"))
    qm_atoms, settings, prep_kw = load_setup(os.path.join(a.work, "setup.json"))
    _, spec = prepare_qmmm_system(system, qm_atoms, topology=pdb.topology, **prep_kw)
    frame = EmbeddingFrame(spec, settings)

    dt = 1.0 * unit.femtosecond  # flexible QM X-H bonds: 1 fs
    integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 1.0 / unit.picosecond, dt)
    integ.setRandomNumberSeed(a.seed)
    plat = [mm.Platform.getPlatformByName(a.platform)] if a.platform else []
    sim = app.Simulation(pdb.topology, system, integ, *plat)
    with open(os.path.join(a.work, "equilibrated.xml")) as fh:
        sim.context.setState(mm.XmlSerializer.deserialize(fh.read()))

    stride = int(round(a.stride_ps * 1000 * unit.femtosecond / dt))
    n_mm = []

    def progress(atoms, k):
        n_mm.append(len(atoms.info["mm_charges"]))
        if (k + 1) % 50 == 0 or k + 1 == a.n_frames:
            print(f"  frame {k + 1}/{a.n_frames}: {len(atoms)} QM+link atoms, {n_mm[-1]} point charges")

    frames = frames_from_simulation(sim, frame, a.n_frames, stride, callback=progress)
    out = os.path.join(a.work, "frames.xyz")
    write_extxyz(out, frames)
    print(f"wrote {len(frames)} frames to {out}; point charges per frame: {min(n_mm)}-{max(n_mm)} "
          f"(mean {np.mean(n_mm):.0f})")


if __name__ == "__main__":
    main()
