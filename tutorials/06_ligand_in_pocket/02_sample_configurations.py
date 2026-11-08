#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 2: classical MD of the solvated complex; store the ligand + its MM surroundings as frames.

Every --stride-ps a frame is stored in FieldMACE's format (frame_to_atoms with the embedding of
setup.json): the 18 ligand atoms, and the protein, water and ion charges within the cutoff (switched).
Every second frame gets random displacements of the ligand atoms (sigma 0.03 or 0.06 A), so the
data also cover geometries away from the force-field minimum (the ML/MM dynamics will visit them).

Output: work/frames.xyz, work/frames_md.dcd (the MD trajectory, for viewing)
"""
import argparse
import os
import time

import numpy as np
import openmm as mm
from openmm import app, unit

from mace.mlmm import EmbeddingFrame, prepare_qmmm_system
from mace.mlmm.dataset import frame_to_atoms, load_setup, write_extxyz

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-frames", type=int, default=160)
    p.add_argument("--stride-ps", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    a = p.parse_args()
    pdb = app.PDBFile(os.path.join(a.work, "equilibrated.pdb"))
    with open(os.path.join(a.work, "system_classical.xml")) as fh:
        system = mm.XmlSerializer.deserialize(fh.read())
    qm, settings, prep_kw = load_setup(os.path.join(a.work, "setup.json"))
    _, spec = prepare_qmmm_system(system, qm, topology=pdb.topology, **prep_kw)
    frame = EmbeddingFrame(spec, settings)

    integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integ.setRandomNumberSeed(a.seed + 1)
    sim = app.Simulation(pdb.topology, system, integ)
    sim.context.setPositions(pdb.positions)
    sim.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin, a.seed + 1)
    sim.reporters.append(app.DCDReporter(os.path.join(a.work, "frames_md.dcd"), int(a.stride_ps * 500)))
    rng = np.random.default_rng(a.seed)
    frames = []
    t0 = time.time()
    for k in range(a.n_frames):
        sim.step(int(a.stride_ps * 500))
        st = sim.context.getState(getPositions=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer).copy()
        sigma = 0.0
        if k % 2 == 1:  # displaced ligand (the MD itself continues undisturbed)
            sigma = [0.03, 0.06][(k // 2) % 2]
            pos[qm] += rng.normal(scale=sigma / 10.0, size=(len(qm), 3))
        at = frame_to_atoms(frame, pos, st.getPeriodicBoxVectors(asNumpy=True),
                            extra_info={"md_time_ps": (k + 1) * a.stride_ps, "displacement_sigma": sigma,
                                        "config_type": "pocket"})
        frames.append(at)
        if (k + 1) % 40 == 0:
            print(f"  {k + 1}/{a.n_frames} frames ({time.time() - t0:.0f} s)", flush=True)
    write_extxyz(os.path.join(a.work, "frames.xyz"), frames)
    n_mm = [len(f.info["mm_charges"]) for f in frames]
    print(f"wrote {len(frames)} frames to {a.work}/frames.xyz: {len(qm)} QM atoms, {min(n_mm)}-{max(n_mm)} point "
          f"charges per frame ({a.n_frames * a.stride_ps:.0f} ps of MD)")


if __name__ == "__main__":
    main()
