#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 3: DFT labels with PySCF - the ligand in the field of the protein, water and ions.

For every frame: energy (including the interaction with the point charges), forces on the 18 ligand
atoms and forces on every point charge. Default level: PBE/def2-SVP with density fitting, about
20-30 s per frame on 2 cores (B3LYP/6-31G* without density fitting is ~3x slower). 10% of the frames
are also labelled without point charges (gas phase), so the model knows the bare ligand too.

The labelled data of this tutorial are shipped (data/*.xyz.gz) - this step is only needed for new
data. It can be stopped and restarted (work/partial.xyz). On a cluster, split the frames with
--start/--stop and concatenate the partial files.

Output: data/train.xyz, data/test.xyz
"""
import argparse
import os
import time

import ase.io
import numpy as np

from mace.mlmm.dataset import attach_reference, write_extxyz
from mace.mlmm.labelers import pyscf_qmmm

HERE = os.path.dirname(os.path.abspath(__file__))


def compact(at, decimals=7):
    at.positions = np.round(at.positions, decimals)
    for k, v in list(at.info.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            at.info[k] = np.round(v, decimals) + 0.0
    return at


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default=os.path.join(HERE, "work", "frames.xyz"))
    p.add_argument("--xc", default="pbe")
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--no-density-fit", action="store_true")
    p.add_argument("--gas-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None, help="label only this many (quick tests)")
    p.add_argument("--partial", default=os.path.join(HERE, "work", "partial.xyz"))
    p.add_argument("--out", default=os.path.join(HERE, "data"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    frames = ase.io.read(a.frames, index=":")
    stop = len(frames) if a.stop is None else a.stop
    if a.max_frames:
        stop = min(stop, a.start + a.max_frames)
    rng = np.random.default_rng(a.seed)
    gas_ids = set(rng.choice(len(frames), int(round(a.gas_fraction * len(frames))), replace=False).tolist())
    done = {}
    if os.path.exists(a.partial):
        for at in ase.io.read(a.partial, index=":"):
            done[(int(at.info["frame_id"]), at.info["config_type"])] = at
    kw = dict(xc=a.xc, basis=a.basis, density_fit=not a.no_density_fit)
    t0, n_new = time.time(), 0
    for i in range(a.start, stop):
        jobs = [("pocket", frames[i])]
        if i in gas_ids:
            g = frames[i].copy()
            g.info = {k: v for k, v in frames[i].info.items() if k not in ("mm_positions", "mm_charges", "mm_indices")}
            g.info["config_type"] = "gas"
            jobs.append(("gas", g))
        for ctype, at in jobs:
            if (i, ctype) in done:
                continue
            at = at.copy()
            at.info["frame_id"] = i
            at.info["config_type"] = ctype
            mm_pos = at.info.get("mm_positions", np.zeros((0, 3)))
            mm_q = at.info.get("mm_charges", np.zeros(0))
            try:
                e, f, fm = pyscf_qmmm(at.get_chemical_symbols(), at.positions, mm_pos, mm_q, **kw)
            except RuntimeError as err:
                print(f"  frame {i} ({ctype}) skipped: {err}")
                continue
            attach_reference(at, e, f, fm)
            at.info["level_of_theory"] = f"{a.xc}/{a.basis}" + ("" if a.no_density_fit else " (density fitting)")
            compact(at)
            write_extxyz(a.partial, [at], append=True)
            done[(i, ctype)] = at
            n_new += 1
            if n_new % 10 == 0:
                print(f"  {n_new} new labels, {(time.time() - t0) / n_new:.1f} s per calculation", flush=True)
    labelled = [done[k] for k in sorted(done)]
    if a.stop is not None and a.stop < len(frames):
        print(f"labelled frames {a.start}-{stop - 1}; {len(labelled)} in {a.partial}. Run without --stop to finish.")
        return
    os.makedirs(a.out, exist_ok=True)
    ids = sorted({int(f.info["frame_id"]) for f in labelled})
    perm = np.random.default_rng(a.seed + 1).permutation(ids)
    test_ids = set(perm[: int(round(a.test_fraction * len(ids)))].tolist())
    test = [f for f in labelled if int(f.info["frame_id"]) in test_ids]
    train = [f for f in labelled if int(f.info["frame_id"]) not in test_ids]
    write_extxyz(os.path.join(a.out, "train.xyz"), train)
    write_extxyz(os.path.join(a.out, "test.xyz"), test)
    print(f"wrote {len(train)} training and {len(test)} test frames to {a.out}/ "
          f"({sum(f.info['config_type'] == 'gas' for f in labelled)} gas phase)")


if __name__ == "__main__":
    main()
