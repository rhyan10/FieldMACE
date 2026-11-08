#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 3: QM labels for every frame with PySCF (DFT in the field of the MM point charges).

For each frame FieldMACE needs (eV, Angstrom):
  REF_energy     QM energy INCLUDING the interaction with the point charges
                 (electrons-charges and nuclei-charges; this is what QM codes report)
  REF_forces     forces on the QM atoms and the H link atoms
  REF_mm_forces  forces on the point charges (recommended: they teach the model the field response)

PySCF is used because it installs with pip and is good enough for a tutorial. Any QM code
that takes external point charges works the same way; mace.mlmm.dataset has ORCA helpers
(write_orca_pointcharges / write_orca_input / read_orca_engrad / read_orca_pcgrad).

The script can be stopped and restarted: finished frames are kept in work/labelled_partial.xyz.
At the end the data are split into data/train.xyz and data/test.xyz.

Cost: about 3-5 s per frame (8 atoms, ~200 charges, B3LYP/6-31G*, 2 cores).
"""
import argparse
import os
import shutil
import time

import ase.io
import numpy as np

from mace.mlmm.dataset import BOHR_TO_A, HARTREE_TO_EV, attach_reference, write_extxyz

HERE = os.path.dirname(os.path.abspath(__file__))


def pyscf_qmmm(symbols, qm_pos, mm_pos, mm_q, xc="b3lyp", basis="6-31g*", charge=0, spin=0):
    """Energy (eV), QM forces and point-charge forces (eV/A) from PySCF with point charges."""
    from pyscf import dft, gto, qmmm

    mol = gto.M(atom=list(zip(symbols, qm_pos)), basis=basis, unit="Angstrom", charge=charge, spin=spin, verbose=0)
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-10
    if len(mm_q):
        mf = qmmm.mm_charge(mf, mm_pos, mm_q, unit="Angstrom")  # adds electron- and nucleus-charge terms
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF not converged")
    g = mf.nuc_grad_method()
    g.verbose = 0
    g_qm = g.kernel()  # Eh/bohr, includes the point-charge field
    g_mm = None
    if len(mm_q):
        dm = mf.make_rdm1()
        g_mm = g.grad_hcore_mm(dm) + g.grad_nuc_mm()  # Eh/bohr on the charges
    conv = HARTREE_TO_EV / BOHR_TO_A
    return e * HARTREE_TO_EV, -g_qm * conv, None if g_mm is None else -g_mm * conv


def compact(at, decimals=7):
    """Round floats before writing (smaller files; 1e-7 eV or A is far below DFT accuracy)."""
    at.positions = np.round(at.positions, decimals)
    for k, v in list(at.info.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            at.info[k] = np.round(v, decimals) + 0.0
    return at


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    p.add_argument("--frames", default=None, help="unlabelled frames (default: work/frames.xyz)")
    p.add_argument("--partial", default=None, help="restart file (default: work/labelled_partial.xyz)")
    p.add_argument("--out", default=os.path.join(HERE, "data"))
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--basis", default="6-31g*")
    p.add_argument("--charge", type=int, default=0, help="charge of the QM region incl. link atoms")
    p.add_argument("--spin", type=int, default=0, help="2S")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from pyscf import lib
    lib.num_threads(a.threads)
    frames = ase.io.read(a.frames or os.path.join(a.work, "frames.xyz"), index=":")
    if a.max_frames:
        frames = frames[: a.max_frames]
    partial = a.partial or os.path.join(a.work, "labelled_partial.xyz")
    done = ase.io.read(partial, index=":") if os.path.exists(partial) else []
    print(f"{len(frames)} frames, {len(done)} already labelled; {a.xc}/{a.basis}, {a.threads} threads")

    t0, n0 = time.time(), len(done)
    for k in range(len(done), len(frames)):
        at = frames[k]
        e, f, fm = pyscf_qmmm(at.get_chemical_symbols(), at.positions, at.info["mm_positions"],
                              at.info["mm_charges"], a.xc, a.basis, a.charge, a.spin)
        attach_reference(at, e, f, fm)
        at.info["qm_method"] = f"{a.xc}/{a.basis}"
        compact(at)
        write_extxyz(partial, [at], append=True)
        done.append(at)
        if (k + 1) % 10 == 0 or k + 1 == len(frames):
            per = (time.time() - t0) / (k + 1 - n0)
            print(f"  {k + 1}/{len(frames)} labelled, {per:.1f} s/frame, "
                  f"~{per * (len(frames) - k - 1) / 60:.0f} min left")

    # ---- split -------------------------------------------------------------------------------
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(done))
    n_test = int(round(a.test_fraction * len(done)))
    test = [done[i] for i in sorted(idx[:n_test])]
    train = [done[i] for i in sorted(idx[n_test:])]
    write_extxyz(os.path.join(a.out, "train.xyz"), train)
    if test:
        write_extxyz(os.path.join(a.out, "test.xyz"), test)
    if os.path.exists(os.path.join(a.work, "setup.json")):  # the embedding these data were made with
        shutil.copy(os.path.join(a.work, "setup.json"), os.path.join(a.out, "setup.json"))
    e = np.array([float(np.ravel(x.info["REF_energy"])[0]) for x in done])
    print(f"wrote {len(train)} training and {len(test)} test frames to {a.out}/")
    print(f"energy range {e.min():.3f} .. {e.max():.3f} eV (spread {np.ptp(e):.3f} eV)")


if __name__ == "__main__":
    main()
