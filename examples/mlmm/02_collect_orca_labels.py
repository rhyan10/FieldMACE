#!/usr/bin/env python
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 2: after running ORCA in every data/qm/frame_XXXXX/ (``orca qm.inp > qm.out``),
collect energies, QM forces and point-charge forces into a FieldMACE training file.

    python 02_collect_orca_labels.py data            -> data/train.xyz

Any other QM code works the same way: produce, per frame, the energy (eV, including the
interaction with the point charges), forces on the QM(+link) atoms and on the point
charges (eV/A), and call ``attach_reference``.
"""
import os
import sys

import ase.io

from mace.mlmm.dataset import attach_reference, read_orca_engrad, read_orca_pcgrad, write_extxyz

root = sys.argv[1] if len(sys.argv) > 1 else "data"
frames = ase.io.read(os.path.join(root, "frames.xyz"), ":")
done = []
for k, atoms in enumerate(frames):
    d = os.path.join(root, "qm", f"frame_{k:05d}")
    engrad, pcgrad = os.path.join(d, "qm.engrad"), os.path.join(d, "qm.pcgrad")
    if not os.path.exists(engrad):
        print("skip (no ORCA result):", d)
        continue
    e, f = read_orca_engrad(engrad)
    fm = read_orca_pcgrad(pcgrad) if os.path.exists(pcgrad) else None
    done.append(attach_reference(atoms, e, f, fm))
write_extxyz(os.path.join(root, "train.xyz"), done)
print(f"{len(done)} labelled frames -> {os.path.join(root, 'train.xyz')}")
print("Train with e.g.:\n  mace_run_train --train_file=data/train.xyz "
      "--model=PerAtomFieldEMACE --multipole_max_ell=1 --n_energies=1 --E0s=average ...")
