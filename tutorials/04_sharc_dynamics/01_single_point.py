#!/usr/bin/env python3
"""
Step 1: one SHARC single point with FieldMACE, compared with the LVC reference.

Creates singlepoint/FIELDMACE and singlepoint/LVC, each with the same QM.in (CH2S + 300 water
point charges from a tutorial-02 test frame, coordinates in bohr) and runs

    $SHARC/SHARC_FIELDMACE.py QM.in      and      $SHARC/SHARC_LVC.py QM.in

exactly as sharc.x would every time step. Both QM.out files are read with SHARC's own parser
and compared: energies, SOCs, gradients (QM atoms and point charges), NACs, dipoles.

This checks the whole chain: the SHARC interface, the state mapping (4 spatial model states ->
8 SHARC states with M_S), the units, and the quality of the model.
"""
import argparse
import os
import shutil
import subprocess
import sys

import ase.io
import numpy as np

from sharc_tools import BOHR_TO_A, HARTREE_TO_EV, T02, copy_lvc, sharc_dir, write_fieldmace_template

HERE = os.path.dirname(os.path.abspath(__file__))
REQUESTS = ["h", "soc", "dm", "grad", "nacdr"]


def write_qm_in(path, elements, coords_bohr, pc_file):
    with open(path, "w") as fh:
        fh.write(f"{len(elements)}\nsingle point\n")
        for el, (x, y, z) in zip(elements, coords_bohr):
            fh.write(f"{el} {x:.10f} {y:.10f} {z:.10f}\n")
        fh.write("unit bohr\nstates 2 0 2\ncharge 0 1 0\nsavedir ./SAVE\nstep 0\ninit\n")
        fh.write(f"point_charges {pc_file}\n" + "\n".join(REQUESTS) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(T02, "models", "ch2s_es.model"))
    p.add_argument("--frame", type=int, default=None,
                   help="solvated test frame of tutorial 02 (default: the first without a water closer than 2.3 A)")
    a = p.parse_args()
    sharc = sharc_dir()
    if not os.path.isfile(os.path.join(sharc, "SHARC_FIELDMACE.py")):
        raise SystemExit("SHARC_FIELDMACE.py is not in $SHARC: run 00_install.sh")
    from qmout import QMout

    frames = [f for f in ase.io.read(os.path.join(T02, "data", "test.xyz"), index=":") if "mm_charges" in f.info]
    dmin = [np.min(np.linalg.norm(f.info["mm_positions"][:, None] - f.positions[None], axis=-1)) for f in frames]
    k = a.frame if a.frame is not None else next(i for i, d in enumerate(dmin) if d > 2.3)
    at = frames[k]
    print(f"test frame {k}: closest water charge {dmin[k]:.2f} A from a QM atom")
    root = os.path.join(HERE, "singlepoint")
    shutil.rmtree(root, ignore_errors=True)
    out = {}
    for iface in ("FIELDMACE", "LVC"):
        d = os.path.join(root, iface)
        os.makedirs(d)
        with open(os.path.join(d, "pc.txt"), "w") as fh:  # x y z (bohr) q
            for (x, y, z), q in zip(at.info["mm_positions"] / BOHR_TO_A, at.info["mm_charges"]):
                fh.write(f"{x:.10f} {y:.10f} {z:.10f} {q:.10f}\n")
        write_qm_in(os.path.join(d, "QM.in"), at.get_chemical_symbols(), at.positions / BOHR_TO_A, "pc.txt")
        if iface == "FIELDMACE":
            write_fieldmace_template(os.path.join(d, "FIELDMACE.template"), a.model)
        else:
            copy_lvc(d)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(sharc, "..", "lib"), sharc,
                                                            os.environ.get("PYTHONPATH", "")]))
        r = subprocess.run([sys.executable, os.path.join(sharc, f"SHARC_{iface}.py"), "QM.in"], cwd=d,
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise SystemExit(f"SHARC_{iface}.py failed:\n{(r.stdout + r.stderr)[-3000:]}")
        q = QMout(filepath=os.path.join(d, "QM.out"), states=[2, 0, 2], natom=len(at), npc=len(at.info["mm_charges"]))
        out[iface] = q
        print(f"ran SHARC_{iface}.py -> {os.path.relpath(d, HERE)}/QM.out")

    f, l = out["FIELDMACE"], out["LVC"]
    names = ["S0", "S1", "T1-", "T2-", "T10", "T20", "T1+", "T2+"]
    print("\nenergies relative to LVC S0 (eV):")
    e0 = np.real(l.h[0, 0])
    print("  state     " + " ".join(f"{n:>7s}" for n in names))
    print("  FieldMACE " + " ".join(f"{(np.real(f.h[i, i]) - e0) * HARTREE_TO_EV:7.3f}" for i in range(8)))
    print("  LVC       " + " ".join(f"{(np.real(l.h[i, i]) - e0) * HARTREE_TO_EV:7.3f}" for i in range(8)))
    # Every state's wave-function sign is arbitrary (LVC's diagonalisation here vs the convention the model
    # learned), so the SOC matrix is compared after choosing the best sign for each state: h_ab -> s_a s_b h_ab.
    # NACs were trained with a free sign per pair, so they are compared pair by pair.
    import itertools
    best = max(itertools.product((1, -1), repeat=8),
               key=lambda sg: np.sum(np.real(np.conj(f.h) * l.h) * np.outer(sg, sg)))
    S = np.outer(best, best)
    iu = np.triu_indices(8, 1)
    sf, sl = (f.h * S)[iu], l.h[iu]
    print(f"SOCs: max |SOC| FieldMACE {np.abs(sf).max() * 219474.63:.1f} cm-1, LVC {np.abs(sl).max() * 219474.63:.1f} cm-1, "
          f"max error {np.abs(sf - sl).max() * 219474.63:.1f} cm-1 (after choosing the state signs)")
    print(f"gradients QM atoms: MAE {np.abs(f.grad - l.grad).mean():.2e} Eh/bohr (mean |grad| {np.abs(l.grad).mean():.2e})")
    print(f"gradients on point charges: MAE {np.abs(f.grad_pc - l.grad_pc).mean():.2e} Eh/bohr "
          f"(mean |grad_pc| {np.abs(l.grad_pc).mean():.2e})")
    for i, j, n in ((0, 1, "S0-S1"), (2, 3, "T1-T2")):
        nf, nl = f.nacdr[i, j], l.nacdr[i, j]
        nf = nf if np.sum(nf * nl) >= 0 else -nf
        print(f"NAC {n}: |FieldMACE| {np.linalg.norm(nf):.4f}, |LVC| {np.linalg.norm(nl):.4f} 1/bohr, "
              f"cosine {np.sum(nf * nl) / np.linalg.norm(nf) / np.linalg.norm(nl):.3f}")
    print(f"permanent dipole S0 (e bohr): FieldMACE {np.real(f.dm[:, 0, 0])}, LVC {np.real(l.dm[:, 0, 0])}")


if __name__ == "__main__":
    main()
