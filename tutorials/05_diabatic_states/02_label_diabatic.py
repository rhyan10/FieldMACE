#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
"""
Step 2: diabatic labels - the diabatic Hamiltonian, its derivatives, diabatic dipoles and SOCs.

The reference is SHARC's LVC model of CH2S again (as in tutorial 02), which is diabatic by
construction: it builds a diabatic Hamiltonian H(R, point charges) and diagonalises it. Here we
store H itself. A QM code that gives adiabatic energies E, gradients g and NACs d works the same way
if you also have the diabatic-to-adiabatic rotation U (from a diabatisation scheme: propagation
diabatisation / overlaps, block diagonalisation, a fitted LVC/vibronic model, ...):

    H = U diag(E) U^T
    dH/dR = U A U^T  with  A_kk = g_k,  A_kl = (E_l - E_k) d_kl       (d_kl = <k|d/dR|l>)

(mace.tools.diabatic.labels_from_adiabatic). The same formula on the point-charge gradients and NACs
gives dH/dR on the point charges. This script checks it against finite differences once.

States: 4 spatial diabatic states S_a, S_b, T_a, T_b (SHARC "states 2 0 2"). Couplings between a
singlet and a triplet are zero (different spin); spin-orbit coupling is separate.

Labels per frame (FieldMACE units eV, Angstrom, e):
  REF_energy     (1, 10)     [H_00 H_11 H_22 H_33 | H_01 H_02 H_03 H_12 H_13 H_23]
  REF_forces     (4, 10, 3)  -dH/dR for every element, QM atoms
  REF_mm_forces  (300, 10, 3) -dH/dR on the point charges
  REF_dipoles    (10, 3)     diabatic dipole matrix, same element order
  REF_socs       (56,)       SOC matrix of the 8 diabatic M_S states, upper triangle, (re, im) pairs
and, for the evaluation only, the adiabatic reference: REF_adiabatic_energy (1, 4), REF_nacs (4, 6, 3).

Diabatic quantities have no arbitrary signs and no singularities, so no phase tricks or smooth NACs
are needed.
"""
import argparse
import importlib.util
import os
import shutil
import tempfile

import ase.io
import numpy as np

from mace.mlmm.dataset import write_extxyz
from mace.tools.diabatic import labels_from_adiabatic, pack, state_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
T02 = os.path.join(HERE, "..", "02_excited_states")
HARTREE_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903
SP = [0, 1, 2, 3]  # SHARC states S0 S1 T1(-1) T2(-1): one per spatial state


def load_lvc_labeler():
    spec = importlib.util.spec_from_file_location("lvc02", os.path.join(T02, "02_label_with_lvc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LVCLabeler


def diabatic_labels(q):
    """LVC output (atomic units, 8 SHARC states) -> diabatic labels in eV / Angstrom."""
    E = np.real(np.diag(q["h"]))
    U = q["U"]
    lab = labels_from_adiabatic(E[SP], U[np.ix_(SP, SP)], q["grad"][SP], q["nacdr"][np.ix_(SP, SP)],
                                q.get("grad_pc", None)[SP] if "grad_pc" in q else None,
                                q["nacdr_pc"][np.ix_(SP, SP)] if "nacdr_pc" in q else None)
    f = HARTREE_TO_EV / BOHR_TO_A
    out = {
        "REF_energy": pack(lab["H"]).reshape(1, -1) * HARTREE_TO_EV,
        "REF_forces": -np.moveaxis(pack(lab["dH"]), 0, 1) * f,
    }
    if "dH_pc" in lab:
        out["REF_mm_forces"] = -np.moveaxis(pack(lab["dH_pc"]), 0, 1) * f
    mu_ad = np.moveaxis(np.real(q["dm"]), 0, -1)[np.ix_(SP, SP)]  # [4, 4, 3], e bohr
    Us = U[np.ix_(SP, SP)]
    out["REF_dipoles"] = pack(np.einsum("ik,klx,jl->ijx", Us, mu_ad, Us)) * BOHR_TO_A
    soc_ad = np.array(q["h"]) - np.diag(np.diag(q["h"]))  # SOC part, adiabatic M_S basis
    soc_dia = U @ soc_ad @ U.T
    iu = np.triu_indices(soc_dia.shape[0], 1)
    v = soc_dia[iu] * HARTREE_TO_EV
    out["REF_socs"] = np.stack([v.real, v.imag], axis=1).reshape(-1)
    # adiabatic reference for the evaluation
    out["REF_adiabatic_energy"] = (E[SP] * HARTREE_TO_EV).reshape(1, -1)
    out["REF_nacs"] = np.stack([q["nacdr"][i, j] for i, j in state_pairs(4)], axis=1) / BOHR_TO_A
    return out


def compact(at, decimals=8):
    at.positions = np.round(at.positions, decimals)
    for k, v in list(at.info.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            at.info[k] = np.round(v, decimals) + 0.0
    return at


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    p.add_argument("--out", default=os.path.join(HERE, "data"))
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--gas-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not os.environ.get("SHARC"):
        raise SystemExit("This step needs SHARC 4 (the LVC reference): set $SHARC")
    ex = os.path.join(os.environ["SHARC"], "..", "examples", "SHARC_QMMM_LVC_OPENMM")
    frames = ase.io.read(os.path.join(a.work, "frames.xyz"), index=":")
    n_pc = len(frames[0].info["mm_charges"])
    tmp = tempfile.mkdtemp(prefix="lvc_")
    lab = load_lvc_labeler()(ex, frames[0].get_chemical_symbols(), n_pc, tmp)

    def run(at, pc_pos, pc_q):
        return lab(at.positions / BOHR_TO_A, pc_pos / BOHR_TO_A, pc_q)

    # one finite-difference check of dH/dR (labels and their derivatives must belong together)
    at0 = frames[0]
    lab0 = diabatic_labels(run(at0, at0.info["mm_positions"], at0.info["mm_charges"]))
    h = 1e-4
    x1, x2 = at0.copy(), at0.copy()
    x1.positions[1, 2] += h
    x2.positions[1, 2] -= h
    fd = (diabatic_labels(run(x1, at0.info["mm_positions"], at0.info["mm_charges"]))["REF_energy"]
          - diabatic_labels(run(x2, at0.info["mm_positions"], at0.info["mm_charges"]))["REF_energy"]) / (2 * h)
    err = np.abs(fd.ravel() + lab0["REF_forces"][1, :, 2]).max()
    print(f"finite-difference check of dH/dR: max error {err:.2e} eV/A")
    if err > 1e-4:
        raise SystemExit("diabatic labels and their derivatives do not match")

    for k, at in enumerate(frames):
        for key, v in diabatic_labels(run(at, at.info["mm_positions"], at.info["mm_charges"])).items():
            at.info[key] = v
        at.info["config_type"] = "solvated"
        compact(at)
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(frames)} labelled")

    rng = np.random.default_rng(a.seed + 1)
    far, zero = np.full((n_pc, 3), 1000.0), np.zeros(n_pc)
    gas = []
    for k in rng.choice(len(frames), int(round(a.gas_fraction * len(frames))), replace=False):
        at = frames[k].copy()
        for key in ("mm_positions", "mm_charges", "mm_indices", "REF_mm_forces"):
            at.info.pop(key, None)
        for key, v in diabatic_labels(run(at, far, zero)).items():
            if key != "REF_mm_forces":
                at.info[key] = v
        at.info["config_type"] = "gas"
        gas.append(compact(at))
    frames += gas
    shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(a.out, exist_ok=True)
    idx = np.random.default_rng(a.seed).permutation(len(frames))
    n_test = int(round(a.test_fraction * len(frames)))
    write_extxyz(os.path.join(a.out, "test.xyz"), [frames[i] for i in sorted(idx[:n_test])])
    write_extxyz(os.path.join(a.out, "train.xyz"), [frames[i] for i in sorted(idx[n_test:])])
    H = np.array([f.info["REF_energy"].ravel() for f in frames if f.info["config_type"] == "solvated"])
    print(f"wrote {len(frames) - n_test} training and {n_test} test frames to {a.out}/")
    print("diabatic couplings |H_01| (S_a-S_b), |H_23| (T_a-T_b): mean "
          f"{1000 * np.abs(H[:, 4]).mean():.1f}, {1000 * np.abs(H[:, 9]).mean():.1f} meV; "
          f"max {1000 * np.abs(H[:, 4]).max():.1f}, {1000 * np.abs(H[:, 9]).max():.1f} meV (zero in the gas phase: "
          "here they come from the water's field)")


if __name__ == "__main__":
    main()
