#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 5: use the trained model from Python - ASE calculator and SHARC calculator.

  1. ASE (MACECalculator): all states' energies, forces of a chosen state, NACs, dipoles, SOCs,
     for a QM region in the field of point charges given in atoms.info.
  2. Solvent effect: the same geometries with and without the water charges.
  3. SharcCalculator: the 8x8 SHARC Hamiltonian (energies + SOCs, atomic units), gradients,
     NACs and dipoles in SHARC's layout - exactly what SHARC_FIELDMACE.py hands to SHARC.
  4. A C=S bond scan in the field of one water configuration (compared with LVC if $SHARC is set).

Outputs: printed numbers, results/scan_CS.png
"""
import argparse
import importlib.util
import os
import tempfile

import ase.io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mace.calculators import MACECalculator  # noqa: E402
from mace.calculators.sharc_calculator import SharcCalculator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BOHR_TO_A = 0.529177210903
STATES = ["S0", "S1", "T1", "T2"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.path.join(HERE, "models", "ch2s_es.model"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--frame", type=int, default=None, help="solvated test frame (default: first typical one)")
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    frames = ase.io.read(os.path.join(HERE, "data", "test.xyz"), index=":")
    solv = [f for f in frames if "mm_charges" in f.info]
    # a typical frame: no water hydrogen-bonded closer than 2.3 A (those frames have larger errors,
    # see README); pass --frame to look at any other one
    dmin = [np.min(np.linalg.norm(f.info["mm_positions"][:, None] - f.positions[None], axis=-1)) for f in solv]
    k = a.frame if a.frame is not None else next(i for i, d in enumerate(dmin) if d > 2.3)
    solv = [solv[k]] + solv[:k] + solv[k + 1:]
    print(f"test frame {k}: closest water charge {dmin[k]:.2f} A from a QM atom")

    # ---- 1. ASE ----------------------------------------------------------------------------
    print("=== 1. ASE calculator ===")
    calc = MACECalculator(a.model, device=a.device, state=1)  # state=1: S1 drives energy/forces
    at = solv[0].copy()
    at.info = {"mm_positions": solv[0].info["mm_positions"], "mm_charges": solv[0].info["mm_charges"]}
    at.calc = calc
    print("E(S1) =", round(at.get_potential_energy(), 4), "eV   (atoms.get_potential_energy uses state=1)")
    r = calc.results
    print("energies of all states (eV):", r["energies"], "  LVC:", np.ravel(solv[0].info["REF_energy"]))
    print("forces on the QM atoms in S1 (eV/A):\n", r["forces"])
    print("forces on the first 3 point charges, all states (eV/A):\n", r["mm_forces"][:3])
    print("smooth NAC S0-S1, (E1 - E0) * NAC (eV/A, sign arbitrary):\n", r["nacs"][:, 0])
    print("permanent dipoles (e A):\n", r["dipoles"][:4], "\ntransition dipole S0-S1:", r["dipoles"][4])
    print("SOC vector (eV, 28 (re, im) pairs, first 6 values):", r["socs"][:6])

    # ---- 2. solvent effect ------------------------------------------------------------------
    print("\n=== 2. same geometries with and without the water ===")
    train = ase.io.read(os.path.join(HERE, "data", "train.xyz"), index=":")
    gas = {f.info["md_frame"]: f for f in train + frames if f.info.get("config_type") == "gas"}
    pairs = [(f, gas[f.info["md_frame"]]) for f in train + frames
             if "mm_charges" in f.info and f.info["md_frame"] in gas][:40]
    calc.state = 0
    shift_ml, shift_ref = [], []
    for f_solv, f_gas in pairs:
        x = f_solv.copy()
        x.info = {"mm_positions": f_solv.info["mm_positions"], "mm_charges": f_solv.info["mm_charges"]}
        calc.calculate(x)
        e_s = calc.results["energies"]
        y = f_gas.copy()
        y.info = {}  # no point charges = gas phase
        calc.calculate(y)
        e_g = calc.results["energies"]
        shift_ml.append((e_s - e_s[0]) - (e_g - e_g[0]))
        rs, rg = np.ravel(f_solv.info["REF_energy"]), np.ravel(f_gas.info["REF_energy"])
        shift_ref.append((rs - rs[0]) - (rg - rg[0]))
    print("solvent shift of the excitation energies (meV), mean over", len(pairs), "geometries:")
    for s in range(1, 4):
        print(f"  {STATES[s]}: FieldMACE {1000 * np.mean(np.array(shift_ml)[:, s]):7.1f}   "
              f"LVC {1000 * np.mean(np.array(shift_ref)[:, s]):7.1f}")

    # ---- 3. SHARC calculator ----------------------------------------------------------------
    print("\n=== 3. SharcCalculator (what SHARC receives; atomic units) ===")
    sc = SharcCalculator(["C", "S", "H", "H"], a.model, device=a.device, states=[2, 0, 2], nac_type="smooth",
                         properties=["energy", "forces", "nacs", "dipoles", "socs"])
    print(f"mapping: model has {sc.n_model_states} states -> SHARC states {sc.states} ({sc.nmstates} with M_S), "
          f"mode '{sc.mode}'")
    f0 = solv[0]
    qm = sc.get_qm(sc.predict(f0.positions / BOHR_TO_A, f0.info["mm_positions"], f0.info["mm_charges"]),
                   as_arrays=True)
    print("diag(h) (Hartree):", np.real(np.diag(qm["h"])))
    print("|SOC| block S0,S1 x T1(M_S=-1..+1) (cm-1):\n", np.abs(qm["h"][:2, 2:]) * 219474.63)
    print("shapes: grad", qm["grad"].shape, " grad_pc", qm["grad_pc"].shape, " nacdr", qm["nacdr"].shape,
          " dm", qm["dm"].shape)
    soc_ref = f0.info["REF_socs"].reshape(-1, 2) @ np.array([1, 1j]) / 27.211386245988
    h_ref = np.zeros((8, 8), complex)
    h_ref[np.triu_indices(8, 1)] = soc_ref
    err = min(np.abs(np.triu(qm["h"], 1) - s * h_ref).max() for s in (1, -1))  # overall sign is arbitrary
    print(f"max |SOC error| vs LVC: {err * 219474.63:.1f} cm-1 (largest SOC {np.abs(h_ref).max() * 219474.63:.1f} cm-1)")

    # ---- 4. C=S scan -----------------------------------------------------------------------
    print("\n=== 4. C=S bond scan in the field of one water configuration ===")
    base = f0.copy()
    d = base.positions[1] - base.positions[0]
    r0 = np.linalg.norm(d)
    u = d / r0
    rs = np.linspace(r0 - 0.15, r0 + 0.35, 21)
    e_ml = []
    for r in rs:
        x = base.copy()
        x.positions[1] = x.positions[0] + r * u
        x.info = {"mm_positions": base.info["mm_positions"], "mm_charges": base.info["mm_charges"]}
        calc.calculate(x)
        e_ml.append(calc.results["energies"])
    e_ml = np.array(e_ml)
    e_lvc = None
    if os.environ.get("SHARC"):
        spec = importlib.util.spec_from_file_location("lvc", os.path.join(HERE, "02_label_with_lvc.py"))
        lvc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lvc)
        ex = os.path.join(os.environ["SHARC"], "..", "examples", "SHARC_QMMM_LVC_OPENMM")
        lab = lvc.LVCLabeler(ex, ["C", "S", "H", "H"], len(base.info["mm_charges"]), tempfile.mkdtemp())
        e_lvc = []
        for r in rs:
            pos = base.positions.copy()
            pos[1] = pos[0] + r * u
            q = lab(pos / BOHR_TO_A, base.info["mm_positions"] / BOHR_TO_A, base.info["mm_charges"])
            e_lvc.append(np.real(np.diag(q["h"]))[[0, 1, 2, 3]] * 27.211386245988)
        e_lvc = np.array(e_lvc)
    ref0 = e_ml[:, 0].min()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in range(4):
        ax.plot(rs, e_ml[:, s] - ref0, "-", color=f"C{s}", label=f"{STATES[s]} FieldMACE")
        if e_lvc is not None:
            ax.plot(rs, e_lvc[:, s] - ref0, "o", ms=3, color=f"C{s}", label=f"{STATES[s]} LVC")
    r_train = np.array([np.linalg.norm(f.positions[1] - f.positions[0]) for f in train])
    lo, hi = np.percentile(r_train, [1, 99])
    ax.axvspan(lo, hi, color="0.9", zorder=0)
    ax.set(xlabel="r(C-S) (A)", ylabel="energy (eV)", title="C=S scan in water (grey: 98% of the training data)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "scan_CS.png"), dpi=120)
    if e_lvc is not None:
        mid = (rs >= lo) & (rs <= hi)
        print(f"scan MAE vs LVC inside the training range {lo:.2f}-{hi:.2f} A (meV):",
              np.round(1000 * np.abs(e_ml - e_lvc)[mid].mean(0), 1))
    print("wrote results/scan_CS.png")


if __name__ == "__main__":
    main()
