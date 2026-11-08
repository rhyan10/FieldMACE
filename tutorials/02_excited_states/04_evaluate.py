#!/usr/bin/env python3
"""
Step 4: test-set errors for every property, and parity plots.

Couplings between two states (NACs, transition dipoles, SOCs) have an arbitrary sign, so they
are compared after choosing the better sign: per state pair for NACs and transition dipoles,
once per frame for the SOC vector (the same convention the training loss uses).

Outputs: printed error table, results/parity_*.png
"""
import argparse
import os

import ase.io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mace.calculators import MACECalculator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = ["S0", "S1", "T1", "T2"]
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def best_sign(ref, pred, axis_pairs=None):
    """Flip pred where that matches ref better. axis_pairs: axis holding the pairs (None: one sign)."""
    if axis_pairs is None:
        return pred if np.sum(ref * pred) >= 0 else -pred
    pred = np.array(pred, copy=True)
    for p in range(pred.shape[axis_pairs]):
        r, q = np.take(ref, p, axis=axis_pairs), np.take(pred, p, axis=axis_pairs)
        if np.sum(r * q) < 0:
            idx = [slice(None)] * pred.ndim
            idx[axis_pairs] = p
            pred[tuple(idx)] *= -1
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.path.join(HERE, "models", "ch2s_es.model"))
    p.add_argument("--test", default=os.path.join(HERE, "data", "test.xyz"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    calc = MACECalculator(a.model, device=a.device)
    frames = ase.io.read(a.test, index=":")
    R = {k: [] for k in ("E", "F", "Fmm", "nac", "dip", "soc")}
    P = {k: [] for k in R}
    for at in frames:
        ref = at.info
        x = at.copy()
        x.info = {k: ref[k] for k in ("mm_positions", "mm_charges") if k in ref}  # none: gas phase
        calc.calculate(x)
        res = calc.results
        R["E"].append(np.ravel(ref["REF_energy"]))
        P["E"].append(res["energies"])
        R["F"].append(ref["REF_forces"])
        P["F"].append(res["forces_all"])
        if "REF_mm_forces" in ref:
            R["Fmm"].append(ref["REF_mm_forces"])
            P["Fmm"].append(res["mm_forces"])
        R["nac"].append(ref["REF_smooth_nacs"])
        P["nac"].append(best_sign(ref["REF_smooth_nacs"], res["nacs"], axis_pairs=1))
        dref, dpred = ref["REF_dipoles"], np.array(res["dipoles"])
        dpred[4:] = best_sign(dref[4:], dpred[4:], axis_pairs=0)
        R["dip"].append(dref)
        P["dip"].append(dpred)
        R["soc"].append(ref["REF_socs"])
        P["soc"].append(best_sign(ref["REF_socs"], res["socs"]))
    R = {k: np.array(v) for k, v in R.items()}
    P = {k: np.array(v) for k, v in P.items()}
    n_gas = sum("mm_charges" not in f.info for f in frames)

    # energies relative to the mean S0 energy (absolute offsets are arbitrary)
    off = R["E"][:, 0].mean()
    gap_r, gap_p = R["E"] - R["E"][:, :1], P["E"] - P["E"][:, :1]
    coupled = [k for k, (i, j) in enumerate(PAIRS) if (i < 2) == (j < 2)]  # same multiplicity
    soc_nonzero = np.abs(R["soc"]).max(0) > 1e-6

    def mae(x, y):
        return np.mean(np.abs(x - y))

    rows = [
        ("energies (meV)", 1000 * mae(R["E"], P["E"])),
        ("excitation energies (meV)", 1000 * mae(gap_r[:, 1:], gap_p[:, 1:])),
        ("QM forces (meV/A)", 1000 * mae(R["F"], P["F"])),
        ("point-charge forces (meV/A)", 1000 * mae(R["Fmm"], P["Fmm"])),
        ("smooth NACs S0-S1, T1-T2 (eV/A)", mae(R["nac"][:, :, coupled], P["nac"][:, :, coupled])),
        ("permanent dipoles (e A)", mae(R["dip"][:, :4], P["dip"][:, :4])),
        ("transition dipoles (e A)", mae(R["dip"][:, 4:][:, coupled], P["dip"][:, 4:][:, coupled])),
        ("SOCs, non-zero elements (meV)", 1000 * mae(R["soc"][:, soc_nonzero], P["soc"][:, soc_nonzero])),
    ]
    print(f"test set: {len(frames)} frames ({n_gas} gas phase), model {os.path.basename(a.model)}")
    for name, v in rows:
        print(f"  MAE {name:32s} {v:10.4f}")
    for s in range(4):
        print(f"  MAE energy {STATES[s]}: {1000 * mae(R['E'][:, s], P['E'][:, s]):7.1f} meV")

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for s in range(4):
        ax[0, 0].scatter(R["E"][:, s] - off, P["E"][:, s] - off, s=6, label=STATES[s])
    ax[0, 0].set(xlabel="LVC energy - <E(S0)> (eV)", ylabel="FieldMACE (eV)", title="state energies")
    ax[0, 0].legend()
    ax[0, 1].scatter(R["F"].ravel(), P["F"].ravel(), s=2)
    ax[0, 1].set(xlabel="LVC force (eV/A)", ylabel="FieldMACE", title="QM forces, all states")
    ax[0, 2].scatter(R["Fmm"].ravel(), P["Fmm"].ravel(), s=1, alpha=0.3)
    ax[0, 2].set(xlabel="LVC force (eV/A)", ylabel="FieldMACE", title="forces on the point charges")
    ax[1, 0].scatter(R["nac"][:, :, coupled].ravel(), P["nac"][:, :, coupled].ravel(), s=2)
    ax[1, 0].set(xlabel="LVC (E_j - E_i) NAC_ij (eV/A)", ylabel="FieldMACE", title="smooth NACs (sign-aligned)")
    ax[1, 1].scatter(R["dip"].ravel(), P["dip"].ravel(), s=2)
    ax[1, 1].set(xlabel="LVC dipole (e A)", ylabel="FieldMACE", title="dipoles (transition: sign-aligned)")
    ax[1, 2].scatter(1000 * R["soc"][:, soc_nonzero].ravel(), 1000 * P["soc"][:, soc_nonzero].ravel(), s=2)
    ax[1, 2].set(xlabel="LVC SOC (meV)", ylabel="FieldMACE", title="SOC matrix elements")
    for x in ax.ravel():
        lo, hi = x.get_xlim()
        x.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "parity_excited_states.png"), dpi=120)
    print(f"wrote {a.out}/parity_excited_states.png")


if __name__ == "__main__":
    main()
