#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
"""
Step 4: evaluate the diabatic model - diabatic elements, and the adiabatic quantities that follow.

  * diabatic: MAE of the diagonal energies, of the couplings and of their gradients
  * adiabatic (what dynamics uses): energies, gradients and NACs after diagonalisation, compared with
    the LVC adiabatic reference. NACs come from the model without ever being trained on them.

The ASE calculator does the diagonalisation: for a diabatic model, results["energies"], ["forces_all"],
["nacs"], ["mm_forces"], ["dipoles"] are adiabatic, and results["diabatic_hamiltonian"],
["diabatic_forces"], ... hold the raw model output. diabatic_blocks=[[0, 1], [2, 3]] diagonalises the
singlets and the triplets separately.

Output: error table, results/parity_diabatic.png
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
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
COUPLED = [k for k, (i, j) in enumerate(PAIRS) if (i < 2) == (j < 2)]  # S_a-S_b and T_a-T_b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(HERE, "models", "ch2s_diabatic.model"))
    p.add_argument("--test", default=os.path.join(HERE, "data", "test.xyz"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    calc = MACECalculator(a.model, device=a.device, diabatic_blocks=[[0, 1], [2, 3]])
    frames = ase.io.read(a.test, index=":")
    R = {k: [] for k in ("Hd", "Hc", "dH", "E", "nac")}
    P = {k: [] for k in R}
    for at in frames:
        x = at.copy()
        x.info = {k: at.info[k] for k in ("mm_positions", "mm_charges") if k in at.info}
        calc._diabatic_U = None  # independent frames: no sign continuity
        calc.calculate(x)
        r = calc.results
        Href = np.ravel(at.info["REF_energy"])
        Hm = np.array([r["diabatic_hamiltonian"][i, i] for i in range(4)]
                      + [r["diabatic_hamiltonian"][i, j] for i, j in PAIRS])
        R["Hd"].append(Href[:4]); P["Hd"].append(Hm[:4])
        R["Hc"].append(Href[4:][COUPLED]); P["Hc"].append(Hm[4:][COUPLED])
        R["dH"].append(np.asarray(at.info["REF_forces"])); P["dH"].append(r["diabatic_forces"])
        R["E"].append(np.ravel(at.info["REF_adiabatic_energy"])); P["E"].append(r["energies"])
        ref_nac = np.asarray(at.info["REF_nacs"])[:, COUPLED]
        nac = r["nacs"][:, COUPLED].copy()
        for k in range(nac.shape[1]):  # NAC signs follow the (arbitrary) adiabatic state signs
            if np.sum(nac[:, k] * ref_nac[:, k]) < 0:
                nac[:, k] *= -1
        R["nac"].append(ref_nac); P["nac"].append(nac)
    R = {k: np.array(v) for k, v in R.items()}
    P = {k: np.array(v) for k, v in P.items()}

    def mae(x, y):
        return float(np.mean(np.abs(x - y)))

    print(f"test set: {len(frames)} frames ({sum('mm_charges' not in f.info for f in frames)} gas phase)")
    print(f"  diabatic energies (diagonal)      MAE {1000 * mae(R['Hd'], P['Hd']):7.2f} meV")
    print(f"  diabatic couplings H_01, H_23     MAE {1000 * mae(R['Hc'], P['Hc']):7.2f} meV "
          f"(mean |H_ij| {1000 * np.abs(R['Hc']).mean():.1f} meV, correlation {np.corrcoef(R['Hc'].ravel(), P['Hc'].ravel())[0, 1]:.2f})")
    print(f"  gradients of all diabatic elements MAE {1000 * mae(R['dH'], P['dH']):7.2f} meV/A")
    print(f"  adiabatic energies                MAE {1000 * mae(R['E'], P['E']):7.2f} meV")
    print(f"  adiabatic NACs (never trained)    MAE {mae(R['nac'], P['nac']):.4f} 1/A "
          f"(mean |NAC| {np.abs(R['nac']).mean():.4f} 1/A)")

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    e0 = R["Hd"][:, 0].mean()
    ax[0].scatter(R["Hd"].ravel() - e0, P["Hd"].ravel() - e0, s=4)
    ax[0].set(title="diabatic energies H_ii", xlabel="LVC (eV)", ylabel="FieldMACE (eV)")
    ax[1].scatter(1000 * R["Hc"].ravel(), 1000 * P["Hc"].ravel(), s=4)
    ax[1].set(title="diabatic couplings H_01, H_23", xlabel="LVC (meV)", ylabel="FieldMACE (meV)")
    ax[2].scatter(R["E"].ravel() - e0, P["E"].ravel() - e0, s=4)
    ax[2].set(title="adiabatic energies (diagonalised)", xlabel="LVC (eV)", ylabel="FieldMACE (eV)")
    ax[3].scatter(R["nac"].ravel(), P["nac"].ravel(), s=3)
    ax[3].set(title="adiabatic NACs S0-S1, T1-T2 (from H)", xlabel="LVC (1/A)", ylabel="FieldMACE (1/A)")
    for x in ax:
        lo, hi = x.get_xlim()
        x.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "parity_diabatic.png"), dpi=120)
    print(f"wrote {a.out}/parity_diabatic.png")


if __name__ == "__main__":
    main()
