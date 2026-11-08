#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 5: how well does the model know the ligand, and the pocket's field?

  * test-set MAEs: energy, forces on the ligand atoms, forces on the point charges
  * the field's effect: for frames labelled both in the pocket and in the gas phase, the
    interaction energy E(pocket) - E(gas) from DFT and from the model (the same model, evaluated
    with and without the point charges)

Output: error table, results/parity_pocket.png
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


def predict(calc, at, with_field=True):
    x = at.copy()
    x.info = {k: at.info[k] for k in ("mm_positions", "mm_charges") if with_field and k in at.info}
    calc.calculate(x)
    r = calc.results
    return float(r["energy"]), np.array(r["forces"]), (np.array(r["mm_forces"]).reshape(-1, 3)
                                                        if "mm_forces" in r and with_field and "mm_charges" in at.info
                                                        else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(HERE, "models", "pxylene_pocket.model"))
    p.add_argument("--data", default=os.path.join(HERE, "data"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    calc = MACECalculator(a.model, device=a.device, default_dtype="float64")
    test = ase.io.read(os.path.join(a.data, "test.xyz"), index=":")
    E, F, M = ([], []), ([], []), ([], [])
    for at in test:
        e, f, fm = predict(calc, at)
        E[0].append(float(np.ravel(at.info["REF_energy"])[0])); E[1].append(e)
        F[0].append(np.asarray(at.info["REF_forces"]).reshape(-1, 3)); F[1].append(f)
        if fm is not None and "REF_mm_forces" in at.info:
            M[0].append(np.asarray(at.info["REF_mm_forces"]).reshape(-1, 3)); M[1].append(fm)
    E = np.array(E)
    n_at = len(test[0])
    F = [np.concatenate(x) for x in F]
    M = [np.concatenate(x) for x in M]
    print(f"test set: {len(test)} frames")
    print(f"  energy        MAE {1000 * np.mean(np.abs(E[0] - E[1])) / n_at:6.2f} meV/atom "
          f"({1000 * np.mean(np.abs(E[0] - E[1])):.1f} meV per frame)")
    print(f"  ligand forces MAE {1000 * np.mean(np.abs(F[0] - F[1])):6.1f} meV/A")
    print(f"  MM forces     MAE {1000 * np.mean(np.abs(M[0] - M[1])):6.2f} meV/A (mean |F| {1000 * np.mean(np.abs(M[0])):.1f})")

    # field effect: frames that have both a pocket and a gas-phase label (train and test)
    every = ase.io.read(os.path.join(a.data, "train.xyz"), index=":") + test
    by_id = {}
    for at in every:
        by_id.setdefault(int(at.info["frame_id"]), {})[at.info["config_type"]] = at
    pairs = [v for v in by_id.values() if "gas" in v and "pocket" in v]
    dft, ml = [], []
    for v in pairs:
        dft.append(float(np.ravel(v["pocket"].info["REF_energy"])[0]) - float(np.ravel(v["gas"].info["REF_energy"])[0]))
        ml.append(predict(calc, v["pocket"])[0] - predict(calc, v["gas"], with_field=False)[0])
    dft, ml = np.array(dft), np.array(ml)
    print(f"  interaction with the pocket, E(pocket) - E(gas), {len(pairs)} frames: DFT {1000 * dft.mean():.0f} "
          f"+- {1000 * dft.std():.0f} meV, model MAE {1000 * np.mean(np.abs(dft - ml)):.1f} meV")

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    e0 = E[0].mean()
    ax[0].scatter(E[0] - e0, E[1] - e0, s=6)
    ax[0].set(title="energy (test)", xlabel="DFT (eV)", ylabel="FieldMACE (eV)")
    ax[1].scatter(F[0].ravel(), F[1].ravel(), s=2)
    ax[1].set(title="forces on the ligand", xlabel="DFT (eV/A)", ylabel="FieldMACE (eV/A)")
    ax[2].scatter(1000 * M[0].ravel(), 1000 * M[1].ravel(), s=1)
    ax[2].set(title="forces on the point charges", xlabel="DFT (meV/A)", ylabel="FieldMACE (meV/A)")
    ax[3].scatter(1000 * dft, 1000 * ml, s=10)
    ax[3].set(title="E(pocket) - E(gas)", xlabel="DFT (meV)", ylabel="FieldMACE (meV)")
    for x in ax:
        lo, hi = x.get_xlim()
        x.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "parity_pocket.png"), dpi=120)
    print(f"wrote {a.out}/parity_pocket.png")


if __name__ == "__main__":
    main()
