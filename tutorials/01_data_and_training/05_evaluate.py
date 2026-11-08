#!/usr/bin/env python3
"""
Step 5: evaluate the trained models on the test set.

  * error table: energies (meV per frame and per atom), QM forces, forces on the point charges
  * parity plots in results/
  * the same through the command-line tool:  mace_eval_configs --configs=data/test.xyz
        --model=models/aladip_PerAtomFieldEMACE.model --output=results/pred.xyz
    (writes MACE_energy, MACE_forces, MACE_mm_forces into pred.xyz)
"""
import argparse
import glob
import os

import ase.io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mace.calculators import MACECalculator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def predict(calc, frames):
    E, F, Fmm = [], [], []
    for at in frames:
        x = at.copy()
        x.info = {k: at.info[k] for k in ("mm_positions", "mm_charges") if k in at.info}
        calc.calculate(x)
        E.append(calc.results["energies"][0])
        F.append(calc.results["forces"])
        Fmm.append(calc.results.get("mm_forces", np.zeros((0, 1, 3)))[:, 0])
    return np.array(E), F, Fmm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=sorted(
        f for f in glob.glob(os.path.join(HERE, "models", "aladip_*.model")) if not f.endswith("_compiled.model")))
    p.add_argument("--test", default=os.path.join(HERE, "data", "test.xyz"))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if not a.models:
        raise SystemExit("no models found in models/: run 04_train.sh first")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    frames = ase.io.read(a.test, index=":")
    n_at = len(frames[0])
    E_ref = np.array([float(np.ravel(f.info["REF_energy"])[0]) for f in frames])
    F_ref = [np.asarray(f.info["REF_forces"]).reshape(len(f), 3) for f in frames]
    Fmm_ref = [np.asarray(f.info["REF_mm_forces"]).reshape(-1, 3) for f in frames]

    fig, ax = plt.subplots(len(a.models), 3, figsize=(12, 3.8 * len(a.models)), squeeze=False)
    print(f"test set: {len(frames)} frames, {n_at} atoms (QM + link)")
    print(f"{'model':28s} {'E meV':>8s} {'E meV/atom':>11s} {'F meV/A':>9s} {'F_mm meV/A':>11s}")
    for row, path in enumerate(a.models):
        calc = MACECalculator(path, device=a.device)
        E, F, Fmm = predict(calc, frames)
        # absolute energies include arbitrary per-element offsets (E0s); compare after removing the mean shift
        dE = (E - E_ref) - np.mean(E - E_ref)
        fe = np.concatenate([x - y for x, y in zip(F, F_ref)])
        fm = np.concatenate([x - y for x, y in zip(Fmm, Fmm_ref)])
        name = os.path.basename(path).replace(".model", "")
        print(f"{name:28s} {1000 * np.mean(np.abs(E - E_ref)):8.1f} {1000 * np.mean(np.abs(E - E_ref)) / n_at:11.2f} "
              f"{1000 * np.mean(np.abs(fe)):9.1f} {1000 * np.mean(np.abs(fm)):11.2f}")
        print(f"{'':28s} (energy MAE after removing a constant shift: {1000 * np.mean(np.abs(dE)):.1f} meV)")
        e0 = E_ref.mean()
        ax[row, 0].scatter(E_ref - e0, E - e0, s=8)
        ax[row, 0].set(xlabel="DFT energy - mean (eV)", ylabel="FieldMACE", title=f"{name}: energy")
        ax[row, 1].scatter(np.concatenate(F_ref).ravel(), np.concatenate(F).ravel(), s=3)
        ax[row, 1].set(xlabel="DFT force (eV/A)", ylabel="FieldMACE", title="forces on QM + link atoms")
        ax[row, 2].scatter(np.concatenate(Fmm_ref).ravel(), np.concatenate(Fmm).ravel(), s=2, alpha=0.4)
        ax[row, 2].set(xlabel="DFT force (eV/A)", ylabel="FieldMACE", title="forces on the point charges")
    for x in ax.ravel():
        lo, hi = x.get_xlim()
        x.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "parity_ground_state.png"), dpi=120)
    print("wrote results/parity_ground_state.png")


if __name__ == "__main__":
    main()
