#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 4: analyse the trajectories - state populations and energy conservation.

    python 04_analyze.py runs/gas_FIELDMACE runs/gas_LVC

For every run directory: the fraction of trajectories in S0, S1, T1, T2 (MCH states, the triplet
M_S components summed) as a function of time, the number of hops, and the total-energy
conservation of each trajectory. Several run directories are plotted together, e.g. FieldMACE
against the LVC reference with the same initial conditions.

With a handful of short trajectories the populations are noisy; production runs use 100+.
Output: results/populations.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sharc_tools import read_lis, trajectories  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# SHARC's M_S-resolved MCH order for "states 2 0 2": S0 S1 T1(-1) T2(-1) T1(0) T2(0) T1(+1) T2(+1)
SPATIAL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 2, 6: 3, 7: 2, 8: 3}
NAMES = ["S0", "S1", "T1", "T2"]


def analyse(run):
    trajs = trajectories(run)
    data = [read_lis(os.path.join(t, "output.lis")) for t in trajs if os.path.exists(os.path.join(t, "output.lis"))]
    if not data:
        raise SystemExit(f"no output.lis in {run}: run 03_run_trajectories.sh first")
    n = min(len(d) for d in data)
    t = data[0][:n, 1]
    pops = np.zeros((n, 4))
    hops, drift = [], []
    for d in data:
        mch = d[:n, 3].astype(int)
        for k, s in enumerate(mch):
            pops[k, SPATIAL[s]] += 1
        # hops that change the spatial state (S0, S1, T1, T2); switches between the M_S components of
        # one triplet are bookkeeping of the degenerate manifold, not population transfer
        sp = np.array([SPATIAL[int(s)] for s in d[:, 3]])
        hops.append(int(np.sum(np.diff(sp) != 0)))
        drift.append(np.abs(d[:, 6] - d[0, 6]).max())
    pops /= len(data)
    return t, pops, hops, drift, len(data)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fig, ax = plt.subplots(1, len(a.runs), figsize=(5 * len(a.runs), 3.8), squeeze=False, sharey=True)
    for i, run in enumerate(a.runs):
        t, pops, hops, drift, n = analyse(run)
        name = os.path.basename(os.path.normpath(run))
        print(f"{name}: {n} trajectories, {t[-1]:.1f} fs; changes of the spatial state per trajectory {hops}; "
              f"max |E_tot - E_tot(0)| per trajectory (eV): {np.round(drift, 4)}")
        print("  final populations: " + ", ".join(f"{NAMES[s]} {pops[-1, s]:.2f}" for s in range(4)))
        for s in range(4):
            ax[0, i].plot(t, pops[:, s], label=NAMES[s])
        ax[0, i].set(title=name, xlabel="time (fs)")
    ax[0, 0].set_ylabel("population (fraction of trajectories)")
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "populations.png"), dpi=120)
    print("wrote results/populations.png")


if __name__ == "__main__":
    main()
