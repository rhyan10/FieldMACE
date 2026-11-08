#!/usr/bin/env python3
"""
Step 2: excited-state labels from SHARC's LVC model of CH2S, in the field of the water charges.

The reference "QM method" is SHARC 4's linear vibronic coupling (LVC) model of thioformaldehyde
with a multipolar density fit, which gives state energies, gradients, NACs, dipoles and SOCs
that all respond to external point charges. It costs milliseconds, so the tutorial can make
real excited-state data without a quantum-chemistry code. With a real QM code (e.g.
SA-CASSCF, TDDFT, MRCI via SHARC's other interfaces) the steps are the same.

States: SHARC "states 2 0 2" = S0, S1, T1, T2 (4 spatial states, 8 with triplet M_S components).
FieldMACE is trained on the 4 spatial states (n_energies = 4); the SHARC interface copies them
to the M_S components. Labels written per frame (FieldMACE units: eV, Angstrom, e):

  REF_energy     (1, 4)       state energies incl. the interaction with the point charges
  REF_forces     (4, 4, 3)    forces on the QM atoms, per state     [atoms, states, xyz]
  REF_mm_forces  (300, 4, 3)  forces on the point charges, per state
  REF_nacs       (4, 6, 3)    NACs (1/A) for the 6 state pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3);
                              pairs of different multiplicity are zero
  REF_smooth_nacs (4, 6, 3)   smooth NACs (E_j - E_i) * NAC_ij (eV/A): what the model is trained on.
                              NACs grow as 1/dE near crossings (here up to ~100 1/A), smooth NACs
                              stay below 1 eV/A; SHARC gets NACs back with nac_type smooth
  REF_dipoles    (10, 3)      4 permanent dipoles, then 6 transition dipoles (e A), same pair order
  REF_socs       (56,)        SOC matrix of all 8 SHARC states (eV): the upper triangle
                              h[i, j], i < j, as (re, im) pairs  ->  train with --soc_num=56

Phases. Each adiabatic state's sign is arbitrary. FieldMACE's losses do not care for NACs and
transition dipoles (per pair) or SOCs (one sign per frame), but a single sign for the whole
SOC vector is only right if the states' signs are consistent between frames. Here they are made
consistent by aligning every state with the LVC states at the reference geometry (overlap of the
diabatic-to-adiabatic matrices); with a QM code use wave-function overlaps (e.g. SHARC's WFoverlap).

Requires SHARC 4 ($SHARC, with $SHARC/../lib and $SHARC on PYTHONPATH). The labelled data are
shipped in data/, so the next steps run without SHARC.
"""
import argparse
import os
import shutil
import sys
import tempfile

import ase.io
import numpy as np

from mace.mlmm.dataset import write_extxyz

HERE = os.path.dirname(os.path.abspath(__file__))
HARTREE_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903
SPATIAL = [0, 1, 2, 3]  # SHARC states S0, S1, T1(M_S=-1), T2(M_S=-1) represent the 4 spatial states


def pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


class LVCLabeler:
    def __init__(self, example_dir, elements, n_pc, workdir):
        sharc = os.environ["SHARC"]
        for path in (sharc, os.path.join(sharc, "..", "lib")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from SHARC_LVC import SHARC_LVC

        self.dir = workdir
        for f in ("LVC.template", "LVC.resources", "V0.txt"):
            shutil.copy(os.path.join(example_dir, "LVC", f), workdir)
        ref = np.array([[float(v) for v in ln.split()[2:5]]
                        for ln in open(os.path.join(example_dir, "LVC", "V0.txt")).read().splitlines()[1:5]])
        with open(os.path.join(workdir, "pc.txt"), "w") as fh:
            for _ in range(n_pc):
                fh.write("100.0 100.0 100.0 0.0\n")
        with open(os.path.join(workdir, "QM.in"), "w") as fh:
            fh.write(f"{len(elements)}\nLVC labels\n")
            fh.write("".join(f"{e} {x:.10f} {y:.10f} {z:.10f}\n" for e, (x, y, z) in zip(elements, ref)))
            fh.write("unit bohr\nstates 2 0 2\ncharge 0 1 0\nsavedir ./SAVE\nstep 0\ninit\npoint_charges pc.txt\n"
                     "h\nsoc\ndm\ngrad\nnacdr\n")
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            it = SHARC_LVC(loglevel=40)
            it.setup_mol("QM.in")
            it.read_resources()
            it.read_template()
            it.setup_interface()
            it.read_requests("QM.in")
            it.set_coords("QM.in")
            it.run()  # reference geometry, zero charges far away
            self.U_ref = it._U.copy()
        finally:
            os.chdir(cwd)
        self.it = it

    def __call__(self, qm_bohr, pc_bohr, pc_q):
        it = self.it
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            it.set_coords(np.asarray(qm_bohr))
            it.set_coords(np.asarray(pc_bohr), pc=True)
            it.set_pccharges(list(map(float, pc_q)))
            it.run()
            q = it.getQMout()
            sign = np.sign(np.einsum("in,in->n", self.U_ref, it._U))  # align with the reference states
            sign[sign == 0] = 1.0
            out = {"h": np.array(q.h) * np.outer(sign, sign), "grad": np.array(q.grad),
                   "grad_pc": np.array(q.grad_pc), "nacdr": np.array(q.nacdr) * np.outer(sign, sign)[:, :, None, None],
                   "dm": np.array(q.dm) * np.outer(sign, sign)[None]}
        finally:
            os.chdir(cwd)
        return out


def to_fieldmace_labels(at, q):
    """SHARC QM.out arrays (atomic units, 8 M_S states) -> FieldMACE labels (eV, A; 4 spatial states)."""
    s = SPATIAL
    force_conv = HARTREE_TO_EV / BOHR_TO_A
    h = q["h"]
    energies = np.real(np.diag(h))[s] * HARTREE_TO_EV
    forces = -np.transpose(q["grad"][s], (1, 0, 2)) * force_conv  # [N, 4, 3]
    mm_forces = -np.transpose(q["grad_pc"][s], (1, 0, 2)) * force_conv  # [M, 4, 3]
    prs = pairs(len(s))
    nacs = np.stack([q["nacdr"][s[i], s[j]] for i, j in prs], axis=1) / BOHR_TO_A  # [N, 6, 3], 1/A
    dip = [q["dm"][:, k, k] for k in s] + [q["dm"][:, s[i], s[j]] for i, j in prs]
    dipoles = np.array(dip) * BOHR_TO_A  # [10, 3], e A
    iu = np.triu_indices(h.shape[0], 1)
    soc = h[iu] * HARTREE_TO_EV
    socs = np.stack([soc.real, soc.imag], axis=1).reshape(-1)  # (re, im) pairs, 56 values
    at.info["REF_energy"] = energies.reshape(1, -1)
    at.info["REF_forces"] = forces
    at.info["REF_mm_forces"] = mm_forces
    at.info["REF_nacs"] = nacs
    # smooth NACs (E_j - E_i) * NAC_ij in eV/A: finite at near-degeneracies, the training target
    gaps = np.array([energies[j] - energies[i] for i, j in prs])
    at.info["REF_smooth_nacs"] = nacs * gaps[None, :, None]
    at.info["REF_dipoles"] = dipoles
    at.info["REF_socs"] = socs
    return compact(at)


def compact(at, decimals=7):
    """Round floats before writing (keeps the files small; 1e-7 eV / A is far below any QM accuracy)."""
    at.positions = np.round(at.positions, decimals)
    for k, v in list(at.info.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            at.info[k] = np.round(v, decimals) + 0.0
    return at


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example-dir", default=None, help="default: $SHARC/../examples/SHARC_QMMM_LVC_OPENMM")
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    p.add_argument("--out", default=os.path.join(HERE, "data"))
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--gas-fraction", type=float, default=0.1,
                   help="add this fraction of frames again without point charges (gas phase)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not os.environ.get("SHARC") and not a.example_dir:
        raise SystemExit("This step needs SHARC 4: set $SHARC. The labelled data are already in data/.")
    ex = a.example_dir or os.path.join(os.environ["SHARC"], "..", "examples", "SHARC_QMMM_LVC_OPENMM")

    frames = ase.io.read(os.path.join(a.work, "frames.xyz"), index=":")
    n_pc = len(frames[0].info["mm_charges"])
    if any(len(f.info["mm_charges"]) != n_pc for f in frames):
        raise SystemExit("this labeller expects the same point charges in every frame (no cutoff)")
    tmp = tempfile.mkdtemp(prefix="lvc_")
    lab = LVCLabeler(ex, frames[0].get_chemical_symbols(), n_pc, tmp)
    for k, at in enumerate(frames):
        q = lab(at.positions / BOHR_TO_A, at.info["mm_positions"] / BOHR_TO_A, at.info["mm_charges"])
        to_fieldmace_labels(at, q)
        at.info["qm_method"] = "LVC(CH2S, SHARC 4 example)"
        at.info["config_type"] = "solvated"
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(frames)} labelled")
    # gas-phase copies: same QM geometry, no point charges (LVC with all charges set to zero).
    # They teach the model the zero-field limit, e.g. for solvent shifts.
    rng = np.random.default_rng(a.seed + 1)
    zero_q = np.zeros(n_pc)
    far = np.full((n_pc, 3), 1000.0)
    gas = []
    for k in rng.choice(len(frames), int(round(a.gas_fraction * len(frames))), replace=False):
        at = frames[k].copy()
        q = lab(at.positions / BOHR_TO_A, far, zero_q)
        for key in ("mm_positions", "mm_charges", "mm_indices", "REF_mm_forces"):
            at.info.pop(key, None)
        to_fieldmace_labels(at, q)
        at.info.pop("REF_mm_forces", None)
        at.info["config_type"] = "gas"
        gas.append(at)
    frames = frames + gas
    print(f"  + {len(gas)} gas-phase frames")
    shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(frames))
    n_test = int(round(a.test_fraction * len(frames)))
    write_extxyz(os.path.join(a.out, "test.xyz"), [frames[i] for i in sorted(idx[:n_test])])
    write_extxyz(os.path.join(a.out, "train.xyz"), [frames[i] for i in sorted(idx[n_test:])])
    e = np.array([np.ravel(f.info["REF_energy"]) for f in frames])
    print(f"wrote {len(frames) - n_test} training and {n_test} test frames to {a.out}/")
    print("mean state energies (eV, relative to S0):", np.round(e.mean(0) - e[:, 0].mean(), 3))
    print("energy spread per state (eV):", np.round(e.std(0), 3))


if __name__ == "__main__":
    main()
