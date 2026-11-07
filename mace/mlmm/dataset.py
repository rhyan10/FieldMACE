"""
Generate FieldMACE training data from OpenMM simulations with any force field.

Workflow
--------
1. Build an OpenMM System with your force field (Amber, OPLS, CHARMM, ...).
2. ``prepare_qmmm_system`` -> (mm_system, spec); choose ``EmbeddingSettings``.
3. Sample configurations (classical MD, or ML/MM MD with an earlier model for
   active learning) and convert each with :func:`frame_to_atoms`.  The QM
   atoms (+ H link atoms) and the MM point charges are written in exactly the
   frame the OpenMM force will use later.
4. Run your QM code on every frame with the point charges in
   ``atoms.info["mm_positions"]`` / ``atoms.info["mm_charges"]``
   (e.g. ORCA via :func:`write_orca_pointcharges`), then store the results
   with :func:`attach_reference` (energy incl. QM-point-charge interaction,
   QM forces, forces on the point charges).
5. Write with :func:`write_extxyz` and train with ``mace_run_train`` / ``scripts/run_train.py``.
   Frames may have different numbers of point charges; no padding is needed.

All lengths are Angstrom, energies eV, charges e (FieldMACE conventions).
"""
from __future__ import annotations

import json
from typing import Callable, Iterable, List, Optional, Sequence

import ase
import ase.io
import numpy as np
import openmm as mm
from openmm import unit

from .embedding import EmbeddingFrame, EmbeddingSettings
from .system import QMMMSpec

HARTREE_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903


def _to_angstrom(x) -> np.ndarray:
    """Quantity / list of Vec3 Quantities / plain nm array -> float ndarray in Angstrom."""
    if isinstance(x, unit.Quantity):
        return np.asarray(x.value_in_unit(unit.angstrom), dtype=float)
    if len(x) and isinstance(x[0], unit.Quantity):
        return np.array([v.value_in_unit(unit.angstrom) for v in x], dtype=float)
    return np.asarray(x, dtype=float) * 10.0


def frame_to_atoms(frame: EmbeddingFrame, positions, box=None, extra_info: Optional[dict] = None) -> ase.Atoms:
    """One configuration -> ase.Atoms in FieldMACE's extxyz convention.

    positions : (N, 3) all particles of the OpenMM system (Quantity or nm array)
    box       : box vectors (Quantity or nm array) for periodic systems
    """
    positions = _to_angstrom(positions)
    if box is not None:
        box = _to_angstrom(box)
    qm, mmp, q, sel = frame.numpy(np.asarray(positions, dtype=float), box)
    atoms = ase.Atoms(numbers=frame.spec.model_atomic_numbers, positions=qm, pbc=False)
    atoms.info["mm_positions"] = mmp
    atoms.info["mm_charges"] = q
    # bookkeeping (ignored by FieldMACE training, useful for analysis / relabelling)
    atoms.info["mm_indices"] = np.asarray(frame.spec.mm_atoms)[sel]
    atoms.info["qm_indices"] = np.asarray(frame.spec.qm_atoms)
    atoms.info["n_link_atoms"] = len(frame.spec.link_atoms)
    atoms.info["embedding"] = json.dumps(frame.settings.to_dict())
    if extra_info:
        atoms.info.update(extra_info)
    return atoms


def frames_from_simulation(simulation, frame: EmbeddingFrame, n_frames: int, stride: int,
                           callback: Optional[Callable[[ase.Atoms, int], None]] = None) -> List[ase.Atoms]:
    """Run ``simulation`` and grab a configuration every ``stride`` steps."""
    out = []
    for k in range(n_frames):
        simulation.step(stride)
        st = simulation.context.getState(getPositions=True)
        box = st.getPeriodicBoxVectors(asNumpy=True) if frame.spec.periodic else None
        atoms = frame_to_atoms(frame, st.getPositions(asNumpy=True), box,
                               extra_info={"md_step": simulation.currentStep})
        if callback is not None:
            callback(atoms, k)
        out.append(atoms)
    return out


def frames_from_trajectory(traj_positions: Iterable, frame: EmbeddingFrame, boxes: Optional[Iterable] = None):
    """Convert an existing trajectory (iterable of (N,3) nm arrays / Quantities)."""
    boxes = boxes if boxes is not None else iter(lambda: None, 1)
    return [frame_to_atoms(frame, p, b) for p, b in zip(traj_positions, boxes)]


def attach_reference(atoms: ase.Atoms, energy: float, forces: np.ndarray,
                     mm_forces: Optional[np.ndarray] = None, energies: Optional[Sequence[float]] = None) -> ase.Atoms:
    """Store QM/MM reference labels where FieldMACE's loader expects them.

    energy    : eV, QM energy *including* the interaction with the point charges
    forces    : (n_qm + n_link, 3) eV/A on the atoms of ``atoms``
    mm_forces : (M, 3) eV/A on the point charges (optional, recommended)
    energies  : several states (n_energies > 1); overrides ``energy``
    """
    e = np.atleast_1d(np.asarray(energy if energies is None else energies, dtype=float))
    n_states = len(e)
    f = np.asarray(forces, dtype=float)
    if f.ndim == 2:
        f = f[:, None, :]  # FieldMACE expects (n_atoms, n_states, 3)
    if f.shape != (len(atoms), n_states, 3):
        raise ValueError(f"forces must have shape ({len(atoms)}, {n_states}, 3), got {f.shape}")
    atoms.info["REF_energy"] = e.reshape(1, n_states)  # batches to (n_graphs, n_states) like the model output
    atoms.info["REF_forces"] = f
    if mm_forces is not None:
        mf = np.asarray(mm_forces, dtype=float)
        if mf.ndim == 2:
            mf = mf[:, None, :]  # (M, n_states, 3)
        if mf.shape != (len(atoms.info["mm_charges"]), n_states, 3):
            raise ValueError(f"mm_forces must have shape (M, {n_states}, 3), got {mf.shape}")
        atoms.info["REF_mm_forces"] = mf
    return atoms


def label_frames(frames: Sequence[ase.Atoms], qm_callback: Callable) -> List[ase.Atoms]:
    """Label frames with any QM code.

    ``qm_callback(symbols, qm_positions, mm_positions, mm_charges)`` must return
    (energy_eV, qm_forces_eV_per_A, mm_forces_eV_per_A or None).
    """
    for atoms in frames:
        e, f, fm = qm_callback(atoms.get_chemical_symbols(), atoms.positions,
                               atoms.info["mm_positions"], atoms.info["mm_charges"])
        attach_reference(atoms, e, f, fm)
    return list(frames)


def pad_mm_charges(frames: Sequence[ase.Atoms], n_max: Optional[int] = None,
                   pad_position: float = 1.0e3) -> List[ase.Atoms]:
    """Pad every frame to the same number of point charges.

    Not needed for FieldMACE >= 2 (batches may mix different numbers of charges); kept for
    training with older FieldMACE versions. Padding charges are 0 e placed far away, so they
    change neither energies nor forces. Do this *after* QM labelling."""
    n_max = n_max or max(len(a.info["mm_charges"]) for a in frames)
    for a in frames:
        m = len(a.info["mm_charges"])
        if m > n_max:
            raise ValueError(f"frame has {m} > n_max={n_max} point charges")
        k = n_max - m
        if k == 0:
            continue
        a.info["mm_charges"] = np.concatenate([a.info["mm_charges"], np.zeros(k)])
        a.info["mm_positions"] = np.concatenate([a.info["mm_positions"], np.full((k, 3), pad_position)])
        if "mm_indices" in a.info:
            a.info["mm_indices"] = np.concatenate([a.info["mm_indices"], -np.ones(k, dtype=int)])
        if "REF_mm_forces" in a.info:
            mf = a.info["REF_mm_forces"]
            a.info["REF_mm_forces"] = np.concatenate([mf, np.zeros((k,) + mf.shape[1:])])
    return list(frames)


def write_extxyz(path: str, frames: Sequence[ase.Atoms], append: bool = False):
    ase.io.write(path, list(frames), format="extxyz", append=append)


# ---------------------------------------------------------------------- #
# ORCA helpers (point charges in Angstrom, gradients in Eh/bohr)
# ---------------------------------------------------------------------- #
def write_orca_pointcharges(path: str, atoms: ase.Atoms):
    """ORCA point-charge file: first line count, then 'q x y z' (Angstrom)."""
    q = atoms.info["mm_charges"]
    r = atoms.info["mm_positions"]
    with open(path, "w") as fh:
        fh.write(f"{len(q)}\n")
        for qi, (x, y, z) in zip(q, r):
            fh.write(f"{qi: .8f} {x: .8f} {y: .8f} {z: .8f}\n")


def write_orca_input(path: str, atoms: ase.Atoms, pc_file: str, method: str = "B3LYP D3BJ def2-SVP",
                     charge: int = 0, multiplicity: int = 1, extra: str = ""):
    with open(path, "w") as fh:
        fh.write(f"! {method} EnGrad\n")
        fh.write(f'%pointcharges "{pc_file}"\n')
        if extra:
            fh.write(extra.rstrip() + "\n")
        fh.write(f"* xyz {charge} {multiplicity}\n")
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions):
            fh.write(f"{s:2s} {x: .8f} {y: .8f} {z: .8f}\n")
        fh.write("*\n")


def read_orca_engrad(path: str):
    """Energy (eV) and forces (eV/A) from an ORCA .engrad file."""
    vals = []
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    blocks, cur = [], []
    for ln in lines:
        if ln.startswith("#"):
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    n = int(blocks[0][0])
    energy = float(blocks[1][0]) * HARTREE_TO_EV
    grad = np.array([float(x) for x in blocks[2]]).reshape(n, 3)
    return energy, -grad * HARTREE_TO_EV / BOHR_TO_A


def read_orca_pcgrad(path: str):
    """Forces on the point charges (eV/A) from an ORCA .pcgrad file (Eh/bohr gradient)."""
    with open(path) as fh:
        n = int(fh.readline().split()[0])
        g = np.loadtxt(fh, max_rows=n).reshape(n, 3)
    return -g * HARTREE_TO_EV / BOHR_TO_A


# ---------------------------------------------------------------------- #
# keep data generation and simulation consistent
# ---------------------------------------------------------------------- #
def save_setup(path: str, spec: QMMMSpec, settings: EmbeddingSettings, **prepare_kwargs):
    """Write the QM selection, link atoms and embedding settings used for the training data.

    Load it with :func:`load_setup` when running ML/MM MD so the model is evaluated in
    exactly the frame it was trained in.
    """
    d = {
        "qm_atoms": list(map(int, spec.qm_atoms)),
        "qm_atomic_numbers": list(map(int, spec.qm_atomic_numbers)),
        "link_atoms": [[lk.qm_atom, lk.mm_atom, lk.ratio] for lk in spec.link_atoms],
        "embedding": settings.to_dict(),
        "prepare_kwargs": {k: v for k, v in prepare_kwargs.items() if isinstance(v, (str, int, float, bool, list))},
        "units": {"length": "Angstrom", "energy": "eV", "charge": "e"},
    }
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)


def load_setup(path: str):
    """Returns (qm_atoms, EmbeddingSettings, prepare_kwargs) as saved by :func:`save_setup`."""
    with open(path) as fh:
        d = json.load(fh)
    kw = dict(d.get("prepare_kwargs", {}))
    kw["link_atoms"] = [(int(a), int(b)) for a, b, _ in d["link_atoms"]] or False
    return d["qm_atoms"], EmbeddingSettings(**d["embedding"]), kw
