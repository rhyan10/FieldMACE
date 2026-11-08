# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
SHARC interface for FieldMACE: energies, gradients, NACs, dipoles and SOCs of a QM region
embedded in MM point charges, returned in SHARC's QM.out layout (atomic units).

Used by the SHARC 4 interface sharc/SHARC_FIELDMACE.py, but works on its own as well:

    calc = SharcCalculator("CH2S", "model.model", device="cuda", states=[2, 0, 2],
                           properties=["energy", "forces", "nacs", "dipoles", "socs"])
    qm_out = calc.calculate(coords_bohr, mm_positions, mm_charges)
    # h       [M][M]        complex  energies on the diagonal, SOCs off-diagonal (Hartree)
    # grad    [M][N][3]              gradients (Hartree / bohr)
    # grad_pc [M][Npc][3]            gradients on the point charges (Hartree / bohr)
    # nacdr   [M][M][N][3]           NACs (antisymmetric; 1/bohr unless nac_to_au is changed)
    # dm      [3][M][M]              permanent (diagonal) and transition dipoles (e bohr)
  with M = SHARC's number of states including the M_S components (nmstates).

States. ``states`` is SHARC's list [n_singlets, n_doublets, n_triplets, ...]. The model's
states are read in the same order (all singlets, then doublets, then triplets):
  * spatial model  (n_energies == sum(states)): energies, gradients, NACs and dipoles are
    copied to every M_S component; couplings only connect states of the same multiplicity
    and M_S. Pair-wise SOCs (default SOC layout) need an M_S-resolved model, see next point.
  * M_S-resolved model (n_energies == nmstates): model state k is SHARC state k.
  * one multiplicity only and n_energies > sum(states): the first sum(states) model states.
SOCs with a free-form vector (model trained with --soc_num, X-MACE convention) fill the upper
triangle of the M x M spin-orbit matrix: soc_num = M(M-1)/2 real values, or M(M-1) values
as (re, im) pairs.

Couplings between two states have an arbitrary sign. Between steps, every NAC, transition
dipole and SOC is aligned with the previous step's prediction so that it changes smoothly
(``last_prediction``; SHARC_FIELDMACE stores it in the save directory between calls).
"""
import os
from typing import Dict, List, Optional, Union

import ase
import numpy as np
import torch

from mace.calculators.mace import MACECalculator

HARTREE_PER_EV = 1.0 / 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
PROPERTIES = ("energy", "forces", "nacs", "dipoles", "socs")


def _pairs(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def sharc_mstates(states: List[int]):
    """SHARC's state list: (multiplicity, index within multiplicity (0-based), M_S) per M_S state."""
    out = []
    for i, n in enumerate(states):
        mult = i + 1
        for k in range(mult):
            for j in range(n):
                out.append((mult, j, k - i / 2.0))
    return out


class SharcCalculator:
    def __init__(
        self,
        atom_types,
        modelpath: Union[str, List[str]],
        device: str = "cpu",
        cutoff: float = 10.0,
        n_states: Optional[Union[int, Dict[str, int]]] = None,
        properties: Optional[List[str]] = None,
        nac_key: str = "smooth_nacs",
        energy_unit: str = "eV",
        distance_unit: str = "Angstrom",
        nac_to_au: Optional[float] = None,
        dipole_to_au: Optional[float] = None,
        soc_to_au: Optional[float] = None,
        mm_positions_in_angstrom: bool = True,
        states: Optional[List[int]] = None,
        soc_selection_rules: bool = True,
        nac_type: str = "nac",
        min_gap: float = 1e-4,
    ):
        """
        atom_types : chemical formula ("CH2S"), list of symbols, or atomic numbers of the QM atoms
        modelpath  : FieldMACE model file(s); several files -> committee average
        states     : SHARC states [n_singlets, n_doublets, n_triplets, ...] (default: all model
                     states as singlets). ``n_states`` (int or {"n_singlets", "n_triplets"}) is the
                     older way of saying the same.
        properties : subset of ["energy", "forces", "nacs", "dipoles", "socs"]
        energy_unit / distance_unit : units the model was trained in ("eV"/"Hartree", "Angstrom"/"Bohr")
        nac_to_au, dipole_to_au, soc_to_au : conversion of the model's NACs, dipoles and SOCs to
            atomic units. Defaults assume NACs in 1/length, dipoles in e*length (model length unit)
            and SOCs in the model energy unit.
        nac_type : "nac" - the model predicts NACs <i|d/dR|j>;
                   "smooth" - the model predicts smooth NACs (E_j - E_i) * NAC_ij for each pair i < j,
                   which stay finite at near-degeneracies and are much easier to learn. They are
                   divided by the model's own energy gap (at least ``min_gap``, model energy unit)
                   before they go to SHARC.
        mm_positions_in_angstrom : MM positions given to calculate() are in Angstrom (default) or bohr.
        soc_selection_rules : set SOC elements that vanish exactly to zero instead of using the model's
            (small, noisy) prediction: between two singlets, between the M_S components of the same
            spin multiplet (same spatial state; zero for real, non-degenerate states), and for
            |Delta M_S| > 1 (the spin-orbit operator has spin rank 1). Noise between exactly degenerate
            M_S components would otherwise mix them and cause spurious hops.
        """
        energy_unit = {"ev": "eV", "hartree": "Hartree"}.get(str(energy_unit).lower(), energy_unit)
        distance_unit = {"ang": "Angstrom", "angstrom": "Angstrom", "bohr": "Bohr"}.get(
            str(distance_unit).lower(), distance_unit)
        energy_to_hartree = {"eV": HARTREE_PER_EV, "Hartree": 1.0}[energy_unit]
        bohr_to_model = {"Angstrom": 1.0 / BOHR_PER_ANGSTROM, "Bohr": 1.0}[distance_unit]
        self.energy_unit_conversion = energy_to_hartree
        self.distance_unit_conversion = bohr_to_model  # SHARC coordinates (bohr) -> model length
        self.gradient_conversion = energy_to_hartree * bohr_to_model  # model force unit -> Hartree/bohr
        self.nac_to_au = bohr_to_model if nac_to_au is None else nac_to_au
        self.dipole_to_au = (1.0 / bohr_to_model) if dipole_to_au is None else dipole_to_au
        self.soc_to_au = energy_to_hartree if soc_to_au is None else soc_to_au
        self.mm_positions_in_angstrom = mm_positions_in_angstrom

        self.modelpath = modelpath
        if properties and isinstance(properties[0], (list, tuple)):  # SHARC template parser nests lists
            properties = list(properties[0])
        self.properties = [p.lower() for p in (properties or ["energy", "forces"])]
        self.properties = ["nacs" if "nac" in p else p for p in self.properties]
        unknown = set(self.properties) - set(PROPERTIES)
        if unknown:
            raise ValueError(f"unknown properties {sorted(unknown)}; choose from {PROPERTIES}")
        if isinstance(atom_types, str):
            numbers = ase.Atoms(atom_types).get_atomic_numbers()
        elif len(atom_types) and isinstance(atom_types[0], str):
            numbers = ase.Atoms(symbols=[s.capitalize() for s in atom_types]).get_atomic_numbers()
        else:
            numbers = np.asarray(atom_types, dtype=int)
        self.atom_types = np.asarray(numbers)
        self.molecule = ase.Atoms(numbers=self.atom_types)
        self.n_atoms = len(self.atom_types)
        self.nac_key = nac_key
        if nac_type not in ("nac", "smooth"):
            raise ValueError("nac_type must be 'nac' or 'smooth'")
        self.nac_type = nac_type
        self.min_gap = float(min_gap)

        self.calc = MACECalculator(model_paths=modelpath, device=device)
        model = self.calc.models[0]
        self.model = model
        self.diabatic = bool(getattr(model, "diabatic", False))
        self.n_model_states = int(getattr(model, "n_states", getattr(model, "n_energies", 1)))
        for prop, flag in {"nacs": "compute_nacs", "dipoles": "compute_dipoles", "socs": "compute_socs"}.items():
            if prop == "nacs" and self.diabatic:
                continue  # NACs follow from the diabatic Hamiltonian
            if prop in self.properties and not getattr(model, flag, False):
                raise ValueError(f"'{prop}' requested but the model was trained without {flag}")
        if self.diabatic and "forces" in self.properties and "nacs" not in self.properties:
            self.properties.append("nacs")
        if self.diabatic:
            self.nac_type = "nac"  # NACs come from the diabatic Hamiltonian; nac_type does not apply

        if states is None:
            if isinstance(n_states, dict):
                states = [n_states.get("n_singlets", self.n_model_states), 0, n_states.get("n_triplets", 0)]
            elif n_states is not None:
                states = [int(n_states)]
            else:
                states = [self.n_model_states]
        while len(states) > 1 and states[-1] == 0:
            states = list(states[:-1])
        self.states = [int(n) for n in states]
        self.mstates = sharc_mstates(self.states)
        self.nmstates = len(self.mstates)
        self.nstates = sum(self.states)
        self._setup_state_map()
        if self.diabatic:
            if self.mode != "spatial":
                raise NotImplementedError("diabatic models need one model state per spatial SHARC state")
            # diagonalise each multiplicity separately (no electronic coupling between multiplicities)
            offsets = np.concatenate([[0], np.cumsum(self.states)])
            blocks = [list(range(int(offsets[m]), int(offsets[m + 1]))) for m in range(len(self.states))
                      if self.states[m] > 0]
            if sum(len(b) for b in blocks) < self.n_model_states:  # one multiplicity, extra model states
                blocks = [list(range(self.n_model_states))]
            self.calc.diabatic_blocks = blocks
        self.soc_selection_rules = bool(soc_selection_rules)
        self._soc_forbidden = np.array([[(ma == mb == 1) or (ma == mb and ja == jb) or abs(sa - sb) > 1
                                         for (mb, jb, sb) in self.mstates] for (ma, ja, sa) in self.mstates])
        self.last_prediction: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ #
    def _setup_state_map(self):
        n_mod, n_sp, n_ms = self.n_model_states, self.nstates, self.nmstates
        one_mult = sum(1 for n in self.states if n > 0) == 1
        offsets = np.concatenate([[0], np.cumsum(self.states)])[:-1]
        if n_mod == n_ms and n_ms != n_sp:
            self.mode = "ms"
            self.model_index = list(range(n_ms))
        elif n_mod == n_sp or (one_mult and n_mod > n_sp):
            self.mode = "spatial"
            self.model_index = [int(offsets[m - 1] + j) for m, j, _ in self.mstates]
        else:
            raise ValueError(
                f"SHARC states {self.states} ({n_sp} spatial, {n_ms} with M_S components) do not match "
                f"the model's {n_mod} states. Train with n_energies = {n_sp} (spatial) or {n_ms} (M_S resolved)."
            )
        if "socs" in self.properties and getattr(self.model, "soc_layout", "pairs") == "pairs":
            if self.mode == "spatial" and n_ms != n_sp:
                raise ValueError(
                    "Pair-wise SOCs of a spatial model cannot be distributed over M_S components. "
                    f"Train an M_S-resolved model (n_energies={n_ms}) or a free-form SOC vector (--soc_num)."
                )
        if "socs" in self.properties and getattr(self.model, "soc_layout", "pairs") == "vector":
            n = int(self.model.soc_group_size)
            if n not in (n_ms * (n_ms - 1) // 2, n_ms * (n_ms - 1)):
                raise ValueError(
                    f"The model's SOC vector has {n} values; for {n_ms} SHARC states it needs "
                    f"{n_ms * (n_ms - 1) // 2} (real upper triangle) or {n_ms * (n_ms - 1)} ((re, im) pairs)."
                )
        self._pair_index = {p: k for k, p in enumerate(_pairs(n_mod))}

    def _coupled(self, a: int, b: int) -> bool:
        """Do SHARC states a and b couple through NACs / transition dipoles?"""
        if self.mode == "ms":
            return True
        (ma, _, sa), (mb, _, sb) = self.mstates[a], self.mstates[b]
        return ma == mb and sa == sb

    # ------------------------------------------------------------------ #
    def predict(self, sharc_coords, mm_positions=None, mm_charges=None) -> Dict[str, np.ndarray]:
        """Model predictions (model units, model state order, phase-aligned to the last step)."""
        self.molecule.set_positions(np.asarray(sharc_coords, dtype=float).reshape(-1, 3) * self.distance_unit_conversion)
        info = {}
        if mm_positions is not None and mm_charges is not None and len(mm_charges) > 0:
            mm = np.asarray(mm_positions, dtype=float).reshape(-1, 3)
            if not self.mm_positions_in_angstrom:
                mm = mm * self.distance_unit_conversion
            info = {"mm_positions": mm, "mm_charges": np.asarray(mm_charges, dtype=float).reshape(-1)}
        self.molecule.info = info
        if self.diabatic and self.last_prediction is not None and "U" in self.last_prediction:
            self.calc._diabatic_U = np.asarray(self.last_prediction["U"])  # eigenvector sign continuity
        self.calc.calculate(self.molecule)
        res = self.calc.results
        out = {"energy": np.asarray(res["energies"]), "forces": np.asarray(res["forces_all"])}
        for key in ("mm_forces", "nacs", "dipoles", "socs", "mm_nacs", "U"):
            if key in res and res[key] is not None:
                out[key] = np.asarray(res[key])
        if self.diabatic:
            if "diabatic_socs" in res:
                out["socs"] = np.asarray(res["diabatic_socs"])  # still in the diabatic basis
            if self.last_prediction is not None and "U" in self.last_prediction:
                out["U_previous"] = np.asarray(self.last_prediction["U"])
            self.last_prediction = out  # signs are fixed by the eigenvectors, no per-coupling flipping
            return out
        if self.last_prediction is not None:
            for key in ("nacs", "dipoles", "socs"):
                if key in out and key in self.last_prediction and np.shape(out[key]) == np.shape(self.last_prediction[key]):
                    out[key] = self._adjust_phase(key, self.last_prediction[key], out[key])
        self.last_prediction = out
        return out

    def calculate(self, sharc_coords, mm_positions=None, mm_charges=None) -> Dict[str, list]:
        """Predict and return SHARC's QM.out dictionary (atomic units)."""
        return self.get_qm(self.predict(sharc_coords, mm_positions, mm_charges))

    # ------------------------------------------------------------------ #
    def get_qm(self, pred: Dict[str, np.ndarray], as_arrays: bool = False) -> Dict[str, list]:
        M, idx = self.nmstates, self.model_index
        h = np.diag(np.asarray(pred["energy"], dtype=complex)[idx] * self.energy_unit_conversion)
        U_ms = self._u_ms(pred["U"]) if self.diabatic else None
        if "socs" in self.properties and "socs" in pred:
            h = h + self._soc_matrix(pred["socs"], U_ms)
        out = {"h": h}
        if "forces" in self.properties:
            f = np.asarray(pred["forces"])  # [N, S_model, 3]
            out["grad"] = -np.transpose(f[:, idx, :], (1, 0, 2)) * self.gradient_conversion
            if "mm_forces" in pred:
                fm = np.asarray(pred["mm_forces"])  # [Npc, S_model, 3]
                out["grad_pc"] = -np.transpose(fm[:, idx, :], (1, 0, 2)) * self.gradient_conversion
        if "nacs" in self.properties and "nacs" in pred:
            nac = np.zeros((M, M, self.n_atoms, 3))
            for a in range(M):
                for b in range(a + 1, M):
                    i, j = idx[a], idx[b]
                    if i == j or not self._coupled(a, b):
                        continue
                    lo, hi = min(i, j), max(i, j)
                    v = pred["nacs"][:, self._pair_index[(lo, hi)]] * self.nac_to_au
                    if self.nac_type == "smooth":  # (E_hi - E_lo) * NAC_lo,hi  ->  NAC_lo,hi
                        gap = float(pred["energy"][hi] - pred["energy"][lo])
                        v = v / (np.sign(gap) or 1.0) / max(abs(gap), self.min_gap)
                    nac[a, b] = v if i < j else -v
                    nac[b, a] = -nac[a, b]
            out["nacdr"] = nac
            if self.diabatic and "mm_nacs" in pred:  # diabatic models also give NACs on the point charges
                npc = pred["mm_nacs"].shape[0]
                nac_pc = np.zeros((M, M, npc, 3))
                for a in range(M):
                    for b in range(a + 1, M):
                        i, j = idx[a], idx[b]
                        if i == j or not self._coupled(a, b):
                            continue
                        v = pred["mm_nacs"][:, self._pair_index[(min(i, j), max(i, j))]] * self.nac_to_au
                        nac_pc[a, b] = v if i < j else -v
                        nac_pc[b, a] = -nac_pc[a, b]
                out["nacdr_pc"] = nac_pc
        if self.diabatic:  # <previous step | this step>, for SHARC's overlap-based couplings
            out["overlap"] = (self._u_ms(pred["U_previous"]).T @ U_ms if "U_previous" in pred
                              else np.eye(M))
        if "dipoles" in self.properties and "dipoles" in pred:
            S = self.n_model_states
            dm = np.zeros((3, M, M))
            for a in range(M):
                dm[:, a, a] = pred["dipoles"][idx[a]] * self.dipole_to_au
                for b in range(a + 1, M):
                    i, j = idx[a], idx[b]
                    if i == j or not self._coupled(a, b):
                        continue
                    dm[:, a, b] = dm[:, b, a] = pred["dipoles"][S + self._pair_index[(min(i, j), max(i, j))]] * self.dipole_to_au
            out["dm"] = dm
        # SHARC works in double precision; float32 models give float32 arrays, which pysharc's C
        # extension would read as garbage (it takes the raw buffer), so always hand over float64/complex128
        out = {k: np.ascontiguousarray(v, dtype=np.complex128 if k == "h" else np.float64) for k, v in out.items()}
        if not self.diabatic:
            out.pop("overlap", None)
        if as_arrays:
            return out
        return {k: v.tolist() for k, v in out.items()}

    def _u_ms(self, U: np.ndarray) -> np.ndarray:
        """Diabatic-to-adiabatic matrix of the model states, expanded to SHARC's M_S states."""
        M, idx = self.nmstates, self.model_index
        U_ms = np.zeros((M, M))
        for a in range(M):
            for b in range(M):
                if self.mstates[a][0] == self.mstates[b][0] and self.mstates[a][2] == self.mstates[b][2]:
                    U_ms[a, b] = U[idx[a], idx[b]]
        return U_ms

    def _soc_matrix(self, socs: np.ndarray, U_ms: Optional[np.ndarray] = None) -> np.ndarray:
        """SOC part of SHARC's Hamiltonian. Diabatic models predict it in the diabatic basis; U_ms
        transforms it to the adiabatic states."""
        M = self.nmstates
        soc = np.zeros((M, M), dtype=complex)
        iu = np.triu_indices(M, 1)
        if getattr(self.model, "soc_layout", "pairs") == "vector":
            v = np.asarray(socs, dtype=float).reshape(-1) * self.soc_to_au
            soc[iu] = v if v.size == len(iu[0]) else v[0::2] + 1j * v[1::2]
        else:  # one complex value per pair of (M_S-resolved) model states
            v = np.asarray(socs).reshape(-1)
            for a, b in zip(*iu):
                i, j = self.model_index[a], self.model_index[b]
                if i != j:
                    val = v[self._pair_index[(min(i, j), max(i, j))]] * self.soc_to_au
                    soc[a, b] = val if i < j else np.conj(val)
        if self.soc_selection_rules:
            soc[self._soc_forbidden] = 0.0
        soc = soc + soc.conj().T
        if U_ms is not None:
            soc = U_ms.T @ soc @ U_ms
            if self.soc_selection_rules:
                soc[self._soc_forbidden | self._soc_forbidden.T] = 0.0
        return soc

    # ------------------------------------------------------------------ #
    def _adjust_phase(self, key: str, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Flip the sign of every coupling whose overlap with the previous step is negative."""
        current = np.array(current, copy=True)
        if key == "nacs":  # [N, P, 3]
            for p in range(current.shape[1]):
                if np.sum(previous[:, p] * current[:, p]) < 0:
                    current[:, p] *= -1
        elif key == "dipoles":  # [S + P, 3]; permanent dipoles have no phase
            for p in range(self.n_model_states, current.shape[0]):
                if np.dot(previous[p], current[p]) < 0:
                    current[p] *= -1
        elif key == "socs":
            if np.iscomplexobj(current):  # pairs: complex [P]
                for p in range(current.shape[0]):
                    if np.real(np.conj(previous[p]) * current[p]) < 0:
                        current[p] *= -1
            elif np.dot(previous.reshape(-1), current.reshape(-1)) < 0:  # vector: one phase
                current *= -1
        return current

    def save_state(self, path: str) -> None:
        """Store the last prediction (phase reference for the next step)."""
        if self.last_prediction is not None:
            np.savez(path, **self.last_prediction)

    def load_state(self, path: str) -> None:
        if os.path.isfile(path):
            with np.load(path) as f:
                self.last_prediction = {k: f[k] for k in f.files}

    def _check_modelpath(self) -> None:
        paths = [self.modelpath] if isinstance(self.modelpath, str) else self.modelpath
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"'{path}' does not exist!")

    def _write_xyz(self, coords: Union[np.ndarray, torch.Tensor]) -> None:
        """Write a rejected geometry to aborted.xyz."""
        with open("aborted.xyz", "w", encoding="utf-8") as output:
            output.write(f"{self.n_atoms}\nRejected geometry\n")
            for z, val in zip(self.molecule.get_chemical_symbols(), np.asarray(coords)):
                output.write(f"{z}\t{val[0]:12.8f}\t{val[1]:12.8f}\t{val[2]:12.8f}\n")
