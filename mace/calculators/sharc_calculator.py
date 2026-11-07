"""
SHARC interface for FieldMACE: energies, gradients, NACs, dipoles and SOCs of a QM region
embedded in MM point charges, returned in SHARC's QM.out layout (atomic units).

    calc = SharcCalculator("CH2O", "model.model", device="cuda",
                           properties=["energy", "forces", "nacs", "dipoles", "socs"])
    qm_out = calc.calculate(coords_bohr, mm_positions, mm_charges)
    # qm_out["h"]       [S][S]        complex  energies on the diagonal, SOCs off-diagonal (Hartree)
    # qm_out["grad"]    [S][N][3]              gradients (Hartree / bohr)
    # qm_out["grad_pc"] [S][M][3]              gradients on the point charges (Hartree / bohr)
    # qm_out["nacdr"]   [S][S][N][3]           NACs (antisymmetric; 1/bohr unless nac_to_au is changed)
    # qm_out["dm"]      [3][S][S]              permanent (diagonal) and transition dipoles (e bohr)

Couplings between two states have an arbitrary sign. Between steps, every NAC, transition
dipole and SOC is aligned with the previous step's prediction so that it changes smoothly.
"""
import os
from typing import Dict, List, Optional, Union

import ase
import numpy as np
import torch

from mace.calculators.mace import MACECalculator

HARTREE_PER_EV = 1.0 / 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def _pairs(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


class SharcCalculator:
    def __init__(
        self,
        atom_types,
        modelpath: Union[str, List[str]],
        device: str,
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
    ):
        """
        atom_types : chemical formula ("CH2O"), list of symbols, or atomic numbers of the QM atoms
        modelpath  : FieldMACE model file(s); several files -> committee average
        n_states   : number of states passed to SHARC (default: all states of the model)
        properties : subset of ["energy", "forces", "nacs", "dipoles", "socs"]
        energy_unit / distance_unit : units the model was trained in ("eV"/"Hartree", "Angstrom"/"Bohr")
        nac_to_au, dipole_to_au, soc_to_au : conversion of the model's NACs, dipoles and SOCs to atomic
            units. Defaults assume NACs in 1/length, dipoles in e*length (model length unit) and SOCs
            in the model energy unit. Set nac_to_au yourself for smooth NACs (NAC * dE).
        mm_positions_in_angstrom : MM positions given to calculate() are in Angstrom (default) or bohr.
        """
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
        self.properties = list(properties) if properties is not None else ["energy", "forces"]
        if isinstance(atom_types, str):
            numbers = ase.Atoms(atom_types).get_atomic_numbers()
        else:
            numbers = ase.Atoms(symbols=list(atom_types)).get_atomic_numbers() if len(atom_types) and \
                isinstance(atom_types[0], str) else np.asarray(atom_types, dtype=int)
        self.atom_types = np.asarray(numbers)
        self.molecule = ase.Atoms(numbers=self.atom_types)
        self.n_atoms = len(self.atom_types)
        self.nac_key = nac_key

        self.calc = MACECalculator(model_paths=modelpath, device=device)
        model = self.calc.models[0]
        self.n_total_states = int(getattr(model, "n_energies", 1))
        if isinstance(n_states, dict):
            n_states = n_states.get("n_singlets", self.n_total_states)
        self.n_states = self.n_total_states if n_states is None else int(n_states)
        if not 1 <= self.n_states <= self.n_total_states:
            raise ValueError(f"n_states={self.n_states} but the model predicts {self.n_total_states}")
        flags = {"nacs": "compute_nacs", "dipoles": "compute_dipoles", "socs": "compute_socs"}
        for prop, flag in flags.items():
            if prop in self.properties and not getattr(model, flag, False):
                raise ValueError(f"'{prop}' requested but the model was trained without {flag}")
        self.last_prediction: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ #
    def predict(self, sharc_coords, mm_positions=None, mm_charges=None) -> Dict[str, np.ndarray]:
        """Predictions for the first n_states states (model units, phase-aligned to the last step)."""
        self.molecule.set_positions(np.asarray(sharc_coords, dtype=float).reshape(-1, 3) * self.distance_unit_conversion)
        info = {}
        if mm_positions is not None and mm_charges is not None and len(mm_charges) > 0:
            mm = np.asarray(mm_positions, dtype=float).reshape(-1, 3)
            if not self.mm_positions_in_angstrom:
                mm = mm * self.distance_unit_conversion
            info = {"mm_positions": mm, "mm_charges": np.asarray(mm_charges, dtype=float).reshape(-1)}
        self.molecule.info = info
        self.calc.calculate(self.molecule)
        res = self.calc.results
        k = self.n_states
        keep = [n for n, (i, j) in enumerate(_pairs(self.n_total_states)) if j < k]
        out = {"energy": res["energies"][:k], "forces": res["forces_all"][:, :k, :]}
        if "mm_forces" in res:
            out["mm_forces"] = res["mm_forces"][:, :k, :]
        if "nacs" in res:
            out["nacs"] = res["nacs"][:, keep, :]
        if "dipoles" in res:
            n_tot = self.n_total_states
            out["dipoles"] = np.concatenate([res["dipoles"][:k], res["dipoles"][n_tot:][keep]], axis=0)
        if "socs" in res:
            out["socs"] = res["socs"][keep]
        if self.last_prediction is not None:
            for key in ("nacs", "dipoles", "socs"):
                if key in out and key in self.last_prediction:
                    out[key] = self._adjust_phase(key, self.last_prediction[key], out[key])
        self.last_prediction = out
        return out

    def calculate(self, sharc_coords, mm_positions=None, mm_charges=None) -> Dict[str, list]:
        """Predict and return SHARC's QM.out dictionary (atomic units)."""
        return self.get_qm(self.predict(sharc_coords, mm_positions, mm_charges))

    # ------------------------------------------------------------------ #
    def get_qm(self, pred: Dict[str, np.ndarray]) -> Dict[str, list]:
        S = self.n_states
        h = np.diag(np.asarray(pred["energy"], dtype=complex) * self.energy_unit_conversion)
        if "socs" in self.properties and "socs" in pred:
            for n, (i, j) in enumerate(_pairs(S)):
                h[i, j] = pred["socs"][n] * self.soc_to_au
                h[j, i] = np.conj(h[i, j])
        qm_out = {"h": h.tolist()}
        if "forces" in self.properties:
            qm_out["grad"] = (-np.transpose(pred["forces"], (1, 0, 2)) * self.gradient_conversion).tolist()
            if "mm_forces" in pred:
                qm_out["grad_pc"] = (-np.transpose(pred["mm_forces"], (1, 0, 2)) * self.gradient_conversion).tolist()
        if "nacs" in self.properties and "nacs" in pred:
            nac = np.zeros((S, S, self.n_atoms, 3))
            for n, (i, j) in enumerate(_pairs(S)):
                nac[i, j] = pred["nacs"][:, n] * self.nac_to_au
                nac[j, i] = -nac[i, j]
            qm_out["nacdr"] = nac.tolist()
        if "dipoles" in self.properties and "dipoles" in pred:
            dm = np.zeros((3, S, S))
            for i in range(S):
                dm[:, i, i] = pred["dipoles"][i] * self.dipole_to_au
            for n, (i, j) in enumerate(_pairs(S)):
                dm[:, i, j] = dm[:, j, i] = pred["dipoles"][S + n] * self.dipole_to_au
            qm_out["dm"] = dm.tolist()
        return qm_out

    # ------------------------------------------------------------------ #
    def _adjust_phase(self, key: str, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Flip the sign of every coupling whose overlap with the previous step is negative."""
        current = np.array(current, copy=True)
        S = self.n_states
        if key == "nacs":  # [N, P, 3]
            for p in range(current.shape[1]):
                if np.sum(previous[:, p] * current[:, p]) < 0:
                    current[:, p] *= -1
        elif key == "dipoles":  # [S + P, 3]; permanent dipoles have no phase
            for p in range(S, current.shape[0]):
                if np.dot(previous[p], current[p]) < 0:
                    current[p] *= -1
        elif key == "socs":  # complex [P]
            for p in range(current.shape[0]):
                if np.real(np.conj(previous[p]) * current[p]) < 0:
                    current[p] *= -1
        return current

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
