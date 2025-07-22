from glob import glob
from pathlib import Path
from typing import Union
import ase
import numpy as np
import torch
import os
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
import ase.io
from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from typing import Dict, Union, List
from mace.calculators.mace import MACECalculator

symbols = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F"]


class SharcCalculator:

    def __init__(
        self,
        atom_types,
        modelpath: str,
        device: str,
        cutoff: float = 10.0,
        n_states: Dict[str, int] = None,
        properties: List[str] = None,
        nac_key: str = "smooth_nacs",
        energy_unit: str = "eV",
        distance_unit: str = "Angstrom",
    ):

        distance_units = {"Angstrom": 0.529177249, "Bohr": 1.0}
        energy_units = {"eV": 0.0367493, "Hartree": 1.0}
        
        self.energy_unit_conversion = energy_units[energy_unit]
        self.distance_unit_conversion = distance_units[distance_unit]

        # Load model and setup molecule
        self.modelpath = modelpath
        self.properties = (
            properties
            if properties is not None
            else [
                "energy",
                "forces",
                #"nacs",
                #"dipoles",
            ]
        )

        if isinstance(atom_types, str):
            atom_types = np.array([symbols.index(c) for c in atom_types.upper()])

        self.molecule = ase.Atoms(symbols=atom_types)

        self.atom_types = atom_types
        self.nac_key = nac_key
        #self._check_modelpath()

        self.n_states = n_states
        self.n_total_states = 11
        self.n_atoms = len(atom_types)

        # self.nac_idx = np.triu_indices(self.n_states["n_singlets"], 1)
        # self.dm_idx = np.triu_indices(self.n_states["n_singlets"], 0)
        # self.soc_idx = np.triu_indices(self.n_total_states, 1)

        self.last_prediction = None
        self.calc = MACECalculator(model_paths=modelpath, n_energies=self.n_total_states, device=device)
            
    def calculate(
        self, sharc_coords, mm_positions, mm_charges
    ) -> Dict[str, List[np.ndarray]]:
        """
        Calculate properties from new positions
        If multiple models are used, the average values between
        the two predictions with the lowest NAC MAE will be returned
        """
        self.molecule.set_positions(np.array(sharc_coords)*self.distance_unit_conversion)
        self.molecule.info = {"mm_charges": mm_charges, "mm_positions": mm_positions}
        self.calc.calculate(self.molecule)
        results = self.calc.results 
        results["energy"] = results["energy"][:,:5] * self.energy_unit_conversion
        print(self.energy_unit_conversion)
        results["forces"] = results["forces"][:,:5,:] * (self.energy_unit_conversion/self.distance_unit_conversion)
        #results["nacs"] = results["nacs"] * self.distance_unit_conversion

            # Save first prediction for phase tracking
        if self.last_prediction is None:
            self.last_prediction = results

        for prop in self.properties:
            if prop not in ["energy", "forces"]:
                results[prop] = self._adjust_phase(
                    self.last_prediction[prop], results[prop]
                )
        #self.last_prediction = r
        return self.get_qm(results)

    def get_qm(self, spainn_output: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Calculate QM string for SHARC
        with predictions from model
        """
        states = self.n_total_states
        n_singlets = self.n_states

        qm_out = {}
        # Convert energy array to complex diagonal matrix
        qm_out["h"] = np.diag(np.array(spainn_output["energy"].squeeze(), dtype=complex)).tolist()

        # Reshape force array from [atoms, states, coords] to [states, atoms, coords]
        qm_out["grad"] = np.einsum("ijk->jik", -spainn_output["forces"]).tolist()

        # nacs_v = np.einsum("ijk->jik", spainn_output["nacs"])
        # nacs_m = np.zeros((states, states, self.n_atoms, 3))

        # if n_triplets == 0:
        #     nacs_m[self.nac_idx] = nacs_v
        #     nacs_m -= np.transpose(nacs_m, axes=(1, 0, 2, 3))
        # else:
        #     nacs_singlet = np.zeros((n_singlets, n_singlets, self.n_atoms, 3))
        #     nacs_singlet[self.nac_idx] = nacs_v[
        #         0 : int(n_singlets * (n_singlets - 1) / 2)
        #     ]
        #     nacs_singlet -= nacs_singlet.T

        #     nacs_m[0:n_singlets, 0:n_singlets] = nacs_singlet

        #     nacs_trip_sub = np.zeros((n_triplets, n_triplets, self.n_atoms, 3))
        #     sub_idx = np.triu_indices(n_triplets, 1)
        #     nacs_trip_sub[sub_idx] = nacs_v[int(n_singlets * (n_singlets - 1) / 2) :]
        #     nacs_trip_sub -= nacs_trip_sub.T

        #     nacs_trip = np.zeros((3 * n_triplets, 3 * n_triplets, self.n_atoms, 3))

        #     for i in range(3):
        #         for j in range(i, 3):
        #             nacs_trip[
        #                 i * n_triplets : (i + 1) * n_triplets,
        #                 j * n_triplets : (j + 1) * n_triplets,
        #             ] = nacs_trip_sub

        #     trip_idx = np.tril_indices(3 * n_triplets)
        #     nacs_trip[trip_idx] = 0
        #     nacs_trip -= nacs_trip.T

        #     nacs_m[n_singlets:, n_singlets:] = nacs_trip

        # qm_out["nacdr"] = nacs_m.tolist()

        n_triplets = 0

        if "dipoles" in self.properties:
            dm_m = np.zeros((states, states, 3), dtype=complex)
            if n_triplets == 0:
                dm_m[self.dm_idx] = spainn_output["dipoles"]
                dm_m += dm_m.T
                dm_m[self.dm_idx] = spainn_output["dipoles"]
            else:
                dm_singlets = np.zeros((n_singlets, n_singlets, 3), dtype=complex)
                dm_singlets[self.dm_idx] = spainn_output["dipoles"][
                    : int(n_singlets * (n_singlets - 1) / 2) + n_singlets
                ]
                dm_singlets += dm_singlets.T
                dm_singlets[self.dm_idx] = spainn_output["dipoles"][
                    : int(n_singlets * (n_singlets - 1) / 2) + n_singlets
                ]

                dm_m[0:n_singlets, 0:n_singlets] = dm_singlets

                dm_triplets = np.zeros((n_triplets, n_triplets, 3), dtype=complex)
                trip_idx = np.triu_indices(n_triplets, 0)
                dm_triplets[trip_idx] = spainn_output["dipoles"][
                    int(n_singlets * (n_singlets - 1) / 2) + n_singlets :
                ]
                dm_triplets += dm_triplets.T
                dm_triplets[trip_idx] = spainn_output["dipoles"][
                    int(n_singlets * (n_singlets - 1) / 2) + n_singlets :
                ]

                for i in range(0, 3):
                    dm_m[
                        n_singlets + i * n_triplets : n_singlets + i * n_triplets
                    ] = dm_triplets

            dm_m = np.einsum("ijk->kij", dm_m)
            qm_out["dm"] = dm_m.tolist()

        # if "socs" in self.properties:
        #     soc_m = np.zeros((states, states), dtype=complex)
        #     soc_m[self.soc_idx] = spainn_output["socs"]
        #     soc_m += soc_m.T

        #     qm_out["h"] += soc_m

        return qm_out

    def _check_modelpath(self) -> None:
        """
        Check if valid path(s) given
        """
        for path in self.modelpath:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"'{path}' does not exist!")

    def _adjust_phase(
        self, primary_phase: np.ndarray, secondary_phase: np.ndarray
    ) -> np.ndarray:
        """
        Function to align the phases of the two predictions
        """

        # Make sure only NACS are transformed
        is_nac = bool(len(primary_phase.shape) > 2)
        if is_nac:
            primary_phase = np.einsum("ijk->jik", primary_phase)
            secondary_phase = np.einsum("ijk->jik", secondary_phase)

        # Adjust phases
        for idx, val in enumerate(secondary_phase):
            if np.vdot(val, primary_phase[idx]) < 0:
                secondary_phase[idx] *= -1

        return np.einsum("ijk->jik", secondary_phase) if is_nac else secondary_phase

    def _write_xyz(self, coords: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Write rejected geometry to xyz file
        """
        with open("aborted.xyz", "w", encoding="utf-8") as output:
            output.write(f"{self.n_atoms}\n")
            output.write("Rejected geometry\n")
            for idx, val in enumerate(coords):
                output.write(
                    f"{symbols[self.atom_types[idx]]}\t{val[0]:12.8f}\t{val[1]:12.8f}\t{val[2]:12.8f}\n"
                )
