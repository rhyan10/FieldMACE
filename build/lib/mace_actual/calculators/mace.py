###########################################################################################
# The ASE Calculator for MACE
# Authors: Ilyes Batatia, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from glob import glob
from pathlib import Path
from typing import Union
from ase import units
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_load


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


class MACECalculator(Calculator):
    """MACE ASE Calculator
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        charges_key: str, Array field of atoms object where atomic charges are stored
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        model_paths: Union[list, str],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model_type = model_type
        self.use_hessian_approx = True
        if model_type in ["FieldEMACE", "PerAtomFieldEMACE"]:
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
            ]

        if "model_path" in kwargs:
            print("model_path argument deprecated, use model_paths")
            model_paths = kwargs["model_path"]

        if isinstance(model_paths, str):
            # Find all models that satisfy the wildcard (e.g. mace_model_*.pt)
            model_paths_glob = glob(model_paths)
            if len(model_paths_glob) == 0:
                raise ValueError(f"Couldn't find MACE model files: {model_paths}")
            model_paths = model_paths_glob
        elif isinstance(model_paths, Path):
            model_paths = [model_paths]
        if len(model_paths) == 0:
            raise ValueError("No mace file names supplied")
        self.num_models = len(model_paths)
        if len(model_paths) > 1:
            print(f"Running committee mace with {len(model_paths)} models")
            if model_type in ["MACE", "EnergyDipoleMACE"]:
                self.implemented_properties.extend(
                    ["energies", "energy_var", "forces_comm", "stress_var"]
                )
            elif model_type == "DipoleMACE":
                self.implemented_properties.extend(["dipole_var"])
        if compile_mode is not None:
            print(f"Torch compile is enabled with mode: {compile_mode}")
            self.models = [
                torch.compile(
                    prepare(extract_load)(f=model_path, map_location=device),
                    mode=compile_mode,
                    fullgraph=fullgraph,
                )
                for model_path in model_paths
            ]
            self.use_compile = True
        else:
            self.models = [
                torch.load(f=model_path, map_location=device, weights_only=False)
                for model_path in model_paths
            ]
            self.use_compile = True
        for model in self.models:
            model.to(device)  # shouldn't be necessary but seems to help with GPU
        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        assert np.all(
            r_maxs == r_maxs[0]
        ), "committee r_max are not all the same {' '.join(r_maxs)}"
        self.r_max = float(r_maxs[0])
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )
        self.charges_key = charges_key
        model_dtype = get_model_dtype(self.models[0])
        if default_dtype == "":
            print(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            print(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.models = [model.double() for model in self.models]
            elif default_dtype == "float32":
                self.models = [model.float() for model in self.models]
        torch_tools.set_default_dtype(default_dtype)
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def compute_nacs_from_hessian(
        self, schnet_outputs, hamiltonian, index, n_singlets, n_atoms, threshold_dE_S
    ):
        """
        Compute NAC approximations from the Hessian and energy gaps for singlet states only.
        
        Parameters
        ----------
        schnet_outputs : dict
            Contains 'energy', 'hessian', and either 'forces' or 'gradients' for all states.
        hamiltonian : np.ndarray
            Hamiltonian matrix containing at least the diagonal energies of states.
        index : np.ndarray
            Sorted indices of states by energy.
        n_singlets : int
            Number of singlet states.
        n_atoms : int
            Number of atoms.
        threshold_dE_S : float
            Energy difference threshold for singlets.
        nacs_approx_method : int
            Method flag for NAC approximation (e.g., 1 or 2).

        Returns
        -------
        nacs_approx : np.ndarray
            NAC approximations as a (n_singlets, n_singlets, n_atoms, 3) numpy array.
        """

        if "forces" in schnet_outputs:
            prop_ = "forces"
            convert_ = -1
        elif "gradients" in schnet_outputs:
            prop_ = "gradients"
            convert_ = 1
        else:
            raise ValueError("Neither 'forces' nor 'gradients' found in schnet_outputs.")

        dH_2 = []
        all_magnitude = []
        hopping_direction = np.zeros((n_singlets, n_singlets, n_atoms, 3))
        nacs_approx = np.zeros((n_singlets, n_singlets, n_atoms, 3))

        indexh = -1
        EPS = 1e-10

        # Loop over singlet pairs
        for istate in range(n_singlets):
            for jstate in range(istate + 1, n_singlets):
                # Check energy difference threshold
                Ei = np.real(hamiltonian[index[istate], index[istate]])
                Ej = np.real(hamiltonian[index[jstate], index[jstate]])
                if abs(Ei - Ej) <= threshold_dE_S:
                    indexh += 1
                    Hi = schnet_outputs['hessian'][0][index[istate]]
                    Hj = schnet_outputs['hessian'][0][index[jstate]]

                    Ei_schnet = schnet_outputs['energy'][0][index[istate]]
                    Ej_schnet = schnet_outputs['energy'][0][index[jstate]]
                    dE = Ei_schnet - Ej_schnet
                    if dE == 0:
                        dE = EPS

                    Gi = convert_ * schnet_outputs[prop_][0][index[istate]]
                    Gj = convert_ * schnet_outputs[prop_][0][index[jstate]]

                    GiGi = np.dot(Gi.reshape(-1, 1), Gi.reshape(-1, 1).T)
                    GjGj = np.dot(Gj.reshape(-1, 1), Gj.reshape(-1, 1).T)
                    GiGj = np.dot(Gi.reshape(-1, 1), Gj.reshape(-1, 1).T)

                    G_diff = 0.5 * (Gi - Gj)
                    G_diff2 = np.dot(G_diff.reshape(-1, 1), G_diff.reshape(-1, 1).T)

                    # dH_2_ij calculation
                    dH_2_ij = 0.5 * (dE * (Hi - Hj) + GiGi + GjGj - 2 * GiGj)
                    dH_2.append(dH_2_ij)
                    magnitude = dH_2_ij / 2 - G_diff2
                    all_magnitude.append(magnitude)

                    # Perform SVD and get direction
                    u, s, vh = np.linalg.svd(magnitude)
                    ev = vh[0]

                    # Determine sign to ensure consistent phase
                    phase_check = max(ev[0:2].min(), ev[0:2].max(), key=abs)
                    if phase_check < 0.0:
                        ev = -ev
                    ew = s[0]

                    # Assign hopping directions
                    iterator = 0
                    for iatom in range(n_atoms):
                        for xyz in range(3):
                            hopping_direction[istate, jstate, iatom, xyz] = ev[iterator]
                            hopping_direction[jstate, istate, iatom, xyz] = -ev[iterator]
                            iterator += 1

                    hopping_magnitude = np.sqrt(ew) / dE

                    nacs_approx[istate, jstate] = hopping_direction[istate, jstate] * hopping_magnitude
                    nacs_approx[jstate, istate] = - nacs_approx[istate, jstate]

        return nacs_approx

    def _create_result_tensors(
        self, model_type: str, num_models: int, num_atoms: int
    ) -> dict:
        """
        Create tensors to store the results of the committee
        :param model_type: str, type of model to load
            Options: [MACE, DipoleMACE, EnergyDipoleMACE]
        :param num_models: int, number of models in the committee
        :return: tuple of torch tensors
        """
        dict_of_tensors = {}
        if model_type in ["MACE", "EnergyDipoleMACE"]:
            energies = torch.zeros(num_models, device=self.device)
            node_energy = torch.zeros(num_models, num_atoms, device=self.device)
            forces = torch.zeros(num_models, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_models, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["EnergyDipoleMACE", "DipoleMACE"]:
            dipole = torch.zeros(num_models, 3, device=self.device)
            dict_of_tensors.update({"dipole": dipole})
        return dict_of_tensors

    def _atoms_to_batch(self, atoms):
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        return batch

    def _clone_batch(self, batch):
        batch_clone = batch.clone()
        if self.use_compile:
            batch_clone["node_attrs"].requires_grad_(True)
            batch_clone["positions"].requires_grad_(True)
        return batch_clone

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        if self.model_type in ["FieldEMACE", "PerAtomFieldEMACE"]:
            batch = self._clone_batch(batch_base)
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            out = model(
                batch.to_dict(),
                compute_stress=compute_stress,
                compute_hessian=False,
                training=self.use_compile,
            )

#            H = out["hessian"].to("cpu")
#            m = torch.tensor(atoms.get_masses()) * units._amu
#            M = m.repeat_interleave(3).sqrt()
#            H_si = H.reshape(132, 132) * 16.02176634
#            H_mass = H_si / (M[:,None] * M[None,:])
#            vals, vecs = torch.linalg.eigh(H_mass)
#            num_modes = vecs.shape[1] - 6  # Excluding 6 rotational and translational modes
#            normal_modes = vecs[:, 6:]  # Remove the first 6 columns
#            normal_modes = normal_modes.T.reshape(num_modes, 44, 3)  # Shape: (-1, num_atoms, 3)
#            vals = vals[6:]
#            c_cm_s = units._c * 100.0  # convert m/s to cm/s
#            freqs_cm = torch.sqrt(vals) / (2 * 3.141592653589793 * c_cm_s)
            # 2) Temperature
#            T = 298.15  # K  ← change this to whatever T you need

            # 3) Constants
 #           h   = 6.62607015e-34   # J·s
 #           c   = 2.99792458e10    # cm/s
 #           kB  = 1.380649e-23     # J/K
 #           NA  = 6.02214076e23    # 1/mol
 #           R   = NA * kB          # J/(mol·K)

            # 4) Filter out the NaNs
 #           ν = freqs_cm[~torch.isnan(freqs_cm)]

            # 5) Dimensionless x_i = h c ν_i / (kB T)
 #           x = (h * c * ν) / (kB * T)

            # 6) Per-mode entropy: s_i = R * [ x/(exp(x)−1) − ln(1−exp(−x)) ]
 #           s = R * ( x/(torch.exp(x) - 1) - torch.log1p(-torch.exp(-x)) )

            # 7) Sum to get S_vib (J/mol·K)
 #           S_vib = s.sum()
 #           print(f"Vibrational entropy = {(S_vib * T)/4184:.2f} kcal/mol")
 #           coords = torch.tensor(atoms.positions) * 1e-10           # (44,3), m
 #           masses = torch.tensor(atoms.get_masses()) * units._amu  # (44,), kg

            # 2) Center‐of‐mass shift
 #           M_tot = masses.sum()
 #           com = (coords * masses[:,None]).sum(dim=0) / M_tot
 #           r = coords - com  # (44,3)

            # 3) Build inertia tensor I = Σ m [(r·r)I − (r⊗r)]
 #           rr    = r.unsqueeze(2) * r.unsqueeze(1)               # (44,3,3)
 #           r2    = (r*r).sum(dim=1)                              # (44,)
 #           I_tens = (masses[:,None,None] * (r2[:,None,None]*torch.eye(3) - rr)).sum(dim=0)

            # 4) Principal moments (kg·m²)
 #           I_vals = torch.linalg.eigvalsh(I_tens)  # (3,)

            # 5) Rotational temperatures Θ_i = h²/(8π² I_i kB)
 #           Θ = h**2 / (8 * torch.pi**2 * I_vals * kB)

            # 6) q_rot for non‐linear rotor: (√π/σ) T^(3/2)/(Θ_aΘ_bΘ_c)^(1/2)
 #           sigma = 1  # change if you know a higher symmetry
 #           ln_qrot = 0.5*torch.log(torch.tensor(torch.pi)) - torch.log(torch.tensor(sigma)) \
 #                   + 1.5*torch.log(torch.tensor(T)) \
 #                   - 0.5*torch.log(Θ.prod())

            # 7) Entropy S_rot = R [ ln q_rot + 3/2 ]
 #           S_rot = R * (ln_qrot + 1.5)  # J/(mol·K)

 #           print(f"Rotational T·S term = {(S_rot * T)/4184:.2f} kcal/mol")

            if self.model_type in ["FieldEMACE", "PerAtomFieldEMACE"]:
                ret_tensors["energies"] = out["energy"].detach()
                ret_tensors["forces"] = out["forces"].detach()
                #ret_tensors['frequencies'] = freqs_cm.detach()
                #ret_tensors["normal_modes"] = normal_modes.detach()
                #ret_tensors["dipoles"] = out["dipoles"].cpu().detach()
                #print(
        # if self.n_energies < 1:
        #     if self.use_hessian_approx:
        #         self.results["nacs"] = self.compute_nacs_from_hessian()
        #     else:
        #         self.results["nacs"] = out["nacs"].detach()

        self.results = {}
        if self.model_type in ["FieldEMACE", "PerAtomFieldEMACE"]:
            self.results["energy"] = ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
            
            self.results["free_energy"] = self.results["energy"]
            
            self.results["forces"] = ret_tensors["forces"].squeeze().cpu().numpy() * self.energy_units_to_eV
            
            #self.results["frequencies"] = ret_tensors['frequencies'].numpy()
            #self.results["normal_modes"] = ret_tensors["normal_modes"].numpy()
            #self.results["dipoles"] = ret_tensors["dipoles"].numpy()
            if self.num_models > 1:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )

        return self.results

    def get_hessian(self, atoms=None):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        batch = self._atoms_to_batch(atoms)
        hessians = [
            model(
                self._clone_batch(batch).to_dict(),
                compute_hessian=True,
                compute_stress=False,
                training=self.use_compile,
            )["hessian"]
            for model in self.models
        ]
        hessians = [hessian.detach().cpu().numpy() for hessian in hessians]
        if self.num_models == 1:
            return hessians[0]
        return hessians

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param invariants_only: bool, if True only the invariant descriptors are returned
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray (num_atoms, num_interactions, invariant_features) of invariant descriptors if num_models is 1 or list[np.ndarray] otherwise
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        if num_layers == -1:
            num_layers = int(self.models[0].num_interactions)
        batch = self._atoms_to_batch(atoms)
        descriptors = [model(batch.to_dict())["node_feats"] for model in self.models]
        if invariants_only:
            irreps_out = self.models[0].products[0].linear.__dict__["irreps_out"]
            l_max = irreps_out.lmax
            num_features = irreps_out.dim // (l_max + 1) ** 2
            descriptors = [
                extract_invariant(
                    descriptor,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )
                for descriptor in descriptors
            ]
        descriptors = [descriptor.detach().cpu().numpy() for descriptor in descriptors]

        if self.num_models == 1:
            return descriptors[0]
        return descriptors
