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


def _phase_aligned_mean(values, pair_axis: int) -> torch.Tensor:
    """Average committee predictions of state couplings after aligning each pair's sign to the first."""
    ref = values[0]
    aligned = [ref]
    for v in values[1:]:
        dims = [d for d in range(v.dim()) if d != pair_axis]
        dot = torch.sum(v * ref, dim=dims, keepdim=True)
        aligned.append(torch.where(dot < 0, -v, v))
    return torch.stack(aligned).mean(0)


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
        state: int = 0,
        **kwargs,
    ):
        kwargs.pop("n_energies", None)  # accepted for backwards compatibility, read from the model
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model_type = model_type
        self.use_hessian_approx = True
        self.state = int(state)  # electronic state returned as "energy"/"forces"
        if model_type == "MACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "energies",
                "forces_all",
                "mm_forces",
                "nacs",
                "dipoles",
                "dipole",
                "socs",
            ]
        elif model_type == "DipoleMACE":
            self.implemented_properties = ["dipole"]
        elif model_type == "EnergyDipoleMACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "dipole",
            ]
        else:
            raise ValueError(
                f"Give a valid model_type: [MACE, DipoleMACE, EnergyDipoleMACE], {model_type} not supported"
            )

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
            self.use_compile = False
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

        For FieldMACE models the MM point charges are read from atoms.info["mm_positions"]
        (Angstrom) and atoms.info["mm_charges"] (e); without them the molecule is in vacuum.
        Results:
          energy / free_energy : energy of ``self.state`` (eV)
          energies             : all states [n_states] (eV)
          forces               : forces of ``self.state`` [n_atoms, 3] (eV/A)
          forces_all           : [n_atoms, n_states, 3]
          mm_forces            : forces on the point charges [M, n_states, 3]
          nacs                 : [n_atoms, n_pairs, 3]   (if the model predicts NACs)
          dipoles / dipole     : [n_states + n_pairs, 3] / permanent dipole of ``self.state``
          socs                 : complex [n_pairs]       (if the model predicts SOCs)
          energy_var / forces_var : committee variances (several models)
        Pairs are ordered (0,1), (0,2), ..., (1,2), ... (see mace.modules.state_pairs).
        """
        Calculator.calculate(self, atoms)
        batch_base = self._atoms_to_batch(atoms)
        state = int(getattr(self, "state", 0))
        energies, forces, mm_forces = [], [], []
        props = {"nacs": [], "dipoles": [], "socs": []}
        for model in self.models:
            batch = self._clone_batch(batch_base)
            out = model(batch.to_dict(), compute_hessian=False, training=False)
            energies.append(out["energy"].detach().reshape(-1))
            forces.append(out["forces"].detach().reshape(len(atoms), -1, 3))
            if out.get("mm_forces") is not None:
                mm_forces.append(out["mm_forces"].detach().reshape(-1, energies[-1].shape[0], 3))
            for key in props:
                if out.get(key) is not None:
                    props[key].append(out[key].detach())
        e = torch.stack(energies)  # [n_models, n_states]
        f = torch.stack(forces)  # [n_models, n_atoms, n_states, 3]
        n_states = e.shape[1]
        if not 0 <= state < n_states:
            raise ValueError(f"state={state} but the model predicts {n_states} state(s)")
        e_conv = self.energy_units_to_eV
        f_conv = self.energy_units_to_eV / self.length_units_to_A
        self.results = {}
        e_mean = (e.mean(0) * e_conv).cpu().numpy()
        f_mean = (f.mean(0) * f_conv).cpu().numpy()
        self.results["energies"] = e_mean
        self.results["energy"] = float(e_mean[state])
        self.results["free_energy"] = self.results["energy"]
        self.results["forces_all"] = f_mean
        self.results["forces"] = f_mean[:, state, :]
        if mm_forces:
            fm = torch.stack(mm_forces).mean(0) * f_conv
            self.results["mm_forces"] = fm.cpu().numpy()
        # excited-state properties; committee members are phase-aligned to the first model
        # (couplings between two states are only defined up to a sign)
        if props["nacs"]:
            n = _phase_aligned_mean([x.reshape(len(atoms), -1, 3) for x in props["nacs"]], pair_axis=1)
            self.results["nacs"] = n.cpu().numpy()  # [n_atoms, n_pairs, 3]
        if props["dipoles"]:
            d = [x.reshape(-1, 3) for x in props["dipoles"]]  # [S + P, 3]
            perm = torch.stack([x[:n_states] for x in d]).mean(0)
            trans = _phase_aligned_mean([x[n_states:] for x in d], pair_axis=0)
            dip = torch.cat([perm, trans], 0).cpu().numpy()
            self.results["dipoles"] = dip
            self.results["dipole"] = dip[state]  # ASE: permanent dipole of the selected state
        if props["socs"]:
            m0 = self.models[0]
            groups = int(getattr(m0, "soc_groups", 1))
            sc = _phase_aligned_mean([x.reshape(groups, -1) for x in props["socs"]], pair_axis=0) * e_conv
            sc = sc.cpu().numpy()
            if getattr(m0, "soc_layout", "pairs") == "pairs":
                self.results["socs"] = sc[:, 0] + 1j * sc[:, 1]  # complex [n_pairs]
            else:
                self.results["socs"] = sc.reshape(-1)  # free-form real vector [soc_num]
        if self.num_models > 1:
            self.results["energy_var"] = (e.var(0, unbiased=False) * e_conv**2).cpu().numpy()
            self.results["forces_var"] = (f.var(0, unbiased=False) * f_conv**2).cpu().numpy()
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
