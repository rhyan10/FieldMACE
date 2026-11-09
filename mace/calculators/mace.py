###########################################################################################
# The ASE Calculator for MACE
# Authors: Ilyes Batatia, David Kovacs
# Modified for FieldMACE by Rhyan Barrett
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


import sys
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


class _Results(dict):
    """calc.results. "energies" holds the energies of all electronic states, while ASE's file writers
    (ase.io.extxyz) treat a result called "energies" as per-atom energies: writing Atoms with this calculator
    attached crashed for several states, or wrote the total energy as every atom's energy. items() leaves
    "energies" out for callers inside ase.io only; everywhere else results behave as a normal dict."""

    _HIDDEN = ("energies",)

    def items(self):
        caller = sys._getframe(1).f_globals.get("__name__", "")  # pylint: disable=protected-access
        if caller.startswith("ase.io"):
            return [(k, v) for k, v in super().items() if k not in self._HIDDEN]
        return super().items()


def _mm_state(atoms):
    """Hashable fingerprint of the MM point charges in atoms.info (None without charges)."""
    q, r = atoms.info.get("mm_charges"), atoms.info.get("mm_positions")
    if q is None and r is None:
        return None
    return (None if q is None else np.asarray(q, dtype=float).tobytes(),
            None if r is None else np.asarray(r, dtype=float).tobytes())


def _check_committee(models, paths):
    """All models of a committee must describe the same states, properties and elements."""
    def sig(m):
        return (int(getattr(m, "n_energies", 1)), bool(getattr(m, "diabatic", False)),
                tuple(int(z) for z in m.atomic_numbers), bool(getattr(m, "compute_nacs", False)),
                bool(getattr(m, "compute_dipoles", False)), bool(getattr(m, "compute_socs", False)))
    if len(models) > 1:
        sigs = [sig(m) for m in models]
        if any(x != sigs[0] for x in sigs[1:]):
            names = ("outputs", "diabatic", "elements", "NACs", "dipoles", "SOCs")
            k = next(i for i in range(len(sigs)) if sigs[i] != sigs[0])
            diff = [f"{n}: {a} vs {b}" for n, a, b in zip(names, sigs[0], sigs[k]) if a != b]
            raise ValueError(f"committee models differ ({paths[0]} vs {paths[k]}): {'; '.join(diff)}")


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
        # diabatic models: groups of states diagonalised separately (e.g. one per multiplicity)
        self.diabatic_blocks = kwargs.pop("diabatic_blocks", None)
        self._diabatic_U = None  # eigenvectors of the last call, for sign continuity
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
            raise NotImplementedError("compile_mode (torch.compile) is not supported for FieldMACE models; "
                                      "leave it unset (TorchScript: mace.mlmm.torchforce.export_torchscript)")
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
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"device='{device}' requested, but PyTorch sees no CUDA GPU here. "
                                   "Use device='cpu' (or run inside a GPU job).")
            self.models = [
                torch.load(f=model_path, map_location=device, weights_only=False)
                for model_path in model_paths
            ]
            self.use_compile = False
        _check_committee(self.models, model_paths)
        for model in self.models:
            model.to(device)  # shouldn't be necessary but seems to help with GPU
        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        if not np.all(r_maxs == r_maxs[0]):
            raise ValueError(f"committee models have different r_max: {[float(r) for r in r_maxs]}")
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
        L = float(self.length_units_to_A)
        if L != 1.0:  # the model works in its own length unit: Angstrom -> model unit
            atoms = atoms.copy()
            atoms.set_cell(np.array(atoms.get_cell()) / L)
            atoms.positions = atoms.positions / L
            if atoms.info.get("mm_positions") is not None:
                atoms.info["mm_positions"] = np.asarray(atoms.info["mm_positions"], dtype=float) / L
        config = data.config_for_inference(atoms)  # labels in atoms.info are not needed (nor parsed)
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

    def check_state(self, atoms, tol=1e-15):
        """ASE only compares positions, numbers, cell, pbc, ...: the MM point charges live in atoms.info,
        so a change of the charges alone would otherwise return the cached results of the old field."""
        changes = super().check_state(atoms, tol=tol)
        if _mm_state(atoms) != getattr(self, "_last_mm_state", None):
            changes = list(changes) + ["mm_charges"]
        return changes

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
        self._last_mm_state = _mm_state(atoms)
        zs = set(int(z) for z in self.z_table.zs)
        unknown = sorted(set(int(z) for z in atoms.numbers) - zs)
        if unknown:
            raise ValueError(f"element(s) Z={unknown} are not in the model (it knows Z={sorted(zs)})")
        r_mm = atoms.info.get("mm_positions")
        if r_mm is not None and np.size(r_mm):
            d = np.linalg.norm(atoms.positions[:, None, :] - np.asarray(r_mm, dtype=float).reshape(1, -1, 3), axis=-1)
            if d.min() < 0.5:
                import warnings

                warnings.warn(f"an MM charge is {d.min():.3f} A from a QM atom - is a QM atom also listed among "
                              "the MM charges?", stacklevel=2)
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
        diabatic = bool(getattr(self.models[0], "diabatic", False))
        n_states = int(getattr(self.models[0], "n_states", e.shape[1])) if diabatic else e.shape[1]
        if not 0 <= state < n_states:
            raise ValueError(f"state={state} but the model predicts {n_states} state(s)")
        e_conv = self.energy_units_to_eV
        f_conv = self.energy_units_to_eV / self.length_units_to_A
        self.results = _Results()
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
        L = float(self.length_units_to_A)
        if props["nacs"]:
            n = _phase_aligned_mean([x.reshape(len(atoms), -1, 3) for x in props["nacs"]], pair_axis=1)
            self.results["nacs"] = n.cpu().numpy() / L  # [n_atoms, n_pairs, 3], 1/A
        if props["dipoles"]:
            d = [x.reshape(-1, 3) for x in props["dipoles"]]  # [S + P, 3]
            perm = torch.stack([x[:n_states] for x in d]).mean(0)
            trans = _phase_aligned_mean([x[n_states:] for x in d], pair_axis=0)
            dip = torch.cat([perm, trans], 0).cpu().numpy() * L  # e*A
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
        if diabatic:
            self._diabatic_to_adiabatic(n_states, state)
        if self.num_models > 1:
            if diabatic:  # spread of the adiabatic energies and forces (what the results hold), not of H
                from mace.tools.diabatic import adiabatize, unpack

                ea, fa = [], []
                for k in range(self.num_models):
                    H = unpack(e[k].cpu().numpy() * e_conv, n_states)
                    dH = -unpack(np.moveaxis(f[k].cpu().numpy() * f_conv, 1, 0), n_states)
                    ad = adiabatize(H, dH, blocks=self.diabatic_blocks)
                    ea.append(ad["energies"])
                    fa.append(-np.moveaxis(ad["grad"], 0, 1))
                self.results["energy_var"] = np.var(np.array(ea), axis=0)
                self.results["forces_var"] = np.var(np.array(fa), axis=0)
            else:
                self.results["energy_var"] = (e.var(0, unbiased=False) * e_conv**2).cpu().numpy()
                self.results["forces_var"] = (f.var(0, unbiased=False) * f_conv**2).cpu().numpy()
            self.results["forces_comm"] = np.asarray(
                [x[:, state, :] for x in (fa if diabatic else (f * f_conv).cpu().numpy())])  # [n_models, N, 3]
        return self.results

    def _diabatic_to_adiabatic(self, n_states: int, state: int):
        """Diabatic model: keep its outputs as diabatic_* and replace energies, forces, NACs and dipoles
        by the adiabatic ones (diagonalisation, Hellmann-Feynman; see mace.tools.diabatic)."""
        from mace.tools.diabatic import adiabatize, state_pairs, unpack

        r = self.results
        S = n_states
        H = unpack(r["energies"], S)
        dH = -unpack(np.moveaxis(r["forces_all"], 1, 0), S)  # [S, S, N, 3]
        dH_pc = -unpack(np.moveaxis(r["mm_forces"], 1, 0), S) if "mm_forces" in r else None
        ad = adiabatize(H, dH, dH_pc, blocks=self.diabatic_blocks, U_ref=self._diabatic_U)
        self._diabatic_U = ad["U"]
        U, pairs = ad["U"], state_pairs(S)
        r["diabatic_hamiltonian"] = H
        r["diabatic_forces"] = r["forces_all"]
        r["U"] = U
        r["energies"] = ad["energies"]
        r["energy"] = r["free_energy"] = float(ad["energies"][state])
        r["forces_all"] = -np.moveaxis(ad["grad"], 0, 1)  # [N, S, 3]
        r["forces"] = r["forces_all"][:, state, :]
        r["nacs"] = np.stack([ad["nac"][i, j] for i, j in pairs], axis=1) if pairs else np.zeros((len(dH[0, 0]), 0, 3))
        if dH_pc is not None:
            r["diabatic_mm_forces"] = r["mm_forces"]
            r["mm_forces"] = -np.moveaxis(ad["grad_pc"], 0, 1)  # [M, S, 3]
            r["mm_nacs"] = np.stack([ad["nac_pc"][i, j] for i, j in pairs], axis=1) if pairs else None
        if "dipoles" in r:  # diabatic dipole matrix -> adiabatic
            d = np.asarray(r["dipoles"])
            r["diabatic_dipoles"] = d
            mu = unpack(d, S)  # [S, S, 3]
            mu_ad = np.einsum("ik,ijx,jl->klx", U, mu, U)
            r["dipoles"] = np.concatenate([np.stack([mu_ad[k, k] for k in range(S)]),
                                           np.stack([mu_ad[i, j] for i, j in pairs])])
            r["dipole"] = r["dipoles"][state]
        if "socs" in r:  # left in the diabatic basis (the SHARC calculator transforms them)
            r["diabatic_socs"] = r.pop("socs")

    def get_hessian(self, atoms=None):
        """Hessian d2E/dR2 of the calculator's state (adiabatic for diabatic models), [3N, N, 3] (eV/A^2).

        The MM point charges are kept fixed. A committee returns one Hessian per model."""
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        batch = self._atoms_to_batch(atoms)
        state = int(getattr(self, "state", 0))
        hessians = []
        for model in self.models:
            d = self._clone_batch(batch).to_dict()
            pos = d["positions"].detach().requires_grad_(True)
            d["positions"] = pos
            out = model(d, training=True, compute_force=False, compute_pc_grads=False)
            e = out["energy"]
            if bool(getattr(model, "diabatic", False)):
                from mace.tools.diabatic import adiabatic_energies_torch

                e = adiabatic_energies_torch(e, int(model.n_states))
            (g,) = torch.autograd.grad(e[0, state], pos, create_graph=True)
            g = g.reshape(-1)
            rows = [torch.autograd.grad(g[k], pos, retain_graph=True)[0] for k in range(g.numel())]
            h = torch.stack(rows) * self.energy_units_to_eV / self.length_units_to_A**2  # [3N, N, 3]
            hessians.append(h.detach().cpu().numpy())
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
        if invariants_only:  # the scalar (0e) channels of every layer, whatever each layer's irreps
            from e3nn import o3

            def invariants(model, x):
                out, offset = [], 0
                for i, product in enumerate(model.products):
                    irreps = o3.Irreps(str(product.linear.irreps_out))
                    if i < num_layers:
                        start = offset
                        for mul, ir in irreps:
                            if ir == o3.Irrep(0, 1):
                                out.append(x[:, start:start + mul])
                            start += mul * ir.dim
                    offset += irreps.dim
                return torch.cat(out, dim=-1)

            descriptors = [invariants(m, d) for m, d in zip(self.models, descriptors)]
        descriptors = [descriptor.detach().cpu().numpy() for descriptor in descriptors]

        if self.num_models == 1:
            return descriptors[0]
        return descriptors
