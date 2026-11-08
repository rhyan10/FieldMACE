###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# Modified for FieldMACE by Rhyan Barrett
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import ase.data
import ase.io
import h5py
import numpy as np
from tqdm import tqdm

from mace.tools import AtomicNumberTable

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ], [3,3], [9, ]
Virials = np.ndarray  # [6, ], [3,3], [9, ]
Charges = np.ndarray  # [..., 1]
Nacs = np.ndarray #[...,...,3]
Cell = np.ndarray  # [3,3]
Mm_charges = np.ndarray
Mm_positions = np.ndarray
Mm_forces = np.ndarray
Center_of_charge = np.ndarray
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipoles: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    pc_array: Optional[Charges] = None 
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None
    nacs: Optional[Nacs] = None  # [n_atoms, n_pairs, 3]
    socs: Optional[np.ndarray] = None  # [n_pairs, 2] (real, imaginary)
    mm_charges: Optional[Mm_charges] = None
    mm_positions: Optional[Mm_positions] = None
    mm_forces: Optional[Mm_forces] = None
    mm_forces_weight: float = 1.0  # weight of the forces on the MM point charges
    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    dipoles_weight: float = 1.0
    pc_weight: float = 1.0
    nacs_weight: float = 1.0
    socs_weight: float = 1.0
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int, work_dir: str
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if len(indices[train_size:]) < 10:
        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with following indices: {indices[train_size:]}"
        )
    else:
        # Save indices to file
        with open(work_dir + f"/valid_indices_{seed}.txt", "w", encoding="utf-8") as f:
            for index in indices[train_size:]:
                f.write(f"{index}\n")

        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with indices saved in: {work_dir}/valid_indices_{seed}.txt"
        )

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="REF_energy",
    forces_key="REF_forces",
    stress_key="REF_stress",
    virials_key="REF_virials",
    dipoles_key="REF_dipoles",
    nacs_key="REF_nacs",
    socs_key="REF_socs",
    charges_key="REF_charges",
    pc_key = "REF_mm_forces",
    config_type_weights: Dict[str, float] = None,
    soc_num: Optional[int] = None,
    n_states: Optional[int] = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in tqdm(atoms_list):
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipoles_key=dipoles_key,
                nacs_key=nacs_key,
                socs_key=socs_key,
                charges_key=charges_key,
                pc_key=pc_key,
                config_type_weights=config_type_weights,
                soc_num=soc_num,
                n_states=n_states,
            )
        )
    return all_configs

def _info_or_array(atoms: ase.Atoms, key: str):
    """Per-configuration labels may live in atoms.info (FieldMACE convention) or atoms.arrays."""
    if key in atoms.info:
        return atoms.info[key]
    return atoms.arrays.get(key, None)


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="REF_energy",
    forces_key="REF_forces",
    stress_key="REF_stress",
    virials_key="REF_virials",
    dipoles_key="dipoles",
    charges_key="REF_charges",
    nacs_key="nacs",
    mm_charges="mm_charges",
    mm_positions="mm_positions",
    pc_key="REF_mm_forces",
    config_type_weights: Dict[str, float] = None,
    socs_key="REF_socs",
    soc_num: Optional[int] = None,
    n_states: Optional[int] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration.

    QM/MM fields (all optional, in atoms.info):
      mm_positions [M, 3] (Angstrom), mm_charges [M] (e), REF_mm_forces [M, 3] or [M, n_states, 3].
    A configuration without point charges (gas phase) is valid and gets M = 0.

    Excited-state labels (optional; S states, P = S(S-1)/2 pairs ordered (0,1), (0,2), ..., (1,2), ...):
      nacs    [n_atoms, P, 3] (or [P, n_atoms, 3])   non-adiabatic couplings
      dipoles [S + P, 3]                             S permanent, then P transition dipoles
      socs    [P] real, [P, 2] (re, im) or complex   spin-orbit couplings (per pair of states)
              any shape with soc_num values           free-form SOC vector (soc_num given, X-MACE style)
    """
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = _info_or_array(atoms, forces_key)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang ^ 3
    virials = atoms.info.get(virials_key, None)
    nacs = _info_or_array(atoms, nacs_key)
    dipoles = atoms.info.get(dipoles_key, None)
    socs = atoms.info.get(socs_key, None)

    q_mm = atoms.info.get(mm_charges, None)
    r_mm = atoms.info.get(mm_positions, None)
    if (q_mm is None) != (r_mm is None):
        raise ValueError(f"'{mm_charges}' and '{mm_positions}' must be given together")
    if q_mm is None:
        q_mm = np.zeros(0)
        r_mm = np.zeros((0, 3))
    q_mm = np.asarray(q_mm, dtype=float).reshape(-1)
    r_mm = np.asarray(r_mm, dtype=float).reshape(-1, 3)
    if len(q_mm) != len(r_mm):
        raise ValueError(f"{len(q_mm)} MM charges but {len(r_mm)} MM positions")
    if not (np.all(np.isfinite(q_mm)) and np.all(np.isfinite(r_mm))):
        raise ValueError("non-finite MM charges or positions")
    f_mm = atoms.info.get(pc_key, None)
    mm_forces_weight = atoms.info.get("config_mm_forces_weight", 1.0)
    if f_mm is not None:
        f_mm = np.asarray(f_mm, dtype=float)
        if f_mm.shape[0] != len(q_mm) or f_mm.shape[-1] != 3:
            raise ValueError(f"'{pc_key}' has shape {f_mm.shape}, expected ({len(q_mm)}, [n_states,] 3)")
        f_mm = f_mm.reshape(len(q_mm), -1, 3) if f_mm.size else None
    if f_mm is None:
        mm_forces_weight = 0.0

    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)
    dipoles_weight = atoms.info.get("config_dipoles_weight", 1.0)
    nacs_weight = atoms.info.get("config_nacs_weight", 1.0)
    socs_weight = atoms.info.get("config_socs_weight", 1.0)

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0

    # label layout: energy [n_states], forces [n_atoms, n_states, 3]
    energy = np.atleast_1d(np.asarray(energy, dtype=float)).reshape(-1)
    forces = np.asarray(forces, dtype=float).reshape(len(atoms), -1, 3)
    n_out = len(energy) if energy_weight != 0.0 or forces_weight == 0.0 else forces.shape[1]
    if forces_weight == 0.0:
        forces = np.zeros((len(atoms), n_out, 3))
    if energy_weight == 0.0:
        energy = np.zeros(forces.shape[1])
    if forces.shape[1] != len(energy):
        raise ValueError(f"{len(energy)} energies but forces for {forces.shape[1]} states")
    if f_mm is None:
        f_mm = np.zeros((len(q_mm), len(energy), 3))
    elif f_mm.shape[1] != len(energy):
        raise ValueError(f"MM forces for {f_mm.shape[1]} states but {len(energy)} energies")

    n_s = int(n_states) if n_states else len(energy)  # diabatic models: n_states < number of outputs
    n_p = n_s * (n_s - 1) // 2
    n_at = len(atoms)
    if nacs is None:
        nacs = np.zeros((n_at, n_p, 3))
        nacs_weight = 0.0
    else:
        nacs = np.asarray(nacs, dtype=float)
        if nacs.size != n_at * n_p * 3:
            raise ValueError(f"'{nacs_key}' has shape {nacs.shape}; expected ({n_at}, {n_p}, 3) for {n_s} states")
        if nacs.ndim == 3 and nacs.shape[0] == n_p and nacs.shape[1] == n_at and n_p != n_at:
            nacs = np.transpose(nacs, (1, 0, 2))  # [P, n_atoms, 3] -> [n_atoms, P, 3]
        nacs = nacs.reshape(n_at, n_p, 3)
    if dipoles is None:
        dipoles = np.zeros((n_s + n_p, 3))
        dipoles_weight = 0.0
    else:
        dipoles = np.asarray(dipoles, dtype=float)
        if dipoles.size != (n_s + n_p) * 3:
            raise ValueError(f"'{dipoles_key}' has shape {dipoles.shape}; expected ({n_s + n_p}, 3): "
                             f"{n_s} permanent then {n_p} transition dipoles")
        dipoles = dipoles.reshape(n_s + n_p, 3)
    if soc_num is not None:  # free-form SOC vector of soc_num real values
        if socs is None:
            socs = np.zeros(soc_num)
            socs_weight = 0.0
        else:
            socs = np.asarray(socs)
            if np.iscomplexobj(socs):
                socs = np.stack([socs.real, socs.imag], axis=-1)
            socs = socs.astype(float).reshape(-1)
            if socs.size != soc_num:
                raise ValueError(f"'{socs_key}' has {socs.size} values but soc_num={soc_num}")
    elif socs is None:
        socs = np.zeros((n_p, 2))
        socs_weight = 0.0
    else:
        socs = np.asarray(socs)
        if np.iscomplexobj(socs):
            socs = np.stack([socs.real, socs.imag], axis=-1)
        socs = socs.astype(float)
        if socs.size == n_p:
            socs = np.stack([socs.reshape(n_p), np.zeros(n_p)], axis=-1)
        if socs.size != 2 * n_p:
            raise ValueError(f"'{socs_key}' has shape {socs.shape}; expected ({n_p},) or ({n_p}, 2) "
                             "(or pass --soc_num for a free-form SOC vector)")
        socs = socs.reshape(n_p, 2)

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipoles=dipoles,
        charges=charges,
        nacs=nacs,
        socs=socs,
        weight=weight,
        mm_charges=q_mm,
        mm_positions=r_mm,
        mm_forces=f_mm,
        mm_forces_weight=mm_forces_weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        dipoles_weight=dipoles_weight,
        nacs_weight=nacs_weight,
        socs_weight=socs_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        if conf.config_type not in all_cts:
            all_cts.append(conf.config_type)
            test_by_ct.append((conf.config_type, [conf]))
        else:
            ind = all_cts.index(conf.config_type)
            test_by_ct[ind][1].append(conf)
    return test_by_ct


def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "REF_virials",
    dipoles_key: str = "REF_dipoles",
    charges_key: str = "REF_charges",
    nacs_key: str = 'REF_nacs',
    socs_key: str = "REF_socs",
    soc_num: Optional[int] = None,
    n_states: Optional[int] = None,
    pc_key = "REF_mm_forces",
    extract_atomic_energies: bool = False,
    keep_isolated_atoms: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")
    if energy_key == "energy":
        logging.warning(
            "Since ASE version 3.23.0b1, using energy_key 'energy' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'energy' to 'REF_energy'. You need to use --energy_key='REF_energy' to specify the chosen key name."
        )
        energy_key = "REF_energy"
        for atoms in atoms_list:
            try:
                atoms.info["REF_energy"] = atoms.get_potential_energy()
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract energy: {e}")
                atoms.info["REF_energy"] = None
    if forces_key == "forces":
        logging.warning(
            "Since ASE version 3.23.0b1, using forces_key 'forces' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'forces' to 'REF_forces'. You need to use --forces_key='REF_forces' to specify the chosen key name."
        )
        forces_key = "REF_forces"
        for atoms in atoms_list:
            try:
                atoms.arrays["REF_forces"] = atoms.get_forces()
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract forces: {e}")
                atoms.arrays["REF_forces"] = None
    if stress_key == "stress":
        logging.warning(
            "Since ASE version 3.23.0b1, using stress_key 'stress' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'stress' to 'REF_stress'. You need to use --stress_key='REF_stress' to specify the chosen key name."
        )
        stress_key = "REF_stress"
        for atoms in atoms_list:
            try:
                atoms.info["REF_stress"] = atoms.get_stress()
            except Exception as e:  # pylint: disable=W0703
                atoms.info["REF_stress"] = None
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            isolated_atom_config = (
                len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"
            )
            if isolated_atom_config:
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = atoms.info[
                        energy_key
                    ]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy will be used."
                    )
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = np.zeros(1)
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")
        if not keep_isolated_atoms:
            atoms_list = atoms_without_iso_atoms

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipoles_key=dipoles_key,
        nacs_key=nacs_key,
        socs_key=socs_key,
        soc_num=soc_num,
        n_states=n_states,
        charges_key=charges_key,
        pc_key = pc_key,
    )
    return atomic_energies_dict, configs


def compute_average_E0s(
    collections_train: Configurations, z_table: AtomicNumberTable, n_diagonal: Optional[int] = None
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s. n_diagonal: diabatic models - average only the first n_diagonal
    energy labels (the diagonal of the diabatic Hamiltonian), not the couplings.
    """
    # configurations without an energy label (energy_weight == 0) carry a dummy energy
    collections_train = [c for c in collections_train if getattr(c, "energy_weight", 1.0) != 0.0]
    len_train = len(collections_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        e = np.ravel(collections_train[i].energy)
        B[i] = np.mean(e[:n_diagonal] if n_diagonal else e)
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(collections_train[i].atomic_numbers == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.error(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict


def save_dataset_as_HDF5(dataset: List, out_name: str) -> None:
    with h5py.File(out_name, "w") as f:
        for i, data in enumerate(dataset):
            grp = f.create_group(f"config_{i}")
            grp["num_nodes"] = data.num_nodes
            grp["edge_index"] = data.edge_index
            grp["positions"] = data.positions
            grp["shifts"] = data.shifts
            grp["unit_shifts"] = data.unit_shifts
            grp["cell"] = data.cell
            grp["node_attrs"] = data.node_attrs
            grp["weight"] = data.weight
            grp["energy_weight"] = data.energy_weight
            grp["forces_weight"] = data.forces_weight
            grp["stress_weight"] = data.stress_weight
            grp["virials_weight"] = data.virials_weight
            grp["forces"] = data.forces
            grp["energy"] = data.energy
            grp["stress"] = data.stress
            grp["virials"] = data.virials
            grp["dipoles"] = data.dipoles
            grp["charges"] = data.charges
            grp["nacs"] = data.nacs
            grp["socs"] = data.socs
            grp["socs_weight"] = data.socs_weight
            grp["nacs_weight"] = data.nacs_weight
            grp["dipoles_weight"] = data.dipoles_weight
            grp["mm_charges"] = data.mm_charges
            grp["mm_positions"] = data.mm_positions
            grp["mm_forces"] = data.mm_forces
            grp["num_mm"] = data.num_mm
            grp["mm_forces_weight"] = data.mm_forces_weight


def save_AtomicData_to_HDF5(data, i, h5_file) -> None:
    grp = h5_file.create_group(f"config_{i}")
    grp["num_nodes"] = data.num_nodes
    grp["edge_index"] = data.edge_index
    grp["positions"] = data.positions
    grp["shifts"] = data.shifts
    grp["unit_shifts"] = data.unit_shifts
    grp["cell"] = data.cell
    grp["node_attrs"] = data.node_attrs
    grp["weight"] = data.weight
    grp["energy_weight"] = data.energy_weight
    grp["forces_weight"] = data.forces_weight
    grp["stress_weight"] = data.stress_weight
    grp["virials_weight"] = data.virials_weight
    grp["forces"] = data.forces
    grp["energy"] = data.energy
    grp["stress"] = data.stress
    grp["virials"] = data.virials
    grp["dipoles"] = data.dipoles
    grp["charges"] = data.charges
    grp["nacs"] = data.nacs
    grp["socs"] = data.socs
    grp["socs_weight"] = data.socs_weight
    grp["nacs_weight"] = data.nacs_weight
    grp["dipoles_weight"] = data.dipoles_weight
    grp["mm_charges"] = data.mm_charges
    grp["mm_positions"] = data.mm_positions
    grp["mm_forces"] = data.mm_forces
    grp["num_mm"] = data.num_mm
    grp["mm_forces_weight"] = data.mm_forces_weight


def save_configurations_as_HDF5(configurations: Configurations, _, h5_file) -> None:
    grp = h5_file.create_group("config_batch_0")
    for j, config in enumerate(configurations):
        subgroup_name = f"config_{j}"
        subgroup = grp.create_group(subgroup_name)
        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        subgroup["energy"] = write_value(config.energy)
        subgroup["forces"] = write_value(config.forces)
        subgroup["stress"] = write_value(config.stress)
        subgroup["virials"] = write_value(config.virials)
        subgroup["dipoles"] = write_value(config.dipoles)
        subgroup["nacs"] = write_value(config.nacs)
        subgroup["socs"] = write_value(config.socs)
        subgroup["nacs_weight"] = write_value(config.nacs_weight)
        subgroup["dipoles_weight"] = write_value(config.dipoles_weight)
        subgroup["socs_weight"] = write_value(config.socs_weight)
        subgroup["charges"] = write_value(config.charges)
        subgroup["cell"] = write_value(config.cell)
        subgroup["pbc"] = write_value(config.pbc)
        subgroup["weight"] = write_value(config.weight)
        subgroup["energy_weight"] = write_value(config.energy_weight)
        subgroup["forces_weight"] = write_value(config.forces_weight)
        subgroup["stress_weight"] = write_value(config.stress_weight)
        subgroup["virials_weight"] = write_value(config.virials_weight)
        subgroup["config_type"] = write_value(config.config_type)
        subgroup["mm_charges"] = write_value(config.mm_charges)
        subgroup["mm_positions"] = write_value(config.mm_positions)
        subgroup["mm_forces"] = write_value(config.mm_forces)
        subgroup["mm_forces_weight"] = write_value(config.mm_forces_weight)


def write_value(value):
    return value if value is not None else "None"
