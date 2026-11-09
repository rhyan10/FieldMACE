# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Sanity checks of a training set before any training starts.

Mistakes in QM/MM data are easy to make and expensive to discover after hours of training: a key
name that does not match (the property is then silently not trained), the wrong number of states,
NaN labels, a QM atom that is also listed as an MM charge, periodic cells from a periodic DFT code.
check_datasets() turns these into clear errors (or warnings, where the data may be intended), with
the file name and frame number.
"""
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

CLOSE_CONTACT = 0.5  # Angstrom: MM charge this close to a QM atom is almost certainly a data error


class InputError(ValueError):
    """A problem with the command-line settings or input files, reported without a traceback."""


class DataError(InputError):
    """A problem with the training data that would make training meaningless."""


XYZ_SUFFIXES = (".xyz", ".extxyz")


def is_xyz(path) -> bool:
    return str(path).lower().endswith(XYZ_SUFFIXES)


def check_arguments(args) -> None:
    """Inconsistent settings and missing files, before anything is loaded."""
    import os

    for opt in ("train_file", "valid_file", "test_file", "statistics_file"):
        path = getattr(args, opt, None)
        if path and not os.path.exists(path):
            raise InputError(f"--{opt}={path}: file not found (working directory {os.getcwd()})")
    tf = getattr(args, "train_file", None)
    if tf and not (is_xyz(tf) or str(tf).lower().endswith((".h5", ".hdf5")) or os.path.isdir(tf)):
        raise InputError(f"--train_file={tf}: use an extended-XYZ file (.xyz / .extxyz) or HDF5 data from "
                         "mace_preprocess_data (.h5 / a directory)")
    if tf and not is_xyz(tf):  # HDF5 from mace_preprocess_data
        if not getattr(args, "valid_file", None):
            raise InputError(f"--train_file={tf} is HDF5: give the validation set too (--valid_file, e.g. the val/ "
                             "folder written by mace_preprocess_data); --valid_fraction only splits XYZ files")
        e0s = getattr(args, "E0s", None)
        if not getattr(args, "statistics_file", None) and (e0s is None or str(e0s).lower() == "average"):
            raise InputError("HDF5 training data: --E0s=average needs the XYZ frames. Pass --statistics_file "
                             "(written by mace_preprocess_data --compute_statistics) or explicit --E0s "
                             "('{1: -13.6, 6: -1029.8}' or a .json file)")
    elif tf and not getattr(args, "valid_file", None):
        vf = getattr(args, "valid_fraction", 0.1)
        if vf is not None and not 0.0 < float(vf) < 1.0:
            raise InputError(f"--valid_fraction={vf}: must be between 0 and 1 (exclusive), or give --valid_file")
    dev = str(getattr(args, "device", "cpu") or "cpu")
    if dev.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise InputError(f"--device={dev} but PyTorch sees no CUDA GPU here. Use --device=cpu, or run on a GPU "
                             "node (check `python -c 'import torch; print(torch.cuda.is_available())'`)")
    n = int(getattr(args, "n_energies", 1) or 1)
    if n < 1:
        raise InputError("--n_energies must be at least 1")
    if getattr(args, "diabatic", False):
        if n < 2:
            raise InputError("--diabatic=True needs --n_energies >= 2 (a diabatic Hamiltonian couples states)")
        if getattr(args, "compute_nacs", False):
            raise InputError("--diabatic=True and --compute_nacs=True: a diabatic model has no NAC outputs, its "
                             "NACs follow from the diabatic Hamiltonian. Drop --compute_nacs")
    if getattr(args, "compute_nacs", False) and n < 2:
        raise InputError("--compute_nacs=True needs --n_energies >= 2 (NACs couple pairs of states)")
    if getattr(args, "compute_socs", False) and getattr(args, "soc_num", None) is None and n < 2:
        raise InputError("--compute_socs=True with one state: give --soc_num (free-form SOC vector) or "
                         "--n_energies >= 2 (one SOC per pair of states)")
    if int(getattr(args, "multipole_max_ell", 1)) < 0:
        raise InputError("--multipole_max_ell must be >= 0")


def _n_out(n_energies: int, diabatic: bool) -> int:
    return n_energies + n_energies * (n_energies - 1) // 2 if diabatic else n_energies


def _suggest_n_energies(n: int, diabatic: bool) -> str:
    if not diabatic:
        return f"For these data use --n_energies={n}"
    for s in range(1, 50):
        if s + s * (s - 1) // 2 == n:
            return f"For these data use --n_energies={s} (diabatic: {s} states give {n} outputs)"
    return (f"{n} values per frame is not a diabatic layout (S + S(S-1)/2 elements of H); without --diabatic "
            f"they would be {n} adiabatic states (--n_energies={n})")


def check_datasets(
    sets: Dict[str, Sequence],
    n_energies: int,
    diabatic: bool = False,
    compute_nacs: bool = False,
    compute_dipoles: bool = False,
    compute_socs: bool = False,
    keys: Optional[Dict[str, str]] = None,
    z_table_zs: Optional[Sequence[int]] = None,
    first_frame_info: Optional[List[str]] = None,
) -> None:
    """sets: {"train": configs, "valid": configs, "test (file)": configs}; raises DataError, logs warnings."""
    keys = keys or {}
    n_out = _n_out(n_energies, diabatic)
    found = f" Keys in the first training frame: {sorted(first_frame_info)}." if first_frame_info else ""
    train = sets.get("train", [])
    if len(train) == 0:
        raise DataError("the training set is empty")
    if "valid" in sets and len(sets["valid"]) == 0:
        raise DataError(f"the validation set is empty ({len(train)} training frames). Use a larger "
                        "--valid_fraction, or give --valid_file")

    # labels that were asked for but are not in the data (the property would silently not be trained)
    def labelled(attr):
        return sum(float(getattr(c, attr, 0.0)) > 0 for c in train)

    if labelled("energy_weight") == 0:
        raise DataError(f"no training frame has energies ('{keys.get('energy', 'REF_energy')}'): check "
                        f"--energy_key. Energies are needed (atomic reference energies, the loss).{found}")
    if labelled("forces_weight") == 0:
        logging.warning(f"No training frame has forces ('{keys.get('forces', 'REF_forces')}'): training on energies "
                        "only needs much more data. Check --forces_key.")
    for flag, attr, key in ((compute_nacs, "nacs_weight", "nacs"), (compute_dipoles, "dipoles_weight", "dipoles"),
                            (compute_socs, "socs_weight", "socs")):
        if flag and labelled(attr) == 0:
            raise DataError(f"--compute_{key}=True but no training frame has '{keys.get(key, key)}' labels: "
                            f"set --{key}_key to the name used in your file.{found}")
    n_mm_frames = sum(len(c.mm_charges) > 0 for c in train)
    if n_mm_frames and labelled("mm_forces_weight") == 0:
        logging.warning("No training frame has forces on the MM charges ('REF_mm_forces'); the model learns the "
                        "field only from energies and QM forces. Point-charge forces are cheap to get from most "
                        "QM codes and help a lot.")
    if n_mm_frames == 0:
        logging.warning("No training frame has MM point charges (mm_positions / mm_charges): this is a gas-phase "
                        "data set, and the model will not learn any response to a field.")

    # elements the model can learn: those in the training set (the z_table may also hold validation elements)
    train_zs = {int(z) for c in train for z in c.atomic_numbers}
    if z_table_zs is not None:
        train_zs &= {int(z) for z in z_table_zs}
    for name, configs in sets.items():
        bad_shape, nonfinite, close, periodic, unknown = [], [], [], [], {}
        zs = train_zs
        for k, c in enumerate(configs):
            e = np.asarray(c.energy).reshape(-1)
            where = getattr(c, "source", None) or f"frame {k}"
            if float(c.energy_weight) > 0 and len(e) != n_out:
                bad_shape.append((where, len(e)))
            for label in ("energy", "forces", "mm_forces", "dipoles", "nacs", "socs"):
                v = getattr(c, label, None)
                if v is not None and not np.all(np.isfinite(np.asarray(v, dtype=float))):
                    nonfinite.append((where, label))
            if len(c.mm_charges):
                d = np.linalg.norm(np.asarray(c.positions)[:, None, :] - np.asarray(c.mm_positions)[None], axis=-1)
                if d.min() < CLOSE_CONTACT:
                    close.append((where, float(d.min())))
            if c.pbc is not None and any(c.pbc):
                periodic.append(where)
            if zs is not None:
                for z in set(int(z) for z in c.atomic_numbers) - zs:
                    unknown.setdefault(z, []).append(where)
        if bad_shape:
            k, n = bad_shape[0]
            raise DataError(f"{name}: {k} has {n} energies but the model has {n_out} energy outputs "
                            f"(--n_energies={n_energies}{', --diabatic' if diabatic else ''}); "
                            f"{len(bad_shape)} frame(s) affected. {_suggest_n_energies(n, diabatic)}")
        if nonfinite:
            k, label = nonfinite[0]
            raise DataError(f"{name}: {k} has NaN or infinite values in '{label}' "
                            f"({len(nonfinite)} occurrence(s)); remove or fix these frames")
        if unknown:
            z, ks = next(iter(unknown.items()))
            raise DataError(f"{name}: {ks[0]} contains element Z={z}, which is not in the training set "
                            f"(elements {sorted(int(x) for x in zs)}). Add frames with Z={z} to the training data")
        if close:
            k, d = close[0]
            logging.warning(f"{name}: {len(close)} frame(s) have an MM charge within {CLOSE_CONTACT} A of a QM atom "
                            f"({k}: {d:.3f} A). Usually a QM atom is also listed among the MM charges, or a "
                            "link host was kept (link scheme Z0). Check these frames")
        if periodic:
            logging.warning(f"{name}: {len(periodic)} frame(s) are periodic (pbc=True, e.g. {periodic[0]}). "
                            "FieldMACE treats the QM region as a cluster in the field of the point charges; periodic "
                            "images of the QM atoms are used for the QM-QM graph, but not for the MM charges. Set "
                            "pbc=False unless the QM region is a periodic solid")
