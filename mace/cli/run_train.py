###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# Modified for FieldMACE by Rhyan Barrett
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import sys
import ast
import glob
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch.distributed
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, modules, tools
from mace.calculators.foundations_models import mace_mp, mace_off
from mace.tools import torch_geometric
from mace.tools.finetuning_utils import load_foundations
from mace.tools.scripts_utils import (
    LRScheduler,
    convert_to_json_format,
    create_error_table,
    extract_config_mace_model,
    get_atomic_energies,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_files_with_suffix,
    print_git_commit,
)
from mace.tools.slurm_distributed import DistributedEnvironment
from mace.tools.utils import AtomicNumberTable


def _build_field_model(name, **kwargs):
    """Settings the model rejects (e.g. vector outputs without vector features) are input errors."""
    from mace.tools.data_checks import InputError

    try:
        return getattr(modules, name)(**kwargs)
    except ValueError as err:
        raise InputError(str(err)) from None


def main() -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args = tools.build_default_arg_parser().parse_args()
    from mace.tools.data_checks import DataError, InputError

    try:
        run(args)
    except InputError as err:  # a problem with the settings or the data: a clear message, no traceback
        what = "the training data" if isinstance(err, DataError) else "the input"
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        logging.error(f"Problem with {what}: {err}")
        sys.exit(1)


def run(args: argparse.Namespace) -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args, input_log_messages = tools.check_args(args)
    tag = tools.get_tag(name=args.name, seed=args.seed)
    if args.distributed:
        try:
            distr_env = DistributedEnvironment()
        except Exception as e:  # pylint: disable=W0703
            from mace.tools.data_checks import InputError

            raise InputError(f"--distributed needs a SLURM job (SLURM_JOB_NODELIST, SLURM_PROCID, ...): "
                             f"failed to set up the distributed environment ({type(e).__name__}: {e})") from None
        world_size = distr_env.world_size
        local_rank = distr_env.local_rank
        rank = distr_env.rank
        if rank == 0:
            print(distr_env)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = int(0)

    # Setup
    tools.set_seeds(args.seed)
    if str(args.log_level).upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        from mace.tools.data_checks import InputError as _LogLevelError

        raise _LogLevelError(f"--log_level={args.log_level}: use DEBUG, INFO, WARNING or ERROR")
    args.log_level = str(args.log_level).upper()
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)
    from mace.tools.data_checks import check_arguments, is_xyz

    check_arguments(args)
    user_scaling = args.scaling != "rms_forces_scaling" or args.mean is not None or args.std is not None

    if args.distributed:
        torch.cuda.set_device(local_rank)
        logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        logging.info(f"Processes: {world_size}")

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    commit = print_git_commit()
    if args.foundation_model is not None:
        if args.foundation_model in ["small", "medium", "large"]:
            logging.info(
                f"Using foundation model mace-mp-0 {args.foundation_model} as initial checkpoint."
            )
            calc = mace_mp(
                model=args.foundation_model,
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        elif args.foundation_model in ["small_off", "medium_off", "large_off"]:
            model_type = args.foundation_model.split("_")[0]
            logging.info(
                f"Using foundation model mace-off-2023 {model_type} as initial checkpoint. ASL license."
            )
            calc = mace_off(
                model=model_type,
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        else:
            model_foundation = tools.torch_tools.load_model(args.foundation_model, device)
            logging.info(
                f"Using foundation model {args.foundation_model} as initial checkpoint."
            )
        # the training dtype (MACE-MP/OFF files are float64; mixing dtypes fails in the first forward pass)
        model_foundation = model_foundation.to(tools.torch_tools.dtype_dict[args.default_dtype])
        args.r_max = model_foundation.r_max.item()

    if args.statistics_file is not None:
        with open(args.statistics_file, "r") as f:  # pylint: disable=W1514
            statistics = json.load(f)
        logging.info("Using statistics json file")
        if args.foundation_model is None and float(statistics["r_max"]) != float(args.r_max):
            logging.warning(f"--r_max={args.r_max} replaced by r_max={statistics['r_max']} of {args.statistics_file} "
                            "(the statistics, e.g. the average number of neighbours, were computed with it)")
        args.r_max = (
            statistics["r_max"] if args.foundation_model is None else args.r_max
        )
        args.atomic_numbers = statistics["atomic_numbers"]
        args.mean = statistics["mean"]
        args.std = statistics["std"]
        args.avg_num_neighbors = statistics["avg_num_neighbors"]
        args.compute_avg_num_neighbors = False
        args.E0s = statistics["atomic_energies"]

    logging.info("")
    logging.info("===========LOADING INPUT DATA===========")
    # Data preparation
    if is_xyz(args.train_file):
        if args.valid_file is not None:
            assert is_xyz(args.valid_file), "valid_file if given must be same format as train_file"
        config_type_weights = get_config_type_weights(args.config_type_weights)
        collections, atomic_energies_dict = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=args.train_file,
            valid_path=args.valid_file,
            valid_fraction=args.valid_fraction,
            config_type_weights=config_type_weights,
            test_path=args.test_file,
            seed=args.seed,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            virials_key=args.virials_key,
            dipoles_key=args.dipoles_key,
            charges_key=args.charges_key,
            nacs_key=args.nacs_key,
            socs_key=args.socs_key,
            soc_num=args.soc_num if args.compute_socs else None,
            n_states=args.n_energies if args.diabatic else None,
            n_outputs=args.n_energies + (args.n_energies * (args.n_energies - 1) // 2 if args.diabatic else 0),
            keep_isolated_atoms=args.keep_isolated_atoms,
        )
        if len(collections.train) < args.batch_size:
            logging.warning(
                f"Batch size ({args.batch_size}) is larger than the number of training frames "
                f"({len(collections.train)}); using --batch_size={len(collections.train)}"
            )
            args.batch_size = len(collections.train)
        if len(collections.valid) < args.valid_batch_size:
            logging.warning(
                f"Validation batch size ({args.valid_batch_size}) is larger than the number of validation data ({len(collections.valid)})"
            )
            args.valid_batch_size = len(collections.valid)

    else:
        atomic_energies_dict = None

    # Atomic number table
    # yapf: disable
    if args.atomic_numbers is None:
        assert is_xyz(args.train_file), "Must specify atomic_numbers when using .h5 train_file input"
        z_table = tools.get_atomic_number_table_from_zs(
            z
            for configs in (collections.train, collections.valid)
            for config in configs
            for z in config.atomic_numbers
        )
    else:
        if args.statistics_file is None:
            logging.info("Using atomic numbers from command line argument")
        else:
            logging.info("Using atomic numbers from statistics file")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)
    # yapf: enable
    logging.info(f"Atomic Numbers used: {z_table.zs}")
    logging.info(f"Model: {args.model}")
    if is_xyz(args.train_file):  # catch data mistakes before training, with file and frame numbers
        from mace.tools.data_checks import check_datasets

        try:
            import ase.io

            first = ase.io.read(args.train_file, index=0)
            first_keys = list(first.info) + [k for k in first.arrays if k not in ("numbers", "positions")]
            if first.calc is not None:  # ASE puts energy=/forces= of an extxyz file into a calculator
                first_keys += list(first.calc.results)
        except Exception:  # noqa: BLE001
            first_keys = None
        sets = {f"training set ({args.train_file})": collections.train,
                "valid": collections.valid}
        for name, configs in collections.tests:
            sets[f"test set '{name}'"] = configs
        check_datasets(
            {("train" if k.startswith("training") else k): v for k, v in sets.items()},
            n_energies=args.n_energies, diabatic=args.diabatic, compute_nacs=args.compute_nacs,
            compute_dipoles=args.compute_dipoles, compute_socs=args.compute_socs,
            keys={"energy": args.energy_key, "forces": args.forces_key, "nacs": args.nacs_key,
                  "dipoles": args.dipoles_key, "socs": args.socs_key},
            z_table_zs=z_table.zs, first_frame_info=first_keys,
            atomic_numbers_given=args.atomic_numbers is not None,
        )
    if atomic_energies_dict:  # from IsolatedAtom frames in the training file
        from mace.tools.scripts_utils import normalise_atomic_energies

        atomic_energies_dict = normalise_atomic_energies(atomic_energies_dict)
        missing = [int(z) for z in z_table.zs if z not in atomic_energies_dict]
        if missing and args.E0s is not None:
            logging.warning(f"IsolatedAtom frames give E0s for Z={sorted(atomic_energies_dict)} but not for "
                            f"Z={missing}: using --E0s={args.E0s} for all elements instead")
            atomic_energies_dict = None
        elif missing:
            from mace.tools.data_checks import InputError

            raise InputError(f"IsolatedAtom frames in the training file give E0s for Z={sorted(atomic_energies_dict)} "
                             f"but not for Z={missing}: add IsolatedAtom frames for them, or pass --E0s "
                             "(e.g. --E0s=average) to use that for all elements")
    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        if args.E0s is None or not str(args.E0s).strip():
            from mace.tools.data_checks import InputError

            raise InputError("no atomic reference energies: pass --E0s=average (least squares from the training "
                             "set), explicit values ('{1: -13.6, 6: -1029.8}', eV) or a .json file")
        if args.E0s.lower() == "foundation":
            from mace.tools.data_checks import InputError

            if args.foundation_model is None:
                raise InputError("--E0s=foundation needs --foundation_model")
            z_table_foundation = AtomicNumberTable(
                [int(z) for z in model_foundation.atomic_numbers]
            )
            unknown = [int(z) for z in z_table.zs if int(z) not in z_table_foundation.zs]
            if unknown:
                raise InputError(f"--E0s=foundation: the foundation model has no E0 for Z={unknown}")
            atomic_energies_dict = {
                z: model_foundation.atomic_energies_fn.atomic_energies[
                    z_table_foundation.z_to_index(z)
                ].item()
                for z in z_table.zs
            }
            logging.info(
                f"Using Atomic Energies from foundation model [z, eV]: {', '.join([f'{z}: {atomic_energies_dict[z]}' for z in z_table.zs])}"
            )
        else:
            if is_xyz(args.train_file):
                atomic_energies_dict = get_atomic_energies(
                    args.E0s, collections.train, z_table,
                    n_diagonal=args.n_energies if args.diabatic else None,
                )
            else:
                atomic_energies_dict = get_atomic_energies(args.E0s, None, z_table)
    if atomic_energies_dict is not None:
        missing = [int(z) for z in z_table.zs if z not in atomic_energies_dict]
        if missing:
            from mace.tools.data_checks import InputError

            raise InputError(f"--E0s has no atomic energy for element(s) Z={missing} (given: "
                             f"{sorted(int(z) for z in atomic_energies_dict)}). Add them, or use --E0s=average")
  
    if args.model == "ExcitedMACE":
        atomic_energies = None
        compute_dipole = True
        compute_energy = True
        args.compute_forces = True
        compute_nacs = True
        compute_virials = False
        args.compute_stress = False

        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )

        logging.info(
            f"Atomic Energies used (z: eV): {{{', '.join([f'{z}: {atomic_energies_dict[z]}' for z in z_table.zs])}}}"
        )
        
    if args.model in ("FieldEMACE", "PerAtomFieldEMACE"):
        compute_energy = True
        args.compute_forces = True
        compute_dipole = bool(args.compute_dipoles)
        compute_nacs = bool(args.compute_nacs)
        compute_socs = bool(args.compute_socs)
        compute_virials = False
        args.compute_stress = False
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(
            f"Atomic Energies used (z: eV): {{{', '.join([f'{z}: {atomic_energies_dict[z]}' for z in z_table.zs])}}}"
        )
        if compute_nacs or compute_dipole or compute_socs:
            logging.info(
                f"Excited-state properties: NACs={compute_nacs} (key {args.nacs_key}), "
                f"dipoles={compute_dipole} (key {args.dipoles_key}), SOCs={compute_socs} (key {args.socs_key})"
            )

    if is_xyz(args.train_file):
        train_set = [
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.train
        ]
        valid_set = [
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.valid
        ]
    else:  # HDF5: one file, or a directory of shards (each of train/valid independently)
        def h5_set(path):
            if os.path.isdir(path):
                return data.dataset_from_sharded_hdf5(path, r_max=args.r_max, z_table=z_table)
            return data.HDF5Dataset(path, r_max=args.r_max, z_table=z_table)

        train_set = h5_set(args.train_file)
        valid_set = h5_set(args.valid_file)
        from mace.tools.data_checks import InputError as _InputError

        n_out = args.n_energies + (args.n_energies * (args.n_energies - 1) // 2 if args.diabatic else 0)
        for name_, set_ in (("--train_file", train_set), ("--valid_file", valid_set)):
            if len(set_) == 0:
                raise _InputError(f"{name_}: the HDF5 data are empty")
            n_lab = int(set_[0].energy.shape[-1])
            if n_lab != n_out:
                from mace.tools.data_checks import _suggest_n_energies

                raise _InputError(f"{name_}: the HDF5 frames have {n_lab} energies but the model has {n_out} energy "
                                 f"outputs. {_suggest_n_energies(n_lab, args.diabatic)}")
    
    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,  # every validation frame is evaluated
            seed=args.seed,
        )
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=(train_sampler is None),
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=valid_set,
        batch_size=args.valid_batch_size,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    logging.info("")
    logging.info("===========MODEL DETAILS===========")

    loss_fn = modules.WeightedEnergyForcesNacsDipoleLoss(
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
        dipoles_weight=args.dipoles_weight,
        nacs_weight=args.nacs_weight,
        mm_forces_weight=args.mm_forces_weight,
        socs_weight=args.socs_weight,
        output_weights=([1.0] * args.n_energies + [float(args.diabatic_coupling_weight)]
                        * (args.n_energies * (args.n_energies - 1) // 2)) if args.diabatic else None,
    )
    
    if args.compute_avg_num_neighbors:
        avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
        if args.distributed:
            num_graphs = torch.tensor(len(train_loader.dataset)).to(device)
            num_neighbors = num_graphs * torch.tensor(avg_num_neighbors).to(device)
            torch.distributed.all_reduce(num_graphs, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                num_neighbors, op=torch.distributed.ReduceOp.SUM
            )
            args.avg_num_neighbors = (num_neighbors / num_graphs).item()
        else:
            args.avg_num_neighbors = avg_num_neighbors
    if args.avg_num_neighbors < 2 or args.avg_num_neighbors > 100:
        logging.warning(
            f"Unusual average number of neighbors: {args.avg_num_neighbors:.1f}"
        )
    else:
        logging.info(f"Average number of neighbors: {args.avg_num_neighbors:.1f}")

    # Selecting outputs
    compute_virials = False  # field models: QM clusters in point charges, no stress/virials

    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
        "nacs": compute_nacs,
        "mm_forces": bool(float(loss_fn.mm_forces_weight) > 0),
    }

    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )

    # field models have no scale/shift block: --scaling, --mean and --std do not apply
    if user_scaling:
        logging.warning("--scaling, --mean and --std are not used by the field models (no scale/shift); ignored")
    # Build model
    if args.foundation_model is not None and args.model in ["MACE", "ScaleShiftMACE"]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_numbers"] = z_table.zs
        model_config_foundation["num_elements"] = len(z_table)
        args.max_L = model_config_foundation["hidden_irreps"].lmax
        args.num_channels = list(
            {irrep.mul for irrep in o3.Irreps(model_config_foundation["hidden_irreps"])}
        )[0]
        model_config_foundation["atomic_inter_shift"] = (
            model_foundation.scale_shift.shift.item()
        )
        model_config_foundation["atomic_inter_scale"] = (
            model_foundation.scale_shift.scale.item()
        )
        model_config_foundation["atomic_energies"] = atomic_energies
        model_config = model_config_foundation  # pylint
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} Å (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} Å)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        logging.info("Building model")
        if args.foundation_model is not None:
            # field models reuse the foundation's MACE layers: their shapes must match
            fcfg = extract_config_mace_model(model_foundation)
            if "error" not in fcfg:
                matched = {
                    "num_radial_basis": fcfg["num_bessel"],
                    "num_cutoff_basis": int(fcfg["num_polynomial_cutoff"]),
                    "max_ell": fcfg["max_ell"],
                    "hidden_irreps": str(fcfg["hidden_irreps"]),
                    "radial_type": fcfg["radial_type"],
                    "radial_MLP": str(list(fcfg["radial_MLP"])),
                    "num_interactions": fcfg["num_interactions"],
                    "correlation": fcfg["correlation"],
                    # the first layer's block too, or its skip connection cannot be transferred
                    "interaction_first": fcfg["interaction_cls_first"].__name__,
                    "interaction": fcfg["interaction_cls"].__name__,
                }
                for key, value in matched.items():
                    current = getattr(args, key)
                    if key == "hidden_irreps":
                        same = o3.Irreps(str(current)) == o3.Irreps(value)
                    else:
                        same = str(current).replace(" ", "") == str(value).replace(" ", "")
                    if not same:
                        logging.warning(f"--{key}={getattr(args, key)} replaced by {value} to match the foundation model")
                        setattr(args, key, value)
                args.hidden_irreps = o3.Irreps(args.hidden_irreps)
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} Å (total receptive field for each atom: {args.r_max * args.num_interactions} Å)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )
        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            atomic_energies=atomic_energies,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
        )

    model: torch.nn.Module

    if args.model == "ExcitedMACE":
        model = modules.ExcitedMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            n_energies=args.n_energies,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            compute_nacs=args.compute_nacs,
            compute_dipoles=args.compute_dipoles,
        )
    elif args.model in ("FieldEMACE", "PerAtomFieldEMACE"):
        model = _build_field_model(
            args.model,
            **model_config,
            pair_repulsion=args.pair_repulsion,
            n_energies=args.n_energies,
            distance_transform=args.distance_transform,
            multipole_max_ell=args.multipole_max_ell,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            field_irreps=o3.Irreps(args.field_irreps),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            compute_nacs=args.compute_nacs,
            compute_dipoles=args.compute_dipoles,
            compute_socs=args.compute_socs,
            soc_num=args.soc_num,
            diabatic=args.diabatic,
            multipole_embedding=args.multipole_embedding,
            num_multipole_basis=args.num_multipole_basis,
            expansion_centre=args.expansion_centre,
        )
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")

    if args.foundation_model is not None:
        model = load_foundations(
            model,
            model_foundation,
            z_table
        )
    model.to(device)

    logging.debug(model)
    logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
    logging.info("")
    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(
        f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}"
    )
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Optimizer: weight decay on the linear weights of interactions, field interactions and
    # products (as in MACE); every other parameter (embeddings, multipole embedding,
    # multipole linear_up, readouts, property heads, ...) is trained without decay.
    # The original FieldMACE left linear_up (and the attention blocks) out of the optimizer,
    # so they were never trained.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decayed = (
            (name.startswith("interactions.") and ("linear.weight" in name or "skip_tp_full.weight" in name))
            or (name.startswith("field_interactions.") and "linear.weight" in name)
            or name.startswith("products.")
        )
        (decay_params if decayed else no_decay_params).append(param)
    n_grouped = len(decay_params) + len(no_decay_params)
    assert n_grouped == sum(1 for p in model.parameters() if p.requires_grad), "parameter missing from optimizer"
    param_options = dict(
        params=[
            {"name": "decay", "params": decay_params, "weight_decay": args.weight_decay},
            {"name": "no_decay", "params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
        betas=(args.beta, 0.999),
    )
    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    elif args.optimizer == "schedulefree":
        try:
            from schedulefree import adamw_schedulefree
        except ImportError as exc:
            raise ImportError(
                "`schedulefree` is not installed. Please install it via `pip install schedulefree` (not mace-torch: that would replace FieldMACE's `mace` package)"
            ) from exc
        _param_options = {k: v for k, v in param_options.items() if k != "amsgrad"}
        optimizer = adamw_schedulefree.AdamWScheduleFree(**_param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = max(1, args.max_num_epochs // 4 * 3)
        logging.info(
            f"Stage Two will start after {args.start_swa} epochs with loss function:"
        )
        # same loss as Stage One with the Stage Two weights (energies weighted up, smaller learning rate)
        loss_fn_energy = modules.WeightedEnergyForcesNacsDipoleLoss(
            energy_weight=args.swa_energy_weight,
            forces_weight=args.swa_forces_weight,
            dipoles_weight=args.swa_dipole_weight,
            nacs_weight=args.nacs_weight,
            # MM forces keep their ratio to the QM forces
            mm_forces_weight=float(loss_fn.mm_forces_weight) * (
                args.swa_forces_weight / args.forces_weight if args.forces_weight > 0 else 1.0),
            socs_weight=args.socs_weight,
            output_weights=loss_fn.output_weights.tolist(),
        )
        logging.info(loss_fn_energy)
        swa = tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(
                optimizer=optimizer,
                swa_lr=args.swa_lr,
                anneal_epochs=1,
                anneal_strategy="linear",
            ),
            start=args.start_swa,
            loss_fn=loss_fn_energy,
        )

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        opt_start_epoch = None
        if args.swa:  # a Stage Two checkpoint, if training got that far
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        if opt_start_epoch is None:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    if args.wandb:
        logging.info("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        args_dict = vars(args)
        args_dict_json = json.dumps(args_dict)
        for key in args.wandb_log_hypers:
            wandb_config[key] = args_dict[key]
        tools.init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config,
            directory=args.wandb_dir,
        )
        wandb.run.summary["params"] = args_dict_json

    if args.distributed:
        distributed_model = DDP(model, device_ids=[local_rank])
    else:
        distributed_model = None

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        distributed=args.distributed,
        distributed_model=distributed_model,
        train_sampler=train_sampler,
        rank=rank,
    )
    logging.info("")
    logging.info("===========RESULTS===========")
    logging.info("Computing metrics for training, validation, and test sets")

    # the training loader drops the last incomplete batch (and shuffles): evaluate every training frame
    train_eval_loader = train_loader if args.distributed else torch_geometric.dataloader.DataLoader(
        dataset=train_set, batch_size=args.valid_batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory and torch.cuda.is_available())
    all_data_loaders = {
        "train": train_eval_loader,
        "valid": valid_loader,
    }

    test_sets = {}
    if is_xyz(args.train_file):
        for name, subset in collections.tests:
            test_sets[name] = [
                data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
                for config in subset
            ]
    elif args.test_dir is not None and not args.multi_processed_test:
        # mace_preprocess_data writes test/<config_type>_<k>.h5 (one file per process): one test set per config_type
        import re
        from torch.utils.data import ConcatDataset

        groups = {}
        for test_file in sorted(glob.glob(os.path.join(args.test_dir, "*.h5"))):
            stem = os.path.splitext(os.path.basename(test_file))[0]
            name = re.sub(r"(_test)?(_\d+)?$", "", stem) or stem
            groups.setdefault(name, []).append(
                data.HDF5Dataset(test_file, r_max=args.r_max, z_table=z_table))
        if not groups:
            logging.warning(f"--test_dir={args.test_dir}: no .h5 files found, no test set")
        for name, sets in groups.items():
            test_sets[name] = sets[0] if len(sets) == 1 else ConcatDataset(sets)
    elif args.test_dir is not None:
        for folder in sorted(glob.glob(os.path.join(args.test_dir, "*"))):
            if os.path.isdir(folder):
                test_sets[os.path.basename(folder)] = data.dataset_from_sharded_hdf5(
                    folder, r_max=args.r_max, z_table=z_table
                )

    for test_name, test_set in test_sets.items():
        test_sampler = None
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
                seed=args.seed,
            )
        test_loader = torch_geometric.dataloader.DataLoader(
            test_set,
            batch_size=args.valid_batch_size,
            sampler=test_sampler,
            shuffle=False,
            drop_last=False,  # every test frame is evaluated
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
        )
        all_data_loaders[test_name] = test_loader

    train_valid_data_loader = {
        k: v for k, v in all_data_loaders.items() if k in ["train", "valid"]
    }
    test_data_loader = {
        k: v for k, v in all_data_loaders.items() if k not in ["train", "valid"]
    }

    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        if swa_eval and epoch is None:
            logging.warning("Stage Two wrote no checkpoint (it never improved on the validation loss): "
                            "no _stagetwo.model is saved")
            continue
        model.to(device)
        if args.distributed:
            distributed_model = DDP(model, device_ids=[local_rank])
        model_to_evaluate = model if not args.distributed else distributed_model
        if swa_eval:
            logging.info(f"Loaded Stage two model from epoch {epoch} for evaluation")
        else:
            logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

        for param in model.parameters():
            param.requires_grad = False

        if epoch is None:
            logging.warning("No checkpoint was written (no training epoch): saving the initial model")
            os.makedirs(args.checkpoints_dir, exist_ok=True)
        if rank == 0:
            # Save entire model
            if swa_eval:
                model_path = Path(args.checkpoints_dir) / (tag + "_stagetwo.model")
            else:
                model_path = Path(args.checkpoints_dir) / (tag + ".model")
            logging.info(f"Saving model to {model_path}")
            model_to_save = deepcopy(model).to("cpu") if args.save_cpu else model
            torch.save(model_to_save, model_path)
            extra_files = {
                "commit.txt": commit.encode("utf-8") if commit is not None else b"",
                "config.yaml": json.dumps(
                    convert_to_json_format(extract_config_mace_model(model_to_save))
                ),
            }
            if swa_eval:
                torch.save(
                    model_to_save, Path(args.model_dir) / (args.name + "_stagetwo.model")
                )
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_stagetwo_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    logging.warning(f"Could not save a TorchScript-compiled model: {e}")
            else:
                torch.save(model_to_save, Path(args.model_dir) / (args.name + ".model"))
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata to {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    logging.warning(f"Could not save a TorchScript-compiled model: {e}")

        # the model is saved first: a failure while evaluating must not lose it
        try:
            table_train = create_error_table(
                table_type=args.error_table,
                all_data_loaders=train_valid_data_loader,
                model=model_to_evaluate,
                loss_fn=loss_fn,
                output_args=output_args,
                log_wandb=args.wandb,
                device=device,
                distributed=args.distributed,
            )
            table_test = create_error_table(
                table_type=args.error_table,
                all_data_loaders=test_data_loader,
                model=model_to_evaluate,
                loss_fn=loss_fn,
                output_args=output_args,
                log_wandb=args.wandb,
                device=device,
                distributed=args.distributed,
            )
            logging.info("Error-table on TRAIN and VALID:\n" + str(table_train))
            if test_data_loader:
                logging.info("Error-table on TEST:\n" + str(table_test))
        except Exception as err:  # pylint: disable=broad-except
            logging.error(f"Evaluating the final model failed ({type(err).__name__}: {err}); the model was saved")
        if args.distributed:
            torch.distributed.barrier()

    logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
