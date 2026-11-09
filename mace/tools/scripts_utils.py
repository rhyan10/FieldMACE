###########################################################################################
# Training utils
# Authors: David Kovacs, Ilyes Batatia
# Modified for FieldMACE by Rhyan Barrett
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import dataclasses
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from e3nn import o3
from prettytable import PrettyTable

from mace import data, modules
from mace.tools import evaluate


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset_from_xyz(
    work_dir: str,
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    keep_isolated_atoms: bool = False,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "virials",
    dipoles_key: str = "dipoles",
    nacs_key: str = "nacs",
    charges_key: str = "charges",
    pc_key: str = "REF_mm_forces",
    socs_key: str = "REF_socs",
    soc_num: Optional[int] = None,
    n_states: Optional[int] = None,
    n_outputs: Optional[int] = None,
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipoles_key=dipoles_key,
        charges_key=charges_key,
        nacs_key = nacs_key,
        socs_key=socs_key,
        soc_num=soc_num,
        n_states=n_states,
        n_outputs=n_outputs,
        pc_key=pc_key,
        extract_atomic_energies=True,
        keep_isolated_atoms=keep_isolated_atoms,
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipoles_key=dipoles_key,
            nacs_key = nacs_key,
            socs_key=socs_key,
        soc_num=soc_num,
        n_states=n_states,
        n_outputs=n_outputs,
            charges_key=charges_key,
            pc_key=pc_key,
            extract_atomic_energies=False,
        )
        train_configs = all_train_configs
    else:
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed, work_dir
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipoles_key=dipoles_key,
            stress_key=stress_key,
            nacs_key = nacs_key,
            socs_key=socs_key,
        soc_num=soc_num,
        n_states=n_states,
        n_outputs=n_outputs,
            virials_key=virials_key,
            charges_key=charges_key,
            pc_key=pc_key,
            extract_atomic_energies=False,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Test set ({len(all_test_configs)} configs) loaded from '{test_path}':"
        )
        #for name, tmp_configs in test_configs:
        #    logging.info(
        #        f"{name}: {len(tmp_configs)} configs, {np.sum([1 if config.energy else 0 for config in tmp_configs])} energy, {np.sum([config.forces.size for config in tmp_configs])} forces"
        #    )

    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )


def get_config_type_weights(ct_weights):
    """
    Get config type weights from command line argument
    """
    try:
        config_type_weights = ast.literal_eval(ct_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
    return config_type_weights


def print_git_commit():
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        commit = repo.head.commit.hexsha
        logging.debug(f"Current Git commit: {commit}")
        return commit
    except Exception as e:  # pylint: disable=W0703
        logging.debug(f"Error accessing Git repository: {e}")
        return "None"


def extract_config_mace_model(model: torch.nn.Module) -> Dict[str, Any]:
    if model.__class__.__name__ != "ScaleShiftMACE":
        return {"error": "Model is not a ScaleShiftMACE model"}

    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        return radial_type

    def radial_to_transform(radial):
        if not hasattr(radial, "distance_transform"):
            return None
        if radial.distance_transform.__class__.__name__ == "AgnesiTransform":
            return "Agnesi"
        if radial.distance_transform.__class__.__name__ == "SoftTransform":
            return "Soft"
        return radial.distance_transform.__class__.__name__

    config = {
        "r_max": model.r_max.item(),
        "num_bessel": len(model.radial_embedding.bessel_fn.bessel_weights),
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics._lmax,  # pylint: disable=protected-access
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": model.num_interactions.item(),
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "MLP_irreps": (
            o3.Irreps(str(model.readouts[-1].hidden_irreps))
            if model.num_interactions.item() > 1
            else 1
        ),
        "gate": (
            model.readouts[-1]  # pylint: disable=protected-access
            .non_linearity._modules["acts"][0]
            .f
            if model.num_interactions.item() > 1
            else None
        ),
        "atomic_energies": model.atomic_energies_fn.atomic_energies.cpu().numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers,
        "correlation": len(
            model.products[0].symmetric_contractions.contractions[0].weights
        )
        + 1,
        "radial_type": radial_to_name(
            model.radial_embedding.bessel_fn.__class__.__name__
        ),
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "pair_repulsion": hasattr(model, "pair_repulsion_fn"),
        "distance_transform": radial_to_transform(model.radial_embedding),
        "atomic_inter_scale": model.scale_shift.scale.item(),
        "atomic_inter_shift": model.scale_shift.shift.item(),
    }
    return config


def extract_load(f: str, map_location: str = "cpu") -> torch.nn.Module:
    from mace.tools.torch_tools import load_model

    model = load_model(f, map_location)
    model_copy = model.__class__(**extract_config_mace_model(model))
    model_copy.load_state_dict(model.state_dict())
    return model_copy.to(map_location)


def extract_model(model: torch.nn.Module, map_location: str = "cpu") -> torch.nn.Module:
    model_copy = model.__class__(**extract_config_mace_model(model))
    model_copy.load_state_dict(model.state_dict())
    return model_copy.to(map_location)


def convert_to_json_format(dict_input):
    for key, value in dict_input.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            dict_input[key] = value.tolist()
        # # check if the value is a class and convert it to a string
        elif hasattr(value, "__class__"):
            dict_input[key] = str(value)
    return dict_input


def convert_from_json_format(dict_input):
    dict_output = dict_input.copy()
    if (
        dict_input["interaction_cls"]
        == "<class 'mace.modules.blocks.RealAgnosticResidualInteractionBlock'>"
    ):
        dict_output["interaction_cls"] = (
            modules.blocks.RealAgnosticResidualInteractionBlock
        )
    if (
        dict_input["interaction_cls"]
        == "<class 'mace.modules.blocks.RealAgnosticInteractionBlock'>"
    ):
        dict_output["interaction_cls"] = modules.blocks.RealAgnosticInteractionBlock
    if (
        dict_input["interaction_cls_first"]
        == "<class 'mace.modules.blocks.RealAgnosticResidualInteractionBlock'>"
    ):
        dict_output["interaction_cls_first"] = (
            modules.blocks.RealAgnosticResidualInteractionBlock
        )
    if (
        dict_input["interaction_cls_first"]
        == "<class 'mace.modules.blocks.RealAgnosticInteractionBlock'>"
    ):
        dict_output["interaction_cls_first"] = (
            modules.blocks.RealAgnosticInteractionBlock
        )
    dict_output["r_max"] = float(dict_input["r_max"])
    dict_output["num_bessel"] = int(dict_input["num_bessel"])
    dict_output["num_polynomial_cutoff"] = float(dict_input["num_polynomial_cutoff"])
    dict_output["max_ell"] = int(dict_input["max_ell"])
    dict_output["num_interactions"] = int(dict_input["num_interactions"])
    dict_output["num_elements"] = int(dict_input["num_elements"])
    dict_output["hidden_irreps"] = o3.Irreps(dict_input["hidden_irreps"])
    dict_output["MLP_irreps"] = o3.Irreps(dict_input["MLP_irreps"])
    dict_output["avg_num_neighbors"] = float(dict_input["avg_num_neighbors"])
    dict_output["gate"] = torch.nn.functional.silu
    dict_output["atomic_energies"] = np.array(dict_input["atomic_energies"])
    dict_output["atomic_numbers"] = dict_input["atomic_numbers"]
    dict_output["correlation"] = int(dict_input["correlation"])
    dict_output["radial_type"] = dict_input["radial_type"]
    dict_output["radial_MLP"] = ast.literal_eval(dict_input["radial_MLP"])
    dict_output["pair_repulsion"] = ast.literal_eval(dict_input["pair_repulsion"])
    dict_output["distance_transform"] = dict_input["distance_transform"]
    dict_output["atomic_inter_scale"] = float(dict_input["atomic_inter_scale"])
    dict_output["atomic_inter_shift"] = float(dict_input["atomic_inter_shift"])

    return dict_output


def load_from_json(f: str, map_location: str = "cpu") -> torch.nn.Module:
    extra_files_extract = {"commit.txt": None, "config.json": None}
    model_jit_load = torch.jit.load(
        f, _extra_files=extra_files_extract, map_location=map_location
    )
    model_load_yaml = modules.ScaleShiftMACE(
        **convert_from_json_format(json.loads(extra_files_extract["config.json"]))
    )
    model_load_yaml.load_state_dict(model_jit_load.state_dict())
    return model_load_yaml.to(map_location)


def get_atomic_energies(E0s, train_collection, z_table, n_diagonal=None) -> dict:
    if isinstance(E0s, str) and not E0s.strip():
        E0s = None
    if E0s is not None:
        logging.info(
            "Isolated Atomic Energies (E0s) not in training file, using command line argument"
        )
        if E0s.lower() == "average":
            logging.info(
                "Computing average Atomic Energies using least squares regression"
            )
            # catch if colections.train not defined above
            try:
                assert train_collection is not None
                atomic_energies_dict = data.compute_average_E0s(
                    train_collection, z_table, n_diagonal=n_diagonal
                )
            except Exception as e:
                from mace.tools.data_checks import InputError

                raise InputError(
                    f"--E0s=average needs the training XYZ frames ({e or 'none given'}): pass explicit --E0s or "
                    "--statistics_file"
                ) from e
        else:
            if E0s.endswith(".json"):
                logging.info(f"Loading atomic energies from {E0s}")
                with open(E0s, "r", encoding="utf-8") as f:
                    atomic_energies_dict = json.load(f)
            else:
                try:
                    atomic_energies_dict = ast.literal_eval(E0s)
                    assert isinstance(atomic_energies_dict, dict)
                except Exception as e:
                    from mace.tools.data_checks import InputError

                    raise InputError(
                        f"--E0s={E0s!r} is not valid: use 'average', a dictionary such as "
                        "'{1: -13.6, 6: -1029.8}' (eV), or a .json file"
                    ) from e
    else:
        from mace.tools.data_checks import InputError

        raise InputError("no atomic reference energies: no IsolatedAtom frames in the training file and no --E0s "
                         "(use --E0s=average, explicit values or a .json file)")
    return normalise_atomic_energies(atomic_energies_dict)


def normalise_atomic_energies(atomic_energies_dict) -> Dict[int, float]:
    """{Z: E0} with int keys (JSON files have string keys) and one float per element.

    An isolated-atom frame of a multi-state data set carries one energy per state; the atomic
    reference energy is that of the first state (the shift is the same for all states)."""
    out = {}
    for z, e in atomic_energies_dict.items():
        e = np.ravel(np.asarray(e, dtype=float))
        out[int(z)] = float(e[0]) if e.size else 0.0
    return out


def get_files_with_suffix(dir_path: str, suffix: str) -> List[str]:
    return [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(suffix)
    ]


def custom_key(key):
    """
    Helper function to sort the keys of the data loader dictionary
    to ensure that the training set, and validation set
    are evaluated first
    """
    if key == "train":
        return (0, key)
    if key == "valid":
        return (1, key)
    return (2, key)


class LRScheduler:
    def __init__(self, optimizer, args) -> None:
        self.scheduler = args.scheduler
        self._optimizer_type = (
            args.optimizer
        )  # Schedulefree does not need an optimizer but checkpoint handler does.
        if args.scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=args.lr_scheduler_gamma
            )
        elif args.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=args.lr_factor,
                patience=args.scheduler_patience,
            )
        else:
            raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")

    def step(self, metrics=None, epoch=None):  # pylint: disable=E1123
        if self._optimizer_type == "schedulefree":
            return  # In principle, schedulefree optimizer can be used with a scheduler but the paper suggests it's not necessary
        if self.scheduler == "ExponentialLR":
            self.lr_scheduler.step(epoch=epoch)
        elif self.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler.step(  # pylint: disable=E1123
                metrics=metrics, epoch=epoch
            )

    def __getattr__(self, name):
        if name == "step":
            return self.step
        return getattr(self.lr_scheduler, name)


# columns of every error table: (header, metric key, scale, format); a metric that was not computed
# (no labels of that kind in the set, e.g. an unlabelled test set) is shown as "-"
_E_ATOM_RMSE = ("RMSE E / meV / atom", "rmse_e_per_atom", 1000.0, "8.1f")
_E_ATOM_MAE = ("MAE E / meV / atom", "mae_e_per_atom", 1000.0, "8.1f")
_F_RMSE = ("RMSE F / meV / A", "rmse_f", 1000.0, "8.1f")
_F_MAE = ("MAE F / meV / A", "mae_f", 1000.0, "8.1f")
_FMM_RMSE = ("RMSE F_mm / meV / A", "rmse_mm_f", 1000.0, "8.1f")
_FMM_MAE = ("MAE F_mm / meV / A", "mae_mm_f", 1000.0, "8.1f")
ERROR_TABLES = {
    "TotalRMSE": [("RMSE E / meV", "rmse_e", 1000.0, "8.1f"), _F_RMSE, ("relative F RMSE %", "rel_rmse_f", 1.0, "8.2f"),
                  _FMM_RMSE],
    "PerAtomRMSE": [_E_ATOM_RMSE, _F_RMSE, ("relative F RMSE %", "rel_rmse_f", 1.0, "8.2f"), _FMM_RMSE],
    "TotalMAE": [("MAE E / meV", "mae_e", 1000.0, "8.1f"), _F_MAE, ("relative F MAE %", "rel_mae_f", 1.0, "8.2f"),
                 _FMM_MAE],
    "PerAtomMAE": [_E_ATOM_MAE, _F_MAE, ("relative F MAE %", "rel_mae_f", 1.0, "8.2f"), _FMM_MAE],
    "DipoleRMSE": [("RMSE MU / e A", "rmse_mu", 1.0, "8.4f"), ("relative MU RMSE %", "rel_rmse_mu", 1.0, "8.1f")],
    "DipoleMAE": [("MAE MU / e A", "mae_mu", 1.0, "8.4f"), ("relative MU MAE %", "rel_mae_mu", 1.0, "8.1f")],
    "EnergyDipoleRMSE": [_E_ATOM_RMSE, _F_RMSE, ("rel F RMSE %", "rel_rmse_f", 1.0, "8.1f"),
                         ("RMSE MU / e A", "rmse_mu", 1.0, "8.4f"), ("rel MU RMSE %", "rel_rmse_mu", 1.0, "8.1f")],
    "EnergyNacsDipoleMAE": [_E_ATOM_MAE, _F_MAE, ("rel F MAE %", "rel_mae_f", 1.0, "8.1f"), _FMM_MAE,
                            ("MAE NACs", "mae_nacs", 1.0, ".4f"), ("MAE dipoles", "mae_mu", 1.0, ".4f"),
                            ("MAE SOCs", "mae_socs", 1.0, ".4g")],
}


def create_error_table(
    table_type: str,
    all_data_loaders: dict,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
    distributed: bool = False,
) -> PrettyTable:
    if log_wandb:
        import wandb
    columns = ERROR_TABLES[table_type]
    table = PrettyTable()
    table.field_names = ["config_type"] + [c[0] for c in columns]

    for name in sorted(all_data_loaders, key=custom_key):
        data_loader = all_data_loaders[name]
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )
        if distributed:
            torch.distributed.barrier()

        del data_loader
        torch.cuda.empty_cache()
        if log_wandb:
            wandb.log({f"{name}_final_{k}": metrics[k] for k in ("rmse_e_per_atom", "rmse_f", "rel_rmse_f")
                       if k in metrics})
        table.add_row([name] + [format(metrics[key] * scale, f) if key in metrics else "-"
                                for _, key, scale, f in columns])
    return table
