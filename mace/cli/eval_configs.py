###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device, weights_only=False)
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad = False
    is_field = hasattr(model, "multipole_max_ell")

    # Load data and prepare input (point charges from atoms.info["mm_positions"/"mm_charges"])
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    energies, forces, mm_forces, stresses, contributions = [], [], [], [], []
    for batch in data_loader:
        batch = batch.to(device)
        kwargs = {"compute_pc_grads": True} if is_field else {"compute_stress": args.compute_stress}
        output = model(batch.to_dict(), training=False, **kwargs)
        e = torch_tools.to_numpy(output["energy"])
        energies.extend(list(e.reshape(e.shape[0], -1)))
        n_states = energies[-1].shape[0]
        f = torch_tools.to_numpy(output["forces"]).reshape(-1, n_states, 3)
        forces.extend(np.split(f, torch_tools.to_numpy(batch.ptr[1:-1])))
        if is_field and output.get("mm_forces") is not None:
            fm = torch_tools.to_numpy(output["mm_forces"]).reshape(-1, n_states, 3)
            bounds = np.cumsum(torch_tools.to_numpy(batch["num_mm"]))[:-1]
            mm_forces.extend(np.split(fm, bounds))
        if not is_field and args.compute_stress:
            stresses.extend(list(torch_tools.to_numpy(output["stress"])))
        if args.return_contributions:
            contributions.extend(list(torch_tools.to_numpy(output["contributions"])))

    assert len(atoms_list) == len(energies) == len(forces)
    p = args.info_prefix
    for i, atoms in enumerate(atoms_list):
        atoms.calc = None  # crucial
        e, f = energies[i], forces[i]
        if len(e) == 1:
            atoms.info[p + "energy"] = float(e[0])
            atoms.arrays[p + "forces"] = f[:, 0, :]
        else:
            atoms.info[p + "energies"] = e  # all states
            atoms.info[p + "energy"] = float(e[0])
            atoms.arrays[p + "forces"] = f[:, 0, :]
            atoms.info[p + "forces_states"] = f  # [n_atoms, n_states, 3]
        if mm_forces:
            atoms.info[p + "mm_forces"] = mm_forces[i] if len(e) > 1 else mm_forces[i][:, 0, :]
        if stresses:
            atoms.info[p + "stress"] = stresses[i]
        if contributions:
            atoms.info[p + "BO_contributions"] = contributions[i]

    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
