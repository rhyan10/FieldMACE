# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Small helpers for tests and benchmarks: randomly initialised field models and inputs."""
from typing import List, Optional, Sequence

import numpy as np
import torch
from e3nn import o3

TEST_ZS = [1, 6, 7, 8]
FIELD_MODELS = ["PerAtomFieldEMACE", "FieldEMACE"]


def build_field_model(
    cls: str = "PerAtomFieldEMACE",
    seed: int = 0,
    n_energies: int = 1,
    channels: int = 16,
    multipole_max_ell: int = 1,
    num_interactions: int = 2,
    atomic_numbers: Sequence[int] = TEST_ZS,
    **kwargs,
) -> torch.nn.Module:
    """A randomly initialised field model (fine for testing the plumbing, not for physics)."""
    from mace import modules

    torch.manual_seed(seed)
    zs = list(atomic_numbers)
    defaults = dict(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=num_interactions,
        num_elements=len(zs),
        hidden_irreps=o3.Irreps(f"{channels}x0e + {channels}x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        field_irreps=o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o"),
        atomic_energies=np.linspace(-1.0, -4.0, len(zs)),
        avg_num_neighbors=3.0,
        atomic_numbers=zs,
        pair_repulsion=False,
        n_energies=n_energies,
        distance_transform="None",
        multipole_max_ell=multipole_max_ell,
        correlation=3,
        gate=torch.nn.functional.silu,
        radial_MLP=[32, 32],
        radial_type="bessel",
        compute_nacs=False,
        compute_dipoles=False,
    )
    defaults.update(kwargs)
    return getattr(modules, cls)(**defaults)


def methane_in_charges(n_mm: int = 30, seed: int = 0, shift=(0.0, 0.0, 0.0), min_dist: float = 2.5):
    """CH4 plus n_mm random point charges at least min_dist from the carbon."""
    rng = np.random.default_rng(seed)
    qm = np.array(
        [[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]], float
    )
    mm = []
    while len(mm) < n_mm:
        p = rng.uniform(-8, 8, 3)
        if np.linalg.norm(p) > min_dist:
            mm.append(p)
    mm = np.array(mm, dtype=float).reshape(-1, 3)
    q = rng.uniform(-0.8, 0.8, n_mm)
    s = np.asarray(shift, dtype=float)
    return "CHHHH", qm + s, mm + s, q


def make_batch(model: torch.nn.Module, configs: List, cutoff: Optional[float] = None, labels: bool = False):
    """configs: list of (symbols, qm_positions, mm_positions, mm_charges) -> Batch."""
    import ase

    from mace import data
    from mace.tools import torch_geometric, utils

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    ds = []
    for sym, qm, mm, q in configs:
        atoms = ase.Atoms(sym, positions=qm)
        if mm is not None:
            atoms.info["mm_positions"] = np.asarray(mm, dtype=float).reshape(-1, 3)
            atoms.info["mm_charges"] = np.asarray(q, dtype=float).reshape(-1)
        cfg = data.config_from_atoms(atoms)
        ds.append(data.AtomicData.from_config(cfg, z_table=z_table, cutoff=cutoff or float(model.r_max)))
    loader = torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds), shuffle=False, drop_last=False)
    return next(iter(loader))
