###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional, Sequence

import torch.utils.data

from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

from .neighborhood import get_neighborhood
from .utils import Configuration
import numpy as np

class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    multipole_moments: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    vectors: torch.Tensor
    scalars: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    vectors_weight: torch.Tensor
    scalars_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        mm_charges: torch.Tensor,
        mm_positions: torch.Tensor,
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        vectors_weight: Optional[torch.Tensor],  # [,]
        scalars_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        vectors: Optional[torch.Tensor],  # [1,3,3]
        scalars: Optional[torch.Tensor],  # [1,3,3]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert weight is None or len(weight.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape[-1] == 3
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "mm_charges": mm_charges,
            "mm_positions": mm_positions,           
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "weight": weight,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "vectors_weight": vectors_weight,
            "scalars_weight": scalars_weight,
            "forces": forces,
            "energy": energy,
            "vectors": vectors,
            "scalars": scalars,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls, config: Configuration, z_table: AtomicNumberTable, cutoff: float
    ) -> "AtomicData":
        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
        )
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )

        # mm_edge_index, mm_shifts, mm_unit_shifts = get_neighborhood(
        #     positions=np.squeeze(config.mm_positions), cutoff=cutoff, pbc=config.pbc, cell=config.cell
        # )

        cell = (
            torch.tensor(config.cell, dtype=torch.get_default_dtype())
            if config.cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else 1
        )

        energy_weight = (
            torch.tensor(config.energy_weight, dtype=torch.get_default_dtype())
            if config.energy_weight is not None
            else 1
        )

        forces_weight = (
            torch.tensor(config.forces_weight, dtype=torch.get_default_dtype())
            if config.forces_weight is not None
            else 1
        )

        vectors_weight = (
            torch.tensor(config.vectors_weight, dtype=torch.get_default_dtype())
            if config.vectors_weight is not None
            else 1
        )

        scalars_weight = (
            torch.tensor(config.scalars_weight, dtype=torch.get_default_dtype())
            if config.scalars_weight is not None
            else 1
        )

        forces = (
            torch.tensor(config.forces, dtype=torch.get_default_dtype())
            if config.forces is not None
            else None
        )
        energy = (
            torch.tensor(config.energy, dtype=torch.get_default_dtype())
            if config.energy is not None
            else None
        )
        scalars = (
            torch.tensor(config.scalars, dtype=torch.get_default_dtype())
            if config.scalars is not None
            else None
        )
        vectors = (
            torch.tensor(config.vectors, dtype=torch.get_default_dtype())
            if config.vectors is not None
            else None
        )
        mm_charges = (
            torch.tensor(config.mm_charges, dtype=torch.get_default_dtype())
            if config.mm_charges[0] is not None
            else None
        )
        mm_positions = (
            torch.tensor(config.mm_positions, dtype=torch.get_default_dtype())
            if config.mm_positions[0] is not None
            else None
        )
        

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            node_attrs=one_hot,
            weight=weight,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            vectors_weight=vectors_weight,
            scalars_weight=scalars_weight,
            forces=forces,
            energy=energy,
            vectors=vectors,
            scalars=scalars,
            mm_charges=mm_charges,
            mm_positions=mm_positions,
        )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
