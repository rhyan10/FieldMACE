###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# Modified for FieldMACE by Rhyan Barrett
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional, Sequence

import numpy as np

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
    stress: torch.Tensor
    virials: torch.Tensor
    nacs: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        mm_charges: torch.Tensor,  # [M]
        mm_positions: torch.Tensor,  # [M, 3]
        mm_forces: torch.Tensor,  # [M, n_states, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        dipoles_weight: Optional[torch.tensor],
        nacs_weight: Optional[torch.tensor],
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipoles: Optional[torch.Tensor],  # [, 3]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
        nacs: Optional[torch.Tensor],
        num_mm: Optional[torch.Tensor] = None,  # [] number of MM charges
        mm_forces_weight: Optional[torch.Tensor] = None,  # []
        socs: Optional[torch.Tensor] = None,  # [1, n_soc_values] (2 per state pair by default)
        socs_weight: Optional[torch.Tensor] = None,  # []
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
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert dipoles_weight is None or len(dipoles_weight.shape) == 0
        assert nacs_weight is None or len(nacs_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape[-1] == 3
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipoles is None or dipoles.shape[-1] == 3
        assert nacs is None or nacs.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        assert mm_positions.dim() == 2 and mm_positions.shape[1] == 3
        assert mm_charges.shape == (mm_positions.shape[0],)
        assert mm_forces.dim() == 3 and mm_forces.shape[0] == mm_positions.shape[0]
        if num_mm is None:
            num_mm = torch.tensor(mm_positions.shape[0], dtype=torch.long)
        if mm_forces_weight is None:
            mm_forces_weight = torch.tensor(1.0, dtype=torch.get_default_dtype())
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "mm_charges": mm_charges,
            "mm_positions": mm_positions,  
            "mm_forces": mm_forces,
            "num_mm": num_mm,
            "mm_forces_weight": mm_forces_weight,
            "socs": socs,
            "socs_weight": socs_weight,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "weight": weight,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "dipoles_weight": dipoles_weight,
            "nacs_weight": nacs_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipoles": dipoles,
            "nacs": nacs,
            "charges": charges,
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

        stress_weight = (
            torch.tensor(config.stress_weight, dtype=torch.get_default_dtype())
            if config.stress_weight is not None
            else 1
        )

        virials_weight = (
            torch.tensor(config.virials_weight, dtype=torch.get_default_dtype())
            if config.virials_weight is not None
            else 1
        )

        nacs_weight = (
            torch.tensor(config.nacs_weight, dtype=torch.get_default_dtype())
            if config.nacs_weight is not None
            else 1
        )

        dipoles_weight = (
            torch.tensor(config.dipoles_weight, dtype=torch.get_default_dtype())
            if config.dipoles_weight is not None
            else 1
        )

        dt = torch.get_default_dtype()
        forces = torch.tensor(np.asarray(config.forces), dtype=dt)
        energy = torch.tensor(np.asarray(config.energy), dtype=dt).reshape(1, -1)  # [1, n_states]
        n_states = energy.shape[1]
        forces = forces.reshape(forces.shape[0], -1, 3)  # [n_atoms, n_states, 3]
        stress = (
            voigt_to_matrix(
                torch.tensor(config.stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.stress is not None
            else None
        )
        virials = (
            voigt_to_matrix(
                torch.tensor(config.virials, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.virials is not None
            else None
        )
        n_pairs = n_states * (n_states - 1) // 2
        dipoles = (
            torch.tensor(np.asarray(config.dipoles), dtype=dt).reshape(1, -1, 3)  # [1, S+P, 3]
            if config.dipoles is not None
            else torch.zeros(1, n_states + n_pairs, 3, dtype=dt)
        )
        nacs = (
            torch.tensor(np.asarray(config.nacs), dtype=dt).reshape(forces.shape[0], -1, 3)  # [N, P, 3]
            if config.nacs is not None and np.asarray(config.nacs).size > 0
            else torch.zeros(forces.shape[0], n_pairs, 3, dtype=dt)
        )
        socs_arr = getattr(config, "socs", None)
        socs = (
            torch.tensor(np.asarray(socs_arr), dtype=dt).reshape(1, -1)  # [1, n_soc_values]
            if socs_arr is not None and np.asarray(socs_arr).size > 0
            else torch.zeros(1, 2 * n_pairs, dtype=dt)
        )
        socs_weight = torch.tensor(
            float(getattr(config, "socs_weight", 1.0)) if socs_arr is not None else 0.0, dtype=dt
        )
        charges = (
            torch.tensor(config.charges, dtype=torch.get_default_dtype())
            if config.charges is not None
            else None
        )
        mm_q = np.zeros(0) if config.mm_charges is None else np.asarray(config.mm_charges, dtype=float)
        mm_r = np.zeros((0, 3)) if config.mm_positions is None else np.asarray(config.mm_positions, dtype=float)
        mm_q = mm_q.reshape(-1)
        mm_r = mm_r.reshape(-1, 3)
        if config.mm_forces is None:
            mm_f = np.zeros((len(mm_q), n_states, 3))
        else:
            mm_f = np.asarray(config.mm_forces, dtype=float)
            mm_f = mm_f.reshape(len(mm_q), n_states, 3) if mm_f.size == 0 else mm_f.reshape(len(mm_q), -1, 3)
        mm_charges = torch.tensor(mm_q, dtype=dt)
        mm_positions = torch.tensor(mm_r, dtype=dt)
        mm_forces = torch.tensor(mm_f, dtype=dt)
        mm_forces_weight = torch.tensor(
            float(getattr(config, "mm_forces_weight", 1.0)) if config.mm_forces is not None else 0.0,
            dtype=dt,
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
            stress_weight=stress_weight,
            virials_weight=virials_weight,
            dipoles_weight=dipoles_weight,
            nacs_weight=nacs_weight,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipoles=dipoles,
            nacs=nacs,
            charges=charges,
            mm_charges=mm_charges,
            mm_positions=mm_positions,
            mm_forces=mm_forces,
            num_mm=torch.tensor(len(mm_q), dtype=torch.long),
            mm_forces_weight=mm_forces_weight,
            socs=socs,
            socs_weight=socs_weight,
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
