###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.o3 import Irreps

from mace.data import AtomicData
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum
import time

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
    ElectrostaticFieldInteractionBlock,
    MultipoleRadialEmbeddingBlock,
    RotInvariantAttention
)
from .utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    compute_multipole_expansion,
    compute_multipole_expansion_attention,
    _calc_multipole_moments,
)

# pylint: disable=C0302

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                    )
                )
            else:
                self.readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        num_graphs = data["ptr"].numel() - 1
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]  # [n_nodes, len(heads)]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)
        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress, hessian = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)
        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(
                readout(node_feats, node_heads)[num_atoms_arange, node_heads]
            )  # {[n_nodes, ], }

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es, node_heads)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        forces, virials, stress, hessian = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )
        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

        return output


@compile_mode("script")
class ExcitedMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        n_energies: int,
        max_ell: int,
        compute_nacs: bool,
        compute_dipoles: bool,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        self.n_energies = n_energies
        self.n_nacs = int(n_energies*(n_energies-1)/2)
        self.n_dipoles = int(n_energies + int(n_energies*(n_energies-1)/2))

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        self.compute_dipoles = compute_dipoles
        self.compute_nacs = compute_nacs

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        if compute_dipoles:
            self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, n_energies, compute_nacs))
        else:
            self.readouts.append(LinearReadoutBlock(hidden_irreps, n_energies, compute_nacs))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                if compute_dipoles:
                    self.readouts.append(NonLinearDipoleReadoutBlock(hidden_irreps_out, MLP_irreps, gate, n_energies, compute_nacs))
                else:
                    self.readouts.append(NonLinearReadoutBlock(hidden_irreps, MLP_irreps, gate, n_energies, compute_nacs))
            else:
                if compute_dipoles:
                    self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, n_energies, compute_nacs))
                else:
                    self.readouts.append(LinearReadoutBlock(hidden_irreps, n_energies, compute_nacs))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_hessian: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )

        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        pair_node_energy = torch.zeros_like(node_e0)
        pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0.unsqueeze(-1).expand(-1, self.n_energies), pair_energy.unsqueeze(-1).expand(-1, self.n_energies)]
        node_energies_list = [node_e0.unsqueeze(-1).expand(-1, self.n_energies), pair_node_energy.unsqueeze(-1).expand(-1, self.n_energies)]
        node_feats_list = []
        node_nacs_list = []
        node_dipoles_list = []

        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_output = readout(node_feats).squeeze(-1)
            node_energies = torch.transpose(node_output[:, :self.n_energies], 0, 1)
            if self.compute_nacs and self.compute_dipoles:
                node_nacs = node_output[:, self.n_energies:self.n_energies + 3*self.n_nacs]
                node_dipoles = node_output[:, self.n_energies + 3*self.n_nacs:]
            elif self.compute_nacs:
                node_nacs = node_output[:, self.n_energies:self.n_energies + 3*self.n_nacs]
                node_dipoles = None
            elif self.compute_dipoles:
                node_dipoles = node_output[:, self.n_energies:]
                node_nacs = None

            node_nacs_list.append(node_nacs.reshape(node_nacs.shape[0], self.n_energies, 3))
            node_dipoles_list.append(node_dipoles)
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(torch.transpose(energy, 0, 1))
            node_energies_list.append(torch.transpose(node_energies, 0, 1))

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=1)
        total_energy = torch.sum(contributions, dim=1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=1)
        node_energy = torch.sum(node_energy_contributions, dim=1)  # [n_nodes, ]

        dipole_contributions = torch.stack(node_dipoles_list, dim=1)
        total_dipoles = torch.sum(dipole_contributions, dim=1)  # [n_graphs, ]

        nacs_contributions = torch.stack(node_nacs_list, dim=1)
        total_nacs = torch.sum(nacs_contributions, dim=1)  # [n_graphs, ]

        # Outputs
        forces, virials, stress, hessian = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "nacs": total_nacs,
            "dipoles": total_dipoles,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }

class FieldEMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        n_energies: int,
        max_ell: int,
        multipole_max_ell: int,
        compute_nacs: bool,
        compute_dipoles: bool,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        field_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        lmax: int = 3,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        self.lmax = lmax
        self.n_energies = n_energies
        self.n_nacs = int(n_energies*(n_energies-1)/2)
        self.n_dipoles = int(n_energies + int(n_energies*(n_energies-1)/2))

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        self.multipole_max_ell = multipole_max_ell
        self.multipole_radial_embedding = MultipoleRadialEmbeddingBlock()  
        multipole_sh_irreps = o3.Irreps.spherical_harmonics(multipole_max_ell)
        self.multipole_spherical_harmonics = o3.SphericalHarmonics(
            multipole_sh_irreps, normalize=True, normalization="component"
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )

        electrofield_inter = ElectrostaticFieldInteractionBlock(
                node_feats_irreps=node_feats_irreps,
                target_irreps=interaction_irreps,
                field_irreps=field_irreps,
        )
        
        self.linear_up = o3.Linear('0e + 1o + 2e + 3o', field_irreps)
        self.interactions = torch.nn.ModuleList([inter])
        self.field_interactions = torch.nn.ModuleList([electrofield_inter])

        self.compute_dipoles = compute_dipoles
        self.compute_nacs = compute_nacs

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )

        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        if compute_dipoles:
            self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, n_energies, compute_nacs))
        else:
            self.readouts.append(LinearReadoutBlock(hidden_irreps, n_energies, compute_nacs))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps

            electrofield_inter = ElectrostaticFieldInteractionBlock(
                node_feats_irreps=hidden_irreps,
                target_irreps=interaction_irreps,
                field_irreps=field_irreps,
            )

            self.field_interactions.append(electrofield_inter)

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )

            self.products.append(prod)
            if i == num_interactions - 2:
                if compute_dipoles:
                    self.readouts.append(NonLinearDipoleReadoutBlock(hidden_irreps_out, MLP_irreps, gate, n_energies, compute_nacs))
                else:
                    self.readouts.append(NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate, n_energies, compute_nacs))
            else:
                if compute_dipoles:
                    self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, n_energies, compute_nacs))
                else:
                    self.readouts.append(LinearReadoutBlock(hidden_irreps, n_energies, compute_nacs))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_hessian: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        mm_spherical_harmonics = self.multipole_spherical_harmonics(data["mm_positions"])
        multipole_moments = _calc_multipole_moments(data["mm_charges"], data["mm_positions"], mm_spherical_harmonics, self.multipole_max_ell)
        multipoles = compute_multipole_expansion(positions_internal=data["positions"], multipoles=multipole_moments, batch=data["batch"], lmax=self.lmax)
        multipole_attrs = self.multipole_spherical_harmonics(data["positions"])
        multipole_feats = self.multipole_radial_embedding(multipoles).squeeze()
        multipole_feats = self.linear_up(multipole_feats)
        pair_node_energy = torch.zeros_like(node_e0)
        pair_energy = torch.zeros_like(e0)
        energies = [e0.unsqueeze(-1).expand(-1, self.n_energies), pair_energy.unsqueeze(-1).expand(-1, self.n_energies)]
        node_energies_list = [node_e0.unsqueeze(-1).expand(-1, self.n_energies), pair_node_energy.unsqueeze(-1).expand(-1, self.n_energies)]
        node_feats_list = []
        node_nacs_list = []
        node_dipoles_list = []

        for interaction, product, readout, field_interaction in zip(
            self.interactions, self.products, self.readouts, self.field_interactions,
        ):
            intermed_node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            
            field_feats = field_interaction(
                node_feats=node_feats,
                multipole_feats=multipole_feats,
            )

            node_feats = intermed_node_feats + field_feats

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            node_feats_list.append(node_feats)
            if self.n_energies == 1:
                node_output = readout(node_feats)
            else:
                node_output = readout(node_feats).squeeze(-1)

            node_energies = torch.transpose(node_output[:, :self.n_energies], 0, 1)
            if self.compute_nacs and self.compute_dipoles:
                node_nacs = node_output[:, self.n_energies:self.n_energies + 3*self.n_nacs]
                node_dipoles = node_output[:, self.n_energies + 3*self.n_nacs:]
                node_nacs_list.append(node_nacs.reshape(node_nacs.shape[0], self.n_energies, 3))
                node_dipoles_list.append(node_dipoles)
            elif self.compute_nacs:
                node_nacs = node_output[:, self.n_energies:self.n_energies + 3*self.n_nacs]
                node_nacs_list.append(node_nacs.reshape(node_nacs.shape[0], self.n_energies, 3))
            elif self.compute_dipoles:
                node_dipoles = node_output[:, self.n_energies:]
                node_dipoles_list.append(node_dipoles)
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(torch.transpose(energy, 0, 1))
            node_energies_list.append(torch.transpose(node_energies, 0, 1))

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=1)
        total_energy = torch.sum(contributions, dim=1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=1)
        node_energy = torch.sum(node_energy_contributions, dim=1)  # [n_nodes, ]

        total_dipoles = None
        total_nacs = None

        # Outputs
        forces, virials, stress, hessian = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "nacs": total_nacs,
            "dipoles": total_dipoles,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }

@compile_mode("script")
class AttentionFieldEMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        n_energies: int,
        max_ell: int,
        multipole_max_ell: int,
        compute_nacs: bool,
        compute_dipoles: bool,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        field_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        lmax: int = 3,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        self.lmax = lmax
        self.n_energies = n_energies
        self.n_nacs = int(n_energies*(n_energies-1)/2)
        self.n_dipoles = int(n_energies + int(n_energies*(n_energies-1)/2))

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        self.multipole_max_ell = multipole_max_ell
        self.multipole_radial_embedding = MultipoleRadialEmbeddingBlock()
        multipole_sh_irreps = o3.Irreps.spherical_harmonics(multipole_max_ell)
        self.multipole_spherical_harmonics = o3.SphericalHarmonics(
            multipole_sh_irreps, normalize=True, normalization="component"
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )

        electrofield_inter = ElectrostaticFieldInteractionBlock(
                node_feats_irreps=node_feats_irreps,
                target_irreps=interaction_irreps,
                field_irreps=field_irreps,
        )

        self.interactions = torch.nn.ModuleList([inter])
        self.field_interactions = torch.nn.ModuleList([electrofield_inter])

        self.compute_dipoles = compute_dipoles
        self.compute_nacs = compute_nacs

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )

        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()

        self.multipole_attention = torch.nn.ModuleList([RotInvariantAttention(irreps_in = '0e + 1o + 2e + 3o', irreps_hidden = field_irreps, irreps_out = field_irreps, 
            node_feats_irreps=node_feats_irreps)])

        if compute_dipoles:
            self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, n_energies, compute_nacs))
        else:
            self.readouts.append(LinearReadoutBlock(hidden_irreps, n_energies, compute_nacs))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps

            electrofield_inter = ElectrostaticFieldInteractionBlock(
                node_feats_irreps=hidden_irreps,
                target_irreps=interaction_irreps,
                field_irreps=field_irreps,
            )

            self.field_interactions.append(electrofield_inter)

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )

            multipole_attention = RotInvariantAttention(irreps_in = '0e + 1o + 2e + 3o', irreps_hidden = field_irreps, irreps_out = field_irreps, 
                node_feats_irreps=hidden_irreps)

            self.products.append(prod)

            self.multipole_attention.append(multipole_attention)

            if i == num_interactions - 2:
                if compute_dipoles:
                    self.readouts.append(NonLinearDipoleReadoutBlock(hidden_irreps_out, MLP_irreps, gate, n_energies, compute_nacs))
                else:
                    self.readouts.append(NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate, n_energies, compute_nacs))
            else:
                if compute_dipoles:
                    self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, n_energies, compute_nacs))
                else:
                    self.readouts.append(LinearReadoutBlock(hidden_irreps, n_energies, compute_nacs))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_hessian: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        mm_spherical_harmonics = self.multipole_spherical_harmonics(data["mm_positions"])
        multipole_moments = _calc_multipole_moments(data["mm_charges"], data["mm_positions"], mm_spherical_harmonics, self.multipole_max_ell)
        print(multipole_moments.shape)
        print(node_feats.shape)
        
        pair_node_energy = torch.zeros_like(node_e0)
        pair_energy = torch.zeros_like(e0)
        # Interactions
        energies = [e0.unsqueeze(-1).expand(-1, self.n_energies), pair_energy.unsqueeze(-1).expand(-1, self.n_energies)]
        node_energies_list = [node_e0.unsqueeze(-1).expand(-1, self.n_energies), pair_node_energy.unsqueeze(-1).expand(-1, self.n_energies)]
        node_feats_list = []
        node_nacs_list = []
        node_dipoles_list = []
        
        for interaction, product, readout, field_interaction, multipole_attention in zip(
            self.interactions, self.products, self.readouts, self.field_interactions, self.multipole_attention,
        ):
            intermed_node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            print(multipole_moments.shape)
            print(node_feats.shape)
            attention_aggregated_moments = multipole_attention(multipole_moments, node_feats)
            multipoles = compute_multipole_expansion_attention(positions_internal=data["positions"], multipoles=attention_aggregated_moments, batch=data["batch"], lmax=self.lmax)
            multipole_feats = self.multipole_radial_embedding(multipoles).squeeze()

            field_feats = field_interaction(
                node_feats=node_feats,
                multipole_feats=multipole_feats,
            )

            node_feats = intermed_node_feats + field_feats

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            node_feats_list.append(node_feats)
            if self.n_energies == 1:
                node_output = readout(node_feats)
            else:
                node_output = readout(node_feats).squeeze(-1)

            node_energies = torch.transpose(node_output[:, :self.n_energies], 0, 1)
            if self.compute_nacs and self.compute_dipoles:
                node_nacs = node_output[:, self.n_energies:self.n_energies + 3*self.n_nacs]
                node_dipoles = node_output[:, self.n_energies + 3*self.n_nacs:]
                node_nacs_list.append(node_nacs.reshape(node_nacs.shape[0], self.n_energies, 3))
                node_dipoles_list.append(node_dipoles)
            elif self.compute_nacs:
                node_nacs = node_output[:, self.n_energies:self.n_energies + 3*self.n_nacs]
                node_nacs_list.append(node_nacs.reshape(node_nacs.shape[0], self.n_energies, 3))
            elif self.compute_dipoles:
                node_dipoles = node_output[:, self.n_energies:]
                node_dipoles_list.append(node_dipoles)
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(torch.transpose(energy, 0, 1))
            node_energies_list.append(torch.transpose(node_energies, 0, 1))

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=1)
        total_energy = torch.sum(contributions, dim=1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=1)
        node_energy = torch.sum(node_energy_contributions, dim=1)  # [n_nodes, ]

        total_dipoles = None
        total_nacs = None

        # Outputs
        forces, virials, stress, hessian = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "nacs": total_nacs,
            "dipoles": total_dipoles,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }
