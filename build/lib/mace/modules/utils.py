###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.utils.data
from scipy.constants import c, e
from torch import linalg as LA
from mace.tools import to_numpy
from mace.tools.scatter import scatter_sum
from mace.tools.torch_geometric.batch import Batch
import time
import math
from .blocks import AtomicEnergiesBlock


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, compute_pc_grads: bool, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy[:,0])]
    excited_gradients = []
    for j, state in enumerate(range(energy.shape[-1])):
        if j < energy.shape[-1] - 1 or training or compute_pc_grads:
            retain = True
        else:
            retain = False
        gradient = torch.autograd.grad(
            outputs=[energy[:,state]],  # [n_graphs, ]
            inputs=[positions],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=retain,  # Make sure the graph is not destroyed during training
            create_graph=retain,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[
        0
        ]  # [n_nodes, 3]
        excited_gradients.append(gradient)
    
    excited_gradients = torch.stack(excited_gradients, dim=0)
    
    if excited_gradients is None:
        return torch.zeros_like(positions)

    return -1 * torch.transpose(excited_gradients, 0, 1)

def _calc_multipole_moments(mm_charges, mm_positions, mm_spherical_harmonics, l_max):
    index = 0
    moments = []
    for l in range(0, l_max+1):
        for m in range(-l, l + 1):
            prefac = torch.sqrt(torch.tensor(4 * np.pi / (2 * l + 1)))
            q_lm = (torch.nan_to_num((mm_charges / (LA.norm(mm_positions, dim=-1) ** (l + 1))), nan=0.0)) * torch.conj(mm_spherical_harmonics)[:,:,index]
            q_lm = torch.where(q_lm.abs() > 1e6, torch.tensor(0.0, dtype=q_lm.dtype, device=q_lm.device), q_lm)
            q_lm *= prefac
            moments.append(q_lm.real)
            index += 1
    moments = torch.stack(moments, dim=-1)
    return moments

def compute_multipole_expansion(
    positions_internal : torch.Tensor,
    multipoles: torch.Tensor,
    batch: torch.Tensor,
    lmax: float = 3,
):
    multipoles = torch.sum(multipoles, dim=1)
    R = torch.linalg.vector_norm(positions_internal, dim=-1).unsqueeze(-1)
    all_mp_contribs = []
    for i in range(0, multipoles.shape[0]):
        range_ = torch.where(batch == i)[0]
        R_mol = R[range_[0]:range_[-1]+1]
        mp_contribs = []
        for l in range(lmax + 1):
            q_l = multipoles[i][l**2: (l+1)**2].unsqueeze(0)
            phi_l = np.sqrt(4 * np.pi / (2 * l + 1)) * q_l * R_mol ** l
            mp_contribs.append(phi_l)
        mp_contribs = torch.cat(mp_contribs, dim=-1)
        all_mp_contribs.append(mp_contribs)
    all_mp_contribs = torch.cat(all_mp_contribs, dim=0)
    return all_mp_contribs

def compute_columb_potential(
    qm_positions: torch.Tensor,   # [2520, 3]
    mm_positions: torch.Tensor,   # [40, 3500, 3]
    mm_charges: torch.Tensor      # [40, 3500]
) -> torch.Tensor:
    """
    Computes Coulomb potential per QM point by looping over batches.
    Returns a flattened tensor of shape [2520, 1].
    """
    B = mm_positions.size(0)
    P = qm_positions.size(0) // B

    # reshape QM → [B, P, 3]
    qm = qm_positions.view(B, P, 3)

    # pairwise distances: [B, P, M]
    dists = torch.cdist(qm, mm_positions, p=2)
    
    # Coulomb: q / r  → [B, P, M]   
    pot = mm_charges.unsqueeze(1) / (dists + 1e-12)

    # sum over M → [B, P]
    pot = pot.sum(dim=2)

    # flatten back to [B*P, 1]
    return pot.view(-1, 1)#.detach()       # [2520, 1]

def compute_ewald_sum_loop(
    qm_positions: torch.Tensor,   # [B*P, 3]
    mm_positions: torch.Tensor,   # [B, M, 3]
    mm_charges:   torch.Tensor,   # [B, M]
    alpha:        float = 0.5,    # damping parameter
    r_cut:        float = None,   # real-space cutoff
    k_cut:        int   = None    # reciprocal-space cutoff
) -> torch.Tensor:
    """
    Vectorized Ewald sum over batch.
    Inputs:
      - qm_positions: flattened ([B*P,3]) queries, where P = pts_per_batch
      - mm_positions: ([B, M, 3]) MM sites per batch
      - mm_charges:   ([B, M])    MM charges per batch
    Returns:
      - potentials ([B*P,1])
    """
    B, M, _ = mm_positions.shape
    Nq = qm_positions.shape[0]
    P = Nq // B

    # --- derive box_length and volume ---
    max_r = mm_positions.norm(dim=-1).max()
    L = (2 * max_r).item()
    V = L**3

    # default cutoffs
    if r_cut is None:
        r_cut = L / 2
    if k_cut is None:
        k_cut = int(2 * alpha * L / math.pi) + 1

    # --- build k‐vector list all at once ---
    device, dtype = qm_positions.device, qm_positions.dtype
    n = torch.arange(-k_cut, k_cut+1, device=device)
    nx, ny, nz = torch.meshgrid(n, n, n, indexing='ij')
    idx = torch.stack([nx.reshape(-1), ny.reshape(-1), nz.reshape(-1)], dim=1)
    # remove zero‐vector
    idx = idx[(idx.abs().sum(dim=1) != 0)]
    two_pi_L = 2*math.pi / L
    kstack = (two_pi_L * idx).to(dtype)           # [K,3]
    ksq    = (kstack**2).sum(dim=1)               # [K]
    rec_factor = (4*math.pi/V) * torch.exp(-ksq/(4*alpha**2)) / ksq  # [K]

    # reshape qm back to [B,P,3]
    qm = qm_positions.view(B, P, 3)

    # --- REAL‐SPACE TERM ---
    # delta: [B, P, M, 3]
    delta = qm[:, :, None, :] - mm_positions[:, None, :, :]
    delta = delta - L * torch.round(delta / L)
    dist  = delta.norm(dim=-1)                    # [B,P,M]
    mask  = (dist <= r_cut).float()
    real_k = torch.erfc(alpha * dist) / (dist + 1e-12)
    real_k = real_k * mask
    real_p = (mm_charges[:, None, :] * real_k).sum(dim=2, keepdim=True)  # [B,P,1]

    # --- RECIPROCAL‐SPACE TERM ---
    # mm phases: [B, M, K]
    phase_mm = torch.einsum('bmi,kj->bmk', mm_positions, kstack)
    c_mm = (torch.cos(phase_mm) * mm_charges[:,:,None]).sum(dim=1)  # [B,K]
    s_mm = (torch.sin(phase_mm) * mm_charges[:,:,None]).sum(dim=1)  # [B,K]

    # qm phases: [B, P, K]
    phase_qm = torch.einsum('bpi,kj->bpk', qm, kstack)
    c_q = torch.cos(phase_qm)
    s_q = torch.sin(phase_qm)

    # combine and sum
    rec_term = c_q * c_mm[:,None,:] + s_q * s_mm[:,None,:]  # [B,P,K]
    rec_p    = (rec_term * rec_factor[None,None,:]).sum(dim=2, keepdim=True)  # [B,P,1]

    # --- SELF‐TERM ---
    self_scalar = - (alpha / math.sqrt(math.pi)) * mm_charges.sum(dim=1)  # [B]
    self_p = self_scalar.view(B,1,1).expand(-1, P, -1)                   # [B,P,1]

    # total potential
    total = real_p + rec_p + self_p  # [B,P,1]
    return total.view(-1,1)#.detach()          # [B*P,1] 


def compute_multipole_expansion_attention(
    positions_internal : torch.Tensor,
    multipoles: torch.Tensor,
    batch: torch.Tensor,
    lmax: float = 3,
    rep_size: float = 16,
):
    R = torch.linalg.vector_norm(positions_internal, dim=-1).unsqueeze(-1)
    all_mp_contribs = []
    for i in range(0, multipoles.shape[0]):
        range_ = torch.where(batch == i)[0]
        R_mol = R[range_[0]:range_[-1]+1]
        mp_contribs = []
        for l in range(lmax + 1):
            q_l = multipoles[i][rep_size*(l**2): rep_size*((l+1)**2)].unsqueeze(0)
            phi_l = np.sqrt(4 * np.pi / (2 * l + 1)) * q_l * R_mol ** l
            mp_contribs.append(phi_l)
        mp_contribs = torch.cat(mp_contribs, dim=-1)
        all_mp_contribs.append(mp_contribs)
    all_mp_contribs = torch.cat(all_mp_contribs, dim=0)
    return all_mp_contribs

def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:

    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor],
    cell: torch.Tensor,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_hessian: bool = False,
    compute_pc_grads: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                compute_pc_grads=compute_pc_grads,
                training=(training or compute_hessian),
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None
    return forces, virials, stress, hessian


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def _check_non_zero(std):
    if std == 0.0:
        logging.warning(
            "Standard deviation of the scaling is zero, Changing to no scaling"
        )
        std = 1.0
    return std


def extract_invariant(x: torch.Tensor, num_layers: int, num_features: int, l_max: int):
    out = []
    for i in range(num_layers - 1):
        out.append(
            x[
                :,
                i
                * (l_max + 1) ** 2
                * num_features : (i * (l_max + 1) ** 2 + 1)
                * num_features,
            ]
        )
    out.append(x[:, -num_features:])
    return torch.cat(out, dim=-1)


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()
    std = _check_non_zero(std)

    return mean, std


def _compute_mean_std_atomic_inter_energy(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
    )
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes
    return atom_energies


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:

    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        ).unsqueeze(-1)
        graph_sizes = (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()
    rms = _check_non_zero(rms)

    return mean, rms


def _compute_mean_rms_energy_forces(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
    )
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes  # {[n_graphs], }
    forces = batch.forces  # {[n_graphs*n_atoms,3], }

    return atom_energies, forces


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def compute_statistics(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float, float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )

    return to_numpy(avg_num_neighbors).item(), mean, rms


def compute_rms_dipoles(
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    dipoles_list = []
    for batch in data_loader:
        dipoles_list.append(batch.dipole)  # {[n_graphs,3], }

    dipoles = torch.cat(dipoles_list, dim=0)  # {[total_n_graphs,3], }
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(dipoles)))).item()
    rms = _check_non_zero(rms)
    return rms


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) / (1e-11 / c / e)  # [N_atoms,3]
    return scatter_sum(
        src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )  # [N_graphs,3]
