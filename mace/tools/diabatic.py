# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
"""
Diabatic models: from the diabatic Hamiltonian to adiabatic energies, gradients and NACs.

A diabatic FieldMACE model (``--diabatic=True``) with S diabatic states predicts the S + P elements
of the diabatic Hamiltonian H (P = S(S-1)/2 pairs), ordered

    [H_00, H_11, ..., H_S-1,S-1,  H_01, H_02, ..., H_12, ...]    (pairs as mace.modules.state_pairs)

and the derivative of every element with respect to the atoms and the point charges (its "forces",
-dH/dR). Diagonalising H = U diag(E) U^T gives (exactly, by the Hellmann-Feynman theorem)

    adiabatic energies   E_k
    adiabatic gradients  dE_k/dR = (U^T dH/dR U)_kk
    NACs                 d_kl = <k|d/dR|l> = (U^T dH/dR U)_kl / (E_l - E_k)     (SHARC's convention)

The same holds for derivatives with respect to the point-charge positions, so a diabatic model also
gives NACs on the MM atoms. Diabatic quantities are smooth and have no arbitrary signs, which makes
them much easier to learn than adiabatic NACs near crossings.

Training labels for a diabatic model, from an electronic-structure method that gives adiabatic
energies E, gradients g, NACs d and the diabatic-to-adiabatic matrix U: see :func:`labels_from_adiabatic`.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def state_pairs(n: int) -> List[List[int]]:
    return [[i, j] for i in range(n) for j in range(i + 1, n)]


def n_outputs(n_states: int) -> int:
    return n_states + n_states * (n_states - 1) // 2


def n_states_from_outputs(n_out: int) -> int:
    n = int(round((np.sqrt(8 * n_out + 1) - 1) / 2))
    if n_outputs(n) != n_out:
        raise ValueError(f"{n_out} outputs is not S + S(S-1)/2 for any S")
    return n


def unpack(values, n_states: int):
    """[..., S + P, *rest] -> symmetric [..., S, S, *rest] (numpy or torch, element axis = -1 - len(rest))."""
    if isinstance(values, torch.Tensor):
        return _unpack_torch(values, n_states)
    v = np.asarray(values)
    return _unpack_np(v, n_states, axis=0)


def _unpack_np(v: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    v = np.moveaxis(v, axis, 0)
    H = np.zeros((n, n) + v.shape[1:], dtype=v.dtype)
    for i in range(n):
        H[i, i] = v[i]
    for k, (i, j) in enumerate(state_pairs(n)):
        H[i, j] = H[j, i] = v[n + k]
    return H


def _unpack_torch(v: torch.Tensor, n: int) -> torch.Tensor:
    """[B, S + P] -> [B, S, S] (differentiable)."""
    b = v.shape[0]
    H = v.new_zeros((b, n, n))
    idx = torch.arange(n, device=v.device)
    H[:, idx, idx] = v[:, :n]
    if n > 1:
        iu = torch.triu_indices(n, n, 1, device=v.device)
        H[:, iu[0], iu[1]] = v[:, n:]
        H[:, iu[1], iu[0]] = v[:, n:]
    return H


def pack(H: np.ndarray) -> np.ndarray:
    """Symmetric [S, S, *rest] -> [S + P, *rest]."""
    n = H.shape[0]
    return np.stack([H[i, i] for i in range(n)] + [H[i, j] for i, j in state_pairs(n)])


def adiabatic_energies_torch(values: torch.Tensor, n_states: int) -> torch.Tensor:
    """Diabatic outputs [B, S + P] -> adiabatic energies [B, S] (eigvalsh, differentiable)."""
    return torch.linalg.eigvalsh(_unpack_torch(values, n_states))


def align_columns(U: np.ndarray, U_ref: Optional[np.ndarray]) -> np.ndarray:
    """Flip eigenvector signs so that every column overlaps positively with the reference."""
    if U_ref is None or U_ref.shape != U.shape:
        # deterministic default: largest component of every column positive
        k = np.argmax(np.abs(U), axis=0)
        s = np.sign(U[k, np.arange(U.shape[1])])
    else:
        s = np.sign(np.einsum("ik,ik->k", U_ref, U))
    s[s == 0] = 1.0
    return U * s[None, :]


def adiabatize(
    H: np.ndarray,
    dH: Optional[np.ndarray] = None,
    dH_pc: Optional[np.ndarray] = None,
    blocks: Optional[Sequence[Sequence[int]]] = None,
    U_ref: Optional[np.ndarray] = None,
    min_gap: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Diagonalise a diabatic Hamiltonian and transform its derivatives.

    H      [S, S]          diabatic Hamiltonian
    dH     [S, S, N, 3]    dH/dR on the atoms (optional)
    dH_pc  [S, S, M, 3]    dH/dR on the point charges (optional)
    blocks lists of state indices diagonalised separately (e.g. one block per multiplicity);
           couplings between blocks are ignored. Default: one block.
    U_ref  previous eigenvectors for sign continuity (columns are compared by overlap)

    Returns energies [S], U [S, S] (columns = adiabatic states, ordered by energy within each block,
    blocks in the given order), grad [S, N, 3], nac [S, S, N, 3] (d_kl = <k|d/dR|l>), and grad_pc /
    nac_pc if dH_pc is given.
    """
    n = H.shape[0]
    blocks = [list(range(n))] if blocks is None else [list(b) for b in blocks]
    order = [i for b in blocks for i in b]
    if sorted(order) != list(range(n)):
        raise ValueError("blocks must partition the states")
    E = np.zeros(n)
    U = np.zeros((n, n))
    for b in blocks:
        e, u = np.linalg.eigh(H[np.ix_(b, b)])
        E[b] = e
        U[np.ix_(b, b)] = u
    U = align_columns(U, U_ref)
    out = {"energies": E, "U": U}
    gap = E[None, :] - E[:, None]  # E_l - E_k
    same = np.zeros((n, n), dtype=bool)
    for b in blocks:
        same[np.ix_(b, b)] = True
    off = same & ~np.eye(n, dtype=bool)
    inv = np.zeros((n, n))
    g = gap[off]
    inv[off] = 1.0 / np.where(np.abs(g) < min_gap, np.copysign(min_gap, g + (g == 0)), g)
    for key, d in (("", dH), ("_pc", dH_pc)):
        if d is None:
            continue
        dd = np.where(same[:, :, None, None], d, 0.0)
        A = np.einsum("ik,ijax,jl->klax", U, dd, U)  # U^T dH U
        out["grad" + key] = np.einsum("kkax->kax", A).copy()
        out["nac" + key] = A * inv[:, :, None, None]
    return out


def labels_from_adiabatic(
    E: np.ndarray,
    U: np.ndarray,
    grad: Optional[np.ndarray] = None,
    nac: Optional[np.ndarray] = None,
    grad_pc: Optional[np.ndarray] = None,
    nac_pc: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Diabatic Hamiltonian and its derivatives from adiabatic data (inverse of :func:`adiabatize`).

    E [S], U [S, S] (diabatic -> adiabatic, columns = adiabatic states), grad [S, N, 3],
    nac [S, S, N, 3] with d_kl = <k|d/dR|l> = (U^T dH U)_kl / (E_l - E_k).
    Returns H [S, S], dH [S, S, N, 3] (and dH_pc). Exact as long as U, E, grad and nac come from the
    same calculation (the same U that defines the NACs' phases).
    """
    n = len(E)
    out = {"H": U @ np.diag(E) @ U.T}
    gap = E[None, :] - E[:, None]
    for key, g, d in (("", grad, nac), ("_pc", grad_pc, nac_pc)):
        if g is None:
            continue
        A = d * gap[:, :, None, None] if d is not None else np.zeros((n, n) + g.shape[1:])
        idx = np.arange(n)
        A[idx, idx] = g
        out["dH" + key] = np.einsum("ik,klax,jl->ijax", U, A, U)
    return out
