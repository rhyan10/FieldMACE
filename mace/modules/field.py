###########################################################################################
# FieldMACE: MACE for QM/MM - QM region embedded in the field of MM point charges
#
# Version 2 of the field models. Compared with the original implementation:
#   * MM point charges use a flat layout ("mm_positions" [sum M, 3], "mm_charges" [sum M],
#     "num_mm" [n_graphs]) so every configuration can have a different number of charges
#     (no padding). The old padded layout [n_graphs, M, 3] is still accepted.
#   * The multipole maths is fully vectorised (no Python loops over graphs or (l, m)).
#   * The multipole features are embedded equivariantly. The original embedding applied a
#     scalar non-linearity to each (l, m) component separately, which breaks rotational
#     invariance for l >= 1 (energies changed by up to ~0.1 eV under rigid rotation).
#   * FieldEMACE expands the MM potential about the QM centroid of each configuration, so
#     the model is translation invariant (originally the expansion centre was (0, 0, 0)).
#   * Forces on QM atoms and on MM charges come from one backward pass per state.
#   * The forward pass is TorchScript compatible.
#
# Models pickled with the original code still load and give the original numbers: they are
# detected in __setstate__ and run in "legacy" mode.
###########################################################################################

import math
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum

from .blocks import (
    AtomicEnergiesBlock,
    ElectrostaticFieldInteractionBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    MultipoleRadialEmbeddingBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RotInvariantAttention,
)
from .radial import ZBLBasis
from .utils import compute_hessians_vmap, get_edge_vectors_and_lengths

FIELD_MODEL_VERSION = 2


# --------------------------------------------------------------------------------------- #
# multipole maths
# --------------------------------------------------------------------------------------- #
def _l_index(lmax: int) -> List[int]:
    return [l for l in range(lmax + 1) for _ in range(2 * l + 1)]


def flat_mm_layout(data: Dict[str, torch.Tensor], num_graphs: int):
    """Return (mm_positions [M,3], mm_charges [M], num_mm [B], mm_batch [M], padded).

    Accepts the flat layout ("num_mm" present) and the original padded layout
    (mm_positions [B, M, 3], mm_charges [B, M]).
    """
    mm_pos = data["mm_positions"]
    mm_q = data["mm_charges"]
    device = mm_pos.device
    if mm_pos.dim() == 3:
        b, m = mm_pos.shape[0], mm_pos.shape[1]
        num_mm = torch.full((b,), m, dtype=torch.long, device=device)
        mm_pos = mm_pos.reshape(b * m, 3)
        mm_q = mm_q.reshape(b * m)
        padded = True
    else:
        if "num_mm" in data:
            num_mm = data["num_mm"].reshape(-1).to(torch.long)
        else:  # single configuration
            num_mm = torch.full((num_graphs,), mm_pos.shape[0], dtype=torch.long, device=device)
        padded = False
    mm_batch = torch.repeat_interleave(torch.arange(num_mm.shape[0], device=device), num_mm)
    return mm_pos, mm_q, num_mm, mm_batch, padded


def qm_mm_pairs(ptr: torch.Tensor, num_mm: torch.Tensor):
    """All (QM atom, MM charge) pairs inside each configuration.

    returns (qm_index [P], mm_index [P], pair_graph [P]) with P = sum_g n_qm_g * n_mm_g
    """
    device = ptr.device
    n_qm = ptr[1:] - ptr[:-1]
    n_pairs = n_qm * num_mm
    graphs = torch.arange(n_qm.shape[0], device=device)
    pair_graph = torch.repeat_interleave(graphs, n_pairs)
    pair_start = torch.cumsum(n_pairs, 0) - n_pairs
    mm_start = torch.cumsum(num_mm, 0) - num_mm
    k = torch.arange(pair_graph.shape[0], device=device) - pair_start[pair_graph]
    nm = num_mm[pair_graph]
    qm_index = ptr[:-1][pair_graph] + torch.div(k, nm, rounding_mode="floor")
    mm_index = mm_start[pair_graph] + torch.remainder(k, nm)
    return qm_index, mm_index, pair_graph


def multipole_terms(
    vectors: torch.Tensor,  # [P, 3] from expansion centre to MM charge
    charges: torch.Tensor,  # [P]
    sh: torch.Tensor,  # [P, K] spherical harmonics of vectors
    l_index: torch.Tensor,  # [K]
    eps: float = 1e-6,
) -> torch.Tensor:
    """sqrt(4 pi / (2l+1)) * q / r^(l+1) * Y_lm(r) for every pair  ->  [P, K].

    Charges closer than eps to the expansion centre are ignored (no NaN/inf).
    """
    r = torch.linalg.norm(vectors, dim=-1)
    ok = r > eps
    r_safe = torch.where(ok, r, torch.ones_like(r))
    lf = l_index.to(vectors.dtype)
    prefac = torch.sqrt(4.0 * math.pi / (2.0 * lf + 1.0))
    lmax = int(l_index[-1]) if l_index.numel() > 0 else 0
    inv = (1.0 / r_safe).unsqueeze(-1).expand(-1, lmax + 1)
    inv_pows = torch.cumprod(inv, dim=-1)  # 1/r, 1/r^2, ..., 1/r^(lmax+1)
    radial = (charges * ok.to(vectors.dtype)).unsqueeze(-1) * inv_pows[:, l_index]
    return prefac * radial * sh


# --------------------------------------------------------------------------------------- #
# equivariant embedding of the multipole features
# --------------------------------------------------------------------------------------- #
@compile_mode("script")
class EquivariantMultipoleEmbedding(torch.nn.Module):
    """phi_l (one irrep per l)  ->  (n_basis + 1) x l  for every l.

    l = 0 :  [phi_0, g_k(phi_0)]                       (signed potential, Gaussian basis)
    l > 0 :  [v_l, v_l * g_k(|v_l|)]                   (equivariant: only the norm is non-linear)
    """

    def __init__(self, lmax: int, num_basis: int = 8, scalar_range: float = 3.0, norm_range: float = 3.0):
        super().__init__()
        self.lmax = lmax
        self.num_basis = num_basis
        dt = torch.get_default_dtype()
        self.scalar_centres = torch.nn.Parameter(torch.linspace(-scalar_range, scalar_range, num_basis, dtype=dt))
        self.scalar_widths = torch.nn.Parameter(torch.ones(num_basis, dtype=dt))
        self.norm_centres = torch.nn.Parameter(torch.linspace(0.0, norm_range, num_basis, dtype=dt))
        self.norm_widths = torch.nn.Parameter(torch.ones(num_basis, dtype=dt))
        self.irreps_out = o3.Irreps(
            [(num_basis + 1, (l, (-1) ** l)) for l in range(lmax + 1)]
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:  # [n, (lmax+1)^2]
        n = phi.shape[0]
        phi0 = phi[:, 0:1]
        g0 = torch.exp(-torch.square(self.scalar_widths) * torch.square(phi0 - self.scalar_centres))
        out = [phi0, g0]
        for l in range(1, self.lmax + 1):
            v = phi[:, l * l:(l + 1) * (l + 1)]  # [n, 2l+1]
            norm = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + 1e-12)
            g = torch.exp(-torch.square(self.norm_widths) * torch.square(norm - self.norm_centres))  # [n, nb]
            gated = torch.cat([torch.ones_like(norm), g], dim=-1)  # [n, nb+1]
            out.append((gated.unsqueeze(-1) * v.unsqueeze(1)).reshape(n, -1))  # mul-major
        return torch.cat(out, dim=-1)


@compile_mode("script")
class _NoAttention(torch.nn.Module):
    """Placeholder so every model has one attention entry per layer (never called)."""

    def __init__(self):
        super().__init__()
        self.to_queries = torch.nn.Identity()
        self.to_keys = torch.nn.Identity()
        self.to_values = torch.nn.Identity()
        self.scores_multi = torch.nn.Identity()
        self.to_out = torch.nn.Identity()


def segment_softmax_context(
    scores: torch.Tensor, values: torch.Tensor, index: torch.Tensor, valid: torch.Tensor, n: int
) -> torch.Tensor:
    """sum_j softmax_j(scores_ij) values_ij for every segment i (pairs with valid=False ignored)."""
    neg = torch.full_like(scores, -1e30)
    masked = torch.where(valid, scores, neg)
    smax = torch.full((n,), -1e30, dtype=scores.dtype, device=scores.device)
    smax = smax.scatter_reduce(0, index, masked, reduce="amax", include_self=True)
    w = torch.exp(masked - smax[index]) * valid.to(scores.dtype)
    denom = scatter_sum(w, index, dim=0, dim_size=n)
    w = w / (denom[index] + 1e-30)
    return scatter_sum(w.unsqueeze(-1) * values, index, dim=0, dim_size=n)


# --------------------------------------------------------------------------------------- #
# base model
# --------------------------------------------------------------------------------------- #
class _FieldMACEBase(torch.nn.Module):
    """Shared implementation. Subclasses only set the expansion mode."""

    # set by subclasses
    _expansion = "global"  # "global" (FieldEMACE) or "per_atom"
    _attention = False

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
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        multipole_embedding: str = "equivariant",
        num_multipole_basis: int = 8,
        expansion_centre: str = "centroid",
    ):
        super().__init__()
        if compute_nacs or compute_dipoles:
            raise NotImplementedError(
                "NACs and dipoles are not implemented for the field models (the readouts never "
                "produced them). Train with compute_nacs=False, compute_dipoles=False."
            )
        if multipole_embedding not in ("equivariant", "legacy"):
            raise ValueError("multipole_embedding must be 'equivariant' or 'legacy'")
        if expansion_centre not in ("centroid", "origin"):
            raise ValueError("expansion_centre must be 'centroid' or 'origin'")
        if multipole_max_ell < 0:
            raise ValueError("multipole_max_ell must be >= 0")
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))
        self.n_energies = n_energies
        self.n_nacs = int(n_energies * (n_energies - 1) / 2)
        self.n_dipoles = int(n_energies + self.n_nacs)
        self.compute_dipoles = False
        self.compute_nacs = False
        self.multipole_max_ell = multipole_max_ell
        self.field_model_version = FIELD_MODEL_VERSION
        self.legacy = multipole_embedding == "legacy"
        self.centre_on_qm = expansion_centre == "centroid" and self._expansion == "global"
        self.attention = self._attention
        self.expansion = self._expansion
        self.pair_repulsion = pair_repulsion

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
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

        multipole_sh_irreps = o3.Irreps.spherical_harmonics(multipole_max_ell)
        self.multipole_spherical_harmonics = o3.SphericalHarmonics(
            multipole_sh_irreps, normalize=True, normalization="component"
        )
        self.register_buffer(
            "multipole_l_index", torch.tensor(_l_index(multipole_max_ell), dtype=torch.long), persistent=False
        )
        if self.legacy:
            self.multipole_radial_embedding = MultipoleRadialEmbeddingBlock()
            emb_irreps = multipole_sh_irreps
        else:
            self.multipole_radial_embedding = EquivariantMultipoleEmbedding(multipole_max_ell, num_multipole_basis)
            emb_irreps = self.multipole_radial_embedding.irreps_out
        self.multipole_attention = torch.nn.ModuleList()
        if self.attention:
            self.linear_up = torch.nn.Identity()
        else:
            self.linear_up = o3.Linear(emb_irreps, field_irreps)

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        use_sc_first = "Residual" in str(interaction_cls_first)
        self.interactions = torch.nn.ModuleList()
        self.field_interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        for i in range(num_interactions):
            first = i == 0
            last = i == num_interactions - 1
            hidden_out = str(hidden_irreps[0]) if (last and num_interactions > 1) else hidden_irreps
            feats_in = node_feats_irreps if first else hidden_irreps
            inter = (interaction_cls_first if first else interaction_cls)(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=feats_in,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            self.field_interactions.append(
                ElectrostaticFieldInteractionBlock(
                    node_feats_irreps=feats_in, target_irreps=interaction_irreps, field_irreps=field_irreps
                )
            )
            if self.attention:
                self.multipole_attention.append(
                    RotInvariantAttention(
                        irreps_in=multipole_sh_irreps,
                        irreps_hidden=field_irreps,
                        irreps_out=field_irreps,
                        node_feats_irreps=feats_in,
                    )
                )
            else:
                self.multipole_attention.append(_NoAttention())
            self.products.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=inter.target_irreps if first else interaction_irreps,
                    target_irreps=hidden_out,
                    correlation=correlation[i],
                    num_elements=num_elements,
                    use_sc=use_sc_first if first else True,
                )
            )
            if last and num_interactions > 1:
                self.readouts.append(NonLinearReadoutBlock(hidden_out, MLP_irreps, gate, n_energies, False))
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps, n_energies, False))

    # ------------------------------------------------------------------ #
    # models pickled by the original FieldMACE code
    # ------------------------------------------------------------------ #
    def __setstate__(self, state):
        super().__setstate__(state)
        if "field_model_version" not in self.__dict__:
            self.field_model_version = 1
            self.legacy = True
            self.centre_on_qm = False
            self.attention = self._attention
            self.expansion = self._expansion
            self.pair_repulsion = "pair_repulsion_fn" in self._modules
            self.register_buffer(
                "multipole_l_index",
                torch.tensor(_l_index(int(self.multipole_max_ell)), dtype=torch.long),
                persistent=False,
            )
            self.compute_nacs = False
            self.compute_dipoles = False
            if not self.attention:
                self.multipole_attention = torch.nn.ModuleList(
                    [_NoAttention() for _ in range(int(self.num_interactions))]
                )

    # ------------------------------------------------------------------ #
    def _field_features(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        mm_pos: torch.Tensor,
        mm_q: torch.Tensor,
        num_mm: torch.Tensor,
        mm_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Per-atom multipole features phi [n_nodes, (lmax+1)^2] (or pair moments for attention)."""
        l_index = self.multipole_l_index
        n_nodes = positions.shape[0]
        if self.expansion == "global":
            if self.centre_on_qm:
                counts = (ptr[1:] - ptr[:-1]).to(positions.dtype).clamp(min=1.0)
                centre = scatter_sum(positions, batch, dim=0, dim_size=num_graphs) / counts.unsqueeze(-1)
            else:
                centre = torch.zeros(num_graphs, 3, dtype=positions.dtype, device=positions.device)
            vec = mm_pos - centre[mm_batch]
            terms = multipole_terms(vec, mm_q, self.multipole_spherical_harmonics(vec), l_index)
            moments = scatter_sum(terms, mm_batch, dim=0, dim_size=num_graphs)  # [B, K]
            R = torch.linalg.norm(positions - centre[batch], dim=-1, keepdim=True)  # [n, 1]
            lf = l_index.to(positions.dtype)
            prefac = torch.sqrt(4.0 * math.pi / (2.0 * lf + 1.0))
            return prefac * moments[batch] * torch.pow(R, lf)
        if num_graphs == 1:
            # single configuration (MD): dense broadcasting, no gather/scatter
            m = mm_pos.shape[0]
            vec = (mm_pos.unsqueeze(0) - positions.unsqueeze(1)).reshape(n_nodes * m, 3)
            q = mm_q.unsqueeze(0).expand(n_nodes, m).reshape(n_nodes * m)
            terms = multipole_terms(vec, q, self.multipole_spherical_harmonics(vec), l_index)
            return terms.reshape(n_nodes, m, l_index.shape[0]).sum(dim=1)
        qi, mj, _ = qm_mm_pairs(ptr, num_mm)
        vec = mm_pos[mj] - positions[qi]
        terms = multipole_terms(vec, mm_q[mj], self.multipole_spherical_harmonics(vec), l_index)
        return scatter_sum(terms, qi, dim=0, dim_size=n_nodes)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_hessian: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_pc_grads: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        positions = data["positions"]
        batch = data["batch"]
        ptr = data["ptr"]
        num_graphs = ptr.numel() - 1
        n_nodes = positions.shape[0]
        mm_pos, mm_q, num_mm, mm_batch, padded = flat_mm_layout(data, num_graphs)
        mm_pos = mm_pos.to(positions.dtype)
        mm_q = mm_q.to(positions.dtype)

        want_qm = compute_force or compute_hessian
        want_mm = compute_force and compute_pc_grads
        if want_qm and not positions.requires_grad:
            positions.requires_grad_(True)
        if want_mm and not mm_pos.requires_grad:
            mm_pos.requires_grad_(True)

        node_attrs = data["node_attrs"]
        node_e0 = self.atomic_energies_fn(node_attrs)
        e0 = scatter_sum(node_e0, batch, dim=-1, dim_size=num_graphs)
        node_feats = self.node_embedding(node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=data["edge_index"], shifts=data["shifts"]
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths, node_attrs, data["edge_index"], self.atomic_numbers)

        energies = [e0.unsqueeze(-1).expand(-1, self.n_energies)]
        node_energies_list = [node_e0.unsqueeze(-1).expand(-1, self.n_energies)]
        if hasattr(self, "pair_repulsion_fn"):
            pair_node = self.pair_repulsion_fn(lengths, node_attrs, data["edge_index"], self.atomic_numbers)
            pair_e = scatter_sum(pair_node, batch, dim=-1, dim_size=num_graphs)
            energies.append(pair_e.unsqueeze(-1).expand(-1, self.n_energies))
            node_energies_list.append(pair_node.unsqueeze(-1).expand(-1, self.n_energies))

        # ---- field features ------------------------------------------------
        multipole_feats = torch.zeros(0, dtype=positions.dtype, device=positions.device)
        pair_terms = torch.zeros(0, dtype=positions.dtype, device=positions.device)
        qi = torch.zeros(0, dtype=torch.long, device=positions.device)
        q_pair = torch.zeros(0, dtype=positions.dtype, device=positions.device)
        m_scale = torch.zeros(0, dtype=torch.long, device=positions.device)
        if self.attention:
            qi, mj, pg = qm_mm_pairs(ptr, num_mm)
            vec = mm_pos[mj] - positions[qi]
            q_pair = mm_q[mj]
            pair_terms = multipole_terms(vec, q_pair, self.multipole_spherical_harmonics(vec), self.multipole_l_index)
            m_scale = num_mm[pg]
        else:
            phi = self._field_features(positions, batch, ptr, mm_pos, mm_q, num_mm, mm_batch, num_graphs)
            multipole_feats = self.linear_up(self.multipole_radial_embedding(phi))

        node_feats_list: List[torch.Tensor] = []
        for interaction, product, readout, field_interaction, att in zip(
            self.interactions, self.products, self.readouts, self.field_interactions, self.multipole_attention
        ):
            if self.attention:
                queries = att.to_queries(pair_terms)  # [P, h]
                keys = att.to_keys(node_feats)  # [n, h]
                scores = att.scores_multi(queries - keys[qi]).squeeze(-1)  # [P]
                if self.legacy:
                    scores = scores / torch.sqrt(m_scale.to(scores.dtype))
                    valid = torch.ones_like(scores, dtype=torch.bool)
                else:
                    valid = q_pair != 0
                context = segment_softmax_context(scores, att.to_values(pair_terms), qi, valid, n_nodes)
                att_out = att.to_out(context)
                if self.legacy:
                    multipole_feats = self.multipole_radial_embedding(att_out)
                else:
                    multipole_feats = att_out
            inter_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            field_feats = field_interaction(node_feats=node_feats, multipole_feats=multipole_feats)
            node_feats = product(node_feats=inter_feats + field_feats, sc=sc, node_attrs=node_attrs)
            node_feats_list.append(node_feats)
            node_out = readout(node_feats).reshape(n_nodes, self.n_energies)
            energies.append(scatter_sum(node_out, batch, dim=0, dim_size=num_graphs))
            node_energies_list.append(node_out)

        contributions = torch.stack(energies, dim=1)  # [B, n_contrib, n_e]
        total_energy = torch.sum(contributions, dim=1)  # [B, n_e]
        node_energy = torch.sum(torch.stack(node_energies_list, dim=1), dim=1)  # [n, n_e]
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        forces: Optional[torch.Tensor] = None
        mm_forces: Optional[torch.Tensor] = None
        hessian: Optional[torch.Tensor] = None
        if want_qm or want_mm:
            f_list: List[torch.Tensor] = []
            fm_list: List[torch.Tensor] = []
            for s in range(self.n_energies):
                keep = training or compute_hessian or (s < self.n_energies - 1)
                grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(total_energy[:, s])]
                if want_mm:
                    g = torch.autograd.grad(
                        [total_energy[:, s]], [positions, mm_pos], grad_outputs=grad_outputs,
                        retain_graph=keep, create_graph=training or compute_hessian, allow_unused=True,
                    )
                    gq = g[0]
                    gm = g[1]
                    f_list.append(-gq if gq is not None else torch.zeros_like(positions))
                    fm_list.append(-gm if gm is not None else torch.zeros_like(mm_pos))
                else:
                    g = torch.autograd.grad(
                        [total_energy[:, s]], [positions], grad_outputs=grad_outputs,
                        retain_graph=keep, create_graph=training or compute_hessian, allow_unused=True,
                    )
                    gq = g[0]
                    f_list.append(-gq if gq is not None else torch.zeros_like(positions))
            forces = torch.stack(f_list, dim=1)  # [n, n_e, 3]
            if want_mm:
                mm_forces = torch.stack(fm_list, dim=1)  # [M, n_e, 3]
                if padded:
                    mm_forces = mm_forces.reshape(num_graphs, -1, self.n_energies, 3)
            if compute_hessian:
                hessian = compute_hessians_vmap(forces[:, 0], positions)

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "nacs": None,
            "dipoles": None,
            "forces": forces,
            "mm_forces": mm_forces,
            "virials": None,
            "stress": None,
            "displacement": None,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class FieldEMACE(_FieldMACEBase):
    """MM potential expanded in multipoles about the QM centroid of each configuration."""

    _expansion = "global"
    _attention = False


@compile_mode("script")
class PerAtomFieldEMACE(_FieldMACEBase):
    """MM potential expanded about every QM atom (translation invariant, the default choice)."""

    _expansion = "per_atom"
    _attention = False


@compile_mode("script")
class AttentionPerAtomFieldEMACE(_FieldMACEBase):
    """Per-atom expansion where each QM atom attends over its MM charges in every layer."""

    _expansion = "per_atom"
    _attention = True


class AttentionFieldEMACE(torch.nn.Module):
    """Removed: the original class could not be constructed (NameError on 'lmax')."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AttentionFieldEMACE was never functional in FieldMACE and has been removed. "
            "Use AttentionPerAtomFieldEMACE, PerAtomFieldEMACE or FieldEMACE."
        )
