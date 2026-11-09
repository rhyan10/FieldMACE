###########################################################################################
# FieldMACE: MACE for QM/MM - QM region embedded in the field of MM point charges
# Author: Rhyan Barrett
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
#   * Excited-state properties (optional): non-adiabatic couplings (per-atom vectors for every
#     pair of states), permanent and transition dipoles, and spin-orbit couplings, predicted
#     with equivariant readouts. Their phase (sign) is arbitrary in the reference data, so the
#     losses are phase-free.
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
)
from .radial import ZBLBasis
from .utils import compute_hessians_vmap, get_edge_vectors_and_lengths

FIELD_MODEL_VERSION = 4


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


def state_pairs(n_states: int) -> List[List[int]]:
    """Order of the state pairs used for NACs, transition dipoles and SOCs: (0,1), (0,2), ..., (1,2), ..."""
    return [[i, j] for i in range(n_states) for j in range(i + 1, n_states)]


# --------------------------------------------------------------------------------------- #
# pseudoscalar readout (diabatic couplings between states of opposite mirror symmetry)
# --------------------------------------------------------------------------------------- #
@compile_mode("script")
class PseudoscalarReadoutBlock(torch.nn.Module):
    """Parity-odd (pseudoscalar, 0o) node outputs for diabatic couplings.

    A coupling between two diabatic states of opposite symmetry under a mirror plane of the molecule
    (e.g. A1 and A2 in C2v) changes sign when the environment is mirrored: it is a pseudoscalar. A
    MACE readout gives only true scalars (0e), which are blind to that sign, so such couplings would
    be learned as zero. For every channel k, with u, v, w learned vectors (1o) of the node features,
    p = v x w (a pseudovector, 1e), and E (1o) and G (2e) learned combinations of the node's field
    features (field and field gradient of the point charges at the atom):

        s_k = u_k . p_k  +  E_k . p_k  +  u_k . (G_k p_k)          (all pseudoscalars)

    e.g. u.(G p) ~ e_x^T G e_y is the field-gradient component that couples A1 and A2 states in C2v.
    The s_k are weighted by an MLP of the node's scalar features (even) and summed: odd.
    """

    def __init__(self, irreps_in: o3.Irreps, n_out: int, field_irreps: Optional[o3.Irreps] = None,
                 n_triple: int = 16, n_hidden: int = 16):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        self.n_out = int(n_out)
        self.n_triple = int(n_triple)
        vec = o3.Irreps(f"{n_triple}x1o")
        self.lin_u = o3.Linear(irreps_in, vec)
        self.lin_v = o3.Linear(irreps_in, vec)
        self.lin_w = o3.Linear(irreps_in, vec)
        self.lin_s = o3.Linear(irreps_in, o3.Irreps(f"{n_hidden}x0e"))
        self.mlp = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(n_hidden, n_triple * n_out))
        field_irreps = o3.Irreps(field_irreps) if field_irreps is not None else o3.Irreps("")
        self.use_field_vec = field_irreps.count(o3.Irrep(1, -1)) > 0
        self.use_field_grad = field_irreps.count(o3.Irrep(2, 1)) > 0
        # always built (an o3.Linear without matching inputs outputs zeros), so TorchScript sees real modules
        self.lin_fe = o3.Linear(field_irreps, vec)
        self.lin_fg = o3.Linear(field_irreps, o3.Irreps(f"{n_triple}x2e"))
        self.tp_gp = o3.ElementwiseTensorProduct(o3.Irreps(f"{n_triple}x2e"), o3.Irreps(f"{n_triple}x1e"), ["1e"])

    def _pseudovectors(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        v = self.lin_v(x).reshape(n, self.n_triple, 3)
        w = self.lin_w(x).reshape(n, self.n_triple, 3)
        return torch.cross(v, w, dim=-1)  # [n, K, 3], 1e

    @torch.jit.export
    def vectors(self, x: torch.Tensor) -> torch.Tensor:
        """Pseudovector (1e) outputs [n, n_out, 3]: sum_k g_k(scalars) (v_k x w_k)."""
        n = x.shape[0]
        g = self.mlp(self.lin_s(x)).reshape(n, self.n_triple, self.n_out)
        return torch.einsum("nkx,nko->nox", self._pseudovectors(x), g)

    def forward(self, x: torch.Tensor, field: Optional[torch.Tensor] = None) -> torch.Tensor:  # -> [n, n_out]
        n = x.shape[0]
        u = self.lin_u(x).reshape(n, self.n_triple, 3)
        p = self._pseudovectors(x)
        s = torch.sum(u * p, dim=-1)  # [n, K], odd
        if field is not None:
            if self.use_field_vec:
                e = self.lin_fe(field).reshape(n, self.n_triple, 3)
                s = s + torch.sum(e * p, dim=-1)
            if self.use_field_grad:
                gp = self.tp_gp(self.lin_fg(field), p.reshape(n, -1)).reshape(n, self.n_triple, 3)  # G p, 1e
                s = s + torch.sum(u * gp, dim=-1)
        g = self.mlp(self.lin_s(x)).reshape(n, self.n_triple, self.n_out)  # even
        return torch.einsum("nk,nko->no", s, g)


# --------------------------------------------------------------------------------------- #
# base model
# --------------------------------------------------------------------------------------- #
class _FieldMACEBase(torch.nn.Module):
    """Shared implementation. Subclasses only set the expansion mode.

    Outputs (B configurations, N atoms, M point charges, S states, P = S(S-1)/2 pairs):
      energy [B, S], forces [N, S, 3], mm_forces [M, S, 3]
               diabatic=True: S diabatic states give S + P outputs, the diabatic Hamiltonian
               [H_00, ..., H_S-1,S-1, then H_ij for the pairs]; forces are -dH/dR of every element
               (see mace.tools.diabatic for turning them into adiabatic energies, gradients, NACs)
      nacs     [N, P, 3]       if compute_nacs     (per-atom couplings for every state pair)
      dipoles  [B, S + P, 3]   if compute_dipoles  (S permanent dipoles, then P transition dipoles)
      socs     [B, P, 2]       if compute_socs     (real and imaginary part for every state pair)
               [B, 1, soc_num] if compute_socs and soc_num is given (free-form SOC vector, e.g. the
                               upper triangle of SHARC's spin-orbit matrix as in X-MACE; one phase)
    """

    odd_layers: List[int]  # interaction layers with a pseudoscalar coupling readout (diabatic models)

    # set by subclasses
    _expansion = "global"  # "global" (FieldEMACE) or "per_atom"

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
        compute_socs: bool = False,
        soc_num: Optional[int] = None,
        diabatic: bool = False,
    ):
        super().__init__()
        if multipole_embedding not in ("equivariant", "legacy"):
            raise ValueError("multipole_embedding must be 'equivariant' or 'legacy'")
        if expansion_centre not in ("centroid", "origin"):
            raise ValueError("expansion_centre must be 'centroid' or 'origin'")
        if multipole_max_ell < 0:
            raise ValueError("multipole_max_ell must be >= 0")
        if n_energies < 1:
            raise ValueError("n_energies must be >= 1")
        if compute_socs and soc_num is not None and int(soc_num) < 1:
            raise ValueError("soc_num must be >= 1")
        if (compute_nacs or (compute_socs and soc_num is None)) and n_energies < 2:
            raise ValueError("NACs and SOCs couple pairs of states: they need n_energies >= 2")
        if diabatic and compute_nacs:
            raise ValueError("diabatic models have no NAC outputs: NACs follow from the diabatic Hamiltonian")
        if diabatic and n_energies < 2:
            raise ValueError("a diabatic model needs at least 2 states")
        if multipole_max_ell > o3.Irreps(field_irreps).lmax:
            raise ValueError(f"multipole_max_ell={multipole_max_ell} but field_irreps={field_irreps} stops at "
                             f"l={o3.Irreps(field_irreps).lmax}: the higher multipoles would be dropped. Add them to "
                             f"--field_irreps (e.g. '+ 4x{multipole_max_ell}{'e' if multipole_max_ell % 2 == 0 else 'o'}') "
                             "or lower --multipole_max_ell")
        # the layers assume the scalars come first ("8x1o + 8x0e" would leave the last layer without scalars)
        hidden_irreps = o3.Irreps(hidden_irreps).sort().irreps.simplify()
        if (compute_nacs or compute_dipoles) and hidden_irreps.count(o3.Irrep(1, -1)) == 0:
            raise ValueError(f"hidden_irreps={hidden_irreps} has no vectors (1o): NACs and dipoles are vectors and "
                             "would be identically zero. Use e.g. --hidden_irreps='32x0e + 32x1o' (or --max_L=1)")
        if diabatic and hidden_irreps.count(o3.Irrep(1, -1)) == 0:
            import warnings

            warnings.warn(f"hidden_irreps={hidden_irreps} has no vectors (1o): the diabatic couplings get no "
                          "pseudoscalar part (couplings between states of opposite mirror symmetry stay zero)")
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))
        # n_states: electronic states; n_energies: energy-like outputs (diabatic: all elements of H)
        self.diabatic = bool(diabatic)
        self.n_states = int(n_energies)
        self.n_nacs = int(n_energies * (n_energies - 1) / 2)
        self.n_dipoles = int(n_energies + self.n_nacs)
        n_energies = self.n_states + self.n_nacs if self.diabatic else self.n_states
        self.n_energies = n_energies
        # atomic reference energies (E0s) enter the diagonal only, not diabatic couplings
        mask = torch.ones(n_energies, dtype=torch.get_default_dtype())
        if self.diabatic:
            mask[self.n_states:] = 0.0
        self.register_buffer("e0_mask", mask, persistent=False)
        self.compute_nacs = bool(compute_nacs)
        self.compute_dipoles = bool(compute_dipoles)
        self.compute_socs = bool(compute_socs)
        self.multipole_max_ell = multipole_max_ell
        self.field_model_version = FIELD_MODEL_VERSION
        self.legacy = multipole_embedding == "legacy"
        self.centre_on_qm = expansion_centre == "centroid" and self._expansion == "global"
        self.expansion = self._expansion
        self.pair_repulsion = pair_repulsion

        # property readout layout: [2*P SOC scalars | P NAC vectors | S+P dipole vectors]
        if self.compute_socs and soc_num is not None:  # free-form vector, one phase per configuration
            self.soc_layout = "vector"
            self.soc_groups, self.soc_group_size = 1, int(soc_num)
        else:  # (re, im) for every pair of states, one phase per pair
            self.soc_layout = "pairs"
            self.soc_groups, self.soc_group_size = self.n_nacs, 2
        self.n_prop_scalars = self.soc_groups * self.soc_group_size if self.compute_socs else 0
        self.n_nac_vectors = self.n_nacs if self.compute_nacs else 0
        self.n_dipole_vectors = self.n_dipoles if self.compute_dipoles else 0
        n_vectors = self.n_nac_vectors + self.n_dipole_vectors
        self.n_prop_out = self.n_prop_scalars + 3 * n_vectors
        prop_irreps = o3.Irreps(
            [(m, ir) for m, ir in [(self.n_prop_scalars, (0, 1)), (n_vectors, (1, -1))] if m > 0]
        )

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
        self.linear_up = o3.Linear(emb_irreps, field_irreps)

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        # vector properties need vectors in the last layer too (MACE normally keeps only scalars)
        keep_vectors = n_vectors > 0
        use_sc_first = "Residual" in str(interaction_cls_first)
        self.interactions = torch.nn.ModuleList()
        self.field_interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        self.property_readouts = torch.nn.ModuleList()
        # diabatic couplings: a true-scalar part (readouts) plus a pseudoscalar part (odd_readouts)
        self.odd_readouts = torch.nn.ModuleList()
        self.odd_dipole_readouts = torch.nn.ModuleList()
        self.odd_layers = []
        for i in range(num_interactions):
            first = i == 0
            last = i == num_interactions - 1
            scalar_last = last and num_interactions > 1 and not keep_vectors
            hidden_out = o3.Irreps(str(hidden_irreps[0])) if scalar_last else hidden_irreps
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
            if self.n_prop_out > 0:
                self.property_readouts.append(o3.Linear(hidden_out, prop_irreps))
            if self.diabatic and self.n_nacs > 0 and o3.Irreps(hidden_out).count(o3.Irrep(1, -1)) > 0:
                self.odd_readouts.append(PseudoscalarReadoutBlock(hidden_out, self.n_nacs, field_irreps))
                if self.compute_dipoles:  # pseudovector part of the off-diagonal diabatic dipoles
                    self.odd_dipole_readouts.append(PseudoscalarReadoutBlock(hidden_out, self.n_nacs))
                self.odd_layers.append(i)

    # ------------------------------------------------------------------ #
    # models pickled by earlier versions
    # ------------------------------------------------------------------ #
    def __setstate__(self, state):
        super().__setstate__(state)
        if "field_model_version" not in self.__dict__:  # FieldMACE 1
            self.field_model_version = 1
            self.legacy = True
            self.centre_on_qm = False
            self.expansion = self._expansion
            self.pair_repulsion = "pair_repulsion_fn" in self._modules
            self.register_buffer(
                "multipole_l_index",
                torch.tensor(_l_index(int(self.multipole_max_ell)), dtype=torch.long),
                persistent=False,
            )
        if "property_readouts" not in self._modules:  # FieldMACE 1 and early 2.0
            self.property_readouts = torch.nn.ModuleList()
            self.compute_nacs = False
            self.compute_dipoles = False
            self.compute_socs = False
            self.soc_layout = "pairs"
            self.soc_groups, self.soc_group_size = 0, 2
            self.n_prop_scalars = 0
            self.n_nac_vectors = 0
            self.n_dipole_vectors = 0
            self.n_prop_out = 0
        if "multipole_attention" in self._modules:
            del self._modules["multipole_attention"]
        if "diabatic" not in self.__dict__:  # before FieldMACE 2.4
            self.diabatic = False
            self.n_states = int(self.n_energies)
        if "odd_readouts" not in self._modules:
            self.odd_readouts = torch.nn.ModuleList()
            self.odd_dipole_readouts = torch.nn.ModuleList()
            self.odd_layers = []
        if "e0_mask" not in self._buffers:
            dt = self.atomic_energies_fn.atomic_energies.dtype
            self.register_buffer("e0_mask", torch.ones(int(self.n_energies), dtype=dt), persistent=False)

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
        """Per-atom multipole features phi [n_nodes, (lmax+1)^2]."""
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

        energies = [e0.unsqueeze(-1) * self.e0_mask]
        node_energies_list = [node_e0.unsqueeze(-1) * self.e0_mask]
        if hasattr(self, "pair_repulsion_fn"):
            pair_node = self.pair_repulsion_fn(lengths, node_attrs, data["edge_index"], self.atomic_numbers)
            pair_e = scatter_sum(pair_node, batch, dim=-1, dim_size=num_graphs)
            energies.append(pair_e.unsqueeze(-1) * self.e0_mask)
            node_energies_list.append(pair_node.unsqueeze(-1) * self.e0_mask)

        phi = self._field_features(positions, batch, ptr, mm_pos, mm_q, num_mm, mm_batch, num_graphs)
        multipole_feats = self.linear_up(self.multipole_radial_embedding(phi))

        node_feats_list: List[torch.Tensor] = []
        for interaction, product, readout, field_interaction in zip(
            self.interactions, self.products, self.readouts, self.field_interactions
        ):
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

        if len(self.odd_layers) > 0:  # pseudoscalar parts of the diabatic couplings
            odd = torch.zeros(n_nodes, self.n_nacs, dtype=positions.dtype, device=positions.device)
            for k, odd_head in enumerate(self.odd_readouts):
                odd = odd + odd_head(node_feats_list[self.odd_layers[k]], multipole_feats)
            odd_node = torch.cat([torch.zeros(n_nodes, self.n_states, dtype=odd.dtype, device=odd.device), odd], dim=1)
            energies.append(scatter_sum(odd_node, batch, dim=0, dim_size=num_graphs))
            node_energies_list.append(odd_node)

        contributions = torch.stack(energies, dim=1)  # [B, n_contrib, n_e]
        total_energy = torch.sum(contributions, dim=1)  # [B, n_e]
        node_energy = torch.sum(torch.stack(node_energies_list, dim=1), dim=1)  # [n, n_e]
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # ---- excited-state properties ------------------------------------
        nacs: Optional[torch.Tensor] = None
        dipoles: Optional[torch.Tensor] = None
        socs: Optional[torch.Tensor] = None
        if self.n_prop_out > 0:
            prop = torch.zeros(n_nodes, self.n_prop_out, dtype=positions.dtype, device=positions.device)
            for i, head in enumerate(self.property_readouts):
                prop = prop + head(node_feats_list[i])
            ns = self.n_prop_scalars
            vec = prop[:, ns:].reshape(n_nodes, -1, 3)  # Cartesian (e3nn 1o basis is x, y, z)
            if self.compute_socs:
                socs = scatter_sum(prop[:, :ns], batch, dim=0, dim_size=num_graphs).reshape(
                    num_graphs, self.soc_groups, self.soc_group_size
                )
            if self.compute_nacs:
                nacs = vec[:, : self.n_nac_vectors]
            if self.compute_dipoles:
                dip_node = vec[:, self.n_nac_vectors:]
                if len(self.odd_dipole_readouts) > 0:  # diabatic: pseudovector part of the off-diagonal dipoles
                    odd_d = torch.zeros(n_nodes, self.n_nacs, 3, dtype=positions.dtype, device=positions.device)
                    for kd, dip_head in enumerate(self.odd_dipole_readouts):
                        odd_d = odd_d + dip_head.vectors(node_feats_list[self.odd_layers[kd]])
                    dip_node = dip_node + torch.cat(
                        [torch.zeros(n_nodes, self.n_states, 3, dtype=odd_d.dtype, device=odd_d.device), odd_d], dim=1)
                dipoles = scatter_sum(dip_node, batch, dim=0, dim_size=num_graphs)

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
                hessian = compute_hessians_vmap(forces[:, 0].contiguous(), positions)

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "nacs": nacs,
            "dipoles": dipoles,
            "socs": socs,
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


@compile_mode("script")
class PerAtomFieldEMACE(_FieldMACEBase):
    """MM potential expanded about every QM atom (translation invariant, the default choice)."""

    _expansion = "per_atom"
