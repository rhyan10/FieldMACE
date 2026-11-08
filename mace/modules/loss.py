###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# Modified for FieldMACE by Rhyan Barrett
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import torch

from mace.tools import TensorDict
from mace.tools.scatter import scatter_sum
from mace.tools.torch_geometric import Batch


def mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    return torch.mean(torch.square(ref["energy"] - pred["energy"]))  # []

def mean_squared_error_pc_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    return torch.mean(torch.square(ref["mm_forces"] - pred["mm_forces"]))  # []

def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    configs_energy_weight = ref.energy_weight  # [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )  # []


def weighted_mean_squared_stress(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)  # [n_graphs, ]
    return torch.mean(
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )  # []

def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    configs_energy_weight = ref.energy_weight  # [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]

    energy_loss = torch.mean(
        configs_weight.unsqueeze(-1)
        * configs_energy_weight.unsqueeze(-1)
        * torch.sum(torch.square((ref["energy"] - pred["energy"])) / num_atoms.unsqueeze(-1), dim=-1)
    )

    return energy_loss

def weighted_mean_squared_virials(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)  # [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )  # []

def phase_rmse_loss(ref: Batch, pred: TensorDict) -> torch.Tensor:
        # nacs: [n_energies*(n_energies-1)/2), 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_nacs_weight = torch.repeat_interleave(
        ref.nacs_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )
    neg = torch.square(ref["nacs"] - pred["nacs"]).unsqueeze(-1)
    pos = torch.square(ref["nacs"] + pred["nacs"]).unsqueeze(-1)
    vec = torch.cat((pos,neg),dim=-1)

    return torch.mean(
        configs_weight.unsqueeze(-1)
        * configs_nacs_weight.unsqueeze(-1)
        * torch.min(vec, dim=-1)[0]
    )  # []

def mean_squared_error_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # forces: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1).unsqueeze(
        -1
    )
    # print(ref["forces"])
    # print(pred["forces"])
    return torch.mean(
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )  # []


def weighted_mean_squared_error_dipole(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dipole: [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)  # [n_graphs,1]
    #print(ref["dipoles"])
    #print(pred["dipoles"])
    #print("##########")
    return torch.mean(torch.square((ref["dipoles"] - pred["dipoles"]) / num_atoms))  # []
    # return torch.mean(torch.square((torch.reshape(ref['dipole'], pred["dipole"].shape) - pred['dipole']) / num_atoms))  # []

def weight_mean_squared_socs(ref: Batch, pred: TensorDict) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_socs_weight = torch.repeat_interleave(
        ref.socs_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )
    return torch.mean(
        configs_weight.unsqueeze(-1)
        * configs_socs_weight.unsqueeze(-1)
        * torch.square(ref["transformed_socs"] - pred["transformed_socs"])
    )  # []

def conditional_mse_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # forces: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]

    # Define the multiplication factors for each condition
    factors = torch.tensor([1.0, 0.7, 0.4, 0.1])

    # Apply multiplication factors based on conditions
    c1 = torch.norm(ref["forces"], dim=-1) < 100
    c2 = (torch.norm(ref["forces"], dim=-1) >= 100) & (
        torch.norm(ref["forces"], dim=-1) < 200
    )
    c3 = (torch.norm(ref["forces"], dim=-1) >= 200) & (
        torch.norm(ref["forces"], dim=-1) < 300
    )

    err = ref["forces"] - pred["forces"]

    se = torch.zeros_like(err)

    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]

    return torch.mean(configs_weight * configs_forces_weight * se)


def conditional_huber_forces(
    ref: Batch, pred: TensorDict, huber_delta: float
) -> torch.Tensor:
    # Define the multiplication factors for each condition
    factors = huber_delta * torch.tensor([1.0, 0.7, 0.4, 0.1])

    # Apply multiplication factors based on conditions
    c1 = torch.norm(ref["forces"], dim=-1) < 100
    c2 = (torch.norm(ref["forces"], dim=-1) >= 100) & (
        torch.norm(ref["forces"], dim=-1) < 200
    )
    c3 = (torch.norm(ref["forces"], dim=-1) >= 200) & (
        torch.norm(ref["forces"], dim=-1) < 300
    )
    c4 = ~(c1 | c2 | c3)

    se = torch.zeros_like(pred["forces"])

    se[c1] = torch.nn.functional.huber_loss(
        ref["forces"][c1], pred["forces"][c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref["forces"][c2], pred["forces"][c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref["forces"][c3], pred["forces"][c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref["forces"][c4], pred["forces"][c4], reduction="none", delta=factors[3]
    )

    return torch.mean(se)


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.energy_weight * weighted_mean_squared_error_energy(
            ref, pred
        ) + self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.stress_weight * weighted_mean_squared_stress(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        return (
            self.energy_weight
            * self.huber_loss(ref["energy"] / num_atoms, pred["energy"] / num_atoms)
            + self.forces_weight * self.huber_loss(ref["forces"], pred["forces"])
            + self.stress_weight * self.huber_loss(ref["stress"], pred["stress"])
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        return (
            self.energy_weight
            * self.huber_loss(ref["energy"] / num_atoms, pred["energy"] / num_atoms)
            + self.forces_weight
            * conditional_huber_forces(ref, pred, huber_delta=self.huber_delta)
            + self.stress_weight * self.huber_loss(ref["stress"], pred["stress"])
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.virials_weight * weighted_mean_squared_virials(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100.0
        )  # multiply by 100 to have the right scale for the loss

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"dipole_weight={self.dipole_weight:.3f})"


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )

def _match(ref_t: torch.Tensor, pred_t: torch.Tensor, name: str) -> torch.Tensor:
    """Bring labels to the prediction shape; fail loudly on a real mismatch (e.g. number of states)."""
    if ref_t.shape == pred_t.shape:
        return ref_t
    if ref_t.numel() == pred_t.numel():
        return ref_t.reshape(pred_t.shape)
    raise ValueError(
        f"{name}: labels have shape {tuple(ref_t.shape)} but the model predicts {tuple(pred_t.shape)}. "
        "Does the number of states in the data match --n_energies?"
    )


def phase_signs(ref: torch.Tensor, pred: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Sign (+1/-1) per (group, pair) that best aligns pred with ref.

    ref, pred: [X, P, C] (X atoms or configurations), index: [X] group (configuration) of each row.
    The phase of each electronic state is arbitrary in QM data, so couplings between two states
    (NACs, transition dipoles, SOCs) are only defined up to a sign per configuration and pair.
    """
    with torch.no_grad():
        plus = scatter_sum(torch.square(ref - pred).sum(-1), index, dim=0, dim_size=n_groups)
        minus = scatter_sum(torch.square(ref + pred).sum(-1), index, dim=0, dim_size=n_groups)
        return torch.where(minus < plus, -torch.ones_like(plus), torch.ones_like(plus))  # [G, P]


def phase_aligned_error(ref: torch.Tensor, pred: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """ref - s * pred with the best sign s per (group, pair)."""
    s = phase_signs(ref, pred, index, n_groups)
    return ref - s[index].unsqueeze(-1) * pred


class WeightedEnergyForcesNacsDipoleLoss(torch.nn.Module):
    """Loss for the field models.

    energies [B, S], forces [N, S, 3], forces on the MM charges [M, S, 3] and, if the model
    predicts them, NACs [N, P, 3], dipoles [B, S+P, 3] (S permanent + P transition) and
    SOCs [B, P, 2]. NACs, transition dipoles and SOCs use a phase-free error (best sign per
    configuration and state pair). Per-configuration weights (config_*_weight) are honoured,
    so configurations without a label (weight 0) do not contribute to that term.
    """

    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipoles_weight=1.0, nacs_weight=1.0,
                 mm_forces_weight=None, socs_weight=1.0, output_weights=None) -> None:
        super().__init__()
        dt = torch.get_default_dtype()
        # per-output weights of the energy-like outputs (diabatic models: diagonal vs couplings)
        self.register_buffer("output_weights", torch.ones(1, dtype=dt) if output_weights is None
                             else torch.as_tensor(output_weights, dtype=dt))
        self.register_buffer("energy_weight", torch.tensor(energy_weight, dtype=dt))
        self.register_buffer("forces_weight", torch.tensor(forces_weight, dtype=dt))
        self.register_buffer("nacs_weight", torch.tensor(nacs_weight, dtype=dt))
        self.register_buffer("dipoles_weight", torch.tensor(dipoles_weight, dtype=dt))
        self.register_buffer("socs_weight", torch.tensor(socs_weight, dtype=dt))
        self.register_buffer(
            "mm_forces_weight",
            torch.tensor(forces_weight if mm_forces_weight is None else mm_forces_weight, dtype=dt),
        )

    def forward(self, ref: Batch, pred: TensorDict, training=False) -> torch.Tensor:
        n_atoms = ref.ptr[1:] - ref.ptr[:-1]
        n_graphs = n_atoms.shape[0]
        w_graph = ref.weight
        graphs = torch.arange(n_graphs, device=n_atoms.device)
        e_ref = _match(ref["energy"], pred["energy"], "energy")
        w_e = (w_graph * ref.energy_weight).unsqueeze(-1)
        w_out = self.output_weights  # [S_out] or [1]
        loss = self.energy_weight * torch.mean(w_e * w_out * torch.square(e_ref - pred["energy"]))

        if pred.get("forces") is not None and self.forces_weight > 0:
            f_ref = _match(ref["forces"], pred["forces"], "forces")
            w_f = torch.repeat_interleave(w_graph * ref.forces_weight, n_atoms)
            w_f = w_f.view(-1, *([1] * (f_ref.dim() - 1)))
            w_fo = w_out.view(-1, 1) if f_ref.dim() == 3 else w_out[:1]
            loss = loss + self.forces_weight * torch.mean(w_f * w_fo * torch.square(f_ref - pred["forces"]))

        if pred.get("mm_forces") is not None and self.mm_forces_weight > 0 and ref["mm_forces"].numel() > 0:
            fm_ref = _match(ref["mm_forces"], pred["mm_forces"], "mm_forces")
            num_mm = ref["num_mm"].reshape(-1)
            w_m = torch.repeat_interleave(w_graph * ref["mm_forces_weight"].reshape(-1), num_mm)
            w_m = w_m.view(-1, *([1] * (fm_ref.dim() - 1)))
            w_mo = w_out.view(-1, 1) if fm_ref.dim() == 3 else w_out[:1]
            loss = loss + self.mm_forces_weight * torch.mean(w_m * w_mo * torch.square(fm_ref - pred["mm_forces"]))

        nacs = pred.get("nacs")
        if nacs is not None and self.nacs_weight > 0 and nacs.numel() > 0:
            n_ref = _match(ref["nacs"], nacs, "nacs")
            err = phase_aligned_error(n_ref, nacs, ref.batch, n_graphs)
            w_n = torch.repeat_interleave(w_graph * ref.nacs_weight, n_atoms).view(-1, 1, 1)
            loss = loss + self.nacs_weight * torch.mean(w_n * torch.square(err))

        dip = pred.get("dipoles")
        if dip is not None and self.dipoles_weight > 0:
            d_ref = _match(ref["dipoles"], dip, "dipoles")
            n_s = e_ref.shape[1]
            err_perm = d_ref[:, :n_s] - dip[:, :n_s]
            err_trans = phase_aligned_error(d_ref[:, n_s:], dip[:, n_s:], graphs, n_graphs)
            err = torch.cat([err_perm, err_trans], dim=1)
            w_d = (w_graph * ref.dipoles_weight).view(-1, 1, 1)
            loss = loss + self.dipoles_weight * torch.mean(w_d * torch.square(err))

        socs = pred.get("socs")
        if socs is not None and self.socs_weight > 0 and socs.numel() > 0:
            s_ref = _match(ref["socs"], socs, "socs")
            err = phase_aligned_error(s_ref, socs, graphs, n_graphs)
            w_s = (w_graph * ref["socs_weight"].reshape(-1)).view(-1, 1, 1)
            loss = loss + self.socs_weight * torch.mean(w_s * torch.square(err))

        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, mm_forces_weight={self.mm_forces_weight:.3f}, "
            f"nacs_weight={self.nacs_weight:.3f}, dipoles_weight={self.dipoles_weight:.3f}, "
            f"socs_weight={self.socs_weight:.3f})"
        )
