# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
FieldMACE as a TorchScript module, for ``openmmtorch.TorchForce`` (openmm-torch) and any other
program that can run a TorchScript potential.

The PythonForce backend (:mod:`mace.mlmm.force`) calls Python every step. TorchForce runs the
model in C++ (libtorch) inside OpenMM: no Python per step, no GIL, CUDA graphs possible, and
the exported ``.pt`` file runs from OpenMM's C++ API or any engine that takes a TorchScript
potential with positions in, energy out. The module does what the PythonForce does, in the same
units and the same embedding frame (:class:`mace.mlmm.embedding.EmbeddingFrame`):

    positions (nm, all particles), [box (nm)], [fieldmace_state]  ->  energy (kJ/mol)

QM region made whole, centred, link atoms placed, MM charges at their minimum image, optional
cutoff with C2 switching, model evaluated, energy of the chosen state (diabatic models: the
eigenvalue of the diabatic Hamiltonian). Forces come from autograd in openmm-torch.

    from mace.mlmm import create_fieldmace_mm_system
    system, spec, force = create_fieldmace_mm_system(mm_system, qm_atoms, "model.model",
                                                      topology=top, backend="torch")

    # or export once and use the file anywhere
    from mace.mlmm.torchforce import export_torchscript
    export_torchscript("model.model", spec, "fieldmace_qmmm.pt")

Installation: openmm-torch is on conda-forge only (``conda install -c conda-forge openmm-torch``),
and needs PyTorch from conda-forge too.
"""
from __future__ import annotations

import copy
import os
from typing import Optional, Union

import torch

from .embedding import EmbeddingSettings
from .force import EV_TO_KJMOL, STATE_PARAMETER, load_fieldmace_model, model_dtype
from .system import QMMMSpec

NM_TO_A = 10.0
_ORIGIN = {"centroid": 0, "com": 1, "none": 2}


def _min_image(d: torch.Tensor, box: Optional[torch.Tensor]) -> torch.Tensor:
    if box is None:
        return d
    d = d - torch.round(d[..., 2:3] / box[2, 2]) * box[2]
    d = d - torch.round(d[..., 1:2] / box[1, 1]) * box[1]
    d = d - torch.round(d[..., 0:1] / box[0, 0]) * box[0]
    return d


def _switch(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


class FieldMACETorchModule(torch.nn.Module):
    """TorchScript-able ML/MM energy: positions (nm) [, box (nm)] [, state] -> energy (kJ/mol).

    ``forward(positions, arg1, arg2)``: for a periodic system arg1 is the box and arg2 the state, for a
    non-periodic one arg1 is the state (openmm-torch passes the box only with periodic boundary
    conditions, then the global parameters in order).
    """

    def __init__(self, model: torch.nn.Module, spec: QMMMSpec, settings: EmbeddingSettings = EmbeddingSettings(),
                 state: int = 0, energy_scale: float = EV_TO_KJMOL, length_scale: float = 1.0,
                 subtract_e0: bool = False, static_graph: Union[bool, str] = "auto"):
        super().__init__()
        from e3nn.util import jit

        dt = model_dtype(model)
        self.n_qm = len(spec.qm_atoms)
        mm_ = list(spec.mm_atoms)
        extra = sorted({lk.mm_atom for lk in spec.link_atoms} - set(mm_))
        particles = list(spec.qm_atoms) + extra + mm_
        local = {g: k for k, g in enumerate(particles)}
        self.register_buffer("particles", torch.tensor(particles, dtype=torch.long))
        self.register_buffer("link_qm", torch.tensor([local[lk.qm_atom] for lk in spec.link_atoms], dtype=torch.long))
        self.register_buffer("link_mm", torch.tensor([local[lk.mm_atom] for lk in spec.link_atoms], dtype=torch.long))
        self.register_buffer("link_ratio", torch.tensor([lk.ratio for lk in spec.link_atoms], dtype=dt).reshape(-1, 1))
        self.register_buffer("mm_local", torch.tensor([local[g] for g in mm_], dtype=torch.long))
        self.register_buffer("mm_charges", torch.tensor(spec.mm_charges, dtype=dt).reshape(-1))
        w = torch.tensor(spec.qm_masses, dtype=dt).reshape(-1, 1)
        self.register_buffer("qm_weights", w / w.sum())
        self.n_links = len(spec.link_atoms)
        self.periodic = bool(spec.periodic)
        if settings.origin not in _ORIGIN:
            raise ValueError("origin must be 'centroid', 'com' or 'none'")
        self.origin = _ORIGIN[settings.origin]
        self.use_cutoff = settings.cutoff is not None
        self.cutoff = float(settings.cutoff) if settings.cutoff is not None else 0.0
        self.switch_width = max(float(settings.switch_width), 1e-12)
        self.cutoff_nearest = settings.cutoff_reference == "nearest"

        zs = [int(z) for z in model.atomic_numbers]
        z_model = spec.model_atomic_numbers
        missing = sorted(set(z_model) - set(zs))
        if missing:
            raise ValueError(f"Elements {missing} of the QM region are not in the model's z_table {zs}")
        n = len(z_model)
        node_attrs = torch.zeros(n, len(zs), dtype=dt)
        node_attrs[torch.arange(n), torch.tensor([zs.index(z) for z in z_model])] = 1.0
        self.register_buffer("node_attrs", node_attrs)
        self.register_buffer("batch", torch.zeros(n, dtype=torch.long))
        self.register_buffer("ptr", torch.tensor([0, n], dtype=torch.long))
        self.register_buffer("cell", torch.zeros(3, 3, dtype=dt))
        self.static_graph = (n <= 128) if static_graph == "auto" else bool(static_graph)
        ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        mask = ii != jj
        self.register_buffer("edge_index_static", torch.stack([ii[mask], jj[mask]]))
        self.r_max = float(model.r_max)
        self.diabatic = bool(getattr(model, "diabatic", False))
        self.n_states = int(getattr(model, "n_states", getattr(model, "n_energies", 1)))
        if not 0 <= int(state) < self.n_states:
            raise ValueError(f"state={state} but the model predicts {self.n_states} state(s)")
        self.state = int(state)
        self.energy_scale = float(energy_scale)
        self.length_scale = float(length_scale)
        e0 = 0.0
        if subtract_e0:
            with torch.no_grad():
                e0 = float(model.atomic_energies_fn(node_attrs.to(next(model.parameters()).device)).sum())
        self.e0 = e0
        m = copy.deepcopy(model).cpu()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = jit.compile(m)

    def _state(self, s: Optional[torch.Tensor]) -> int:
        if s is None:
            return self.state
        k = int(torch.round(s.reshape(-1)[0]).item())
        if k < 0:
            k = 0
        if k >= self.n_states:
            k = self.n_states - 1
        return k

    def forward(self, positions: torch.Tensor, arg1: Optional[torch.Tensor] = None,
                arg2: Optional[torch.Tensor] = None) -> torch.Tensor:
        box: Optional[torch.Tensor] = None
        st: Optional[torch.Tensor] = None
        if self.periodic:
            box = arg1
            st = arg2
        else:
            st = arg1
        dt = self.node_attrs.dtype
        ls = 10.0 * self.length_scale  # nm -> Angstrom -> model length unit
        pos = positions.index_select(0, self.particles).to(dt) * ls
        if box is not None:
            box = box.to(dt) * ls
        qm = pos[: self.n_qm]
        qm = qm[0] + _min_image(qm - qm[0], box)
        if self.origin == 1:
            centre = (self.qm_weights * qm).sum(0)
        else:
            centre = qm.mean(0)
        if self.n_links > 0:
            a = qm.index_select(0, self.link_qm)
            b = a + _min_image(pos.index_select(0, self.link_mm) - a, box)
            qm_all = torch.cat([qm, a + self.link_ratio * (b - a)], 0)
        else:
            qm_all = qm
        mm_rel = _min_image(pos.index_select(0, self.mm_local) - centre, box)
        q = self.mm_charges
        if self.use_cutoff:
            if self.cutoff_nearest:
                d = torch.cdist(mm_rel, qm_all - centre).min(dim=1).values
            else:
                d = torch.linalg.norm(mm_rel, dim=-1)
            keep = d.detach() < self.cutoff
            mm_rel = mm_rel[keep]
            r_on = self.cutoff - self.switch_width
            q = q[keep] * _switch((d[keep] - r_on) / self.switch_width)
        if self.origin == 2:
            qm_in = qm_all
            mm_in = mm_rel + centre
        else:
            qm_in = qm_all - centre
            mm_in = mm_rel
        if self.static_graph:
            edge_index = self.edge_index_static
        else:
            dd = torch.cdist(qm_in.detach(), qm_in.detach())
            n = qm_in.shape[0]
            mask = (dd < self.r_max) & ~torch.eye(n, dtype=torch.bool, device=dd.device)
            idx = torch.nonzero(mask)
            edge_index = torch.stack([idx[:, 0], idx[:, 1]])
        shifts = torch.zeros(edge_index.shape[1], 3, dtype=dt, device=qm_in.device)
        data = {
            "node_attrs": self.node_attrs,
            "positions": qm_in,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": shifts,
            "cell": self.cell,
            "batch": self.batch,
            "ptr": self.ptr,
            "mm_positions": mm_in,
            "mm_charges": q,
            "num_mm": torch.tensor([q.shape[0]], dtype=torch.long, device=q.device),
        }
        out = self.model(data, training=False, compute_force=False, compute_pc_grads=False)
        e = out["energy"]
        assert e is not None
        if self.diabatic:
            ns = self.n_states
            H = e.new_zeros((ns, ns))
            v = e.reshape(-1)
            k = ns
            for i in range(ns):
                H[i, i] = v[i]
                for j in range(i + 1, ns):
                    H[i, j] = v[k]
                    H[j, i] = v[k]
                    k += 1
            energies = torch.linalg.eigvalsh(H)
        else:
            energies = e.reshape(-1)
        return ((energies[self._state(st)] - self.e0) * self.energy_scale).to(torch.float64)


def build_torch_module(model, spec: QMMMSpec, settings: EmbeddingSettings = EmbeddingSettings(), device: str = "cpu",
                       dtype: Optional[torch.dtype] = None, **kwargs) -> torch.jit.ScriptModule:
    """FieldMACE model (path or module) + QM/MM spec -> scripted :class:`FieldMACETorchModule` on ``device``."""
    m = load_fieldmace_model(model, "cpu", dtype)
    module = torch.jit.script(FieldMACETorchModule(m, spec, settings, **kwargs))
    return module.to(device)


def export_torchscript(model, spec: QMMMSpec, path: Union[str, os.PathLike],
                       settings: EmbeddingSettings = EmbeddingSettings(), dtype: Optional[torch.dtype] = None,
                       **kwargs) -> str:
    """Write the TorchScript ML/MM potential to ``path`` (load it with torch.jit.load or openmmtorch.TorchForce)."""
    build_torch_module(model, spec, settings, "cpu", dtype, **kwargs).save(str(path))
    return str(path)


def create_fieldmace_torchforce(model, spec: QMMMSpec, settings: EmbeddingSettings = EmbeddingSettings(),
                                force_group: int = 0, state: int = 0, device: Optional[str] = None,
                                dtype: Optional[torch.dtype] = None, **kwargs):
    """openmmtorch.TorchForce evaluating FieldMACE on the QM region of ``spec``.

    The active state is the global parameter ``fieldmace_state``, as with the PythonForce backend.
    ``device`` is chosen by OpenMM's platform (CUDA platform -> GPU), not here; it is accepted for
    symmetry with the PythonForce backend and ignored.
    """
    try:
        from openmmtorch import TorchForce
    except ImportError as e:
        raise ImportError("backend='torch' needs openmm-torch: conda install -c conda-forge openmm-torch") from e
    module = build_torch_module(model, spec, settings, "cpu", dtype, state=state, **kwargs)
    force = TorchForce(module)
    force.addGlobalParameter(STATE_PARAMETER, float(state))
    force.setUsesPeriodicBoundaryConditions(bool(spec.periodic))
    force.setForceGroup(force_group)
    force.setName("FieldMACE")
    force._module = module
    return force
