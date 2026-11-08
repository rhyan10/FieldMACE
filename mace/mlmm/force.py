# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
OpenMM force that evaluates a trained FieldMACE model (FieldEMACE or PerAtomFieldEMACE)
on the QM region, embedded in the point charges of the MM region.

Implemented with ``openmm.PythonForce`` (OpenMM >= 8.3): the model runs in PyTorch, needs no
TorchScript export, and forces on QM atoms, on MM point charges and through link atoms /
periodic imaging / charge switching come from one backward pass.

Speed: the QM graph is built once with all atom pairs ("static graph"). MACE's radial cutoff
makes pairs beyond r_max contribute exactly zero, so this gives identical results while
avoiding a neighbour-list rebuild (and a GPU->CPU synchronisation) every step.

Properties: if the model predicts them, ``computation.last`` also holds "dipoles" [S+P, 3]
(e*Angstrom), "socs" [P, 2] and "nacs" [N_model_atoms, P, 3] of the last evaluation.

Multi-state: the electronic state that drives the dynamics is the global context parameter
``fieldmace_state`` and can be changed during a run with
``context.setParameter("fieldmace_state", k)``. The energies of all states from the last
evaluation are available in ``computation.last`` and can be written with
:class:`FieldMACEStateReporter`.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import openmm as mm
import torch
from openmm import unit

from .embedding import EmbeddingFrame, EmbeddingSettings
from .system import QMMMSpec, prepare_qmmm_system

EV_TO_KJMOL = 96.48533212331002  # 1 eV in kJ/mol
NM_TO_A = 10.0
STATE_PARAMETER = "fieldmace_state"


def load_fieldmace_model(model: Union[str, os.PathLike, torch.nn.Module], device: str = "cpu",
                         dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    if not isinstance(model, torch.nn.Module):
        model = torch.load(model, map_location=device, weights_only=False)
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if not hasattr(model, "multipole_max_ell"):
        raise TypeError(f"{type(model).__name__} is not a FieldMACE field model")
    return model


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


class FieldMACEComputation:
    """Callable passed to openmm.PythonForce: State -> (energy, forces)."""

    def __init__(
        self,
        model: Union[str, os.PathLike, torch.nn.Module],
        spec: QMMMSpec,
        settings: EmbeddingSettings = EmbeddingSettings(),
        state: int = 0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        energy_scale: float = EV_TO_KJMOL,  # model energy unit -> kJ/mol
        length_scale: float = 1.0,  # model length unit per Angstrom (1 for Angstrom models)
        subtract_e0: bool = False,
        static_graph: Union[bool, str] = "auto",
    ):
        self.model_path = model if not isinstance(model, torch.nn.Module) else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_arg = dtype
        self.model = load_fieldmace_model(model, self.device, dtype)
        self.dtype = model_dtype(self.model)
        self.spec = spec
        self.settings = settings
        self.frame = EmbeddingFrame(spec, settings)
        self.n_states = int(getattr(self.model, "n_energies", 1))
        self.state = self._check_state(state)
        self.energy_scale = float(energy_scale)
        self.length_scale = float(length_scale)
        self.subtract_e0 = bool(subtract_e0)

        zs = [int(z) for z in self.model.atomic_numbers]
        z_model = spec.model_atomic_numbers
        missing = sorted(set(z_model) - set(zs))
        if missing:
            raise ValueError(f"Elements {missing} of the QM region are not in the model's z_table {zs}")
        n = len(z_model)
        node_attrs = torch.zeros(n, len(zs), dtype=self.dtype)
        node_attrs[torch.arange(n), torch.tensor([zs.index(z) for z in z_model])] = 1.0
        dev = self.device
        self.node_attrs = node_attrs.to(dev)
        self.r_max = float(self.model.r_max)
        self.e0 = 0.0
        if subtract_e0:
            with torch.no_grad():
                self.e0 = float(self.model.atomic_energies_fn(self.node_attrs).sum())
        self.batch = torch.zeros(n, dtype=torch.long, device=dev)
        self.ptr = torch.tensor([0, n], dtype=torch.long, device=dev)
        self.cell = torch.zeros(3, 3, dtype=self.dtype, device=dev)
        self.static_graph = (n <= 128) if static_graph == "auto" else bool(static_graph)
        self.static_graph_arg = static_graph
        if self.static_graph:
            ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
            mask = ii != jj
            self.edge_index_static = torch.stack([ii[mask], jj[mask]]).to(dev)
            self.shifts_static = torch.zeros(self.edge_index_static.shape[1], 3, dtype=self.dtype, device=dev)
        self.last = {}

    def _check_state(self, state) -> int:
        s = int(round(float(state)))
        if not 0 <= s < self.n_states:
            raise ValueError(f"state={state} but the model predicts {self.n_states} state(s)")
        return s

    # --------------------------------------------------------------- #
    def build_input(self, qm_pos: torch.Tensor, mm_pos: torch.Tensor, mm_q: torch.Tensor):
        """FieldMACE input dict for one configuration (flat MM layout)."""
        if self.static_graph:
            edge_index, shifts = self.edge_index_static, self.shifts_static
        else:
            with torch.no_grad():
                d = torch.cdist(qm_pos, qm_pos)
                n = len(qm_pos)
                mask = (d < self.r_max) & ~torch.eye(n, dtype=torch.bool, device=d.device)
                sender, receiver = torch.nonzero(mask, as_tuple=True)
            edge_index = torch.stack([sender, receiver])
            shifts = torch.zeros(edge_index.shape[1], 3, dtype=self.dtype, device=self.device)
        return {
            "node_attrs": self.node_attrs,
            "positions": qm_pos,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": shifts,
            "cell": self.cell,
            "batch": self.batch,
            "ptr": self.ptr,
            "mm_positions": mm_pos,
            "mm_charges": mm_q,
            "num_mm": torch.tensor([mm_q.shape[0]], dtype=torch.long, device=self.device),
        }

    def evaluate(self, pos_A: np.ndarray, box_A: Optional[np.ndarray] = None, state: Optional[int] = None):
        """Energy (model units) of ``state`` and its gradient wrt the subset positions (model units / A)."""
        s = self.state if state is None else self._check_state(state)
        pos = torch.as_tensor(np.asarray(pos_A), dtype=self.dtype).to(self.device).requires_grad_(True)
        box = None
        if box_A is not None and self.spec.periodic:
            box = torch.as_tensor(np.asarray(box_A), dtype=self.dtype).to(self.device)
        qm_pos, mm_pos, mm_q, sel = self.frame(pos, box)
        ls = self.length_scale
        data = self.build_input(qm_pos * ls, mm_pos * ls, mm_q)
        out = self.model(data, training=False, compute_force=False, compute_pc_grads=False)
        energies = out["energy"].reshape(-1)
        (grad,) = torch.autograd.grad(energies[s], pos)
        e_all = energies.detach().cpu().numpy() - self.e0
        self.last = {"energies": e_all, "state": s, "n_mm_charges": int(mm_q.shape[0])}
        for key in ("dipoles", "socs", "nacs"):  # excited-state properties, if the model has them
            if out.get(key) is not None:
                v = out[key].detach().cpu().numpy()
                self.last[key] = v[0] if key != "nacs" else v
        return float(e_all[s]), grad.detach().cpu().numpy()

    def __call__(self, state: mm.State):
        params = state.getParameters()
        s = self.state
        if STATE_PARAMETER in params:
            s = self._check_state(params[STATE_PARAMETER])
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer) * NM_TO_A
        box = None
        if self.spec.periodic:
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer) * NM_TO_A
        e, g = self.evaluate(np.asarray(pos), None if box is None else np.asarray(box), state=s)
        energy = e * self.energy_scale  # kJ/mol
        forces = -g * self.energy_scale * NM_TO_A  # kJ/mol/nm
        return energy, forces

    # PythonForce pickles the computation for XmlSerializer: store the path, not the weights
    def __getstate__(self):
        if self.model_path is None:
            raise TypeError("Pass the model as a file path to make the force serialisable")
        return {
            "model_path": self.model_path, "spec": self.spec, "settings": self.settings,
            "state": self.state, "device": self.device, "dtype": self.dtype_arg,
            "energy_scale": self.energy_scale, "length_scale": self.length_scale,
            "subtract_e0": self.subtract_e0, "static_graph": self.static_graph_arg,
        }

    def __setstate__(self, d):
        self.__init__(d["model_path"], d["spec"], d["settings"], d["state"], d["device"], d["dtype"],
                      d["energy_scale"], d["length_scale"], d["subtract_e0"], d.get("static_graph", "auto"))


def create_fieldmace_force(
    model, spec: QMMMSpec, settings: EmbeddingSettings = EmbeddingSettings(), force_group: int = 0, **kwargs
) -> mm.PythonForce:
    """openmm.PythonForce evaluating FieldMACE on the QM region of ``spec``.

    The active state is the global parameter ``fieldmace_state`` (default: ``state``).
    """
    if not hasattr(mm, "PythonForce"):
        raise RuntimeError("openmm.PythonForce requires OpenMM >= 8.3")
    comp = FieldMACEComputation(model, spec, settings, **kwargs)
    force = mm.PythonForce(comp, {STATE_PARAMETER: float(comp.state)})
    force.setParticles(comp.frame.particles)
    force.setUsesPeriodicBoundaryConditions(spec.periodic)
    force.setForceGroup(force_group)
    force.setName("FieldMACE")
    force._computation = comp  # keep a Python reference for introspection
    return force


def create_fieldmace_mm_system(
    system: mm.System,
    qm_atoms,
    model,
    topology=None,
    settings: EmbeddingSettings = EmbeddingSettings(),
    force_group: int = 0,
    model_kwargs: Optional[dict] = None,
    **prepare_kwargs,
):
    """One call: any OpenMM System + QM atom list + FieldMACE model -> ML/MM System.

    Returns (system, spec, computation). ``prepare_kwargs`` are passed to
    :func:`prepare_qmmm_system` (link_atoms, link_charge_scheme, ...).
    """
    new, spec = prepare_qmmm_system(system, qm_atoms, topology=topology, **prepare_kwargs)
    force = create_fieldmace_force(model, spec, settings, force_group, **(model_kwargs or {}))
    new.addForce(force)
    return new, spec, force._computation


class FieldMACEStateReporter:
    """OpenMM reporter writing the FieldMACE energies of all electronic states.

    Columns: step, time (ps), active state, E_0 ... E_{n-1} (eV, from the last force evaluation).
    """

    def __init__(self, file, report_interval: int, computation: FieldMACEComputation):
        self._own = isinstance(file, str)
        self._out = open(file, "w") if self._own else file
        self._interval = int(report_interval)
        self._comp = computation
        n = computation.n_states
        self._out.write("#step,time_ps,active_state," + ",".join(f"E{k}_eV" for k in range(n)) + "\n")

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return {"steps": steps, "periodic": None, "include": ["energy"]}

    def report(self, simulation, state):
        last = self._comp.last
        if not last:
            return
        t = state.getTime().value_in_unit(unit.picosecond)
        es = ",".join(f"{e:.8f}" for e in last["energies"])
        self._out.write(f"{simulation.currentStep},{t:.6f},{last['state']},{es}\n")
        self._out.flush()

    def __del__(self):
        if getattr(self, "_own", False):
            try:
                self._out.close()
            except Exception:  # noqa: BLE001
                pass
