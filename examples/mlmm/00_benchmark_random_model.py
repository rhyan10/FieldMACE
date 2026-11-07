#!/usr/bin/env python
"""
Test the interface without a trained model.

Builds a randomly initialised FieldMACE model of realistic size, puts it on
alanine dipeptide in TIP3P water (Amber, from openmmtools), and checks:
  1. NVE energy conservation (the interface's forces are the exact gradient)
  2. speed in ms/step on the chosen device

The dynamics are NOT physical (random weights) - this only tests the plumbing.

    python 00_benchmark_random_model.py --device cuda
    python 00_benchmark_random_model.py --device cuda --channels 128 --qm sidechain --cutoff 12
"""
import argparse
import time

import numpy as np
import openmm as mm
import torch
from openmm import unit

from mace.mlmm import EmbeddingSettings, create_fieldmace_mm_system

ap = argparse.ArgumentParser()
ap.add_argument("--model-type", default="PerAtomFieldEMACE", choices=["PerAtomFieldEMACE", "FieldEMACE"])
ap.add_argument("--channels", type=int, default=64, help="hidden_irreps = Nx0e + Nx1o")
ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--platform", default=None, help="OpenMM platform (default: CUDA if --device cuda, else CPU)")
ap.add_argument("--qm", default="whole", choices=["whole", "sidechain"],
                help="whole dipeptide (22 atoms) or ALA side chain with 2 link atoms")
ap.add_argument("--cutoff", type=float, default=None, help="point-charge cutoff in Angstrom (default: all)")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--static-graph", default="auto", help="auto | true | false")
a = ap.parse_args()

torch.set_default_dtype(getattr(torch, a.dtype))
from mace.tools.testing import build_field_model  # noqa: E402

model = build_field_model(a.model_type, channels=a.channels, radial_MLP=[64, 64, 64])
torch.save(model, "random_fieldmace.model")
print(f"random {a.model_type}: {sum(p.numel() for p in model.parameters()):,} parameters, {a.dtype}, {a.device}")

import openmmtools.testsystems as T  # noqa: E402

ts = T.AlanineDipeptideExplicit()
qm = list(range(22)) if a.qm == "whole" else [8, 9, 10, 11, 12, 13]
system, spec, comp = create_fieldmace_mm_system(
    ts.system, qm, "random_fieldmace.model", topology=ts.topology,
    settings=EmbeddingSettings(cutoff=a.cutoff),
    model_kwargs=dict(device=a.device, static_graph={"auto": "auto", "true": True, "false": False}[a.static_graph.lower()]))
print(f"QM atoms {spec.n_qm} (+{len(spec.link_atoms)} link), point charges {len(spec.mm_atoms)}")

plat = a.platform or ("CUDA" if a.device.startswith("cuda") else "CPU")
integ = mm.VerletIntegrator(0.25 * unit.femtosecond)
ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(plat))
ctx.setPositions(ts.positions)
mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 100)
ctx.setVelocitiesToTemperature(300 * unit.kelvin, 1)

integ.step(10)  # warm-up
E = []
t0 = time.time()
for _ in range(a.steps // 10):
    integ.step(10)
    st = ctx.getState(getEnergy=True)
    E.append((st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole))
dt = (time.time() - t0) / (10 * len(E))
E = np.array(E)
drift = np.polyfit(np.arange(len(E)), E, 1)[0] * len(E)
print(f"\n{1000 * dt:.1f} ms/step  ({86400 / dt * 0.5e-6:.2f} ns/day at 0.5 fs)")
print(f"NVE total energy: std {E.std():.3f} kJ/mol, drift {drift:+.3f} kJ/mol over {10 * len(E)} steps")
print("PASS" if E.std() < 2.0 and abs(drift) < 2.0 else "CHECK: energy not conserved (try --dtype float64)")
