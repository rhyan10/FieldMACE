#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 6: ML/MM MD of the ligand in the pocket, compared with the classical (GAFF) ligand.

The ML/MM system is the classical one with the ligand replaced by FieldMACE: the ligand's bonded
terms and charges are removed, its LJ with the protein and water stays, and FieldMACE gives the
ligand's energy in the field of the charges within 12 A (and the forces on those charges).

  1. NVT (Langevin) with FieldMACE, then NVE to check energy conservation
  2. the same length of classical MD
  3. compare the ligand's internal geometry: ring C-C and C-CH3 bond lengths, ML/MM (DFT quality)
     vs GAFF

--backend torch runs the model as an openmm-torch TorchForce (conda-forge).

Output: results/mlmm_nvt.dcd, results/classical.dcd, results/ligand_geometry.png
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openmm as mm  # noqa: E402
from openmm import app, unit  # noqa: E402

from mace.mlmm import create_fieldmace_mm_system  # noqa: E402
from mace.mlmm.dataset import load_setup  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RING = [("C1", "C2"), ("C1", "C3"), ("C2", "C5"), ("C3", "C6"), ("C5", "C7"), ("C6", "C7")]
METHYL = [("C1", "C4"), ("C7", "C8")]


def run(sim, n_steps, every, lig, names):
    """MD; returns ring and methyl bond lengths (A) every `every` steps and the ms per step."""
    idx = {nm: k for k, nm in enumerate(names)}
    ring, methyl = [], []
    t0 = time.time()
    for _ in range(n_steps // every):
        sim.step(every)
        x = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)[lig]
        ring += [np.linalg.norm(x[idx[a]] - x[idx[b]]) for a, b in RING]
        methyl += [np.linalg.norm(x[idx[a]] - x[idx[b]]) for a, b in METHYL]
    return np.array(ring), np.array(methyl), 1000 * (time.time() - t0) / max(n_steps, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(HERE, "models", "pxylene_pocket.model"))
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    p.add_argument("--nvt-ps", type=float, default=2.0)
    p.add_argument("--nve-ps", type=float, default=0.5)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--backend", choices=["python", "torch"], default="python")
    p.add_argument("--device", default=None)
    p.add_argument("--platform", default=None)
    a = p.parse_args()
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    pdb = app.PDBFile(os.path.join(a.work, "equilibrated.pdb"))
    with open(os.path.join(a.work, "system_classical.xml")) as fh:
        classical = mm.XmlSerializer.deserialize(fh.read())
    qm, settings, prep_kw = load_setup(os.path.join(a.work, "setup.json"))
    atoms = list(pdb.topology.atoms())
    names = [atoms[i].name for i in qm]
    system, spec, comp = create_fieldmace_mm_system(classical, qm, a.model, topology=pdb.topology, settings=settings,
                                                    backend=a.backend, model_kwargs=dict(device=a.device), **prep_kw)
    plat = [mm.Platform.getPlatformByName(a.platform)] if a.platform else []
    dt = a.dt_fs * unit.femtosecond
    every = max(1, int(10 / a.dt_fs))  # positions every 10 fs

    # 1. ML/MM NVT, then NVE
    sim = app.Simulation(pdb.topology, system, mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, dt),
                         *plat)
    sim.context.setPositions(pdb.positions)
    sim.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    sim.minimizeEnergy(maxIterations=100)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    sim.reporters.append(app.DCDReporter(os.path.join(out, "mlmm_nvt.dcd"), every * 10))
    n_nvt = int(a.nvt_ps * 1000 / a.dt_fs)
    ring_ml, methyl_ml, ms = run(sim, n_nvt, every, qm, names)
    print(f"ML/MM NVT {a.nvt_ps} ps ({a.backend} backend, {sim.context.getPlatform().getName()}): {ms:.1f} ms/step")
    st = sim.context.getState(getPositions=True, getVelocities=True)
    nve = app.Simulation(pdb.topology, system, mm.VerletIntegrator(dt), *plat)
    nve.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
    nve.context.setPositions(st.getPositions())
    nve.context.setVelocities(st.getVelocities())
    etot = []
    for _ in range(int(a.nve_ps * 1000 / a.dt_fs) // every):
        nve.step(every)
        s = nve.context.getState(getEnergy=True)
        etot.append((s.getPotentialEnergy() + s.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole))
    etot = np.array(etot)
    drift = np.polyfit(np.arange(len(etot)) * every * a.dt_fs / 1000, etot, 1)[0] if len(etot) > 2 else 0.0
    print(f"ML/MM NVE {a.nve_ps} ps: total energy std {etot.std():.2f} kJ/mol, drift {drift:.1f} kJ/mol/ps "
          f"({classical.getNumParticles()} atoms)")

    # 2. classical MD of the same length (2 fs, X-H constrained as in the sampling)
    csim = app.Simulation(pdb.topology, classical, mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                                                2 * unit.femtosecond), *plat)
    csim.context.setPositions(pdb.positions)
    csim.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    csim.context.setVelocitiesToTemperature(300 * unit.kelvin, 2)
    csim.reporters.append(app.DCDReporter(os.path.join(out, "classical.dcd"), 50))
    ring_mm, methyl_mm, _ = run(csim, int(a.nvt_ps * 500), 5, qm, names)

    print(f"ring C-C   ML/MM {ring_ml.mean():.4f} +- {ring_ml.std():.4f} A, GAFF {ring_mm.mean():.4f} +- {ring_mm.std():.4f} A")
    print(f"C-CH3      ML/MM {methyl_ml.mean():.4f} +- {methyl_ml.std():.4f} A, GAFF {methyl_mm.mean():.4f} +- {methyl_mm.std():.4f} A")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for x, (ml, cl, title) in zip(ax, ((ring_ml, ring_mm, "ring C-C"), (methyl_ml, methyl_mm, "C-CH3"))):
        bins = np.linspace(min(ml.min(), cl.min()), max(ml.max(), cl.max()), 40)
        x.hist(ml, bins=bins, alpha=0.6, density=True, label="FieldMACE (ML/MM)")
        x.hist(cl, bins=bins, alpha=0.6, density=True, label="GAFF (classical)")
        x.set(title=title, xlabel="bond length (A)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ligand_geometry.png"), dpi=120)
    print(f"wrote {out}/ligand_geometry.png")


if __name__ == "__main__":
    main()
