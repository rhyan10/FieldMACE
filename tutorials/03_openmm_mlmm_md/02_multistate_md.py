#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Part 2: excited-state ML/MM MD with the 4-state model from tutorial 02 (CH2S in water).

  * The state that drives the dynamics is the global context parameter "fieldmace_state";
    it can be changed at any time: context.setParameter("fieldmace_state", k).
  * FieldMACEStateReporter writes the energies of ALL states every report interval.
  * computation.last holds the latest NACs, dipoles and SOCs (if the model predicts them).

The script runs NVE on S1 (state 1), then makes a "hop" to S0 by hand. A hop changes the
potential energy, so the velocities must be rescaled to conserve the total energy - the
script shows the total energy with and without rescaling. (Real surface hopping, with hopping
probabilities from NACs and SOCs, is SHARC's job: tutorial 04.)

Outputs: results/multistate.csv (state energies), results/multistate.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openmm as mm  # noqa: E402
from openmm import app, unit  # noqa: E402

from mace.mlmm import STATE_PARAMETER, FieldMACEStateReporter, create_fieldmace_mm_system  # noqa: E402
from mace.mlmm.dataset import load_setup  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T02 = os.path.join(HERE, "..", "02_excited_states")
KJ_PER_EV = 96.48533212331002


def total_energy(ctx):
    st = ctx.getState(getEnergy=True)
    return (st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole) / KJ_PER_EV


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(T02, "models", "ch2s_es.model"))
    p.add_argument("--work", default=os.path.join(T02, "work"), help="tutorial 02 work dir (system_mm.xml, ...)")
    p.add_argument("--ps-per-segment", type=float, default=0.25)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(a.work, "system_mm.xml")) as fh:
        mm_system = mm.XmlSerializer.deserialize(fh.read())
    pdb = app.PDBFile(os.path.join(a.work, "droplet.pdb"))
    qm_atoms, settings, prep_kw = load_setup(os.path.join(a.work, "setup.json"))
    system, spec, comp = create_fieldmace_mm_system(mm_system, qm_atoms, a.model, topology=pdb.topology,
                                                    settings=settings, model_kwargs=dict(device=a.device, state=1),
                                                    **prep_kw)
    print(f"{comp.n_states} states; starting in state {comp.state} (S1)")

    dt = a.dt_fs * unit.femtosecond
    sim = app.Simulation(pdb.topology, system, mm.VerletIntegrator(dt), mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 3)
    sim.step(1)  # the system's CMMotionRemover removes the centre-of-mass motion of the random velocities here
    n = int(a.ps_per_segment * 1000 / a.dt_fs)
    rep = FieldMACEStateReporter(os.path.join(out, "multistate.csv"), max(1, n // 50), comp)
    sim.reporters.append(rep)

    # --- segment 1: NVE on S1 -----------------------------------------------------------------
    e_tot = [total_energy(sim.context)]
    sim.step(n)
    e_tot.append(total_energy(sim.context))
    last = comp.last
    print(f"after {a.ps_per_segment} ps on S1: E(S0..T2) = {np.round(last['energies'], 3)} eV, "
          f"total energy change {1000 * (e_tot[-1] - e_tot[0]):.1f} meV")
    if "nacs" in last:
        print(f"  |model NAC output S0-S1| = {np.linalg.norm(last['nacs'][:, 0]):.3f} (tutorial 02 model: smooth NAC, "
              f"(E1 - E0) * NAC in eV/A), "
              f"|transition dipole S0-S1| = {np.linalg.norm(last['dipoles'][4]):.3f} e A, "
              f"max |SOC| = {1000 * np.abs(last['socs']).max():.1f} meV")

    # --- hop S1 -> S0 --------------------------------------------------------------------------
    e_before = total_energy(sim.context)
    dE = last["energies"][1] - last["energies"][0]  # eV released by the hop
    sim.context.setParameter(STATE_PARAMETER, 0)
    e_after_no_rescale = total_energy(sim.context)
    ekin = sim.context.getState(getEnergy=True).getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / KJ_PER_EV
    scale = np.sqrt((ekin + dE) / ekin)  # put the released energy into kinetic energy (all atoms here)
    v = sim.context.getState(getVelocities=True).getVelocities(asNumpy=True)
    sim.context.setVelocities(v * scale)
    e_after = total_energy(sim.context)
    print(f"hop S1 -> S0: dE = {dE:.3f} eV; total energy {e_before:.3f} -> {e_after_no_rescale:.3f} eV without "
          f"rescaling, {e_after:.3f} eV with velocities x {scale:.4f}")

    # --- segment 2: NVE on S0 -----------------------------------------------------------------
    sim.step(n)
    e_tot.append(total_energy(sim.context))
    print(f"after {a.ps_per_segment} ps on S0: total energy change since the hop "
          f"{1000 * (e_tot[-1] - e_after):.1f} meV")
    del rep  # flush

    d = np.genfromtxt(os.path.join(out, "multistate.csv"), delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, name in enumerate(["S0", "S1", "T1", "T2"]):
        ax.plot(d["time_ps"], d[f"E{k}_eV"] - d["E0_eV"][0], label=name)
    ax.plot(d["time_ps"], np.where(d["active_state"] == 1, d["E1_eV"], d["E0_eV"]) - d["E0_eV"][0], "k:", lw=2,
            label="active state")
    ax.set(xlabel="time (ps)", ylabel="energy (eV)", title="ML/MM MD on S1, hop to S0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "multistate.png"), dpi=120)
    print("wrote results/multistate.csv and results/multistate.png")


if __name__ == "__main__":
    main()
