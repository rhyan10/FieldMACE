#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 2: set up surface-hopping trajectories for SHARC 4.

  --mode gas    CH2S alone (gas phase); the QM program is FIELDMACE (or LVC)
  --mode qmmm   CH2S in the water droplet with SHARC_QMMM: QM program FIELDMACE (or LVC),
                MM program OpenMM (SHARC_OPENMM), electrostatic embedding, subtractive scheme
                (the QM/MM input files come from SHARC's example SHARC_QMMM_LVC_OPENMM)

  --interface LVC  sets up the SAME trajectories with the LVC reference, so the populations of
                   FieldMACE and of the reference can be compared (04_analyze.py)

Initial conditions (positions + velocities) are the classical-MD snapshots of tutorial 02
(work/initconds.npz); every trajectory starts in T2 (--state 4, SHARC's MCH numbering), which
decays to T1 and mixes with S1 within 200 fs. (Started in S1, CH2S stays there for picoseconds.)
For production use SHARC's own tools instead: wigner.py / amber2init.py -> setup_init.py ->
excite.py -> setup_traj.py (setup_traj.py knows the FIELDMACE interface).

Directory layout created (runs/<mode>_<interface>/TRAJ_00001, ...):
  input  geom  veloc                      SHARC input, geometry (bohr), velocities (bohr/atu)
  QM/runQM.sh                             how sharc.x calls the interface each step
  QM/FIELDMACE.template                   (gas)  model file, properties, device
  QM/QMMM.template QMMM.table MML/ MMS/ FM/   (qmmm) SHARC_QMMM inputs, FM/FIELDMACE.template
"""
import argparse
import os
import shutil

import numpy as np

from sharc_tools import T02, copy_lvc, example_dir, sharc_dir, write_fieldmace_template, write_geom, write_input, \
    write_veloc

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["gas", "qmmm"], default="gas")
    p.add_argument("--interface", choices=["FIELDMACE", "LVC"], default="FIELDMACE")
    p.add_argument("--model", default=os.path.join(T02, "models", "ch2s_es.model"))
    p.add_argument("--initconds", default=os.path.join(T02, "work", "initconds.npz"))
    p.add_argument("--ntraj", type=int, default=8)
    p.add_argument("--tmax", type=float, default=200.0, help="fs")
    p.add_argument("--dt", type=float, default=0.5, help="fs")
    p.add_argument("--state", type=int, default=4,
                   help="initial state, SHARC MCH numbering: 1 S0, 2 S1, 3/5/7 T1 (M_S=-1/0/+1), 4/6/8 T2 "
                        "(default 4: T2, which decays within 200 fs; from S1, intersystem crossing takes ps)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--runs", default=os.path.join(HERE, "runs"))
    a = p.parse_args()
    sharc_dir()
    ic = np.load(a.initconds)
    n_ic = len(ic["positions_bohr"])
    if a.ntraj > n_ic:
        raise SystemExit(f"only {n_ic} initial conditions in {a.initconds} (tutorial 02 step 1: --n-initconds)")
    qm = list(ic["qm_atoms"])
    run = os.path.join(a.runs, f"{a.mode}_{a.interface}")
    shutil.rmtree(run, ignore_errors=True)

    for k in range(a.ntraj):
        traj = os.path.join(run, f"TRAJ_{k + 1:05d}")
        qmdir = os.path.join(traj, "QM")
        os.makedirs(qmdir)
        if a.mode == "gas":
            sel = qm
            extra = ""
            iface = a.interface
            if a.interface == "FIELDMACE":
                write_fieldmace_template(os.path.join(qmdir, "FIELDMACE.template"), a.model, device=a.device)
            else:
                copy_lvc(qmdir)
        else:
            sel = list(range(len(ic["elements"])))
            # SHARC_QMMM (SHARC 4.0/4.1) does not pass the QM program's phases on -> sharc.x must not ask
            extra = "nophases_from_interface"
            iface = "QMMM"
            ex = example_dir()
            shutil.copy(os.path.join(ex, "QMMM.table"), qmdir)
            for sub in ("MML", "MMS"):
                shutil.copytree(os.path.join(ex, sub), os.path.join(qmdir, sub))
                for f in os.listdir(os.path.join(qmdir, sub)):  # SHARC_OPENMM reads OPENMM.template
                    if f.startswith("OPENMM_") and f.endswith(".template"):
                        shutil.move(os.path.join(qmdir, sub, f), os.path.join(qmdir, sub, "OPENMM.template"))
            sub = "FM" if a.interface == "FIELDMACE" else "LVC"
            with open(os.path.join(qmdir, "QMMM.template"), "w") as fh:
                fh.write(f"qmmm_table QMMM.table\nqm-program {a.interface}\nmm-program openmm\nembedding subtractive\n"
                         f"mms-dir ./MMS\nmml-dir ./MML\nqm-dir ./{sub}\nmm_dipole False\n")
            if a.interface == "FIELDMACE":
                os.makedirs(os.path.join(qmdir, "FM"))
                write_fieldmace_template(os.path.join(qmdir, "FM", "FIELDMACE.template"), a.model, device=a.device)
            else:
                copy_lvc(os.path.join(qmdir, "LVC"))
        write_geom(os.path.join(traj, "geom"), ic["elements"][sel], ic["positions_bohr"][k][sel], ic["masses_amu"][sel])
        write_veloc(os.path.join(traj, "veloc"), ic["velocities_au"][k][sel])
        write_input(os.path.join(traj, "input"), tmax=a.tmax, dt=a.dt, state=a.state, seed=1000 + k, extra=extra)
        with open(os.path.join(qmdir, "runQM.sh"), "w") as fh:
            fh.write(f"cd QM\n$SHARC/SHARC_{iface}.py QM.in >> QM.log 2>> QM.err\nerr=$?\nexit $err\n")
        os.chmod(os.path.join(qmdir, "runQM.sh"), 0o755)
        with open(os.path.join(traj, "interface"), "w") as fh:  # for 03_run_trajectories.sh (pysharc)
            fh.write(iface + "\n")
    print(f"set up {a.ntraj} trajectories in {os.path.relpath(run, HERE)}/ ({a.mode}, {a.interface}, "
          f"{len(sel)} atoms, {a.tmax} fs)")


if __name__ == "__main__":
    main()
