# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Small helpers shared by the SHARC tutorial scripts (SHARC 4 file formats)."""
import glob
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T02 = os.path.join(HERE, "..", "02_excited_states")
BOHR_TO_A = 0.529177210903
HARTREE_TO_EV = 27.211386245988


def sharc_dir():
    sharc = os.environ.get("SHARC")
    if not sharc or not os.path.isdir(sharc):
        raise SystemExit("Set $SHARC to the bin directory of SHARC 4 (and source $SHARC/../sharcvars.sh if you have it)")
    for path in (sharc, os.path.join(sharc, "..", "lib")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return sharc


def example_dir():
    return os.path.join(sharc_dir(), "..", "examples", "SHARC_QMMM_LVC_OPENMM")


def write_geom(path, elements, coords_bohr, masses_amu):
    """SHARC native geometry: symbol, nuclear charge, x, y, z (bohr), mass (amu)."""
    from ase.data import atomic_numbers

    with open(path, "w") as fh:
        for el, (x, y, z), m in zip(elements, coords_bohr, masses_amu):
            el = el.capitalize()
            fh.write(f" {el:2s} {float(atomic_numbers[el]):5.1f} {x:16.10f} {y:16.10f} {z:16.10f} {m:14.8f}\n")


def write_veloc(path, velocities_au):
    with open(path, "w") as fh:
        for vx, vy, vz in velocities_au:
            fh.write(f" {vx:18.12f} {vy:18.12f} {vz:18.12f}\n")


SHARC_INPUT = """printlevel   2
geomfile     "geom"
veloc        external
velocfile    "veloc"
nstates      2 0 2
actstates    2 0 2
charge       0 1 0
state        {state} mch
coeff        auto
rngseed      {seed}
tmax         {tmax}
stepsize     {dt}
nsubsteps    25
surf         sharc
coupling     {coupling}
nogradcorrect
ekincorrect  parallel_vel
reflect_frustrated none
decoherence_scheme edc
{extra}
output_format ascii
"""


def write_input(path, tmax=50.0, dt=0.5, state=2, seed=1234, extra="", coupling="ddr"):
    """coupling: "ddr" (NACs from the interface) or "overlap" (wave-function overlaps; needs an
    interface that gives them: LVC, or a diabatic FieldMACE model)."""
    with open(path, "w") as fh:
        fh.write(SHARC_INPUT.format(tmax=tmax, dt=dt, state=state, seed=seed, extra=extra, coupling=coupling))


def write_fieldmace_template(path, model, properties=("energy", "forces", "nacs", "dipoles", "socs"), device="cpu"):
    with open(path, "w") as fh:
        fh.write(f"model_file     {os.path.abspath(model)}\n"
                 f"properties     {' '.join(properties)}\n"
                 f"device         {device}\n"
                 "energy_unit    eV\n"
                 "distance_unit  Angstrom\n"
                 "nac_type       smooth\n")  # the model predicts (E_j - E_i) * NAC_ij


def copy_lvc(dest):
    os.makedirs(dest, exist_ok=True)
    for f in ("LVC.template", "LVC.resources", "V0.txt"):
        shutil.copy(os.path.join(example_dir(), "LVC", f), dest)


def read_lis(path):
    """output.lis -> array with columns step, time (fs), diag state, MCH state, E_kin, E_pot, E_tot (eV), ..."""
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]
    return np.array([[float(x) for x in r[:9]] for r in rows])


def trajectories(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "TRAJ_*")))
