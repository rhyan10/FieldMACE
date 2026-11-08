#!/usr/bin/env python3
"""Write tutorial.ipynb in every tutorial folder (the notebook version of the README + scripts).

    python tutorials/make_notebooks.py

The notebooks run the same scripts as the README, with the same defaults, and look at the results
in between. With the environment variable FIELDMACE_TUTORIAL_QUICK=1 they use tiny settings
(this is how check_tutorials.py --notebooks tests them).
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = '''import os, subprocess, sys
import numpy as np
from IPython.display import Image, display

QUICK = os.environ.get("FIELDMACE_TUTORIAL_QUICK") == "1"   # tiny settings for automatic checks
os.environ.setdefault("OMP_NUM_THREADS", "4")

def run(cmd, env=None):
    """Run a tutorial step and show its output (stops the notebook if the step fails)."""
    print("$", cmd)
    p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env={**os.environ, **(env or {})})
    lines = [l for l in p.stdout.splitlines() if "Warning" not in l and "warnings.warn" not in l]
    print("\\n".join(lines[-60:]))
    if p.returncode:
        raise RuntimeError(f"step failed: {cmd}")'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def nb01():
    return [
        md("""
# Tutorial 01: QM/MM training data from any force field, and training

Alanine dipeptide in TIP3P water (Amber14). The QM region is the alanine side chain (CA, HA, CB, HB1–3),
cut from the backbone, with two hydrogen link atoms. This notebook runs the same steps as the scripts in this
folder; the [README](README.md) explains every option in more detail.

**Energy model.** FieldMACE predicts the QM energy **including** the interaction with the MM point charges;
everything else is the force field. The *embedding* (which charges, link atoms, cutoff and switching, origin) is
saved in `setup.json` and used again for ML/MM MD."""),
        code(SETUP),
        md("""
## Step 1: classical system and QM region

`01_prepare_system.py` reads any OpenMM input (PDB + ffxml as here, Amber prmtop, GROMACS/OPLS top, CHARMM psf),
selects the QM atoms, finds the QM–MM bonds that need link atoms, and equilibrates the system. Try
`--qm "index 6-15"` (more of the backbone) or `--cutoff 0` (all charges)."""),
        code('run(f"python 01_prepare_system.py --equil-ps {0.2 if QUICK else 20}")'),
        code('''import json
setup = json.load(open("work/setup.json"))
print(json.dumps(setup, indent=1))'''),
        md("""
## Step 2: sampling

Classical MD; every frame is written in FieldMACE's format with `frame_to_atoms` (QM + link atoms, MM point
charges in `atoms.info`, already switched by the cutoff function)."""),
        code('run(f"python 02_sample_configurations.py --n-frames {6 if QUICK else 300} --stride-ps {0.02 if QUICK else 0.5}")'),
        code('''import ase.io
frames = ase.io.read("work/frames.xyz", index=":")
a = frames[0]
print(a.get_chemical_symbols(), "(last two H are link atoms)")
print("info keys:", sorted(a.info))
n_mm = [len(f.info["mm_charges"]) for f in frames]
print("point charges per frame:", min(n_mm), "-", max(n_mm))
q = a.info["mm_charges"]; r = np.linalg.norm(a.info["mm_positions"], axis=1)
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3)); plt.scatter(r, q, s=4)
plt.xlabel("distance from the QM centroid (A)"); plt.ylabel("charge seen by FieldMACE (e)")
plt.title("cutoff 8 A, switched from 6 A"); plt.show()'''),
        md("""
## Step 3: DFT labels with PySCF

Energy (incl. the point-charge interaction), forces on QM + link atoms, forces on the point charges, in eV and Å.
This takes ~5 s per frame on a few cores. The full data set (300 frames, B3LYP/6-31G*) ships with the tutorial
in `data/*.xyz.gz`; here only a few frames are labelled again, into `work/`, to show how it works."""),
        code('run(f"python 03_label_with_pyscf.py --max-frames {2 if QUICK else 5} --basis {\'sto-3g\' if QUICK else \'6-31g*\'} --out work/demo_data --partial work/demo_partial.xyz")'),
        code('''lab = ase.io.read("work/demo_partial.xyz", index=":")
x = lab[0]
print("REF_energy (eV):", x.info["REF_energy"])
print("REF_forces (eV/A), shape", np.shape(x.info["REF_forces"]))
print("REF_mm_forces (eV/A), shape", np.shape(x.info["REF_mm_forces"]))'''),
        md("""
## Step 4: training

Two models: `PerAtomFieldEMACE` (multipole expansion of the MM potential about every QM atom, recommended) and
`FieldEMACE` (one expansion about the QM centroid). On a GPU use `DEVICE=cuda` and more epochs/channels."""),
        code('''env = {"EPOCHS": "2", "CHANNELS": "8"} if QUICK else {}
run("bash 04_train.sh", env=env)'''),
        md("## Step 5: evaluation"),
        code('run("python 05_evaluate.py")\ndisplay(Image("results/parity_ground_state.png"))'),
        md("""
## Step 6 (optional): HDF5 for large data sets

`mace_preprocess_data` converts the data once (point charges and their forces included); training then reads
the HDF5 files."""),
        code('run("bash 06_hdf5.sh", env={"EPOCHS": "1"})'),
        md("""
## Next

* Tutorial 03, part 1: ML/MM MD in OpenMM with `models/aladip_PerAtomFieldEMACE.model`.
* Tutorial 02: excited states."""),
    ]


def nb02():
    return [
        md("""
# Tutorial 02: excited states – energies, NACs, dipoles and SOCs

CH₂S in a droplet of 100 waters (SHARC 4's QM/MM example). The labels come from SHARC's LVC model of CH₂S,
which gives energies, gradients, NACs, dipoles and SOCs of S0, S1, T1, T2 in the field of the water charges.
Steps 1–2 need SHARC 4 (`$SHARC`). Details: [README](README.md)."""),
        code(SETUP + '\nassert os.environ.get("SHARC"), "set $SHARC (SHARC 4 bin directory) before starting Jupyter"'),
        md("## Step 1: sampling"),
        code('run(f"python 01_sample_configurations.py --n-frames {80 if QUICK else 1000} --equil-ps {0.5 if QUICK else 20} --n-initconds {4 if QUICK else 10}")'),
        md("""
## Step 2: LVC labels

4 spatial states (S0, S1, T1, T2). SOCs: the upper triangle of the 8×8 SOC matrix of all SHARC states with M_S,
as (re, im) pairs: 56 numbers per frame (`--soc_num=56`). The states' signs are aligned with the LVC
reference states, so the SOC vector has consistent phases; 10% gas-phase frames are added."""),
        code('run("python 02_label_with_lvc.py")'),
        code('''import ase.io
frames = ase.io.read("data/train.xyz", index=":")
x = [f for f in frames if f.info.get("config_type") == "solvated"][0]
for k in ("REF_energy", "REF_forces", "REF_mm_forces", "REF_nacs", "REF_dipoles", "REF_socs"):
    print(f"{k:14s}", np.shape(x.info[k]))
E = np.array([np.ravel(f.info["REF_energy"]) for f in frames])
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
for s, n in enumerate(["S0", "S1", "T1", "T2"]):
    plt.hist(E[:, s] - E[:, 0].mean(), bins=40, alpha=0.6, label=n)
plt.xlabel("energy - <E(S0)> (eV)"); plt.legend(); plt.show()
soc = x.info["REF_socs"].reshape(-1, 2) @ np.array([1, 1j])
H = np.zeros((8, 8), complex); H[np.triu_indices(8, 1)] = soc
print("|SOC| matrix (cm-1), upper triangle:"); print(np.round(np.abs(H) * 8065.54, 1))'''),
        md("## Step 3: training"),
        code('''env = {"EPOCHS": "2", "CHANNELS": "8"} if QUICK else {}
run("bash 03_train.sh", env=env)'''),
        md("## Step 4: evaluation (couplings compared with the best sign, as in the loss)"),
        code('run("python 04_evaluate.py")\ndisplay(Image("results/parity_excited_states.png"))'),
        md("""
## Step 5: ASE and SHARC calculators

`MACECalculator` for all properties; `SharcCalculator` for SHARC's QM.out layout (what the SHARC interface
returns)."""),
        code('run("python 05_calculators.py")\ndisplay(Image("results/scan_CS.png"))'),
        code('''# the same by hand
from mace.calculators import MACECalculator
calc = MACECalculator("models/ch2s_es.model", device="cpu", state=1)
test = [f for f in ase.io.read("data/test.xyz", index=":") if "mm_charges" in f.info]
a = test[0].copy(); a.info = {"mm_positions": test[0].info["mm_positions"], "mm_charges": test[0].info["mm_charges"]}
calc.calculate(a)
print("energies (eV):", calc.results["energies"])
print("LVC        :", np.ravel(test[0].info["REF_energy"]))
print("S1 forces (eV/A):\\n", calc.results["forces"])'''),
    ]


def nb03():
    return [
        md("""
# Tutorial 03: ML/MM molecular dynamics in OpenMM

FieldMACE as an `openmm.PythonForce`, with any force field for the rest. Part 1 needs the model from tutorial 01,
part 2 the model and `work/` files from tutorial 02. Details: [README](README.md)."""),
        code(SETUP),
        md("""
## Part 1: ground state – NVT, then NVE

`create_fieldmace_mm_system(mm_system, qm_atoms, model, topology=..., settings=..., **prepare_kwargs)` removes
QM–QM bonded terms, zeroes the QM charges (QM–QM and QM–MM Coulomb come from FieldMACE), keeps QM–MM
Lennard-Jones, places link atoms and adds the model. `setup.json` from the training data guarantees the same
embedding."""),
        code('run(f"python 01_mlmm_md.py --nvt-ps {0.01 if QUICK else 2} --nve-ps {0.01 if QUICK else 1}")'),
        code('''import matplotlib.pyplot as plt
d = np.loadtxt("results/mlmm_nve.csv", delimiter=",")
plt.figure(figsize=(5, 3))
for k, n in ((1, "E_pot"), (2, "E_kin"), (3, "E_tot")):
    plt.plot(d[:, 0], d[:, k] - d[0, k], label=n)
plt.xlabel("time (ps)"); plt.ylabel("energy - initial (kJ/mol)"); plt.legend(); plt.title("NVE"); plt.show()'''),
        md("""
## Part 2: excited states

The active state is the context parameter `fieldmace_state`; `FieldMACEStateReporter` writes all states'
energies. The script runs on S1, switches to S0 and rescales the velocities."""),
        code('''if os.path.exists("../02_excited_states/models/ch2s_es.model"):
    run(f"python 02_multistate_md.py --ps-per-segment {0.005 if QUICK else 0.25}")
    display(Image("results/multistate.png"))
else:
    print("train the model of tutorial 02 first")'''),
        md("""
## Part 3: active learning

ML/MM MD with one model, a committee evaluates the frames, the most uncertain ones are written for labelling."""),
        code('run(f"python 03_active_learning.py --n-frames {4 if QUICK else 40} --every {5 if QUICK else 100} --n-select {2 if QUICK else 10}")'),
    ]


def nb04():
    return [
        md("""
# Tutorial 04: surface hopping with SHARC 4

The CH₂S model from tutorial 02 in SHARC 4: single point, gas-phase and QM/MM (SHARC_QMMM + OpenMM)
trajectories with pysharc or `sharc.x`, compared with the LVC reference. Details: [README](README.md)."""),
        code(SETUP + '\nassert os.environ.get("SHARC"), "set $SHARC (SHARC 4 bin directory) before starting Jupyter"'),
        md("## Step 0: link the interface into SHARC and check the installation"),
        code('run("bash 00_install.sh")'),
        md("## Step 1: one QM.in through SHARC_FIELDMACE.py and SHARC_LVC.py"),
        code('run("python 01_single_point.py")'),
        code('print(open("singlepoint/FIELDMACE/QM.out").read()[:2500])'),
        md("""
## Steps 2–4: trajectories (gas phase), FieldMACE and LVC, same initial conditions

8 trajectories of 200 fs, starting in T2 (SHARC MCH state 4), which decays to T1 and mixes with S1."""),
        code('''nt, tmax = (2, 2) if QUICK else (8, 200)
for iface in ("FIELDMACE", "LVC"):
    run(f"python 02_setup_trajectories.py --mode gas --interface {iface} --ntraj {nt} --tmax {tmax}")
    run(f"bash 03_run_trajectories.sh runs/gas_{iface}")
run("python 04_analyze.py runs/gas_FIELDMACE runs/gas_LVC")
display(Image("results/populations.png"))'''),
        md("""
## QM/MM: SHARC_QMMM with FieldMACE + OpenMM

`qm-program FIELDMACE`, `mm-program openmm`; `nophases_from_interface` in the SHARC input (SHARC_QMMM does
not pass phases on)."""),
        code('''nt, tmax = (1, 1) if QUICK else (4, 100)
for iface in ("FIELDMACE", "LVC"):
    run(f"python 02_setup_trajectories.py --mode qmmm --interface {iface} --ntraj {nt} --tmax {tmax}")
    run(f"bash 03_run_trajectories.sh runs/qmmm_{iface}")
run("python 04_analyze.py runs/qmmm_FIELDMACE runs/qmmm_LVC")
display(Image("results/populations.png"))'''),
        code('print(open("runs/qmmm_FIELDMACE/TRAJ_00001/QM/QMMM.template").read())\nprint(open("runs/qmmm_FIELDMACE/TRAJ_00001/input").read())'),
    ]


def main():
    for folder, cells in (("01_data_and_training", nb01()), ("02_excited_states", nb02()),
                          ("03_openmm_mlmm_md", nb03()), ("04_sharc_dynamics", nb04())):
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = os.path.join(HERE, folder, "tutorial.ipynb")
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
