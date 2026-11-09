# FieldMACE tutorials

*Author: Rhyan Barrett*

Six tutorials, one folder each. Every folder has a `README.md` (the tutorial text), numbered
scripts that do the work, and `tutorial.ipynb`, a notebook version of the same steps.

| folder | what you learn | needs |
|---|---|---|
| [`01_data_and_training`](01_data_and_training/) | QM/MM training data from any OpenMM force field (link atoms, cutoff, point charges); DFT labels with PySCF; training `PerAtomFieldEMACE` and `FieldEMACE`; evaluation; HDF5 for big data sets | PySCF (only for new labels; data are included) |
| [`02_excited_states`](02_excited_states/) | several electronic states with NACs, transition dipoles and SOCs; phases; gas-phase frames; ASE and SHARC calculators | SHARC 4 for the reference labels (LVC model) |
| [`03_openmm_mlmm_md`](03_openmm_mlmm_md/) | ML/MM MD in OpenMM with any force field: NVT/NVE, multi-state MD and state switching, active learning (manual and automated); how the interface works (energy partition, link atoms, embedding frame, exact forces); Amber/GROMACS/XML inputs; MD recipes (NPT, reporters, restarts, cost) | models from 01 and 02 |
| [`04_sharc_dynamics`](04_sharc_dynamics/) | surface hopping with SHARC 4: single points, gas phase and QM/MM (`SHARC_QMMM` + OpenMM), `sharc.x` and pysharc, comparison with the reference | SHARC 4, model from 02 |
| [`05_diabatic_states`](05_diabatic_states/) | diabatic models: learning the diabatic Hamiltonian and its couplings, diabatic labels from adiabatic data, adiabatic energies/NACs (also on MM atoms) by diagonalisation, SHARC dynamics with NACs or overlaps | SHARC 4, scripts of 02 and 04 |
| [`06_ligand_in_pocket`](06_ligand_in_pocket/) | a ligand in a protein pocket in explicit water: protein–ligand set-up with GAFF ligand parameters, embedding with ~700 charges, PySCF labels, training, the pocket's field, ML/MM MD vs GAFF | openmmtools (system), PySCF (only for new labels; data are included) |

Order: 01 → 03 part 1 (ground state) → 06 (protein–ligand), and 02 → 03 part 2 → 04 (excited states) → 05 (diabatic).

## The systems

* **Ground state (01, 03):** alanine dipeptide in 715 TIP3P waters, Amber14. The QM region is the alanine
  side chain (CA, HA, CB, HB1–3). It is cut out of the backbone, so two hydrogen **link atoms** cap the bonds
  CA–N and CA–C. The labels are B3LYP/6-31G* in the field of all MM charges within 8 Å.
* **Excited states (02, 03, 04, 05):** thioformaldehyde (CH₂S) in a droplet of 100 waters, the QM/MM example of
  SHARC 4. The labels come from SHARC's linear vibronic coupling (LVC) model of CH₂S with a multipolar density
  fit. It gives energies, gradients, NACs, dipoles and SOCs of S0, S1, T1 and T2, all of which respond to the
  water's point charges. LVC takes milliseconds per frame, so real excited-state data can be made without a
  quantum-chemistry code. It also serves as the reference the trained model is checked against in 04.

* **Protein–ligand (06):** p-xylene in the L99A cavity of T4 lysozyme, in 8400 TIP3P waters (27 835 atoms),
  Amber14 + GAFF. QM region: the ligand; ~700 point charges of protein, water and ions within 12 Å. Labels:
  PBE/def2-SVP in PySCF.

These are small so that everything runs on a laptop CPU. The workflow is the same for a protein active site
with TDDFT or CASSCF labels; only the QM step and the amount of data change.

## Setup

```bash
pip install -e ".[mlmm,test]"          # FieldMACE with OpenMM support
pip install pyscf matplotlib jupyter   # PySCF (tutorial 01 labels), plots, notebooks
bash sharc/install_sharc4.sh --env venv --prefix ~/software   # SHARC 4, if you don't have it (or --env conda)
source ~/software/sharc4/sharc_env.sh  # sets SHARC, PYTHONPATH, LD_LIBRARY_PATH; or by hand:
export SHARC=/path/to/sharc4/bin       # SHARC 4 for tutorials 02 (labels) and 04
export PYTHONPATH=$SHARC/../lib:$SHARC:$SHARC/../pysharc:$PYTHONPATH
export LD_LIBRARY_PATH=$SHARC/../lib:$LD_LIBRARY_PATH   # pysharc
```

Each script prints what it does and accepts `--help`. Training scripts read environment variables:
`DEVICE=cuda`, `EPOCHS=...`, `CHANNELS=...`. The defaults are small, so a CPU finishes in minutes; for
real models use a GPU, more epochs and wider layers.

On a cluster (e.g. ZIH), run the scripts inside a job:

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DEVICE=cuda bash 01_data_and_training/04_train.sh
```

## Checking the tutorials

`python tutorials/check_tutorials.py` runs every step of every tutorial in a temporary copy, with tiny
settings, and reports what passed. It takes about 25 minutes on 2 CPU cores. Steps that need SHARC are
skipped without `$SHARC`. `--full` uses the real settings.

## Run times (2 CPU cores, defaults)

| step | time |
|---|---|
| 01: system + 300 frames of classical MD (150 ps) | ~30 min (GPU: a few min) |
| 01: PySCF labels for 300 frames | ~20–40 min (data are included) |
| 01: training, each model, 200 epochs | ~15 min |
| 02: sampling 1000 frames + LVC labels | ~3 min |
| 02: training, 120 epochs | ~45 min |
| 03: ML/MM MD, 3 ps (6000 steps) | ~6 min on CPU (~55 ms/step) |
| 04: 8 trajectories × 200 fs gas phase, FieldMACE / LVC (pysharc) | ~5 min / ~1 min |
| 04: 4 trajectories × 100 fs QM/MM, FieldMACE or LVC (pysharc) | ~6 min each |
| 05: diabatic labels / training, 100 epochs | ~3 min / ~70 min |
| 05: dynamics, gas 8 × 200 fs + QM/MM 4 × 100 fs, FieldMACE and LVC | ~25 min |
| 06: system + 40 ps of classical MD | ~45 min (GPU: minutes) |
| 06: PySCF labels, 176 calculations | ~1.5–2.5 h (data are included) |
| 06: training, 150 epochs / ML/MM MD, 2 ps | ~25 min / ~15 min |
