# FieldMACE

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![tests](https://github.com/rhyan10/FieldMACE/actions/workflows/tests.yml/badge.svg)](https://github.com/rhyan10/FieldMACE/actions/workflows/tests.yml)
[![docs](https://github.com/rhyan10/FieldMACE/actions/workflows/docs.yml/badge.svg)](https://rhyan10.github.io/FieldMACE/)

FieldMACE is a deep-learning framework for QM/MM data.

* **Model.** It extends the Message Passing Atomic Cluster Expansion (MACE) architecture. A multipole expansion encodes the electrostatic field that the MM point charges create at the QM region.
* **Outputs.** It predicts energies (ground and excited states), forces on the QM atoms and forces on the MM point charges. It can also predict non-adiabatic couplings, permanent and transition dipoles, and spin-orbit couplings, or a **diabatic Hamiltonian** whose diagonalisation gives all of these.
* **OpenMM.** It ships an OpenMM interface for ML/MM molecular dynamics with any fixed-charge force field OpenMM can build: Amber, CHARMM, OPLS, GAFF/OpenFF and others. Implicit solvent and polarisable force fields (AMOEBA, Drude) are not supported as the MM environment. The model runs as a `PythonForce` or as an openmm-torch `TorchForce`, and it can be exported as one TorchScript file.
* **SHARC 4.** Surface hopping, in the gas phase and in QM/MM with `SHARC_QMMM`.
* **Active learning.** `fieldmace_active_learning` automates committee training, ML/MM MD, selection and labelling.

Documentation: [rhyan10.github.io/FieldMACE](https://rhyan10.github.io/FieldMACE/) (built from this README and the tutorials). Changes: [CHANGELOG.md](CHANGELOG.md).

For the main MACE parameters see [ACEsuit/mace](https://github.com/ACEsuit/mace).

Version 2 changes the default model behaviour; read [What changed in version 2](#what-changed-in-version-2) before comparing with results from version 1.

**Tutorials.** [`tutorials/`](tutorials/) has six step-by-step tutorials, each a folder with a README, runnable scripts and a notebook:

1. QM/MM training data from any force field, and training;
2. excited states (NACs, dipoles, SOCs);
3. ML/MM MD in OpenMM, and automated active learning;
4. surface hopping with SHARC 4;
5. diabatic states: learning the diabatic Hamiltonian, and SHARC dynamics with it;
6. a ligand in a protein pocket (p-xylene in T4 lysozyme, explicit water).

---

## Installation

```bash
git clone https://github.com/rhyan10/FieldMACE.git
cd FieldMACE
python -m venv venv && source venv/bin/activate       # or a conda env, Python >= 3.10
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the CUDA build for your driver
pip install -e ".[mlmm,test]"     # mlmm: OpenMM >= 8.3 interface, test: pytest + openmmtools
python scripts/check_env.py       # add --gpu inside a GPU job
pytest -q tests/                  # ~10 min on 2 CPU cores
```

The `mlmm` extra installs OpenMM. For GPU MD you also need the OpenMM CUDA plugin, e.g. `pip install "openmm[cuda12]"`.
Without `mlmm`, everything except `mace.mlmm` works.

## Quick start

About 10 minutes on a laptop CPU, with the data that ship with tutorial 01 (DFT labels of an alanine
side chain in water):

```bash
cd tutorials/01_data_and_training
python 01_prepare_system.py                          # solvated system, QM region = side chain (~2 min)
MODELS=PerAtomFieldEMACE EPOCHS=20 bash 04_train.sh  # train on the shipped data (~2 min; the tutorial uses 200 epochs)
python 05_evaluate.py                                # errors on the test set, results/parity_ground_state.png
cd ../03_openmm_mlmm_md
python 01_mlmm_md.py --nvt-ps 0.5 --nve-ps 0.2       # ML/MM MD in OpenMM with the new model
```

Then follow [the tutorials](tutorials/) for your own system. If something is wrong with your data or
settings, FieldMACE stops with a message that names the file, the frame and the option to change; see
[Troubleshooting](docs/troubleshooting.md) for the common ones.

---

## Data format (extended XYZ)

Each frame holds the QM atoms. Everything QM/MM-specific goes in `atoms.info`:

| key | shape | unit | |
|---|---|---|---|
| `mm_positions` | `(M, 3)` | Å | MM point-charge positions |
| `mm_charges` | `(M,)` | e | MM point charges |
| `REF_energy` | scalar or `(n_states,)` / `(1, n_states)` | eV | QM energy **including** the interaction with the point charges |
| `REF_forces` | `(N, 3)` or `(N, n_states, 3)` | eV/Å | forces on the QM atoms (`atoms.info` or `atoms.arrays`) |
| `REF_mm_forces` | `(M, 3)` or `(M, n_states, 3)` | eV/Å | forces on the point charges (optional) |
| `config_*_weight` | scalar | | per-frame loss weights (`energy`, `forces`, `mm_forces`) |

The number of point charges M may differ between frames. Frames without `mm_positions` and `mm_charges` are gas-phase configurations. Padding is no longer needed.
A missing label gets weight 0, so that term is skipped for that frame only.

## Training

```bash
mace_run_train --name=protein --train_file=train.xyz --valid_fraction=0.1 --model=PerAtomFieldEMACE \
  --multipole_max_ell=1 --n_energies=1 --E0s=average --r_max=5.0 --batch_size=40 --lr=0.01 \
  --correlation=3 --max_num_epochs=1000 --ema --ema_decay=0.99 --default_dtype=float32 --device=cuda \
  --hidden_irreps="32x0e + 32x1o" --MLP_irreps="32x0e" --num_radial_basis=8 --num_interactions=2 \
  --forces_weight=100 --mm_forces_weight=100 --energy_weight=300 --error_table=EnergyNacsDipoleMAE \
  --scheduler=ReduceLROnPlateau --lr_factor=0.5 --scheduler_patience=10
```

**Models**

* `PerAtomFieldEMACE` (recommended): expands the MM potential about every QM atom.
* `FieldEMACE`: a single multipole expansion about the QM centroid.

**Starting from a MACE foundation model.** `--foundation_model` initialises a field model with the MACE layers of a pretrained model: the node embedding, the radial basis, the interaction and product blocks. The field interactions, the readouts and any excited-state heads start from scratch. The architecture options (`r_max`, channels, `max_ell`, layers, correlation order, radial basis) are replaced by the foundation model's, with a warning, and every element of your data must be one the foundation model knows.

| `--foundation_model` | model | where it comes from | status |
|---|---|---|---|
| `medium` | MACE-MP-0 medium (2023-12-03) | bundled in `mace/calculators/foundations_models/` | tested |
| `small`, `large` | MACE-MP-0 small / large (2023) | downloaded on first use | worked in FieldMACE 1, not re-tested |
| `small_off`, `medium_off`, `large_off` | MACE-OFF23 (ASL license) | downloaded on first use | worked in FieldMACE 1, not re-tested |
| a file path | a `.model` file | | only models made with the same (2023) MACE version |

**Newer foundation models do not work yet.** That includes MACE-MPA-0, MACE-OMAT-0, MACE-MP-0b3, MACE-MATPES, MACE-OFF24 and the multi-head MH models. FieldMACE is built on a 2023 version of MACE, which lacks the newer blocks; loading MACE-MPA-0, for example, fails with `Can't get attribute 'RealAgnosticDensityInteractionBlock'`. Supporting them means updating the MACE core and is planned on a development branch. The ASE helpers `mace_mp()`, `mace_off()` and `mace_anicc()` in `mace.calculators` have the same limits; they are upstream MACE code, not FieldMACE models.

**Field-specific options**

| option | default | |
|---|---|---|
| `--multipole_max_ell` | 1 | highest multipole order *l* |
| `--n_energies` | 1 | number of electronic states |
| `--mm_forces_weight` | `= forces_weight` | weight of the point-charge forces; `0` skips computing them (faster) |
| `--multipole_embedding` | `equivariant` | `legacy` reproduces version 1 (not rotation invariant) |
| `--num_multipole_basis` | 8 | Gaussian basis size of the multipole embedding |
| `--expansion_centre` | `centroid` | FieldEMACE only; `origin` reproduces version 1 (not translation invariant) |
| `--compute_nacs` / `--compute_dipoles` / `--compute_socs` | `False` | excited-state properties, see below |
| `--nacs_weight` / `--dipoles_weight` / `--socs_weight` | 100 / 1 / 1 | their loss weights |
| `--nacs_key` / `--dipoles_key` / `--socs_key` | `smooth_nacs` / `REF_dipoles` / `REF_socs` | their keys in `atoms.info` |
| `--soc_num` | – | SOCs as a free-form vector of this length (X-MACE layout), see below |
| `--diabatic` / `--diabatic_coupling_weight` | `False` / 10 | diabatic model, see [Diabatic models](#diabatic-models) |
| `--energy_key` / `--forces_key` | `REF_energy` / `REF_forces` | label keys (`REF_mm_forces` is fixed) |

### Excited-state properties

With S = `--n_energies` states and P = S(S−1)/2 state pairs, ordered (0,1), (0,2), …, (1,2), …:

| property | label in `atoms.info` | model output | how it is predicted |
|---|---|---|---|
| NACs | `[N, P, 3]` (or `[P, N, 3]`) | `nacs [N, P, 3]` | per-atom equivariant vectors (l = 1), one per pair |
| dipoles | `[S + P, 3]`: S permanent, then P transition | `dipoles [B, S+P, 3]` | sum of per-atom equivariant vectors |
| SOCs | `[P]` real, `[P, 2]` (re, im) or complex `[P]` | `socs [B, P, 2]` | sum of per-atom invariant scalars |
| SOCs, `--soc_num=L` | any `L` real numbers per frame | `socs [B, 1, L]` | as above, one phase per frame |

`--soc_num` takes SOCs in a free layout, as in X-MACE: for SHARC, the upper triangle of the M_S-resolved SOC matrix, either real or as (re, im) pairs. See [SHARC 4](#surface-hopping-with-sharc-4).

* **Arbitrary phases.** The sign of every coupling between two states (NACs, transition dipoles, SOCs) is arbitrary in QM data. The losses and error metrics therefore use the better of the two signs, chosen separately for each configuration and pair. Labels do not need consistent phases.
* **Units.** Use whatever units your data has.
* **Smooth NACs (recommended).** NACs grow as 1/ΔE near crossings and then dominate the loss. Train on smooth NACs instead: (E_j − E_i) · NAC_ij for each pair i < j, stored under the NAC key. The SHARC interface converts them back with `nac_type smooth`.
* **Missing labels.** Frames without a property label get weight 0 for that term.
* **Field dependence.** All properties respond to the MM field, like the energies.
* **SOCs are rotation-invariant scalars.** This is the usual ML convention, as in SchNarc and SPaiNN. If your SOCs are resolved by triplet M_S component in a fixed laboratory frame, that is an approximation.
* **Architecture change.** Models that predict NACs or dipoles keep l = 1 features in the last layer as well, because MACE normally reduces the last layer to scalars.

```bash
mace_run_train ... --n_energies=3 --compute_nacs=True --compute_dipoles=True --compute_socs=True \
  --nacs_weight=100 --dipoles_weight=10 --socs_weight=10
```

Large data sets can be preprocessed to HDF5 first:

```bash
mace_preprocess_data --train_file=train.xyz --h5_prefix=h5/ --r_max=5.0 --valid_fraction=0.1 --compute_statistics
mace_run_train --train_file=h5/train/train_0.h5 --valid_file=h5/val/val_0.h5 --statistics_file=h5/statistics.json ...
```

**Evaluation**

```bash
mace_eval_configs --configs=test.xyz --model=protein.model --output=pred.xyz
```

`pred.xyz` contains `MACE_energy` (plus `MACE_energies` for several states), `MACE_forces`, `MACE_mm_forces` and, if the model predicts them, `MACE_nacs`, `MACE_dipoles` and `MACE_socs`. For a diabatic model these are the raw outputs (the elements of H); the ASE calculator below gives the adiabatic quantities.

**ASE**

```python
from mace.calculators import MACECalculator
calc = MACECalculator("protein.model", device="cuda", state=0)  # several paths -> committee
atoms.info["mm_positions"], atoms.info["mm_charges"] = mm_xyz, mm_q
atoms.calc = calc
atoms.get_potential_energy()
calc.results["energies"], calc.results["mm_forces"]
calc.results["nacs"], calc.results["dipoles"], calc.results["socs"]   # if the model predicts them
```

For a committee, each member's couplings are sign-aligned to the first model before averaging.

**SHARC** (QM.out layout in atomic units; see [SHARC 4](#surface-hopping-with-sharc-4) for running dynamics):

```python
from mace.calculators.sharc_calculator import SharcCalculator
sc = SharcCalculator("CH2O", "model.model", device="cuda", states=[2, 0, 2],
                     properties=["energy", "forces", "nacs", "dipoles", "socs"])
qm_out = sc.calculate(coords_bohr, mm_positions_angstrom, mm_charges)
# h (energies + SOCs, Hermitian), grad, grad_pc, nacdr (antisymmetric), dm
```

* **Phase continuity.** Couplings are sign-aligned with the previous step, so they change smoothly along a trajectory.
* **Unit conversions.** `nac_to_au`, `dipole_to_au` and `soc_to_au` set them; the defaults assume eV and Å.

### Diabatic models

`--diabatic=True --n_energies=S` trains a model of the diabatic Hamiltonian H(R, MM charges). It has S + S(S−1)/2 energy outputs, ordered `[H_00 … H_S−1,S−1, H_01, H_02, …, H_12, …]`, each with forces on the atoms and on the point charges (`REF_energy`, `REF_forces`, `REF_mm_forces` in that layout). Adiabatic quantities follow from diagonalisation, exactly:

```
H = U diag(E) Uᵀ,   dE_k/dR = (Uᵀ dH/dR U)_kk,   NAC d_kl = (Uᵀ dH/dR U)_kl / (E_l − E_k)
```

* **Symmetry.** Couplings between states of opposite mirror symmetry (n→π\* and π→π\*, A1 and A2, …) change sign when the environment is mirrored, so they are pseudoscalars. Diabatic models therefore add a pseudoscalar readout to the usual scalar one. `--diabatic_coupling_weight` (default 10) gives the small couplings enough weight in the loss.
* **No NAC labels needed.** The model never sees NACs, so there are no phase problems. The 1/ΔE singularity at crossings is exact, not learned. NACs come out on the MM atoms too.
* **Labels from adiabatic data.** `mace.tools.diabatic.labels_from_adiabatic(E, U, grad, nac, …)` turns energies, gradients, NACs and a diabatisation U from any QM method into labels.
* **Where it works.** The ASE calculator, the SHARC interface (with `coupling ddr` or `coupling overlap`) and the OpenMM force all take diabatic models. See [tutorial 05](tutorials/05_diabatic_states/).

---

## Surface hopping with SHARC 4

`sharc/SHARC_FIELDMACE.py` is a SHARC 4 interface (`SHARC_FAST`, like SPaiNN's). It works with every way SHARC 4 calls an interface:

| how | command | notes |
|---|---|---|
| `sharc.x`, one interface call per step | `QM/runQM.sh`: `$SHARC/SHARC_FIELDMACE.py QM.in` | Python and the model reload every step |
| pysharc, in-process | `$SHARC/driver.py -i FIELDMACE input` | model stays loaded; same trajectory as `sharc.x`, ~50× faster here |
| QM/MM with `SHARC_QMMM` | `qm-program FIELDMACE` in `QMMM.template`, e.g. with `mm-program openmm` | MM charges enter FieldMACE as point charges; `grad_pc` goes back to the MM atoms |
| point charges in `QM.in` | `point_charges pc.txt` | |

**Installing SHARC 4.** `sharc/install_sharc4.sh` installs SHARC 4 with `sharc.x` and pysharc. It works in a conda environment (compilers and libraries from conda-forge) or in a Python virtual environment (system compilers or cluster modules). It also installs FieldMACE into that environment, links the interface, writes `sharc_env.sh` and runs a test trajectory:

```bash
bash sharc/install_sharc4.sh --env conda --conda-env sharc4 --prefix ~/software          # conda
bash sharc/install_sharc4.sh --env venv --venv ../venv-fieldmace --prefix ~/software      # existing venv
source ~/software/sharc4/sharc_env.sh                                                      # in every shell / job
```

See `bash sharc/install_sharc4.sh --help` for all options. On the ZIH clusters (TU Dresden), `bash sharc/install_sharc4_zih.sh` does the same with the right modules and adds them to `sharc_env.sh`.

**By hand.** Link the interface into SHARC's `bin` directory, with FieldMACE in the same Python environment:

```bash
ln -s /path/to/FieldMACE/sharc/SHARC_FIELDMACE.py $SHARC/
cp /path/to/FieldMACE/sharc/FIELDMACE.template QM/        # then edit model_file and properties
```

`FIELDMACE.template`:

```
model_file     /path/to/model.model      # or "modelpath" in FIELDMACE.resources
properties     energy forces nacs dipoles socs
device         cuda
energy_unit    eV                        # units the model was trained in
distance_unit  Angstrom
nac_type       smooth                    # the model predicts (E_j - E_i) * NAC_ij; "nac" for plain NACs
# soc_selection_rules true               # default: SOCs that vanish by spin symmetry are set to zero
```

**States.** The SHARC `states` line (e.g. `states 2 0 2`, 8 states with M_S) is matched to the model:

* **Spatial model.** `n_energies` = number of spatial states (here 4: S0, S1, T1, T2). Every M_S component of a triplet gets the triplet's energy and gradient. NACs couple states of the same multiplicity and M_S; all other NACs are zero. SOCs need a model trained with `--soc_num`: the upper triangle of the nmstates × nmstates SOC matrix, X-MACE layout. Use nmstates(nmstates−1)/2 real values (here 28), or nmstates(nmstates−1) values as (re, im) pairs (here 56).
* **M_S-resolved model.** `n_energies` = nmstates (here 8). NACs, dipoles and SOCs are taken pair by pair.
* **Other combinations.** These raise an error that says what does not match. With singlets only, SOC requests return zeros.

**SOC selection rules.** By default, SOC elements that vanish exactly are set to zero instead of taking the model's small, noisy values:

* between two singlets;
* between the M_S components of one multiplet;
* for |ΔM_S| > 1.

Noise between exactly degenerate M_S components would otherwise mix them and cause spurious hops.

**Smooth NACs.** With `nac_type smooth` the interface divides the model's (E_j − E_i) · NAC_ij by the model's own energy gap.

**What the interface answers.** `h` (with SOCs if `soc` is requested), `grad`, `grad_pc`, `nacdr`, `dm` (zeros and a warning without a dipole model) and `phases`; diabatic models also `nacdr_pc` and `overlap`. Requests the model cannot answer (`nacdr` without NACs, SOCs between multiplicities without a SOC model) fail with a clear error.

**Phases.** FieldMACE aligns the signs of NACs, transition dipoles and SOCs with the previous step itself, so it reports all phases as +1. The reference is kept in memory and in `SAVE/FIELDMACE_previous.npz`, so file-based runs and restarts continue with consistent signs.

**QM/MM.** `SHARC_QMMM` in SHARC 4.0 does not pass the QM program's phases on. Put `nophases_from_interface` in the SHARC `input` file for `sharc.x` QM/MM runs, or `sharc.x` stops with `'QMout' object has no attribute 'phases'`. Nothing is lost, because FieldMACE keeps the signs consistent itself.

**Tests.** `tests/sharc` runs against a real SHARC 4 installation and is skipped without one:

```bash
export SHARC=/path/to/sharc4/bin PYTHONPATH=$SHARC/../lib:$SHARC:$SHARC/../pysharc LD_LIBRARY_PATH=$SHARC/../lib
pytest tests/sharc
```

They cover:

* `QM.in` → `QM.out` against the ASE calculator, with and without point charges;
* spatial, M_S-resolved and singlet-only models;
* error messages;
* finite-difference checks of `grad` and `grad_pc`, and of the full `SHARC_QMMM` (FieldMACE + OpenMM) gradient;
* phase continuity between calls;
* `setup_traj.py` hooks;
* surface hopping with `sharc.x` and pysharc, gas phase and QM/MM, with float64 and float32 models.

[Tutorial 04](tutorials/04_sharc_dynamics/) runs a trained model through all of this and compares the dynamics with the reference.

---

## ML/MM molecular dynamics in OpenMM (`mace.mlmm`)

```python
from openmm import app, unit
import openmm as mm
from mace.mlmm import create_fieldmace_mm_system, EmbeddingSettings, FieldMACEStateReporter

prm = app.AmberPrmtopFile("complex.prmtop"); crd = app.AmberInpcrdFile("complex.inpcrd")
system = prm.createSystem(nonbondedMethod=app.PME, constraints=app.HBonds)   # any force field
qm = [a.index for a in prm.topology.atoms() if a.residue.name == "LIG"]

system, spec, comp = create_fieldmace_mm_system(
    system, qm, "protein.model", topology=prm.topology,
    settings=EmbeddingSettings(cutoff=10.0, switch_width=2.0),   # the same as for the training data
    model_kwargs=dict(device="cuda", state=0))                   # (save_setup / load_setup keep them together)

sim = app.Simulation(prm.topology, system, mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.5*unit.femtosecond))
sim.context.setPositions(crd.positions); sim.context.setPeriodicBoxVectors(*crd.boxVectors)
sim.reporters.append(FieldMACEStateReporter("states.csv", 100, comp))   # energies of all states
sim.step(10000)
sim.context.setParameter("fieldmace_state", 1)        # switch the driving electronic state
```

**What the interface does**

* **Electrostatic embedding.** QM charges are zeroed in the MM force field, and QM–QM pairs are excluded from every nonbonded force. Bonded terms that involve only QM atoms are removed (including RB torsions, CMAP and Custom* forces). QM–MM Lennard-Jones is kept.
* **Link atoms.** H link atoms cap covalent QM–MM bonds. The `Z0` to `Z3` options choose which MM charges near the link are left out.
* **Virtual sites.** These are handled, including the TIP4P M-site.
* **Periodic boxes.** Point charges use the minimum image, with an optional smooth cutoff.
* **Training data from the same setup.** `examples/mlmm/01–03` sample with a classical force field, write QM inputs (ORCA example), collect labels and run ML/MM MD. They use the same embedding frame (`setup.json`).
* **Two backends.** The default is `openmm.PythonForce` (OpenMM ≥ 8.3, nothing else to install). With `backend="torch"`, the same model runs as an `openmmtorch.TorchForce` in libtorch, with no Python per step. It needs openmm-torch and PyTorch from conda-forge; the PyPI openmm-torch fails with recent PyTorch. Both give identical energies and forces.
* **Other programs.** `mace.mlmm.export_torchscript(model, spec, "fieldmace_qmmm.pt")` writes the whole ML/MM potential (embedding frame, link atoms, cutoff, model, state selection) as one TorchScript file: positions in nm (plus the box, plus the state) in, energy in kJ/mol out. Any engine that takes a TorchScript potential can run it, as can OpenMM's C++ API.

`examples/mlmm/00_benchmark_random_model.py --device cuda` measures ms/step and checks energy conservation without a trained model.

---

## What changed in version 2

**Correctness fixes**

* **Rotation invariance.** The multipole embedding applied a scalar non-linearity to each (l, m) component separately, which breaks rotational invariance for `multipole_max_ell ≥ 1`: energies changed by about 0.05–0.1 eV under rigid rotation (random weights). The new default embedding is equivariant: only the norm of each l-block enters non-linearly.
* **Translation invariance.** `FieldEMACE` expanded the MM potential about (0, 0, 0), so energies depended on where the system sat in the box. It now expands about the QM centroid of each frame.
* **Untrained parameters.** The optimiser never contained `linear_up`, which maps multipole features into the network, or the attention blocks. They stayed at their random initial values. All parameters are now trained, and a check enforces this.
* **Loss.** The training loss crashed for every field model (`None` dipoles/NACs), and the forces/NAC conditions were shifted by one line. Old `(N,3)` force labels were compared against `(N,1,3)` predictions, which gave an N×N broadcast. All of this is fixed, and per-frame weights are honoured.
* **Command-line booleans.** `type=bool` command-line flags turned the string "False" into True. `--n_energies` had no default.
* **Silent export failure.** The TorchScript export silently failed (`except: pass`). Field models now compile, and failures are logged.
* **Pipeline bugs.** Fixed the HDF5 pipeline (MM fields missing, wrong keyword), `preprocess_data` (missing arguments, numpy ≥ 2 statistics), E0 averaging over unlabelled frames, and energy statistics with several states.
* **Attention models removed.** `AttentionFieldEMACE` (could not be constructed) and `AttentionPerAtomFieldEMACE` are gone, since neither is part of the publication. FieldMACE now has `FieldEMACE` (one multipole expansion) and `PerAtomFieldEMACE`.
* **NACs, dipoles and SOCs.** These were never produced by the field models (the readouts returned `None`). They are now implemented, see [Excited-state properties](#excited-state-properties).

**Version 2.4.1**

* **Input checks.** Mistakes in data, settings, QM/MM set-ups and SHARC templates stop with a message that names the file, frame and option to change; see [Troubleshooting](docs/troubleshooting.md).
* **Fix.** The ASE calculator recomputes when only the MM charges change (before, ASE's cache returned the old field's results).

**Version 2.4**

* **Diabatic models.** See [Diabatic models](#diabatic-models). The SHARC interface diagonalises H every step and also gives NACs on MM atoms and wave-function overlaps.
* **OpenMM.** A TorchForce backend (`backend="torch"`) and `export_torchscript`. OpenMM 8.3–8.5 work again.
* **Active learning.** `fieldmace_active_learning`, an automated, resumable loop: committee → ML/MM MD → selection → labels.
* **Tutorials.** 05 (diabatic states) and 06 (a ligand in a protein pocket).
* **Infrastructure.** CI (GitHub Actions), PyPI releases on tags, and a documentation site. See [CHANGELOG.md](CHANGELOG.md).

**Performance and functionality**

* **Variable charge counts.** Frames in a batch may have different numbers of point charges, or none (flat MM layout with `num_mm`). The old padded `[B, M, 3]` input is still accepted.
* **Faster maths.** The multipole maths is vectorised (no Python loops over graphs or (l, m)). A dense path handles single configurations in MD. Forces on QM atoms and point charges come from one backward pass per state.
* **TorchScript.** Field models compile with TorchScript.
* **Tutorials (2.3).** [`tutorials/`](tutorials/) has four tutorials (data and training, excited states, OpenMM ML/MM, SHARC 4), each with a README, scripts and a notebook, all checked by `tutorials/check_tutorials.py`.
* **SHARC fixes and options (2.3).**
  * pysharc received zero gradients from float32 models; the calculator now always returns double precision.
  * `nac_type smooth` for models trained on (E_j − E_i) · NAC_ij.
  * SOC selection rules (on by default).
  * `sharc/install_sharc4.sh` installs SHARC 4 (`sharc.x` + pysharc) in a conda environment or a virtual environment and tests it. It also fixes an upstream input-reader bug (an `intent(out)` status left unset) that made pysharc read no input once PyTorch was imported.
* **SHARC 4.** The new interface `sharc/SHARC_FIELDMACE.py` replaces the unfinished `mace/calculators/sharcinterface.py`. It works with `sharc.x`, pysharc and `SHARC_QMMM` (e.g. with OpenMM), and handles multiplicities and point-charge gradients. SOCs can also be learned as a free-form vector (`--soc_num`, X-MACE layout).
* **Repository cleanup.** Removed `build/`, `*.pyc`, editor swap files and the egg-info. Added `.gitignore`. Declared every dependency (tqdm, python-hostlist, torchmetrics, configargparse, h5py, …). The package is `fieldmace`; the import name is still `mace`.
* **Tests.** 160+ tests: invariances and equivariances (including NACs and dipoles), finite-difference forces, batching, padding, gas phase, coincident charges, single atoms, float32, TorchScript, every loss term including the phase-free ones, CLI training (xyz, HDF5, transfer learning from the bundled MACE-MP-0 medium, all properties), the ASE and SHARC calculators, SHARC 4 (`sharc.x`, pysharc, `SHARC_QMMM` with OpenMM) and OpenMM ML/MM.

**Old models and data**

* **Old models.** `FieldEMACE` and `PerAtomFieldEMACE` models saved with version 1 load unchanged in legacy mode and give the same numbers as before. This was checked to machine precision, see `tests/legacy/`. Version 1 attention models no longer load.
* **Old data.** Old datasets (padded or not, `(N,3)` or `(N,1,3)` labels) load as before.
* **Reproducing version 1.** To train a model that behaves like version 1, use `--multipole_embedding=legacy --expansion_centre=origin`.

## License

MIT
