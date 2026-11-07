# FieldMACE

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

FieldMACE is a deep-learning framework for QM/MM data.

* **Model.** It extends the Message Passing Atomic Cluster Expansion (MACE) architecture. A multipole expansion encodes the electrostatic field that the MM point charges create at the QM region.
* **Outputs.** It predicts energies (ground and excited states), forces on the QM atoms and forces on the MM point charges.
* **OpenMM.** It ships an OpenMM interface for ML/MM molecular dynamics with any OpenMM force field: Amber, CHARMM, OPLS, GAFF/OpenFF and others.

For the main MACE parameters see [ACEsuit/mace](https://github.com/ACEsuit/mace).

Version 2 changes the default model behaviour; read [What changed in version 2](#what-changed-in-version-2) before comparing with results from version 1.

---

## Installation

```bash
git clone https://github.com/rhyan10/FieldMACE.git
cd FieldMACE
python -m venv venv && source venv/bin/activate       # or a conda env, Python >= 3.10
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the CUDA build for your driver
pip install -e ".[mlmm,test]"     # mlmm: OpenMM >= 8.3 interface, test: pytest + openmmtools
python scripts/check_env.py       # add --gpu inside a GPU job
pytest -q tests/                  # ~5 min on CPU
```

The `mlmm` extra installs OpenMM. For GPU MD you also need the OpenMM CUDA plugin, e.g. `pip install "openmm[cuda12]"`.
Without `mlmm`, everything except `mace.mlmm` works.

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
* `FieldEMACE`: a single expansion about the QM centroid.
* `AttentionPerAtomFieldEMACE`: each QM atom attends over its point charges.

Transfer learning works as before with `--foundation_model=medium_off` (or `small`, `medium`, `large`, `*_off`, or a path).

**Field-specific options**

| option | default | |
|---|---|---|
| `--multipole_max_ell` | 1 | highest multipole order *l* |
| `--n_energies` | 1 | number of electronic states |
| `--mm_forces_weight` | `= forces_weight` | weight of the point-charge forces; `0` skips computing them (faster) |
| `--multipole_embedding` | `equivariant` | `legacy` reproduces version 1 (not rotation invariant) |
| `--num_multipole_basis` | 8 | Gaussian basis size of the multipole embedding |
| `--expansion_centre` | `centroid` | FieldEMACE only; `origin` reproduces version 1 (not translation invariant) |

Large data sets can be preprocessed to HDF5 first:

```bash
mace_preprocess_data --train_file=train.xyz --h5_prefix=h5/ --r_max=5.0 --valid_fraction=0.1 --compute_statistics
mace_run_train --train_file=h5/train/train_0.h5 --valid_file=h5/val/val_0.h5 --statistics_file=h5/statistics.json ...
```

**Evaluation**

```bash
mace_eval_configs --configs=test.xyz --model=protein.model --output=pred.xyz
```

`pred.xyz` contains `MACE_energy` (plus `MACE_energies` for several states), `MACE_forces` and `MACE_mm_forces`.

**ASE**

```python
from mace.calculators import MACECalculator
calc = MACECalculator("protein.model", device="cuda", state=0)  # several paths -> committee
atoms.info["mm_positions"], atoms.info["mm_charges"] = mm_xyz, mm_q
atoms.calc = calc
atoms.get_potential_energy()
calc.results["energies"], calc.results["mm_forces"]
```

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
    settings=EmbeddingSettings(cutoff=None),          # must match how the training data was made
    model_kwargs=dict(device="cuda", state=0))

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
* **Implementation.** It runs through `openmm.PythonForce`. OpenMM's TorchScript route (`TorchForce`) was tested and is not used: the PyPI openmm-torch 1.5 fails at runtime with recent PyTorch.

`examples/mlmm/00_benchmark_random_model.py --device cuda` measures ms/step and checks energy conservation without a trained model.

---

## What changed in version 2

**Correctness fixes**

* **Rotation invariance.** The multipole embedding applied a scalar non-linearity to each (l, m) component separately, which breaks rotational invariance for `multipole_max_ell ≥ 1`: energies changed by about 0.05–0.1 eV under rigid rotation (random weights). The new default embedding is equivariant: only the norm of each l-block enters non-linearly.
* **Translation invariance.** `FieldEMACE` expanded the MM potential about (0, 0, 0), so energies depended on where the system sat in the box. It now expands about the QM centroid of each frame.
* **Untrained parameters.** The optimiser never contained `linear_up`, which maps multipole features into the network, or the attention blocks. They stayed at their random initial values. All parameters are now trained, and a check enforces this.
* **Loss.** The training loss crashed for every field model (`None` dipoles/NACs), and the forces/NAC conditions were shifted by one line. Old `(N,3)` force labels were compared against `(N,1,3)` predictions, which gave an N×N broadcast. All of this is fixed, and per-frame weights are honoured.
* **Attention padding.** `AttentionPerAtomFieldEMACE` normalised its attention over padded zero charges, so results depended on the padding. Zero charges are now ignored.
* **Command-line booleans.** `type=bool` command-line flags turned the string "False" into True. `--n_energies` had no default.
* **Silent export failure.** The TorchScript export silently failed (`except: pass`). Field models now compile, and failures are logged.
* **Pipeline bugs.** Fixed the HDF5 pipeline (MM fields missing, wrong keyword), `preprocess_data` (missing arguments, numpy ≥ 2 statistics), E0 averaging over unlabelled frames, and energy statistics with several states.
* **`AttentionFieldEMACE` removed.** It could not be constructed (NameError) and now raises a clear error.
* **NACs and dipoles.** Their readouts were never implemented for the field models. `compute_nacs` / `compute_dipoles` now raise `NotImplementedError` instead of silently returning `None`.

**Performance and functionality**

* **Variable charge counts.** Frames in a batch may have different numbers of point charges, or none (flat MM layout with `num_mm`). The old padded `[B, M, 3]` input is still accepted.
* **Faster maths.** The multipole maths is vectorised (no Python loops over graphs or (l, m)). A dense path handles single configurations in MD. Forces on QM atoms and point charges come from one backward pass per state.
* **TorchScript.** Field models compile with TorchScript.
* **Repository cleanup.** Removed `build/`, `*.pyc`, editor swap files and the egg-info. Added `.gitignore`. Declared every dependency (tqdm, python-hostlist, torchmetrics, configargparse, h5py, …). The package is now `fieldmace` 2.0.0; the import name is still `mace`.
* **Tests.** 100+ tests: invariances, finite-difference forces, batching, padding, gas phase, coincident charges, single atoms, float32, TorchScript, every loss term, CLI training (xyz and HDF5), the ASE calculator and OpenMM ML/MM.

**Old models and data**

* **Old models.** Models saved with version 1 load unchanged in legacy mode and give the same numbers as before. This was checked to machine precision, see `tests/legacy/`.
* **Old data.** Old datasets (padded or not, `(N,3)` or `(N,1,3)` labels) load as before.
* **Reproducing version 1.** To train a model that behaves like version 1, use `--multipole_embedding=legacy --expansion_centre=origin`.

## License

MIT
