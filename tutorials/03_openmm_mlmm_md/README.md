# Tutorial 03: ML/MM molecular dynamics in OpenMM

*Author: Rhyan Barrett*

FieldMACE runs inside OpenMM as an `openmm.PythonForce` (OpenMM ≥ 8.3), or as an `openmmtorch.TorchForce`. The model describes the QM region in
the field of the MM point charges, and everything else is the force field. Any force field OpenMM reads
works.

```
01_mlmm_md.py          ground state: alanine side chain in water (model from tutorial 01), NVT + NVE
02_multistate_md.py    excited states: CH2S in water (model from tutorial 02), switching states, all-state energies
03_active_learning.py  find configurations the model is unsure about, for the next round of training
04_active_learning_loop.sh  the automated loop: train committee, ML/MM MD, select, label (PySCF), repeat
```

---

## Part 1: ground-state ML/MM MD

```bash
python 01_mlmm_md.py                          # model from tutorial 01, 2 ps NVT + 1 ps NVE
python 01_mlmm_md.py --device cuda --nvt-ps 100
```

The core is one call:

```python
from mace.mlmm import create_fieldmace_mm_system
from mace.mlmm.dataset import load_setup

qm_atoms, settings, prep_kw = load_setup("../01_data_and_training/data/setup.json")
system, spec, comp = create_fieldmace_mm_system(mm_system, qm_atoms, "model.model", topology=topology,
                                                settings=settings, **prep_kw)
```

What it does to the force-field system (a copy; the original is untouched):

| term | QM–QM | QM–MM | MM–MM |
|---|---|---|---|
| Coulomb | FieldMACE | FieldMACE (point charges) | force field |
| Lennard-Jones | removed (FieldMACE) | force field | force field |
| bonds, angles, torsions | removed if all atoms are QM | force field | force field |

* **Link atoms.** Link atoms are placed on the QM–MM bonds every step. Their forces go to the QM and MM
  atoms by the chain rule, and the MM charges hidden by the link scheme are left out.
* **Cutoff.** The cutoff with its switching function and periodic imaging around the QM region are part of
  the computational graph, so the forces are exact derivatives of the energy. The NVE run checks this.
* **Speed.** Forces on QM atoms, link atoms and all point charges come from **one backward pass** on the
  GPU. For QM regions up to 128 atoms the model graph is built once ("static graph"), so no neighbour list
  is rebuilt each step.
* **Same embedding as training.** `setup.json` stores QM atoms, link atoms and their charge scheme,
  cutoff, switching width and origin, as used for the training data. Use it, or the model is evaluated in
  a frame it was not trained in.

**What to look at.**

* **Speed.** The script prints ms/step: ~55 ms on 2 CPU cores, where the model dominates. For
  comparison, the ML/MM benchmark of `scripts/check_env.py --gpu` ran at ~22 ms/step on an H100 MIG slice
  (ZIH Capella).
* **NVE energy conservation.** Expect a total-energy fluctuation much smaller than the kinetic-energy
  fluctuation, and no systematic drift. Here: std 1.0 kJ/mol against 78 kJ/mol for the kinetic energy,
  with a drift of −2.5 kJ/mol/ps for a float32 model at 0.5 fs. If it drifts more, use a smaller time
  step, a `--default_dtype=float64` model, or check that the cutoff in `setup.json` matches training.

**Starting MD from random velocities.** If the system has a `CMMotionRemover` (most force fields add one),
the first step removes the centre-of-mass motion of the random velocities, about 3/2 kT. Measure energy
conservation from the first step on, not from the velocities you set.

**Units inside OpenMM.** The model works in eV and Å. The force converts to kJ/mol and nm (and back for
positions); `energy_scale` and `length_scale` in `model_kwargs` cover models trained in other units.

**Serialization.** The force pickles the model *path* (not the weights), so `XmlSerializer` and
checkpoints work when the model is given as a file.

**TorchForce backend.** `python 01_mlmm_md.py --backend torch` runs the model as an `openmmtorch.TorchForce`
instead: TorchScript in libtorch, with no Python call per step. The device follows the OpenMM platform (CUDA
platform, GPU). Install it from conda-forge, together with PyTorch from conda-forge:

```bash
conda install -c conda-forge openmm-torch "pytorch=*=cuda*"   # or =cpu* without a GPU
```

```python
system, spec, force = create_fieldmace_mm_system(mm_system, qm_atoms, "model.model", topology=top,
                                                 settings=settings, backend="torch")
```

Energies and forces are identical to the PythonForce, and `fieldmace_state` works the same way. Only
`computation.last`, the per-step energies of all states, is not available, so for multi-state reporting use
the default backend. On a CPU both backends take the same time (the model dominates). The TorchForce helps
on GPUs and for running without Python.
`mace.mlmm.export_torchscript(model, spec, "fieldmace_qmmm.pt")` writes the complete ML/MM potential as one
TorchScript file for other programs.

**OpenMM versions.** OpenMM 8.6 passes the force only its own particles. 8.3–8.5 pass all of them, and the
force then picks its own. Both work.

## Part 2: excited states, switching states

```bash
python 02_multistate_md.py                    # model from tutorial 02
```

* **State.** The state that drives the dynamics is the context parameter `fieldmace_state`:
  `context.setParameter("fieldmace_state", 1)`.
* **All-state energies.** `FieldMACEStateReporter(file, interval, comp)` writes the energies of all states
  at every report (CSV: step, time, active state, E0, E1, ...).
* **Couplings.** `comp.last` holds the latest `energies` and, if the model predicts them, `nacs`,
  `dipoles` and `socs`. `nacs` is whatever the model was trained on: for the tutorial-02 model, smooth NACs
  (E_j − E_i) · NAC_ij.

The script runs NVE on S1, then switches to S0 by hand and rescales the velocities so the total energy is
conserved. The total energy changes by 0.2 meV in 0.25 ps on either state. After the hop, the released
~1.9 eV heats the system and the C=S stretch, which shows up as the large oscillations of the excited
states.

![state energies](figures/multistate.png)

Rescaling the velocities at a hop is what a surface-hopping code does too, with the hops decided from
NACs and SOCs. For real nonadiabatic dynamics use SHARC ([tutorial 04](../04_sharc_dynamics/)), which runs FieldMACE
with OpenMM as the MM engine through `SHARC_QMMM`.

## Part 3: active learning

```bash
python 03_active_learning.py
```

The first model was trained on classical-MD geometries, but ML/MM MD visits the model's own distribution.
The script runs ML/MM MD a bit hotter than the data (400 K) and stores frames in exactly the training
format (`frame_to_atoms` with the force's `EmbeddingFrame`). It evaluates a committee of models on them
and writes the frames with the largest disagreement to `work/al_candidates.xyz`. Label these with
`01_data_and_training/03_label_with_pyscf.py --frames work/al_candidates.xyz`, add them to the training
set and retrain. Repeat until the committee agrees along your production trajectories. A committee of
the same architecture with different seeds (`SEED=2 bash 04_train.sh`) is the usual choice.
`MACECalculator` with several model paths returns `energy_var` and `forces_var` directly.

## Part 4: the automated loop

```bash
bash 04_active_learning_loop.sh               # = fieldmace_active_learning --config al.yaml
bash 04_active_learning_loop.sh --status
```

`fieldmace_active_learning` repeats part 3 until the committee agrees. Each round:

1. **train** a committee (`committee_size` models, seeds 1, 2, …) on the current training set with the
   `mace_run_train` flags in `train_args`;
2. **sample**: ML/MM MD with the first model at each temperature in `sampling` (PythonForce, or
   TorchForce with `backend: torch`). Frames are stored in the training format. A trajectory that blows
   up stops; its frames up to then are kept;
3. **select**: committee force disagreement per frame (max over atoms and states). Frames between
   `force_std_min` and `force_std_max` are candidates; the most uncertain ones are taken, at least
   `min_separation` frames apart. Above `force_std_max` the model is so far off that the geometry is
   probably unphysical, so it is not labelled;
4. **label** with `labeller`: `pyscf` (as in tutorial 01), `model:<path>` (a bigger model as the
   reference), `file.py:function` or `module:function` (your QM code: frames in, labelled frames out,
   see `mace/mlmm/labelers.py`), or `external`. With `external` the loop writes
   `round_NN/candidates.xyz` and stops. Label them on your cluster, write `round_NN/labelled.xyz` and run
   the same command again;
5. **merge** into `round_NN/train.xyz`, the training set of the next round.

It stops after `rounds` rounds, or when fewer than `converged_fraction` of the MD frames are uncertain.
`state.json` in the work directory records the progress, so the loop resumes where it stopped (after a
time limit, a crash or external labelling). `fieldmace_active_learning --example` prints an annotated
configuration. For excited-state and diabatic models, the disagreement covers the forces of all states
and `sampling.state` chooses the driving state.

On a cluster, run each round as one job (`--rounds 1`), or split the labelling into its own job array with
`labeller: external`.
