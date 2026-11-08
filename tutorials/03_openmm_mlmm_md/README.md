# Tutorial 03: ML/MM molecular dynamics in OpenMM

FieldMACE runs inside OpenMM as an `openmm.PythonForce` (OpenMM ≥ 8.3). The model describes the QM region in
the field of the MM point charges, and everything else is the force field. Any force field OpenMM reads
works.

```
01_mlmm_md.py          ground state: alanine side chain in water (model from tutorial 01), NVT + NVE
02_multistate_md.py    excited states: CH2S in water (model from tutorial 02), switching states, all-state energies
03_active_learning.py  find configurations the model is unsure about, for the next round of training
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
