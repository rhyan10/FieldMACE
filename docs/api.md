# Python API

*Author: Rhyan Barrett*

The most used entry points. Every function and class has a docstring with the details.

## Models and training

| | |
|---|---|
| `mace_run_train` (`mace.cli.run_train`) | training from the command line. `--model PerAtomFieldEMACE` or `FieldEMACE`, `--n_energies`, `--compute_nacs`, `--compute_dipoles`, `--compute_socs`, `--diabatic`, `--mm_forces_weight`, ... |
| `mace.modules.PerAtomFieldEMACE`, `FieldEMACE` | the models. Inputs: the QM atoms and the MM point charges (flat layout: `mm_positions [M, 3]`, `mm_charges [M]`, `num_mm [B]`) |
| `mace.tools.diabatic` | diabatic models: `adiabatize` (H, dH → E, U, gradients, NACs), `labels_from_adiabatic` (the inverse, for labels), `pack`, `unpack` |

## Using a model

| | |
|---|---|
| `mace.calculators.MACECalculator(path, ...)` | ASE calculator. `atoms.info["mm_positions"]` and `["mm_charges"]` set the field. Results: `energies`, `forces_all`, `mm_forces`, `nacs`, `dipoles`, `socs`. Diabatic models also give `diabatic_hamiltonian`, `U`, `mm_nacs` (`diabatic_blocks=` per multiplicity) |
| `mace.calculators.sharc_calculator.SharcCalculator` | SHARC's QM.out layout (M_S components, SOC selection rules, phases, overlaps) |
| `sharc/SHARC_FIELDMACE.py` | the SHARC 4 interface (`sharc.x`, pysharc `driver.py -i FIELDMACE`, `SHARC_QMMM`) |

## OpenMM (`mace.mlmm`)

| | |
|---|---|
| `create_fieldmace_mm_system(system, qm_atoms, model, topology=, settings=, backend="python" or "torch", **prepare_kwargs)` | any OpenMM System → ML/MM System. Link atoms, charge schemes, virtual sites, periodic boxes |
| `EmbeddingSettings(origin, cutoff, switch_width, cutoff_reference)` | how the MM charges enter the model; save it with the data (`save_setup` / `load_setup`) |
| `FieldMACEStateReporter` | energies of all states during MD; switch the driving state with `context.setParameter("fieldmace_state", k)` |
| `export_torchscript(model, spec, path)` | the whole ML/MM potential as one TorchScript file: positions (nm) [, box] [, state] → energy (kJ/mol) |
| `mace.mlmm.dataset.frame_to_atoms`, `attach_reference`, `write_orca_input`, ... | training data from simulations, labels from any QM code |
| `mace.mlmm.io.load_system` | PDB + force field, Amber, GROMACS, CHARMM or serialized XML |

## Active learning

| | |
|---|---|
| `fieldmace_active_learning --config al.yaml` (`mace.mlmm.active_learning`) | train a committee, run ML/MM MD, select uncertain frames, label, repeat; resumable. `--example` prints a configuration |
| `mace.mlmm.labelers` | built-in labellers (`pyscf`, `model:<path>`, `external`) and the interface for your own |
