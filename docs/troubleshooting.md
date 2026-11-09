# Troubleshooting

*Author: Rhyan Barrett*

FieldMACE checks its inputs before it does anything expensive. These are the messages you are most likely
to see, what they mean and what to do.

## Training (`mace_run_train`)

| message (shortened) | meaning | fix |
|---|---|---|
| `frame k has 2 energies but the model has 1 energy outputs ... use --n_energies=2` | the number of states in `REF_energy` does not match `--n_energies` | use the suggested `--n_energies`; for a diabatic model S states have S + S(S−1)/2 values |
| `frame k: 'REF_forces' has shape (4, 1, 3), expected (5, [n_states,] 3)` | forces for the wrong number of atoms, e.g. link atoms missing in the labels | label the same atoms (QM atoms + link H) as in the frame |
| `frame k has NaN or infinite values in 'forces'` | a failed QM calculation was written to the data | remove the frame (or re-run the QM calculation) |
| `no training frame has energies ('REF_energy')` | the key name does not match | `--energy_key=...` with the name in your file; the message lists the keys it found |
| `--compute_dipoles=True but no training frame has 'REF_dipoles' labels` | same for dipoles / NACs / SOCs | `--dipoles_key` / `--nacs_key` / `--socs_key` (the NAC default is `smooth_nacs`) |
| `'mm_charges' and 'mm_positions' must be given together`, `5 MM charges but 6 MM positions` | inconsistent point charges in a frame | write both arrays for the same charges |
| `contains element Z=7, which is not in the training set` | the test (or validation) set has an element the model never saw | add training frames with that element |
| `the validation set is empty` | too few frames for `--valid_fraction` | more data, a larger `--valid_fraction`, or `--valid_file` |
| `--E0s has no atomic energy for element(s) Z=[6]` | an incomplete `--E0s` dictionary | add the element, or `--E0s=average` |
| `--compute_nacs=True needs --n_energies >= 2`, `--diabatic=True and --compute_nacs=True` | inconsistent options | as the message says; diabatic models get NACs from H |
| `--E0s=... is not valid` / `no atomic reference energies` | a malformed or missing `--E0s` | `--E0s=average`, `'{1: -13.6, 6: -1029.8}'` (eV) or a `.json` file |
| warning: `IsolatedAtom frames give E0s for Z=[1] but not for Z=[6]` | isolated-atom frames for some elements only | `--E0s` is used for all elements; add the missing isolated atoms to use theirs |
| `--valid_fraction=1.0: must be between 0 and 1` | no frames left for training (or validation) | e.g. `--valid_fraction=0.1`, or `--valid_file` |
| `--train_file=... is HDF5: give the validation set too` / `--E0s=average needs the XYZ frames` | HDF5 input | `--valid_file` (the `val/` folder or a file) and `--statistics_file` from `mace_preprocess_data --compute_statistics` |
| `hidden_irreps=32x0e has no vectors (1o)` | NACs/dipoles need vector features | e.g. `--hidden_irreps='32x0e + 32x1o'` |
| `multipole_max_ell=4 but field_irreps=... stops at l=3` | the higher multipoles would be dropped | add `4x4e` to `--field_irreps`, or a smaller `--multipole_max_ell` |
| `pbc=(True, True, True) but lattice vector 0 of the cell is zero` | periodic frames without a cell | `pbc=False` for a QM cluster in point charges |
| `--device=cuda but PyTorch sees no CUDA GPU` | no GPU on this node | `--device=cpu`, or a GPU node |
| warning: `MM charge within 0.5 A of a QM atom` | usually a QM atom is also in the MM charge list | remove QM atoms from `mm_positions` / `mm_charges` |
| warning: `frame(s) are periodic (pbc=True)` | frames from a periodic code | set `pbc=False` for a QM cluster in point charges |
| warning: `No training frame has MM point charges` | gas-phase data only | the model cannot learn a field response without charges |

Training files can be `.xyz` or `.extxyz` (extended XYZ), or HDF5 from `mace_preprocess_data`.

## OpenMM (`mace.mlmm`)

| message | meaning | fix |
|---|---|---|
| `the QM region cuts the bond i-j to a hydrogen` | a hydrogen of a QM atom is MM | include the hydrogen (link atoms replace heavy atoms, never hydrogens) |
| warning: `QM-MM boundary through bond(s) ... sp3 carbon` | the cut is at a heteroatom, a double, aromatic or amide bond | move the boundary to a C–C (or C–N) single bond of an sp3 carbon |
| warning: `link_charge_scheme Z0 keeps the MM link host's charge` | a charge ~0.4 Å from the link H | use Z1 (default) or Z2 |
| `embedding cutoff 20.0 A is larger than half the box` | charges beyond half the box are ambiguous | smaller cutoff or larger box |
| `switch_width ... must be smaller than cutoff` | the switching region does not fit | e.g. cutoff 10, switch_width 2 |
| `Elements [16] of the QM region are not in the model's z_table` | the model was trained without that element | retrain with it |
| `GBSAOBCForce: implicit solvent is not part of the FieldMACE embedding` | implicit solvent or a polarisable force field | use explicit solvent with a fixed-charge force field |
| `PythonForce requires OpenMM >= 8.3` | old OpenMM | `pip install -U openmm` |
| `the system is periodic: pass the box vectors` | `frame_to_atoms` without the box of a periodic system | pass `state.getPeriodicBoxVectors()` |
| warning: `link_atoms=False but the QM region cuts ... covalent bond(s)` | QM atoms with dangling bonds | `link_atoms="auto"` (default) |
| `selection '...': incomplete clause` / `index range 30-20 is reversed` | a typo in the selection | complete the clause / write `20-30` |

## ASE calculator

| message | fix |
|---|---|
| `element(s) Z=[16] are not in the model` | the model was trained without that element |
| `committee models differ` | all committee models must have the same states, elements and properties |
| `device='cuda' requested, but PyTorch sees no CUDA GPU` | `device="cpu"`, or run inside a GPU job |
| warning: `an MM charge is 0.000 A from a QM atom` | a QM atom is also among the MM charges |

## SHARC

| message | fix |
|---|---|
| `unknown keyword 'energy_units' is ignored` | a typo in `FIELDMACE.template`; the message lists the known keywords |
| `energy_unit 'kcal'` | `eV` or `Hartree`, the unit the model was trained in |
| `nac_type must be 'nac' or 'smooth'` | as the message says |
| `'QMout' object has no attribute 'phases'` (sharc.x with SHARC_QMMM) | add `nophases_from_interface` to the SHARC `input` |

## Active learning (`fieldmace_active_learning`)

Unknown keys, misspelled keys in `sampling` / `selection`, and missing files are reported with the key name
before anything runs. Paths in the YAML file, including files in `train_args`
(`E0s`, `valid_file`, `statistics_file`, `foundation_model`), are relative to the YAML file.

| message | fix |
|---|---|
| `... of the frames are uncertain but none lies in [force_std_min, force_std_max]` | the model leaves its training domain during sampling: a smaller time step or temperature, a larger `force_std_max`, or more training data near the start structure |
