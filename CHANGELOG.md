# Changelog

All notable changes to FieldMACE. Versions follow [semantic versioning](https://semver.org/):
a new minor version adds features, a new major version changes behaviour.

## [Unreleased]

## [2.4.3] – 2026-11-09

50 more bug fixes from a second systematic search (regression tests in `tests/test_regressions_243.py`;
numbers as there).

### Fixed – training (`mace_run_train`)
1. `--distributed`: early stopping ran on rank 0 only, so the other ranks waited forever (hang).
2. A NaN validation loss counted as an improvement: the diverged model was saved as the best one and every
   later epoch "improved". NaN epochs are now never saved, with a warning.
3. `--distributed` outside a SLURM job exited with status 0 after logging an error.
4. `--distributed`: validation and test frames were dropped (`drop_last`), up to one per GPU.
6. Inconsistent `--hidden_irreps`, `--num_channels` or `--max_L` ended in bare `AssertionError`s.
12. `--E0s=foundation` crashed for every MACE-MP/OFF model (`KeyError` in a log line).
20. `--statistics_file` replaced `--r_max` silently; now with a warning.
25. An element missing from `--atomic_numbers` was reported as "not in the training set".
26. A missing `--E0s` JSON file or `--foundation_model` file is reported before anything is loaded.
35. HDF5 data with the wrong `--n_energies`/`--diabatic` failed as a traceback in the first loss.
37. The final TRAIN table left out the last incomplete training batch.
41. `--scheduler` accepted any string; a typo failed after the data were loaded.
42. Malformed `--radial_MLP`, `--atomic_numbers` or `--config_type_weights`, or an invalid `--log_level`,
    gave tracebacks after loading the data (or in the logger set-up).
44. `--max_ell=0` crashed while building the model.
45. `--num_interactions=0`, `--correlation=0`, `--r_max<=0` gave obscure errors; `--r_max=-1` trained a
    model without interactions.
49. `--train_file` and `--valid_file` in different formats gave an `AssertionError` or an h5py error.

### Fixed – fine-tuning and foundation models
5. From MACE-MP/OFF: the first layer used another block than the foundation's, so its skip connection
   stayed random (a bare `except: pass` hid it).
11. An element unknown to the foundation model gave `ValueError: 54 is not in list`.
13. MACE-MP downloads went through tinyurl.com (blocked by many cluster proxies), and a failed or
    interrupted download was cached and reused. Direct GitHub links, downloads are atomic.
16. `mace_off(return_raw_model=True)` failed with PyTorch ≥ 2.6.
23. The shipped `mace_anicc()` model could not be loaded on a machine without a GPU (e3nn TorchScript
    buffers loaded without `map_location`); all model loading now maps them to the chosen device.
24. `mace_mp()`, `mace_off()`, `mace_anicc()` could not compute anything (`IndexError` on non-existent
    "heads" in `ScaleShiftMACE.forward`).
30. Fine-tuning with a `--default_dtype` other than the foundation model's crashed.

### Fixed – data
18. Neighbour lists of QM clusters used a box scaled by the largest coordinate: coordinates ~400 Å from
    the origin cost 4.5 GB and seconds per frame.
19. A frame with forces but no energy and the wrong number of states crashed batching (or blamed the
    dipoles); every frame's state count is now checked.
39. Two QM atoms on top of each other (NaN energies) were not reported.
46. `mace_preprocess_data` did not check the elements of validation/test sets or `--atomic_numbers`.
47. Frames without point charges written by `write_extxyz` were corrupted (an empty `mm_charges=` swallowed
    the next key, e.g. the energy).
48. NaN QM positions were not reported (training ran with NaN everywhere).
50. Data written by ASE with a calculator (`energy=`/`forces=`): the "no energies" message did not show those
    keys; it now suggests `--energy_key=energy --forces_key=forces`.

### Fixed – ASE calculator and tools
10. The calculator advertised properties it never computes (stress, node_energy, and NACs/dipoles/SOCs of
    models without them); `node_energy` is now computed.
17. A committee of a float32 and a float64 model failed (only the first model's dtype was checked).
21. `mace_plot_train` wrote empty plots for runs shorter than 50 epochs, into the current directory.
29. `mace_eval_configs` and the other tools showed tracebacks for `--device=cuda` without a GPU and for
    missing files.
34. Changing `calc.state` between calls returned the cached results of the previous state.

### Fixed – OpenMM
7. CHARMM inputs: triclinic boxes became rectangular; a `.crd` file (no box) with PME failed inside OpenMM.
8. `load_system`: prmtop without inpcrd (etc.) gave TypeErrors.
9. `load_system` replaced PME by NoCutoff silently when the input has no box; now a warning.
14. A serialized System with the FieldMACE force could not be loaded from another directory when the
    model path was relative (Python crashed with a segmentation fault).
15. With `cutoff_reference="nearest"`, cutoff + QM radius beyond half the box: charges with two periodic
    images within the cutoff were embedded once; now a warning.
36. A topology without the force field's QM–MM bonds (PDB without CONECT records) got no link atom, silently;
    now a warning.
38. The all-pairs "static" QM graph was used up to 128 atoms, 1.7–5× slower than the cutoff graph above
    ~40 atoms; the automatic limit is 32.

### Fixed – active learning and tutorials
22. A failed step ended in a traceback; now `ERROR: ...`, the status and exit code 1.
27. A missing or misspelt `system` was found only at sampling, after the first committee was trained.
28. Misspelt `labeller_args` were ignored (default QM method used).
31. Misspelt `train_args` were reported only inside a training log.
32. `temperatures: 300` (a number) crashed sampling.
33. Tutorial 01 step 3: a frame whose SCF does not converge blocked the labelling forever; an old
    `data/test.xyz` could survive a re-split.
40. Raising `rounds` after the loop stopped there did nothing.
43. `pyyaml` was not a dependency, but `fieldmace_active_learning` and `mace_run_train --config` need it.

## [2.4.2] – 2026-11-09

50 bug fixes from a systematic search (regression tests in `tests/test_regressions_242.py`). The
numbers are those of the tests.

### Fixed – training (`mace_run_train`)
1. Early stopping (`--patience`) crashed without `--swa` (`'NoneType' object has no attribute 'start'`).
2. `--swa` (Stage Two) crashed at start-up (undefined `dipole_only`, no loss for `--loss=weighted`); it now
   uses the field-model loss with the Stage Two weights.
3. `--loss` was ignored (the field-model loss was always used); only `weighted` is accepted now.
4. `--error_table` `DipoleRMSE`, `EnergyDipoleRMSE`, `PerAtom*stressvirials` crashed at the end of training;
   every table now shows what was computed and `-` otherwise (the stress tables are gone).
12. `--interaction_first` was ignored (its default is now the block that was always used); `--scaling`,
    `--mean`, `--std` were computed but never used and now say so.
13. `--radial_type=gaussian` crashed (it built the legacy multipole basis).
14. `--valid_fraction` of 0 or 1 ended in a bare `AssertionError`.
15. The `schedulefree` hint suggested `pip install mace-torch[...]`, which replaces FieldMACE's `mace` package.
16. `--E0s=file.json` was rejected (JSON keys are strings).
17. `IsolatedAtom` frames for only some elements silently overrode `--E0s`, then failed.
19. Multi-state (or `(1, 1)`-shaped) `IsolatedAtom` energies crashed (`AssertionError`).
25. `--device=cuda` without a GPU, 29. HDF5 data without `--valid_file` or with `--E0s=average`, and
    35. an invalid or empty `--E0s` ended in tracebacks; they are now clear messages.
26. The final model was written after the error tables: an error while evaluating lost it. It is now
    saved first.
33. With `--eval_interval` larger than the number of epochs (or `--max_num_epochs=0`) no checkpoint was
    written and the run crashed at the end; the last epoch is always evaluated.
34. Stage Two checkpoints were named from the epoch after `--start_swa`; a short Stage Two crashed at the end.
37. HDF5 test sets (`--test_dir`) were never found (wrong file pattern); 38. `--multi_processed_test`
    crashed; 39. the last partial batch of a test set was dropped; 40. `--valid_batch_size` was ignored;
    41. a shard folder for training with a single validation file failed.
44. Frames without energies and forces in a multi-state data set (e.g. an unlabelled test set) broke the
    evaluation (one-state dummy labels).
47. An element present only in the validation set was not reported.

### Fixed – models and data
18. Fine-tuning from a FieldMACE model (`--foundation_model`) crashed for correlation ≠ 3 and transferred
    only the MACE layers; the field interactions, multipole embedding and readouts are now transferred too.
27. Scalar-only hidden irreps with NACs or dipoles gave identically zero outputs; this is an error now
    (and a warning for diabatic couplings).
28. Unsorted `--hidden_irreps` (e.g. `8x1o + 8x0e`) left the last layer without scalars.
30. Periodic frames without a cell were given a 1 Å box (thousands of bogus edges); this is an error now.
31. `mace_preprocess_data` crashed with more processes than frames, and averaged shard statistics
    without weights; 32. shards of an earlier run in the same `--h5_prefix` were trained on.
36. `--multipole_max_ell` above the largest l of `--field_irreps` was silently dropped; this is an error now.

### Fixed – ASE calculator and tools
5. `mace_eval_configs` failed for float32 models; it now runs in the model's dtype.
6. `mace_eval_configs` wrote a diabatic model's H elements as energies; it now writes adiabatic energies,
   forces and NACs, plus `MACE_diabatic_hamiltonian`. Reference labels in the file are not read.
7. `get_hessian` crashed for multi-state models and ignored `state` (diabatic: H_00); it now returns the
   Hessian of the calculator's (adiabatic) state.
8. The calculator parsed reference labels in `atoms.info` and failed on layouts it does not need.
9. Committee variances of diabatic models were those of the diabatic elements.
11. A committee r_max error message was broken; `compile_mode` now says it is unsupported.
20. `mace_active_learning_md` crashed at its first report (`forces_comm`); it works with one model or a
    committee and defaults to the CPU.
21. `mace_create_lammps_model` failed to load any model (PyTorch ≥ 2.6) and now explains that field
    models cannot run in LAMMPS.
43. `length_units_to_A` ≠ 1 passed Ångström positions to the model; positions, charges and cell are now
    converted, and NACs and dipoles too.
45. `ase.io.write` of Atoms with the calculator attached crashed for several states, or wrote the total
    energy as every atom's energy.
46. `get_descriptors` returned vector components as "invariants" for models with NAC or dipole outputs.

### Fixed – OpenMM, active learning, labellers
10. The active-learning loop declared convergence when every frame was beyond `force_std_max`.
22. Active learning: paths in `train_args` were not relative to the YAML file; a committee of one could
    slip through; on/off flags with a default of True were passed inverted.
23. TorchForce backend with `length_scale` ≠ 1: the cutoff was applied in model units.
24. TorchForce backend: an invalid `fieldmace_state` was silently clamped; it is an error now.
42. PySCF labeller with `spin` ≠ 0 crashed computing the point-charge forces.
48. `link_atoms=False` on a QM region that cuts bonds now warns (dangling bonds).
49. `frame_to_atoms` on a periodic system without box vectors tore the QM region apart; it is an error now.
50. `select_atoms` ignored a dangling `and` and reversed index ranges.

## [2.4.1] – 2026-11-09

Robustness: wrong inputs now stop with a clear message (file, frame, option to change) instead of
training silently on nonsense or failing with a traceback. One real bug fixed.

### Fixed
- ASE calculator: when only the MM charges changed between two calls (same QM geometry), ASE's
  cache returned the energy and forces of the old field. The calculator now also compares
  `mm_positions` and `mm_charges`.
- `--model=ExcitedMACE` (X-MACE's model, not part of FieldMACE) crashed during training; it is no
  longer offered.
- Training files with the `.extxyz` extension were treated as HDF5.
- A batch size larger than the training set crashed; it is now reduced, with a warning.

### Added
- Checks of the training data before training (`mace/tools/data_checks.py`): wrong number of
  states (with the right `--n_energies`), NaN/inf labels, missing labels for requested properties
  (wrong `--*_key`, with the keys found in the file), unknown elements in validation/test sets, an
  empty validation set, an incomplete `--E0s`; warnings for MM charges within 0.5 Å of a QM atom,
  periodic frames, data without point charges or without point-charge forces. All errors name the
  file and frame.
- Checks of the settings: missing files, `--compute_nacs` with one state, `--diabatic` with
  `--compute_nacs`, SOCs without pairs or `--soc_num`.
- ASE calculator: clear errors for elements the model does not know, committees of different
  models and CUDA requested without a GPU; a warning for a charge on top of a QM atom.
- OpenMM: errors when the QM region cuts a bond to a hydrogen, for an unknown link scheme, an
  embedding cutoff larger than half the box, and invalid cutoff/switching settings; warnings for
  boundaries that are not at an sp3 carbon, for the Z0 link scheme and for periodic systems without
  a cutoff.
- SHARC: unknown keywords in `FIELDMACE.template` are reported; invalid units give a clear error.
- Active learning: missing files, unknown keys (also inside `sampling`/`selection`) and a committee
  of one are reported before anything runs.
- README: a quick start; `docs/troubleshooting.md` lists the messages and their fixes.

### Changed
- Known harmless PyTorch/e3nn warnings (TorchScript deprecation, e3nn annotations) are no longer
  printed (dozens of lines per run); `pin_memory` is only used with a GPU.
- `scripts/check_env.py` no longer prints the pymbar banner.

## [2.4.0] – 2026-11-08

Diabatic models, a TorchScript backend for OpenMM, automated active learning, two new tutorials,
CI, packaging and a documentation site.

### Added
- **Diabatic models** (`--diabatic=True --n_energies=S`). The model predicts the S + S(S−1)/2
  elements of the diabatic Hamiltonian and their gradients on the atoms and the point charges.
  Atomic reference energies enter the diagonal only. Couplings have a true-scalar and a
  pseudoscalar part (triple products of vector features), so couplings between states of opposite
  mirror symmetry (e.g. n→π\* and π→π\*), which flip sign when the environment is mirrored, can
  be learned; the off-diagonal diabatic dipoles get a pseudovector part likewise.
  `--diabatic_coupling_weight` (default 10) weights the couplings in the loss.
  `mace.tools.diabatic` turns them into adiabatic energies, gradients and NACs, exactly
  (Hellmann–Feynman), and back (`labels_from_adiabatic`, for training labels from any QM method).
  - ASE calculator: adiabatic `energies`, `forces_all`, `nacs`, `mm_forces`, `dipoles`, plus
    `mm_nacs` (NACs on the point charges), `diabatic_hamiltonian`, `U`; `diabatic_blocks`
    diagonalises each multiplicity separately.
  - SHARC: the interface diagonalises H every step (one block per multiplicity) and returns
    adiabatic `h` (SOCs rotated, selection rules applied), `grad`, `nacdr`, `nacdr_pc` (NACs on MM
    atoms, used by SHARC_QMMM), `dm`, and `overlap` for `coupling overlap`.
  - OpenMM: the force uses the eigenvalue of the chosen state, differentiable through the
    diagonalisation.
- **TorchForce backend**: `create_fieldmace_mm_system(..., backend="torch")` runs the model as an
  `openmmtorch.TorchForce` (TorchScript in libtorch, no Python per step). Energies and forces are
  identical to the PythonForce. `export_torchscript` writes the complete ML/MM potential
  (embedding frame, link atoms, cutoff, model, state) as one TorchScript file for other programs.
- **Automated active learning**: `fieldmace_active_learning --config al.yaml`. Each round trains a
  committee, runs ML/MM MD, selects frames by committee disagreement and labels them (PySCF, a
  reference model, your own function, or external). It resumes from `state.json`.
  `mace.mlmm.labelers` holds the labellers.
- Tutorial 05: diabatic states, from labels to SHARC dynamics with NACs or overlaps, compared with
  the LVC reference.
- Tutorial 06: a ligand in a protein pocket. p-xylene in T4 lysozyme L99A in explicit water: system
  set-up with GAFF ligand parameters, sampling, PySCF labels, training, field effects, ML/MM MD.
- Tutorial 03: `--backend torch`; part 4, the automated active-learning loop; part 5, how the
  interface works (what it changes in the force field, link atoms and charge schemes, the embedding
  frame, energy bookkeeping, finite-difference force checks); part 6, the same set-up from Amber,
  GROMACS, OpenMM XML and serialized inputs; part 7, MD recipes (platform, time step, NVT/NPT,
  reporters, energy components, restarts, cost).
- `select_atoms`: `name` and `element` keywords, and `and` clauses (`resid 2 and name CA CB`).
- README: which MACE foundation models `--foundation_model` can start from (MACE-MP-0 2023, MACE-OFF23)
  and which it cannot yet (MPA-0, OMAT-0, MP-0b3, MATPES, OFF24, MH); the MM environment must be a
  fixed-charge force field.
- GitHub Actions: tests with pip (Python 3.10–3.13) and with conda-forge openmm-torch, sdist and
  wheel build; weekly SHARC 4 integration tests; release to PyPI on tags (trusted publishing);
  the documentation site (MkDocs, GitHub Pages) built from the READMEs and tutorials.

### Changed
- OpenMM 8.3–8.5 are supported again: there `PythonForce` has no `setParticles`, so the force
  receives all particles and selects its own.
- `pyscf_qmmm` moved to `mace.mlmm.labelers` and gained `density_fit`.
- The tutorial-03 scripts start from tutorial 01's NPT-equilibrated structure. The raw input box has water
  at 0.66 g/mL; the training data were sampled at 0.98 g/mL.

## [2.3.0] – 2026-11-08

Tutorials, and fixes and options for the SHARC 4 interface.

### Added
- `tutorials/`: four end-to-end tutorials, each with a README, scripts and a notebook:
  QM/MM training data from any force field and training (01), excited states with NACs,
  dipoles and SOCs (02), ML/MM MD in OpenMM (03), surface hopping with SHARC 4 (04).
  `tutorials/check_tutorials.py` runs every step and notebook.
- SHARC: `nac_type smooth` for models trained on smooth NACs, (E_j − E_i) · NAC_ij.
- SHARC: SOC selection rules (`soc_selection_rules`, on by default). SOCs between two
  singlets, within one multiplet and for |ΔM_S| > 1 are set to zero.
- `sharc/install_sharc4.sh`: installs SHARC 4 (`sharc.x`, pysharc, NetCDF tools) in a conda
  environment or a virtual environment, links the interface and tests the installation.
- `sharc/install_sharc4_zih.sh`: the same for the ZIH clusters (TU Dresden), with modules.
- Author headers in all files.

### Fixed
- pysharc received zero gradients from float32 models. The SHARC calculator now always
  returns double-precision arrays.
- SHARC 4 input reader (upstream bug, patched by the installer): an unset `intent(out)` status
  made pysharc read no input once PyTorch had been imported.

## [2.2.0] – 2026-11-07

### Added
- SHARC 4 interface `sharc/SHARC_FIELDMACE.py` (replaces the unfinished
  `mace/calculators/sharcinterface.py`). It works with `sharc.x`, pysharc and `SHARC_QMMM`
  (e.g. with OpenMM), maps spatial or M_S-resolved models onto SHARC's states, and returns
  gradients on point charges and phases.
- SOCs as a free-form vector per frame (`--soc_num`, X-MACE layout).

## [2.1.0] – 2026-11-07

### Added
- NACs, permanent and transition dipoles, and SOCs from the field models. The losses ignore the
  arbitrary sign of couplings between states.

### Removed
- The attention models `AttentionFieldEMACE` and `AttentionPerAtomFieldEMACE`.

## [2.0.0] – 2026-11-07

Version 2 changes the default model behaviour. Version 1 models still load and give the same
numbers (legacy mode).

### Added
- `mace.mlmm`: ML/MM molecular dynamics in OpenMM with any force field, link atoms, cutoff
  with switching, training-data export, multi-state MD.
- Frames with different numbers of point charges, and gas-phase frames, in one batch.
- TorchScript export of field models. A test suite.

### Changed
- Equivariant multipole embedding (version 1 was not rotation invariant).
- `FieldEMACE` expands about the QM centroid (version 1 was not translation invariant).
- Package name `fieldmace`; the import name is still `mace`.

### Fixed
- Parameters that were never trained (`linear_up`), loss crashes and shape errors, boolean
  command-line flags, HDF5 pipeline, E0 averaging and statistics with several states.

## [1.0] – 2025-12-19

The original FieldMACE (tag `v1.0`).

[Unreleased]: https://github.com/rhyan10/FieldMACE/compare/v2.4.3...HEAD
[2.4.3]: https://github.com/rhyan10/FieldMACE/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/rhyan10/FieldMACE/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/rhyan10/FieldMACE/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/rhyan10/FieldMACE/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/rhyan10/FieldMACE/compare/v1.0...v2.3.0
[1.0]: https://github.com/rhyan10/FieldMACE/releases/tag/v1.0
