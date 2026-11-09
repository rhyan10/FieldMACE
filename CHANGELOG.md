# Changelog

All notable changes to FieldMACE. Versions follow [semantic versioning](https://semver.org/):
a new minor version adds features, a new major version changes behaviour.

## [Unreleased]

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

[Unreleased]: https://github.com/rhyan10/FieldMACE/compare/v2.4.0...HEAD
[2.4.0]: https://github.com/rhyan10/FieldMACE/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/rhyan10/FieldMACE/compare/v1.0...v2.3.0
[1.0]: https://github.com/rhyan10/FieldMACE/releases/tag/v1.0
