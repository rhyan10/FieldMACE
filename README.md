# FieldMACE

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)   
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)

FieldMACE is a deep learning framework designed to be used in QM/MM datasets. It extends the Message Passing Atomic Cluster Expansion (MACE) architecture by integrating the multipole expansion to encode information from the MM region. An overview of the main parameters used in the MACE architecture can be found [here](https://github.com/ACEsuit/mace).

---

## Overview

The framework is designed to handle long range effects from the MM region on the QM site and predict the resulting energies and forces. The main additional parameters for FieldMACE can be seen below:

Key features include:
- **Multi-level Energy Learning:** Use the `--n_energies` parameter to specify how many energy levels the model should learn.
- **Transfer Learning Capability:** Leverage pre-trained ground state representations by specifying the `--foundation_model` parameter.
- **Multipole Expansion Parameter:** Use the `--multipole_max_ell` parameter for controlling the number of higher order tensors used
---

## System Requirements

### Hardware
- A standard computer with enough RAM for deep learning computations however a GPU enabled machine is strongly recommended. 

### Software
- **Operating System:** Linux, macOS, or Windows (best used with a conda environment)
- **Python:** 3.11 or higher
- **Dependencies:** FieldMACE relies on the typical deep learning and scientific computing stack in Python. All dependencies are installed during installation of the library 

---

## Installation

To install FieldMACE and its dependencies clone the github repo and install locally. The installation should only take a few minutes on a normal computer. The following commands illustate this:

```bash
git clone https://github.com/rhyan10/FieldMACE.git
cd FieldMACE
pip install .
```
It is also highly recommended to create a python environment beforehand, This can be done using the following commands:

```bash
# Clone the repository
git clone https://github.com/rhyan10/FieldMACE.git
cd FieldMACE

# Create and activate a new Python virtual environment using conda
conda create --name mace-env python=3.11 -y
conda activate mace-env

# Install dependencies and X‑MACE
pip install .
``` 

---

## Usage

The following commands allow training of the machine learning models mentioned in the paper with the multipole expansion or multipole moments. Output files containing the loss and validation errors can be seen in the results folder. Run time will depend on the size of the architecture as well as the number of GPUs available but normally will take less than 1 day. The following commands can be replaced with the relevant datasets, number of energy levels and desired higher order tensors.  

### Training the Model with Multipole expansion

```bash
python3 scripts/run_train.py   --name="protein_1_peratom_0.01" --multipole_max_ell=1   --train_file="ligand_protein.xyz"   --seed=100   --E0s="average"   --model="FieldEMACE"   --r_max=5.0   --batch_size=40   --lr=0.01   --n_energies=1   --correlation=3   --max_num_epochs=1000   --ema   --ema_decay=0.99   --default_dtype="float32"   --device=cuda   --hidden_irreps="32x0e +32x1o"   --MLP_irreps="32x0e"   --num_radial_basis=8   --num_interactions=2   --forces_weight=100.0   --energy_weight=300.0   --error_table="EnergyNacsDipoleMAE"   --scheduler="ReduceLROnPlateau"   --lr_factor=0.5   --scheduler_patience=10
```

### Training the Model with Multipole Moments

```bash
python3 scripts/run_train.py   --name="protein_1_peratom_0.01" --multipole_max_ell=1   --train_file="ligand_protein.xyz"   --seed=100   --E0s="average"   --model="PerAtomFieldEMACE"   --r_max=5.0   --batch_size=40   --lr=0.01   --n_energies=1   --correlation=3   --max_num_epochs=1000   --ema   --ema_decay=0.99   --default_dtype="float32"   --device=cuda   --hidden_irreps="32x0e +32x1o"   --MLP_irreps="32x0e"   --num_radial_basis=8   --num_interactions=2   --forces_weight=100.0   --energy_weight=300.0   --error_table="EnergyNacsDipoleMAE"   --scheduler="ReduceLROnPlateau"   --lr_factor=0.5   --scheduler_patience=10
```

### Transfer Learning

```bash
python3 scripts/run_train.py   --name="protein_1_peratom_0.01" --multipole_max_ell=1   --train_file="ligand_protein.xyz"   --seed=100   --E0s="average" --foundation_model="medium_off"  --model="PerAtomFieldEMACE"   --r_max=5.0   --batch_size=40   --lr=0.01   --n_energies=1   --correlation=3   --max_num_epochs=1000   --ema   --ema_decay=0.99   --default_dtype="float32"   --device=cuda   --hidden_irreps="32x0e +32x1o"   --MLP_irreps="32x0e"   --num_radial_basis=8   --num_interactions=2   --forces_weight=100.0   --energy_weight=300.0   --error_table="EnergyNacsDipoleMAE"   --scheduler="ReduceLROnPlateau"   --lr_factor=0.5   --scheduler_patience=10
```

---

## License

This project is licensed under the MIT License

---
