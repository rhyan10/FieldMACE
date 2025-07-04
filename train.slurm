#!/bin/bash
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=30G

source ~/.bashrc
module load release/24.04  GCCcore/11.3.0
module load Python/3.10.4
source myenv/bin/activate

python3 scripts/run_train.py   --name="protein_1_peratom_0.01" --multipole_max_ell=1   --train_file="ligand_protein.xyz"   --seed=100   --E0s="average"   --model="PerAtomFieldEMACE"   --r_max=5.0   --batch_size=40   --lr=0.01   --n_energies=1   --correlation=3   --max_num_epochs=1000   --ema   --ema_decay=0.99   --default_dtype="float32"   --device=cuda   --hidden_irreps="32x0e +32x1o"   --MLP_irreps="32x0e"   --num_radial_basis=8   --num_interactions=2   --forces_weight=100.0   --energy_weight=300.0   --error_table="EnergyNacsDipoleMAE"   --scheduler="ReduceLROnPlateau"   --lr_factor=0.5   --scheduler_patience=10
