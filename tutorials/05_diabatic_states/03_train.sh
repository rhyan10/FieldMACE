#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
#
# Step 3: train a diabatic FieldMACE model: 4 diabatic states, the 10 elements of the diabatic
# Hamiltonian and their gradients, diabatic dipoles and SOCs.
#
#   bash 03_train.sh                          # CPU, ~1 h with the defaults
#   DEVICE=cuda EPOCHS=400 CHANNELS=64 bash 03_train.sh
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-100}
CHANNELS=${CHANNELS:-16}
DTYPE=${DTYPE:-float32}
SEED=${SEED:-1}
OUT=${OUT:-models}
[ -f data/train.xyz ] || { echo "data/train.xyz missing: run 02_label_diabatic.py first"; exit 1; }
mkdir -p "$OUT"

# --diabatic=True --n_energies=4: 4 diabatic states -> 4 + 6 outputs (REF_energy), each with forces
#   (REF_forces, REF_mm_forces). E0s are fitted to the diagonal only; couplings get no atomic offsets.
# --diabatic_coupling_weight=10: the couplings (~20 meV) are tiny next to the spread of the diagonal
#   (eV); without extra weight the loss hardly notices them.
# No --compute_nacs: NACs follow from the diabatic Hamiltonian.
# Dipoles (REF_dipoles) and SOCs (REF_socs, 56 values) are diabatic too.
mace_run_train \
  --name=ch2s_diabatic \
  --work_dir="$OUT" \
  --train_file=data/train.xyz \
  --valid_fraction=0.1 \
  --test_file=data/test.xyz \
  --seed="$SEED" \
  --model=PerAtomFieldEMACE \
  --diabatic=True \
  --n_energies=4 \
  --diabatic_coupling_weight=10 \
  --compute_dipoles=True --dipoles_key=REF_dipoles --dipoles_weight=100 \
  --compute_socs=True --socs_key=REF_socs --soc_num=56 --socs_weight=10000 \
  --energy_weight=100 --forces_weight=100 --mm_forces_weight=10 \
  --multipole_max_ell=2 \
  --E0s=average \
  --r_max=5.0 \
  --num_interactions=2 \
  --correlation=3 \
  --hidden_irreps="${CHANNELS}x0e + ${CHANNELS}x1o" \
  --MLP_irreps="${CHANNELS}x0e" \
  --num_radial_basis=8 \
  --batch_size=20 \
  --lr=0.01 \
  --max_num_epochs="$EPOCHS" \
  --scheduler=ReduceLROnPlateau --lr_factor=0.5 --scheduler_patience=10 \
  --ema --ema_decay=0.99 \
  --error_table=EnergyNacsDipoleMAE \
  --default_dtype="$DTYPE" \
  --device="$DEVICE"
echo "model: $OUT/ch2s_diabatic.model"
