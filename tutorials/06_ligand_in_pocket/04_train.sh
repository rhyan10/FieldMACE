#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
#
# Step 4: train PerAtomFieldEMACE on the ligand in its pocket.
#
#   bash 04_train.sh                                  # CPU, ~30 min with the defaults
#   DEVICE=cuda EPOCHS=500 CHANNELS=64 bash 04_train.sh
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-150}
CHANNELS=${CHANNELS:-32}
DTYPE=${DTYPE:-float32}
SEED=${SEED:-1}
NAME=${NAME:-pxylene_pocket}
for f in data/train.xyz data/test.xyz; do
  if [ ! -f "$f" ] && [ -f "$f.gz" ]; then gunzip -k "$f.gz"; fi
  [ -f "$f" ] || { echo "$f missing: run 01-03 first (or use the shipped data/*.xyz.gz)"; exit 1; }
done
mkdir -p models

# 18 ligand atoms, ~700 point charges per frame (protein, water, ions within 12 A).
# --multipole_max_ell=2: the field of a protein pocket is far from uniform (quadrupoles help)
# --mm_forces_weight: forces on the ~700 point charges - lots of extra information about the field
mace_run_train \
  --name="$NAME" \
  --work_dir=models \
  --train_file=data/train.xyz \
  --valid_fraction=0.1 \
  --test_file=data/test.xyz \
  --seed="$SEED" \
  --model=PerAtomFieldEMACE \
  --multipole_max_ell=2 \
  --E0s=average \
  --r_max=5.0 \
  --num_interactions=2 \
  --correlation=3 \
  --hidden_irreps="${CHANNELS}x0e + ${CHANNELS}x1o" \
  --MLP_irreps="${CHANNELS}x0e" \
  --num_radial_basis=8 \
  --energy_weight=10 --forces_weight=100 --mm_forces_weight=100 \
  --batch_size=8 \
  --lr=0.01 \
  --max_num_epochs="$EPOCHS" \
  --scheduler=ReduceLROnPlateau --lr_factor=0.5 --scheduler_patience=10 \
  --ema --ema_decay=0.99 \
  --error_table=EnergyNacsDipoleMAE \
  --default_dtype="$DTYPE" \
  --device="$DEVICE"
echo "model: models/${NAME}.model"
