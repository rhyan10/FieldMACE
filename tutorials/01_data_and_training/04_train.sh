#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

# Step 4: train ground-state FieldMACE models on the PySCF data.
#
#   bash 04_train.sh                          # both models, CPU (~15 min each with the defaults)
#   MODELS=PerAtomFieldEMACE bash 04_train.sh # only one
#   DEVICE=cuda EPOCHS=500 CHANNELS=64 bash 04_train.sh
#
# Output: models/aladip_<MODEL>.model
set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-200}
CHANNELS=${CHANNELS:-32}
MODELS=${MODELS:-"PerAtomFieldEMACE FieldEMACE"}
DTYPE=${DTYPE:-float32}
SEED=${SEED:-1}

# shipped data are compressed; unpack once
for f in data/train.xyz data/test.xyz; do
  if [ ! -f "$f" ] && [ -f "$f.gz" ]; then gunzip -k "$f.gz"; fi
  [ -f "$f" ] || { echo "$f missing: run 01-03 first (or use the shipped data/*.xyz.gz)"; exit 1; }
done
mkdir -p models

for MODEL in $MODELS; do
  # PerAtomFieldEMACE: multipole expansion of the MM potential about every QM atom (recommended)
  # FieldEMACE:        one expansion about the QM centroid (cheaper, less accurate for big regions)
  # --E0s=average:     per-element reference energies from a least-squares fit to the data
  # --mm_forces_weight: forces on the point charges (from REF_mm_forces) - cheap extra field information
  mace_run_train \
    --name="aladip_${MODEL}" \
    --work_dir=models \
    --train_file=data/train.xyz \
    --valid_fraction=0.1 \
    --test_file=data/test.xyz \
    --seed="$SEED" \
    --model="$MODEL" \
    --n_energies=1 \
    --multipole_max_ell=2 \
    --E0s=average \
    --r_max=5.0 \
    --num_interactions=2 \
    --correlation=3 \
    --hidden_irreps="${CHANNELS}x0e + ${CHANNELS}x1o" \
    --MLP_irreps="${CHANNELS}x0e" \
    --num_radial_basis=8 \
    --energy_weight=10 --forces_weight=100 --mm_forces_weight=100 \
    --batch_size=10 \
    --lr=0.01 \
    --max_num_epochs="$EPOCHS" \
    --scheduler=ReduceLROnPlateau --lr_factor=0.5 --scheduler_patience=10 \
    --ema --ema_decay=0.99 \
    --error_table=EnergyNacsDipoleMAE \
    --default_dtype="$DTYPE" \
    --device="$DEVICE"
  echo "model: models/aladip_${MODEL}.model"
done
