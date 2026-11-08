#!/usr/bin/env bash
# Step 3: train a FieldMACE model for 4 states with NACs, dipoles and SOCs.
#
#   bash 03_train.sh                       # CPU, ~45 min with the defaults (120 epochs)
#   DEVICE=cuda bash 03_train.sh           # GPU
#   EPOCHS=400 CHANNELS=64 bash 03_train.sh   # a better model
#
# Output: models/ch2s_es.model (+ logs/, results/, checkpoints/ in models/; OUT=dir to change)
set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-120}
CHANNELS=${CHANNELS:-16}
MODEL=${MODEL:-PerAtomFieldEMACE}       # or FieldEMACE
DTYPE=${DTYPE:-float32}                 # float64 for tiny models / debugging
SEED=${SEED:-1}
OUT=${OUT:-models}

[ -f data/train.xyz ] || { echo "data/train.xyz missing: run 02_label_with_lvc.py first"; exit 1; }
mkdir -p "$OUT"

# Keys of the labels written by 02_label_with_lvc.py:
#   REF_energy, REF_forces, REF_mm_forces (defaults) and REF_smooth_nacs, REF_dipoles, REF_socs.
# NACs are learned as smooth NACs, (E_j - E_i) * NAC_ij: real NACs diverge at near-degeneracies and
# would dominate the loss. The SHARC interface divides by the model's gap (nac_type smooth).
# --soc_num=56: SOCs as one vector per frame (upper triangle of the 8x8 SOC matrix, (re, im) pairs),
#               the layout the SHARC interface needs for a spatial (4-state) model with triplets.
# Loss weights: the properties have very different sizes (SOCs ~0.01 eV, smooth NACs ~0.05 eV/A,
# energies ~eV), so their weights are raised accordingly.
# --multipole_max_ell=2: potential, field and field gradient at every QM atom. The field gradient lets
# the model represent atomic quadrupoles, which matters for the forces on nearby point charges.
mace_run_train \
  --name=ch2s_es \
  --work_dir="$OUT" \
  --train_file=data/train.xyz \
  --valid_fraction=0.1 \
  --test_file=data/test.xyz \
  --seed="$SEED" \
  --model="$MODEL" \
  --n_energies=4 \
  --compute_nacs=True --nacs_key=REF_smooth_nacs --nacs_weight=300 \
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

echo "model: $OUT/ch2s_es.model"
