#!/usr/bin/env bash
# Step 6 (optional): large data sets - preprocess to HDF5 once, then train from HDF5.
#
# For data sets that do not fit comfortably in memory, or for multi-GPU training, convert the
# extended-XYZ files once. The point charges (mm_positions, mm_charges, REF_mm_forces) are kept.
#
#   bash 06_hdf5.sh            # converts data/ and trains 2 epochs as a demonstration
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS=${EPOCHS:-2}
DEVICE=${DEVICE:-cpu}
for f in data/train.xyz data/test.xyz; do
  if [ ! -f "$f" ] && [ -f "$f.gz" ]; then gunzip -k "$f.gz"; fi
done

rm -rf h5 && mkdir -p h5
mace_preprocess_data \
  --train_file=data/train.xyz \
  --test_file=data/test.xyz \
  --valid_fraction=0.1 \
  --h5_prefix=h5/ \
  --r_max=5.0 \
  --E0s=average \
  --compute_statistics \
  --seed=1 \
  --num_process=1

ls -R h5 | head -20

mace_run_train \
  --name=aladip_h5 \
  --work_dir=models \
  --train_file=h5/train/train_0.h5 \
  --valid_file=h5/val/val_0.h5 \
  --test_dir=h5/test \
  --statistics_file=h5/statistics.json \
  --model=PerAtomFieldEMACE \
  --multipole_max_ell=2 \
  --r_max=5.0 \
  --hidden_irreps="32x0e + 32x1o" --MLP_irreps="32x0e" --num_interactions=2 \
  --energy_weight=10 --forces_weight=100 --mm_forces_weight=100 \
  --batch_size=10 --max_num_epochs="$EPOCHS" \
  --error_table=EnergyNacsDipoleMAE \
  --device="$DEVICE"
echo "model: models/aladip_h5.model"
