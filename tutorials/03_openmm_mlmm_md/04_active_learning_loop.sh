#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
#
# Part 4: the automated active-learning loop (train committee -> ML/MM MD -> select -> label -> repeat).
#
#   bash 04_active_learning_loop.sh                 # al.yaml: 4 rounds, PySCF labels (hours on a CPU)
#   bash 04_active_learning_loop.sh --rounds 1      # one round now; run again to continue
#   bash 04_active_learning_loop.sh --status
#
# Needs the tutorial-01 data (data/train.xyz, unpacked by 01_data_and_training/04_train.sh) and PySCF.
set -euo pipefail
cd "$(dirname "$0")"
for f in ../01_data_and_training/data/train.xyz ../01_data_and_training/data/test.xyz; do
  [ -f "$f" ] || { [ -f "$f.gz" ] && gunzip -k "$f.gz"; } || { echo "$f missing"; exit 1; }
done
CONFIG=al.yaml
if [ "${FIELDMACE_TUTORIAL_QUICK:-0}" = 1 ]; then  # tiny settings for automatic checks
  mkdir -p work
  python - <<'PY'
import yaml
c = yaml.safe_load(open("al.yaml"))
c.update(work_dir="al_quick", committee_size=2, rounds=1, labeller_args={"xc": "b3lyp", "basis": "sto-3g"})
for k in ("setup", "initial_data", "test_file"):
    c[k] = "../" + c[k]
c["system"]["pdb"] = "../" + c["system"]["pdb"]
c["train_args"].update(max_num_epochs=1, hidden_irreps="8x0e + 8x1o", MLP_irreps="8x0e")
c["sampling"].update(temperatures=[300], n_frames=4, every=5)
c["selection"].update(n_select=2, force_std_min=0.0)
yaml.safe_dump(c, open("work/al_quick.yaml", "w"), sort_keys=False)
PY
  CONFIG=work/al_quick.yaml
fi
python -m mace.cli.active_learning --config "$CONFIG" "$@"
