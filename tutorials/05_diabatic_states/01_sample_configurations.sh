#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
#
# Step 1: sample CH2S in the water droplet - the same sampling as tutorial 02 (see its README),
# written to this tutorial's work/ directory. Needs $SHARC (the system comes from SHARC's example).
#
#   bash 01_sample_configurations.sh                 # 1000 frames
#   bash 01_sample_configurations.sh --n-frames 200  # options are passed on
set -euo pipefail
cd "$(dirname "$0")"
python ../02_excited_states/01_sample_configurations.py --work work "$@"
