#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
#
# Step 5: surface hopping with the diabatic model in SHARC 4, compared with the LVC reference.
#
#   bash 05_sharc_dynamics.sh                    # NACs (ddr): gas phase 8 x 200 fs, QM/MM 4 x 100 fs
#   COUPLING=overlap bash 05_sharc_dynamics.sh   # wave-function overlaps instead of NACs
#   MODES=gas NTRAJ=4 TMAX=50 bash 05_sharc_dynamics.sh
#
# SHARC propagates in the adiabatic (MCH) basis. The FIELDMACE interface diagonalises the predicted
# diabatic Hamiltonian every step (singlet and triplet blocks separately) and hands SHARC:
#   h        adiabatic energies + SOCs (U^T SOC_dia U)
#   grad     adiabatic gradients (Hellmann-Feynman, from dH/dR)
#   nacdr    NACs  d_kl = (U^T dH/dR U)_kl / (E_l - E_k)   - also on the MM atoms (nacdr_pc)
#   overlap  U(t-dt)^T U(t)                                 - for "coupling overlap"
#   dm       adiabatic dipoles
# The trajectory scripts are tutorial 04's; only the model and the initial conditions change.
set -euo pipefail
cd "$(dirname "$0")"
: "${SHARC:?set SHARC to the SHARC 4 bin directory}"
MODEL=${MODEL:-models/ch2s_diabatic.model}
COUPLING=${COUPLING:-ddr}
MODES=${MODES:-gas qmmm}
NTRAJ=${NTRAJ:-8}
TMAX=${TMAX:-200}
NTRAJ_QMMM=${NTRAJ_QMMM:-$(( NTRAJ < 4 ? NTRAJ : 4 ))}
TMAX_QMMM=${TMAX_QMMM:-$(( TMAX < 100 ? TMAX : 100 ))}
DEVICE=${DEVICE:-cpu}
RUNS=${RUNS:-runs/$COUPLING}
T04=../04_sharc_dynamics
[ -f "$MODEL" ] || { echo "$MODEL missing: run 03_train.sh first"; exit 1; }
[ -f work/initconds.npz ] || { echo "work/initconds.npz missing: run 01_sample_configurations.sh first"; exit 1; }

for MODE in $MODES; do
  N=$NTRAJ; T=$TMAX
  if [ "$MODE" = qmmm ]; then N=$NTRAJ_QMMM; T=$TMAX_QMMM; fi
  for IFACE in FIELDMACE LVC; do
    python "$T04/02_setup_trajectories.py" --mode "$MODE" --interface "$IFACE" --model "$MODEL" \
      --initconds work/initconds.npz --ntraj "$N" --tmax "$T" --coupling "$COUPLING" \
      --device "$DEVICE" --runs "$RUNS"
    bash "$T04/03_run_trajectories.sh" "$RUNS/${MODE}_$IFACE"
  done
  python "$T04/04_analyze.py" "$RUNS/${MODE}_FIELDMACE" "$RUNS/${MODE}_LVC" --out "results/${COUPLING}_${MODE}"
done
