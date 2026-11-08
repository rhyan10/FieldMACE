#!/usr/bin/env bash
# Step 3: run all trajectories of a run directory.
#
#   bash 03_run_trajectories.sh runs/gas_FIELDMACE              # pysharc (fast, model stays loaded)
#   ENGINE=sharcx bash 03_run_trajectories.sh runs/gas_FIELDMACE  # sharc.x, interface called every step
#
# pysharc needs SHARC built with USE_PYSHARC=true; it runs  $SHARC/driver.py -i <interface> input
# in every TRAJ directory. sharc.x runs QM/runQM.sh every step (slower: Python and the model are
# loaded again each step), but needs nothing beyond a standard SHARC build.
# On a cluster, run the TRAJ directories in parallel instead (one job or array task per trajectory).
set -euo pipefail
RUN=${1:?usage: bash 03_run_trajectories.sh runs/<mode>_<interface>}
ENGINE=${ENGINE:-pysharc}
: "${SHARC:?set SHARC to the SHARC 4 bin directory}"
export PYTHONPATH="$SHARC/../lib:$SHARC:$SHARC/../pysharc:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SHARC/../lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

for T in "$RUN"/TRAJ_*; do
  IFACE=$(cat "$T/interface")
  start=$(date +%s)
  if [ "$ENGINE" = "pysharc" ]; then
    (cd "$T" && python3 "$SHARC/driver.py" -i "$IFACE" input > run.log 2>&1) || { echo "FAILED: $T (see $T/run.log)"; exit 1; }
  else
    (cd "$T" && "$SHARC/sharc.x" input > run.log 2>&1) || { echo "FAILED: $T (see $T/run.log, $T/QM/QM.err)"; exit 1; }
  fi
  echo "$T done in $(( $(date +%s) - start )) s ($ENGINE, $IFACE)"
done
