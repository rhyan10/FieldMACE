#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
#
# Install SHARC 4 for FieldMACE on the ZIH (TU Dresden) clusters, into an existing FieldMACE venv.
# Wraps sharc/install_sharc4.sh with the ZIH modules and paths.
#
#   bash sharc/install_sharc4_zih.sh                      # defaults below
#   bash sharc/install_sharc4_zih.sh --workspace /data/horse/ws/<you>-<ws> --venv venv-fieldmace
#
# Run it on Capella (the GPU cluster you run FieldMACE on): on a Capella login node or inside a
# Capella job. Binaries built against one cluster's modules are not guaranteed to work on another.
#
# Result:  <workspace>/sharc4/            SHARC 4 with sharc.x and pysharc, FieldMACE interface linked
#          <workspace>/sharc4/sharc_env.sh   loads the modules, activates the venv, sets $SHARC etc.
# Use:     source <workspace>/sharc4/sharc_env.sh     (in every shell and job script)
#
# Options
#   --workspace DIR   workspace that holds FieldMACE/ and the venv
#                     (default: the directory above this FieldMACE checkout)
#   --venv NAME|PATH  FieldMACE virtual environment (default: venv-fieldmace in the workspace)
#   --modules "..."   modules to load (default: the GCC 13.2 toolchain of release/24.10)
#   --no-test         skip the test trajectories
#   -h, --help
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FM_REPO=$(cd "$HERE/.." && pwd)
WORKSPACE=$(cd "$FM_REPO/.." && pwd)
VENV=venv-fieldmace
MODULES="release/24.10 GCC/13.2.0 OpenMPI/4.1.6 FlexiBLAS FFTW netCDF"
EXTRA=()

die() { echo -e "\nERROR: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) WORKSPACE=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --modules) MODULES=$2; shift 2 ;;
    --no-test) EXTRA+=(--no-test); shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done
WORKSPACE=$(cd "$WORKSPACE" && pwd) || die "workspace not found"
case "$VENV" in /*) ;; *) VENV=$WORKSPACE/$VENV ;; esac
[ -f "$VENV/bin/activate" ] || die "no virtual environment at $VENV (pass --venv)"
[ -f "$FM_REPO/sharc/install_sharc4.sh" ] || die "install_sharc4.sh not found next to this script"

case "$(hostname -f 2>/dev/null || hostname)" in
  *capella*) ;;
  *) echo "WARNING: this is not a Capella node ($(hostname)). Build SHARC on the cluster you will run it on." ;;
esac

# --- modules (Lmod is not always initialised in non-interactive shells) -------------------------
if ! type module >/dev/null 2>&1; then
  for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
    # shellcheck disable=SC1090
    [ -f "$init" ] && source "$init" && break
  done
fi
type module >/dev/null 2>&1 || die "the 'module' command is not available"
echo ">> module load $MODULES"
# shellcheck disable=SC2086
module load $MODULES || die "module load failed. Check the names with: module spider netCDF"

# --- venv ----------------------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo ">> Python: $(command -v python) ($(python --version 2>&1)), FieldMACE $(python -c 'import mace; print(mace.__version__)' 2>/dev/null || echo 'not installed')"
TORCH_SETUPTOOLS=$(python - <<'EOF' 2>/dev/null || true
from importlib.metadata import requires
for r in requires("torch") or []:
    if r.replace(" ", "").startswith("setuptools") and "<" in r:
        print(r.split(";")[0].replace(" ", ""))
EOF
)

# --- install -------------------------------------------------------------------------------------
bash "$FM_REPO/sharc/install_sharc4.sh" --env venv --venv "$VENV" --prefix "$WORKSPACE" \
  --lapack "-lflexiblas" "${EXTRA[@]}"

# torch pins setuptools; older versions of install_sharc4.sh upgraded it
if [ -n "$TORCH_SETUPTOOLS" ]; then python -m pip install -q "$TORCH_SETUPTOOLS"; fi

# --- sharc_env.sh: load the same modules first ---------------------------------------------------
ENVFILE=$WORKSPACE/sharc4/sharc_env.sh
python - "$ENVFILE" "$MODULES" "$VENV" <<'EOF'
import sys
path, modules, venv = sys.argv[1:]
lines = [l for l in open(path).read().splitlines()
         if not l.startswith("module load") and not l.startswith("source ") and "ZIH modules" not in l]
head = [lines[0], "# ZIH modules SHARC was built with (Lmod)",
        "type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null",
        f"module load {modules}", f"source {venv}/bin/activate"]
open(path, "w").write("\n".join(head + [l for l in lines[1:] if not l.startswith("type module")]) + "\n")
EOF
echo
echo ">> $ENVFILE:"
cat "$ENVFILE"
cat <<EOF

Done. In every shell or job script:
    source $ENVFILE
Check:
    cd $FM_REPO && pytest -q tests/sharc -p no:warnings
EOF
