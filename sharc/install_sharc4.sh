#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
#
# Install SHARC 4 (https://github.com/sharc-md/sharc4, GPL-3) for use with FieldMACE, in a conda
# environment or in a Python virtual environment, with sharc.x and pysharc.
#
#   conda:  bash install_sharc4.sh --env conda [--conda-env sharc4]  [--prefix DIR]
#   venv:   bash install_sharc4.sh --env venv  [--venv PATH]         [--prefix DIR]
#
# What it does
#   1. conda: creates (or reuses) a conda environment with Python, compilers, LAPACK/BLAS, FFTW and
#      NetCDF from conda-forge.  venv: creates (or reuses) a virtual environment and uses the
#      compilers and libraries of the system / your loaded modules (it checks them first).
#   2. clones SHARC 4 (or uses --source), applies small source fixes (uninitialised LP-ZPE array
#      sizes and input-reader status - the latter makes pysharc read no input after "import torch";
#      "==" between logicals for gfortran), and builds
#        - sharc.x, data_extractor.x, laser.x        (file-based SHARC)
#        - pysharc (sharc.sharc + libsharc.so)       (in-process SHARC: driver.py -i FIELDMACE)
#        - data_extractor_NetCDF.x, data_converter.x (NetCDF output, with pysharc)
#   3. installs FieldMACE into the environment (unless it is there already) and links
#      sharc/SHARC_FIELDMACE.py into $SHARC
#   4. writes <prefix>/sharc4/sharc_env.sh  - source it before using SHARC
#   5. runs a short LVC trajectory with sharc.x and pysharc as a test
#
# Options
#   --env conda|venv       environment type (default: venv if a venv is active, conda if a conda
#                          environment other than base is active)
#   --prefix DIR           where SHARC is put: DIR/sharc4 (default: current directory)
#   --conda-env NAME       conda environment to create or reuse (default: sharc4)
#   --venv PATH            virtual environment to create or reuse (default: the active one, else
#                          DIR/venv-sharc4)
#   --python EXE           Python used to create a new venv, or "3.12" etc. for conda (default:
#                          python3 / 3.12); SHARC 4 needs Python >= 3.10
#   --source DIR           use an existing SHARC 4 checkout instead of cloning
#   --ref REF              git branch/tag/commit to check out (default: main)
#   --no-pysharc           build sharc.x only (no NetCDF needed)
#   --no-fieldmace         do not install FieldMACE / link the interface
#   --no-test              skip the test trajectory
#   --lapack "LIBS"        link flags for LAPACK+BLAS (default: auto-detect -llapack -lblas,
#                          -lflexiblas, -lopenblas)
#   --gcc VERSION          conda only: GCC/gfortran version (default 13; pysharc built with
#                          gfortran 16 reads no input, so newer is not the default)
#   -h, --help
#
# On a cluster (venv mode), load compilers and libraries first, e.g.
#   module load GCC FlexiBLAS FFTW netCDF         (names differ: module spider netCDF)
# and keep them loaded when running SHARC (sharc_env.sh records the library paths it found).
set -euo pipefail

# ------------------------------------------------------------------------------------------------
ENV_TYPE=""
PREFIX=$PWD
CONDA_ENV=sharc4
VENV=""
PYTHON=""
SOURCE=""
REF=main
PYSHARC=1
FIELDMACE=1
TEST=1
LAPACK=""
GCC_VERSION=13
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FM_REPO=$(cd "$HERE/.." && pwd)

die() { echo -e "\nERROR: $*" >&2; exit 1; }
say() { echo -e "\n>> $*"; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_TYPE=$2; shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --conda-env) CONDA_ENV=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --no-pysharc) PYSHARC=0; shift ;;
    --no-fieldmace) FIELDMACE=0; shift ;;
    --no-test) TEST=0; shift ;;
    --lapack) LAPACK=$2; shift 2 ;;
    --gcc) GCC_VERSION=$2; shift 2 ;;
    -h|--help) usage ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

if [ -z "$ENV_TYPE" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then ENV_TYPE=venv
  elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != base ]; then ENV_TYPE=conda; CONDA_ENV=$CONDA_DEFAULT_ENV
  else die "choose --env conda or --env venv"; fi
fi
[ "$ENV_TYPE" = conda ] || [ "$ENV_TYPE" = venv ] || die "--env must be conda or venv"
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
SHARCDIR=${SOURCE:-$PREFIX/sharc4}
[ -f "$FM_REPO/sharc/SHARC_FIELDMACE.py" ] || { FIELDMACE=0; echo "(FieldMACE repository not found next to this script: skipping the FieldMACE steps)"; }

# ------------------------------------------------------------------------------------------------
# 1. environment
# ------------------------------------------------------------------------------------------------
if [ "$ENV_TYPE" = conda ]; then
  command -v conda >/dev/null || die "conda not found. Install Miniforge (https://github.com/conda-forge/miniforge) or use --env venv"
  eval "$(conda shell.bash hook)"
  INSTALLER=conda; command -v mamba >/dev/null && INSTALLER=mamba
  # Python packages of SHARC's own conda environment (conda_env_sharc4.1.yml) + compilers and libraries
  PKGS="python=${PYTHON:-3.12} pip setuptools numpy scipy h5py matplotlib pyparsing pyyaml sympy numba ase \
        openmm pyscf opt_einsum threadpoolctl joblib gfortran=$GCC_VERSION gcc=$GCC_VERSION make git fftw libblas liblapack"
  [ $PYSHARC = 1 ] && PKGS="$PKGS libnetcdf netcdf4"
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    say "reusing conda environment '$CONDA_ENV' (adding missing packages)"
    $INSTALLER install -y -n "$CONDA_ENV" -c conda-forge $PKGS
  else
    say "creating conda environment '$CONDA_ENV'"
    $INSTALLER create -y -n "$CONDA_ENV" -c conda-forge $PKGS
  fi
  conda activate "$CONDA_ENV"
  DEPS=$CONDA_PREFIX
  export LIBRARY_PATH="$DEPS/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$DEPS/lib:${LD_LIBRARY_PATH:-}"
  export CPATH="$DEPS/include:${CPATH:-}"
  ACTIVATE="eval \"\$(conda shell.bash hook)\" && conda activate $CONDA_ENV"
else
  if [ -z "$VENV" ]; then VENV=${VIRTUAL_ENV:-$PREFIX/venv-sharc4}; fi
  case "$VENV" in /*) ;; *) VENV=$PWD/$VENV ;; esac  # absolute: sharc_env.sh is sourced from anywhere
  if [ ! -f "$VENV/bin/activate" ]; then
    PY=${PYTHON:-python3}
    command -v "$PY" >/dev/null || die "$PY not found (use --python)"
    say "creating virtual environment $VENV with $($PY --version)"
    "$PY" -m venv "$VENV"
  else
    say "reusing virtual environment $VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -q --upgrade pip  # not setuptools: an existing torch may pin it
  python -m pip install -q numpy scipy h5py matplotlib pyparsing pyyaml sympy numba ase openmm \
    pyscf opt_einsum threadpoolctl joblib
  [ $PYSHARC = 1 ] && python -m pip install -q netCDF4
  DEPS=""
  ACTIVATE="source $VENV/bin/activate"
fi
python - <<'EOF' || die "SHARC 4 needs Python >= 3.10"
import sys; sys.exit(sys.version_info < (3, 10))
EOF
say "Python: $(command -v python) ($(python --version 2>&1))"

# ------------------------------------------------------------------------------------------------
# 2. compilers and libraries
# ------------------------------------------------------------------------------------------------
for tool in gfortran gcc make git; do
  command -v $tool >/dev/null || die "$tool not found.
  Debian/Ubuntu: sudo apt-get install gfortran gcc make git liblapack-dev libblas-dev libfftw3-dev libnetcdf-dev
  Fedora/RHEL:   sudo dnf install gcc-gfortran gcc make git lapack-devel blas-devel fftw-devel netcdf-devel
  Cluster:       module load GCC FlexiBLAS FFTW netCDF   (or use --env conda)"
done
TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
try_link() {  # try_link "<flags>" -> 0 if a Fortran program links with them
  printf 'program t\nend program t\n' > "$TMPD/t.f90"
  gfortran "$TMPD/t.f90" -o "$TMPD/t" ${DEPS:+-L$DEPS/lib} $1 >/dev/null 2>&1
}
if [ -z "$LAPACK" ]; then
  for cand in "-llapack -lblas" "-lflexiblas" "-lopenblas" "-lopenblas -llapack"; do
    if try_link "$cand"; then LAPACK=$cand; break; fi
  done
  [ -n "$LAPACK" ] || die "no LAPACK/BLAS found (tried -llapack -lblas, -lflexiblas, -lopenblas); install one or pass --lapack"
fi
try_link "$LAPACK" || die "cannot link with --lapack '$LAPACK'"
try_link "-lfftw3" || die "FFTW3 not found (Debian: libfftw3-dev, Fedora: fftw-devel, cluster: module load FFTW)"
NCPREFIX=""
if [ $PYSHARC = 1 ]; then
  if [ -n "$DEPS" ]; then NCPREFIX=$DEPS
  elif command -v nc-config >/dev/null; then NCPREFIX=$(nc-config --prefix)
  else NCPREFIX=/usr; fi
  [ -d "$NCPREFIX/include" ] && export CPATH="$NCPREFIX/include:${CPATH:-}"
  [ -d "$NCPREFIX/lib" ] && export LIBRARY_PATH="$NCPREFIX/lib:${LIBRARY_PATH:-}"
  [ -d "$NCPREFIX/lib64" ] && export LIBRARY_PATH="$NCPREFIX/lib64:${LIBRARY_PATH:-}"
  try_link "-lnetcdf" || die "NetCDF (C library) not found, needed for pysharc.
  Debian: libnetcdf-dev, Fedora: netcdf-devel, cluster: module load netCDF; or use --no-pysharc"
  printf '#include <netcdf.h>\nint main(void){return 0;}\n' > "$TMPD/n.c"
  gcc -c "$TMPD/n.c" -o "$TMPD/n.o" 2>/dev/null || die "netcdf.h not found (install the NetCDF development package)"
  python - <<'EOF' || die "Python headers (Python.h) not found: install python3-dev / python3-devel, or use --env conda"
import sysconfig, os, sys
sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))
EOF
fi
say "compilers: $(gfortran --version | head -1); LAPACK/BLAS: $LAPACK"
GFMAJOR=$(gfortran -dumpversion | cut -d. -f1)
if [ $PYSHARC = 1 ] && [ "$GFMAJOR" -ge 16 ]; then
  echo "WARNING: gfortran $GFMAJOR: pysharc built with gfortran 16 did not read its input in our tests."
  echo "         If the pysharc test below fails, load an older GCC (13 works) or use --env conda."
fi

# ------------------------------------------------------------------------------------------------
# 3. SHARC source + gfortran fixes
# ------------------------------------------------------------------------------------------------
if [ -z "$SOURCE" ]; then
  if [ -d "$SHARCDIR/.git" ]; then
    say "updating $SHARCDIR"
    git -C "$SHARCDIR" fetch -q origin && git -C "$SHARCDIR" checkout -q "$REF" && \
      { git -C "$SHARCDIR" pull -q --ff-only 2>/dev/null || true; }
  else
    say "cloning SHARC 4 into $SHARCDIR"
    git clone -q https://github.com/sharc-md/sharc4.git "$SHARCDIR"
    git -C "$SHARCDIR" checkout -q "$REF"
  fi
fi
[ -f "$SHARCDIR/source/Makefile" ] || die "$SHARCDIR does not look like SHARC 4 (no source/Makefile)"
say "SHARC source: $SHARCDIR ($(git -C "$SHARCDIR" log --oneline -1 2>/dev/null || echo 'no git'))"

python - "$SHARCDIR/source" <<'EOF'
# gfortran: the LP-ZPE array sizes are used before they are read from the input, so they hold
# garbage ("Could not allocate lpzpe_ke_ah", crashes when writing restart files). Initialise them,
# size the arrays after reading the input, and guard the restart writes. Idempotent.
import re, sys, os
src = sys.argv[1]
p = os.path.join(src, "input.F90"); s = open(p).read(); changed = False
if "FieldMACE installer fix" not in s:
    a = "  call read_input_list_from_file(u_i_input)\n"
    if a in s:
        s = s.replace(a, a + "  ! FieldMACE installer fix: LP-ZPE sizes are used before they are read\n"
                             "  ctrl%lpzpe_nah=0\n  ctrl%lpzpe_nbc=0\n", 1); changed = True
    m = re.search(r"\n(\s*)! zpe of AH bonds", s)
    if m and "ah_list" in s[:m.start()]:
        ind = m.group(1)
        block = (f"\n{ind}! FieldMACE installer fix: size the trajectory LP-ZPE arrays now\n"
                 f"{ind}if (allocated(traj%lpzpe_ke_ah)) deallocate(traj%lpzpe_ke_ah)\n"
                 f"{ind}if (allocated(traj%lpzpe_ke_bc)) deallocate(traj%lpzpe_ke_bc)\n"
                 f"{ind}allocate(traj%lpzpe_ke_ah(ctrl%lpzpe_nah), traj%lpzpe_ke_bc(ctrl%lpzpe_nbc))\n"
                 f"{ind}traj%lpzpe_ke_ah=-123.d0\n{ind}traj%lpzpe_ke_bc=-123.d0\n")
        s = s[:m.start()] + block + s[m.start():]; changed = True
    if changed:
        open(p, "w").write(s)
p = os.path.join(src, "restart.f90"); s = open(p).read()
pat = re.compile(r"^(\s*)write\(u,\*\) (\((ctrl|traj)%(lpzpe_\w+)\(.*\),\w+=1,ctrl%lpzpe_n(?:ah|bc)\))\s*$", re.M)
def guard(m):
    ind, expr, obj, arr = m.group(1), m.group(2), m.group(3), m.group(4)
    return (f"{ind}if (allocated({obj}%{arr})) then\n{ind}  write(u,*) {expr}\n"
            f"{ind}else\n{ind}  write(u,*)\n{ind}endif")
if "allocated(ctrl%lpzpe_ah)" not in s and "allocated(traj%lpzpe_ke_ah)" not in s:
    s2 = pat.sub(guard, s)
    if s2 != s:
        open(p, "w").write(s2)
# input_list.f90: next_keyword() never sets its intent(out) "stat" on success, so the input reader
# stops early whenever that stack slot is non-zero (seen with pysharc after "import torch":
# "Keyword CHARGE not found"). Any compiler.
p = os.path.join(src, "input_list.f90"); s = open(p).read()
if "    ! stat = 0\n" in s:
    open(p, "w").write(s.replace("    ! stat = 0\n", "    stat = 0  ! FieldMACE installer fix: intent(out) was never set\n", 1))
# gfortran rejects "==" between logicals (ifort accepts it), e.g. in the NetCDF data extractor
lpat = re.compile(r"\s*==\s*(\.(?:true|false)\.)", re.I)
for name in os.listdir(src):
    if name.endswith((".f90", ".F90")):
        p = os.path.join(src, name); s = open(p).read(); s2 = lpat.sub(r" .eqv. \1", s)
        if s2 != s:
            open(p, "w").write(s2)
print("gfortran fixes: applied (or already present)")
EOF

# ------------------------------------------------------------------------------------------------
# 4. build
# ------------------------------------------------------------------------------------------------
MK="make USE_COMPILER=gnu USE_LIBS=gnu"
cd "$SHARCDIR/source"
say "building sharc.x (serial make; takes a few minutes)"
make -s clean >/dev/null 2>&1 || true
$MK USE_PYSHARC=false LD="$LAPACK -lfftw3 -lgfortran" install > "$PREFIX/sharc_build.log" 2>&1 || \
  { tail -30 "$PREFIX/sharc_build.log"; die "build failed, see $PREFIX/sharc_build.log"; }
cp -f "$SHARCDIR/bin/sharc.x" "$PREFIX/.sharc1.x.keep"

if [ $PYSHARC = 1 ]; then
  say "building pysharc"
  make -s clean >/dev/null 2>&1 || true
  NCLINK="-L../pysharc/lib -lsharcnc"
  for d in "$NCPREFIX/lib" "$NCPREFIX/lib64"; do [ -d "$d" ] && NCLINK="$NCLINK -L$d"; done
  NCLINK="$NCLINK -lnetcdf"
  PYMK="$MK USE_PYSHARC=true ANACONDA=$NCPREFIX NETCDF_LIB=\"$NCLINK\" LD=\"$LAPACK -lfftw3 -lgfortran $NCLINK\""
  ( cd "$SHARCDIR/pysharc" && eval "$PYMK install" ) >> "$PREFIX/sharc_build.log" 2>&1 || \
    { tail -30 "$PREFIX/sharc_build.log"; die "pysharc build failed, see $PREFIX/sharc_build.log"; }
  # NetCDF tools; sharc.x itself stays the file-based build from above
  ( eval "$PYMK data_extractor_NetCDF data_converter data_converter_to_ASCII" && \
    cp -f data_extractor_NetCDF.x data_converter.x data_converter_to_ASCII.x "$SHARCDIR/bin/" ) \
    >> "$PREFIX/sharc_build.log" 2>&1 || echo "(NetCDF tools not built; see $PREFIX/sharc_build.log)"
  cp -f "$PREFIX/.sharc1.x.keep" "$SHARCDIR/bin/sharc.x"
fi
rm -f "$PREFIX/.sharc1.x.keep"
cd "$PREFIX"

# ------------------------------------------------------------------------------------------------
# 5. FieldMACE
# ------------------------------------------------------------------------------------------------
if [ $FIELDMACE = 1 ]; then
  if python -c "import mace.mlmm" 2>/dev/null; then
    say "FieldMACE is installed in this environment ($(python -c 'import mace; print(mace.__version__)'))"
  else
    say "installing FieldMACE from $FM_REPO into the environment (pip install -e .[mlmm])"
    python -m pip install -q -e "$FM_REPO[mlmm]"
  fi
  ln -sf "$FM_REPO/sharc/SHARC_FIELDMACE.py" "$SHARCDIR/bin/SHARC_FIELDMACE.py"
  chmod +x "$FM_REPO/sharc/SHARC_FIELDMACE.py"
  echo "linked $SHARCDIR/bin/SHARC_FIELDMACE.py"
fi

# ------------------------------------------------------------------------------------------------
# 6. environment file (with the directories of every non-system library SHARC links against)
# ------------------------------------------------------------------------------------------------
LIBDIRS=""
for f in "$SHARCDIR/bin/sharc.x" "$SHARCDIR/lib/libsharc.so"; do
  [ -f "$f" ] || continue
  for d in $(ldd "$f" 2>/dev/null | awk '/=> \//{print $3}' | xargs -r -n1 dirname | sort -u); do
    case "$d" in /lib|/lib64|/usr/lib|/usr/lib64|/lib/*-linux-gnu|/usr/lib/*-linux-gnu) ;;
      *) case ":$LIBDIRS:" in *":$d:"*) ;; *) LIBDIRS="${LIBDIRS:+$LIBDIRS:}$d" ;; esac ;; esac
  done
done
ENVFILE=$SHARCDIR/sharc_env.sh
cat > "$ENVFILE" <<EOF
# SHARC 4 environment (written by FieldMACE's sharc/install_sharc4.sh). Use:  source $ENVFILE
$ACTIVATE
export SHARC=$SHARCDIR/bin
export SHARCLIB=$SHARCDIR/lib
export PYSHARC=$SHARCDIR/pysharc
export PYTHONPATH=\$SHARCLIB:\$SHARC:\$PYSHARC\${PYTHONPATH:+:\$PYTHONPATH}
export LD_LIBRARY_PATH=\$SHARCLIB${LIBDIRS:+:$LIBDIRS}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF
say "wrote $ENVFILE"

# ------------------------------------------------------------------------------------------------
# 7. test: 2 fs of CH2S with SHARC's LVC model, file-based and (if built) with pysharc
# ------------------------------------------------------------------------------------------------
if [ $TEST = 1 ]; then
  say "test trajectories"
  ( set +u; source "$ENVFILE"; set -u
    T=$(mktemp -d); cd "$T"
    EX=$SHARC/../examples/SHARC_QMMM_LVC_OPENMM/LVC
    [ -d "$EX" ] || { echo "SHARC example LVC files not found; skipping"; exit 0; }
    python - "$EX" <<'PYEOF'
import sys, os, shutil
ex = sys.argv[1]
lines = open(os.path.join(ex, "V0.txt")).read().splitlines()[1:5]
for run in ("x", "py"):
    os.makedirs(f"{run}/QM")
    for f in ("LVC.template", "LVC.resources", "V0.txt"):
        shutil.copy(os.path.join(ex, f), f"{run}/QM/")
    with open(f"{run}/geom", "w") as g:
        for l in lines:
            g.write(" " + l.strip() + "\n")
    open(f"{run}/input", "w").write(
        'geomfile "geom"\nveloc random 0.1\nnstates 2 0 2\nactstates 2 0 2\ncharge 0 1 0\nstate 2 mch\n'
        'coeff auto\nrngseed 7\ntmax 2.0\nstepsize 0.5\nsurf sharc\ncoupling ddr\noutput_format ascii\n')
    open(f"{run}/QM/runQM.sh", "w").write("cd QM\n$SHARC/SHARC_LVC.py QM.in >> QM.log 2>> QM.err\n")
PYEOF
    ok=1
    (cd x && "$SHARC/sharc.x" input > run.log 2>&1) && [ "$(grep -vc '^#' x/output.lis)" -ge 5 ] \
      && echo "[ OK ] sharc.x" || { echo "[FAIL] sharc.x (see $T/x)"; ok=0; }
    if [ -f "$SHARCLIB/libsharc.so" ]; then
      (cd py && python "$SHARC/driver.py" -i LVC input > run.log 2>&1) && [ "$(grep -vc '^#' py/output.lis)" -ge 5 ] \
        && echo "[ OK ] pysharc" || { echo "[FAIL] pysharc (see $T/py)"; ok=0; }
      if [ $ok = 1 ] && cmp -s <(grep -v '^#' x/output.lis | awk '{print $1,$2,$3,$4,$7}') \
                                <(grep -v '^#' py/output.lis | awk '{print $1,$2,$3,$4,$7}'); then
        echo "[ OK ] pysharc and sharc.x give the same trajectory"
      fi
    fi
    if [ -e "$SHARC/SHARC_FIELDMACE.py" ]; then
      python -c "import warnings; warnings.filterwarnings('ignore'); import SHARC_FIELDMACE; print('[ OK ] SHARC_FIELDMACE', SHARC_FIELDMACE.SHARC_FIELDMACE.version())" 2>/dev/null \
        || { echo "[FAIL] import SHARC_FIELDMACE"; ok=0; }
    fi
    [ $ok = 1 ] && rm -rf "$T"
    exit $(( 1 - ok )) ) || die "tests failed"
fi

cat <<EOF

SHARC 4 is installed in $SHARCDIR.
Before using it (every new shell / job):
    source $ENVFILE
Then e.g.
    \$SHARC/sharc.x input                        # file-based SHARC
    python \$SHARC/driver.py -i FIELDMACE input   # pysharc with FieldMACE
    pytest $FM_REPO/tests/sharc                   # FieldMACE's SHARC tests
    bash $FM_REPO/tutorials/04_sharc_dynamics/00_install.sh
EOF
