#!/usr/bin/env bash
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

# Step 0: make FieldMACE known to SHARC 4 and check the installation.
#
#   export SHARC=/path/to/sharc4/bin       # (or: source /path/to/sharc4/bin/sharcvars.sh)
#   bash 00_install.sh
set -euo pipefail
: "${SHARC:?set SHARC to the bin directory of SHARC 4}"
# the FieldMACE repository = where the installed (pip install -e) package lives
REPO=$(python3 -c "import mace, os; print(os.path.dirname(os.path.dirname(os.path.abspath(mace.__file__))))")
IFACE="$REPO/sharc/SHARC_FIELDMACE.py"

# 1. the interface lives in the FieldMACE repository; SHARC finds interfaces in $SHARC
if [ -e "$SHARC/SHARC_FIELDMACE.py" ]; then
  echo "found $SHARC/SHARC_FIELDMACE.py"
elif [ -f "$IFACE" ]; then
  [ -x "$IFACE" ] || chmod +x "$IFACE"
  ln -s "$IFACE" "$SHARC/SHARC_FIELDMACE.py"
  echo "linked $SHARC/SHARC_FIELDMACE.py -> $IFACE"
else
  echo "cannot find $IFACE (is FieldMACE installed with pip install -e from its repository?)"; exit 1
fi

export PYTHONPATH="$SHARC/../lib:$SHARC:$SHARC/../pysharc:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SHARC/../lib:${LD_LIBRARY_PATH:-}"

# 2. checks
python3 - <<'EOF'
import os, sys
sharc = os.environ["SHARC"]
ok = True
def check(name, f):
    global ok
    try:
        msg = f()
        print(f"[ OK ] {name}{': ' + msg if msg else ''}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {e}")
check("FieldMACE importable", lambda: __import__("mace").__version__)
check("SHARC 4 Python library", lambda: (__import__("SHARC_FAST"), "")[1])
check("SHARC_FIELDMACE interface", lambda: __import__("SHARC_FIELDMACE").SHARC_FIELDMACE.version())
check("SHARC_LVC (reference)", lambda: (__import__("SHARC_LVC"), "")[1])
check("SHARC_QMMM + SHARC_OPENMM", lambda: (__import__("SHARC_QMMM"), __import__("SHARC_OPENMM"), "")[2])
def exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return ""
check("sharc.x", lambda: exists(os.path.join(sharc, "sharc.x")))
def pysharc():
    import sharc.sharc  # noqa: F401
    return "driver.py -i FIELDMACE works"
try:
    print("[ OK ] pysharc: " + pysharc())
except Exception as e:  # noqa: BLE001
    print(f"[ -- ] pysharc not available ({e}); use ENGINE=sharcx with sharc.x (slower), or build SHARC with USE_PYSHARC=true")
ex = os.path.join(sharc, "..", "examples", "SHARC_QMMM_LVC_OPENMM")
check("SHARC example SHARC_QMMM_LVC_OPENMM", lambda: exists(ex))
sys.exit(0 if ok else 1)
EOF
