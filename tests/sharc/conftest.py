# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""SHARC 4 integration tests. They run only when SHARC 4 is available:

    export SHARC=/path/to/sharc4/bin      (SHARC_FIELDMACE.py linked or copied into it)
    export PYTHONPATH=$SHARC/../lib:$SHARC:$PYTHONPATH
    pytest tests/sharc

Dynamics tests also need $SHARC/sharc.x; the pysharc tests need SHARC built with USE_PYSHARC=true
(``import sharc.sharc`` working, i.e. $SHARC/../pysharc on PYTHONPATH and $SHARC/../lib on LD_LIBRARY_PATH).
"""
import os
import shutil
import sys

import numpy as np
import pytest
import torch

SHARC = os.environ.get("SHARC")
HAVE_SHARC = bool(SHARC) and os.path.isfile(os.path.join(SHARC, "SHARC_FIELDMACE.py"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if HAVE_SHARC:
    for p in (SHARC, os.path.join(SHARC, "..", "lib")):
        if p not in sys.path:
            sys.path.insert(0, p)

requires_sharc = pytest.mark.skipif(not HAVE_SHARC, reason="SHARC 4 with SHARC_FIELDMACE.py not available ($SHARC)")
requires_sharc_x = pytest.mark.skipif(
    not (HAVE_SHARC and os.path.isfile(os.path.join(SHARC or "", "sharc.x"))), reason="$SHARC/sharc.x not available"
)



def _have_pysharc():
    if not HAVE_SHARC:
        return False
    pysharc_dir = os.path.join(SHARC, "..", "pysharc")
    if os.path.isdir(pysharc_dir) and pysharc_dir not in sys.path:
        sys.path.append(pysharc_dir)
    try:
        import sharc.sharc  # noqa: F401
    except Exception:
        return False
    return os.path.isfile(os.path.join(SHARC, "driver.py"))


HAVE_PYSHARC = _have_pysharc()
requires_pysharc = pytest.mark.skipif(not HAVE_PYSHARC, reason="pysharc (SHARC built with USE_PYSHARC=true) not available")

ZS = [1, 6, 8, 16]
# thioformaldehyde, bohr
CH2S = np.array([[0.01724515, 0.01406483, -0.04028205], [-2.01254877, 1.84662415, 1.61130840],
                 [0.12177534, 0.30511325, -2.11890982], [0.90230198, -1.72757999, 0.73443120]])
CH2S_EL = ["C", "S", "H", "H"]


@pytest.fixture(scope="session")
def sharc_models(tmp_path_factory):
    """Random FieldMACE models for SHARC states [2, 0, 2] (8 states with M_S) and [3]."""
    from mace.tools.testing import build_field_model

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    d = tmp_path_factory.mktemp("sharc_models")
    specs = {
        # spatial model: 2 singlets + 2 triplets, SOCs as free-form upper triangle of the 8x8 matrix
        "spatial": dict(n_energies=4, compute_nacs=True, compute_dipoles=True, compute_socs=True, soc_num=28),
        # M_S-resolved model: 8 states, complex SOC per pair
        "ms": dict(n_energies=8, compute_nacs=True, compute_dipoles=True, compute_socs=True),
        # singlets only
        "singlets": dict(n_energies=3, compute_nacs=True, compute_dipoles=True),
        # energies and forces only
        "plain": dict(n_energies=4),
        # diabatic: 4 diabatic states -> 10 elements of H, diabatic dipoles and SOCs (8x8 upper triangle)
        "diabatic": dict(n_energies=4, diabatic=True, compute_dipoles=True, compute_socs=True, soc_num=28),
    }
    paths = {}
    for name, kw in specs.items():
        m = build_field_model("PerAtomFieldEMACE", seed=3, atomic_numbers=ZS, channels=8, **kw)
        paths[name] = str(d / f"{name}.model")
        torch.save(m, paths[name])
        if name == "spatial":  # float32 copy: models are usually trained in float32
            paths["spatial32"] = str(d / "spatial32.model")
            torch.save(m.float(), paths["spatial32"])
    torch.set_default_dtype(old)
    return paths


def write_qm_in(path, coords_bohr, elements, states, requests, pcfile=None, step=0, savedir="./SAVE", init=True):
    lines = [str(len(elements)), "FieldMACE test"]
    for el, (x, y, z) in zip(elements, coords_bohr):
        lines.append(f"{el} {x:.10f} {y:.10f} {z:.10f}")
    lines += ["unit bohr", "states " + " ".join(map(str, states)), "charge " + " ".join(str(i % 2) for i in range(len(states))),
              f"savedir {savedir}", f"step {step}"]
    if init:
        lines.append("init")
    if pcfile:
        lines.append(f"point_charges {pcfile}")
    lines += requests
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_template(path, model, properties):
    with open(path, "w") as f:
        f.write(f"model_file {model}\nproperties {' '.join(properties)}\ndevice cpu\n"
                "energy_unit eV\ndistance_unit Angstrom\n")


def write_pcfile(path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-12, 12, 3)
        if np.min(np.linalg.norm(CH2S - p, axis=1)) > 5.0:
            pts.append(p)
    pts = np.array(pts)
    q = rng.uniform(-0.8, 0.8, n)
    with open(path, "w") as f:
        for (x, y, z), qi in zip(pts, q):
            f.write(f"{x:.10f} {y:.10f} {z:.10f} {qi:.10f}\n")
    return pts, q
