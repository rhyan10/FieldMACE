#!/usr/bin/env python
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Check that a FieldMACE environment is complete.

    python scripts/check_env.py            # login node (no GPU needed)
    python scripts/check_env.py --gpu      # inside a GPU job: also runs a tiny model on CUDA
"""
import importlib
import platform
import sys
import traceback

GPU = "--gpu" in sys.argv
ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"[ OK ] {name}: {msg}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def ver(mod):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):  # pymbar banner
        m = importlib.import_module(mod)
    return getattr(m, "__version__", "?")


print("python", sys.version.split()[0], platform.platform())
for mod in ["numpy", "scipy", "torch", "e3nn", "ase", "openmm", "openmmtools", "matscipy", "torch_ema",
            "torchmetrics", "prettytable", "opt_einsum_fx", "tqdm", "hostlist", "configargparse", "h5py",
            "pytest"]:
    check(mod, lambda m=mod: ver(m))


def e3nn_pin():
    import e3nn
    assert e3nn.__version__ == "0.5.1", f"FieldMACE pins e3nn==0.5.1, found {e3nn.__version__}"
    return "0.5.1"


def torch_cuda():
    import torch
    s = f"built with CUDA {torch.version.cuda}, cuda available={torch.cuda.is_available()}"
    if torch.cuda.is_available():
        s += f", {torch.cuda.get_device_name(0)}"
    elif GPU:
        raise RuntimeError(s + " (driver too old for this wheel? compare with nvidia-smi)")
    return s


def openmm_platforms():
    import openmm as mm
    assert hasattr(mm, "PythonForce"), "OpenMM >= 8.3 needed (PythonForce missing)"
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if GPU and "CUDA" not in names:
        raise RuntimeError(f"no CUDA platform in {names}; pip install 'openmm[cuda12]' (or cuda13)")
    return f"{mm.__version__}, platforms {names}"


def fieldmace():
    import mace
    from mace.modules import field

    assert mace.__version__.startswith("2."), f"FieldMACE {mace.__version__}: expected version 2.x"
    return f"{mace.__version__} at {mace.__file__} (field model v{field.FIELD_MODEL_VERSION})"


def interface():
    import mace.mlmm

    return "mace.mlmm importable"


def tiny_run():
    import numpy as np
    import openmm as mm
    import openmmtools.testsystems as T
    import torch
    from openmm import unit

    from mace.mlmm import EmbeddingSettings, create_fieldmace_mm_system
    from mace.tools.testing import build_field_model

    torch.set_default_dtype(torch.float32)
    dev = "cuda" if GPU else "cpu"
    model = build_field_model("PerAtomFieldEMACE")
    ts = T.AlanineDipeptideExplicit()
    s, spec, comp = create_fieldmace_mm_system(ts.system, range(22), model, topology=ts.topology,
                                               settings=EmbeddingSettings(cutoff=10.0, switch_width=2.0),
                                               model_kwargs=dict(device=dev))
    plat = "CUDA" if GPU else "CPU"
    ctx = mm.Context(s, mm.VerletIntegrator(0.5 * unit.femtosecond), mm.Platform.getPlatformByName(plat))
    ctx.setPositions(ts.positions)
    st = ctx.getState(getEnergy=True, getForces=True)
    f = st.getForces(asNumpy=True)._value
    assert np.isfinite(f).all()
    return f"model on {dev}, OpenMM {plat}: E = {st.getPotentialEnergy()}"


check("e3nn version pin", e3nn_pin)
check("torch CUDA", torch_cuda)
check("OpenMM", openmm_platforms)
check("FieldMACE", fieldmace)
check("mace.mlmm (OpenMM interface)", interface)
check("end-to-end ML/MM energy+forces", tiny_run)
print("\nALL GOOD" if ok else "\nSOMETHING IS MISSING (see [FAIL] lines)")
