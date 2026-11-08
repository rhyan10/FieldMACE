#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Run every step of every tutorial in a temporary copy and report what works.

    python tutorials/check_tutorials.py              # tiny settings, ~20 min on 2 CPU cores
    python tutorials/check_tutorials.py --full       # the tutorials' real settings (hours on a CPU)
    python tutorials/check_tutorials.py --notebooks  # also execute the notebooks (tiny settings)
    python tutorials/check_tutorials.py --only 01 03 # some tutorials only

Steps that need SHARC 4 are skipped when $SHARC is not set; steps that need the output of a
skipped or failed step are skipped too. The tutorials in the repository are not touched.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def steps(full):
    q = not full
    tiny_train = {"EPOCHS": "2", "CHANNELS": "8"} if q else {}
    return [
        # tutorial 01: data + training
        ("01", "01_prepare_system.py", ["--equil-ps", "0.2"] if q else [], {}, False),
        ("01", "02_sample_configurations.py", ["--n-frames", "6", "--stride-ps", "0.02"] if q else [], {}, False),
        # tiny settings: label a few frames into work/ only, train on the shipped data
        ("01", "03_label_with_pyscf.py", ["--max-frames", "4", "--basis", "sto-3g", "--out", "work/check_data", "--partial", "work/check_partial.xyz"] if q else [], {}, False),
        ("01", "04_train.sh", [], tiny_train, False),
        ("01", "05_evaluate.py", [], {}, False),
        ("01", "06_hdf5.sh", [], {"EPOCHS": "1"}, False),
        # tutorial 02: excited states (labels need SHARC)
        ("02", "01_sample_configurations.py", ["--n-frames", "80", "--equil-ps", "0.5", "--n-initconds", "4"] if q else [], {}, True),
        ("02", "02_label_with_lvc.py", [], {}, True),
        ("02", "03_train.sh", [], tiny_train, True),
        ("02", "04_evaluate.py", [], {}, True),
        ("02", "05_calculators.py", [], {}, True),
        # tutorial 03: OpenMM ML/MM
        ("03", "01_mlmm_md.py", ["--nvt-ps", "0.01", "--nve-ps", "0.01"] if q else [], {}, False),
        ("03", "02_multistate_md.py", ["--ps-per-segment", "0.005"] if q else [], {}, True),
        ("03", "03_active_learning.py", ["--n-frames", "4", "--every", "5", "--n-select", "2"] if q else [], {}, False),
        # tutorial 04: SHARC
        ("04", "00_install.sh", [], {}, True),
        ("04", "01_single_point.py", [], {}, True),
        ("04", "02_setup_trajectories.py", ["--mode", "gas", "--interface", "FIELDMACE", "--ntraj", "2", "--tmax", "2" if q else "50"], {}, True),
        ("04", "03_run_trajectories.sh", ["runs/gas_FIELDMACE"], {}, True),
        ("04", "02_setup_trajectories.py", ["--mode", "gas", "--interface", "LVC", "--ntraj", "2", "--tmax", "2" if q else "50"], {}, True),
        ("04", "03_run_trajectories.sh", ["runs/gas_LVC"], {"ENGINE": "sharcx"}, True),
        ("04", "02_setup_trajectories.py", ["--mode", "qmmm", "--interface", "FIELDMACE", "--ntraj", "1", "--tmax", "1" if q else "50"], {}, True),
        ("04", "03_run_trajectories.sh", ["runs/qmmm_FIELDMACE"], {}, True),
        ("04", "04_analyze.py", ["runs/gas_FIELDMACE", "runs/gas_LVC", "runs/qmmm_FIELDMACE"], {}, True),
    ]


FOLDER = {"01": "01_data_and_training", "02": "02_excited_states", "03": "03_openmm_mlmm_md", "04": "04_sharc_dynamics"}
NEEDS = {  # (tutorial, script) -> earlier steps it depends on
    ("03", "01_mlmm_md.py"): [("01", "04_train.sh")],
    ("03", "03_active_learning.py"): [("01", "04_train.sh")],
    ("03", "02_multistate_md.py"): [("02", "03_train.sh")],
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true")
    p.add_argument("--notebooks", action="store_true")
    p.add_argument("--only", nargs="*", default=["01", "02", "03", "04"])
    p.add_argument("--keep", action="store_true", help="keep the temporary copy")
    a = p.parse_args()
    have_sharc = bool(os.environ.get("SHARC"))
    tmp = tempfile.mkdtemp(prefix="fieldmace_tutorials_")
    for f in FOLDER.values():
        shutil.copytree(os.path.join(HERE, f), os.path.join(tmp, f),
                        ignore=shutil.ignore_patterns("work", "models", "results", "runs", "singlepoint", "h5",
                                                      "__pycache__", "*.xyz", ".ipynb_checkpoints"))
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"), MPLBACKEND="Agg",
               PYTHONWARNINGS="ignore")
    if not a.full:
        env["FIELDMACE_TUTORIAL_QUICK"] = "1"
    print(f"checking tutorials in {tmp} ({'full' if a.full else 'tiny'} settings, SHARC: {have_sharc})\n")
    status, results = {}, []
    t_all = time.time()
    for tut, script, args, extra_env, needs_sharc in steps(a.full):
        if tut not in a.only:
            continue
        key = (tut, script)
        name = f"{FOLDER[tut]}/{script} {' '.join(args)}".strip()
        missing = [d for d in NEEDS.get(key, []) if status.get(d) != "ok"]
        if needs_sharc and not have_sharc:
            results.append(("skip", name, "needs $SHARC", 0.0))
            status[key] = "skip"
            continue
        if missing or (tut == "04" and status.get(("02", "03_train.sh")) != "ok" and script != "00_install.sh"):
            results.append(("skip", name, "needs an earlier step", 0.0))
            status[key] = "skip"
            continue
        cwd = os.path.join(tmp, FOLDER[tut])
        cmd = ["bash", script, *args] if script.endswith(".sh") else [sys.executable, script, *args]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=cwd, env={**env, **extra_env}, capture_output=True, text=True)
        dt = time.time() - t0
        ok = r.returncode == 0
        status[key] = "ok" if ok else "fail"
        tail = "" if ok else (r.stdout + r.stderr)[-2500:]
        results.append(("ok" if ok else "FAIL", name, tail, dt))
        print(f"[{'ok' if ok else 'FAIL':4s}] {name}  ({dt:.0f} s)", flush=True)
        if not ok:
            print(tail)
    if a.notebooks:
        for tut in a.only:
            nb = os.path.join(tmp, FOLDER[tut], "tutorial.ipynb")
            if not os.path.exists(nb):
                continue
            if tut in ("02", "04") and not have_sharc:
                results.append(("skip", f"{FOLDER[tut]}/tutorial.ipynb", "needs $SHARC", 0.0))
                continue
            t0 = time.time()
            r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                                "--ExecutePreprocessor.timeout=7200", "--output", "executed.ipynb", nb],
                               cwd=os.path.dirname(nb), env=env, capture_output=True, text=True)
            ok = r.returncode == 0
            results.append(("ok" if ok else "FAIL", f"{FOLDER[tut]}/tutorial.ipynb", "" if ok else r.stderr[-2500:],
                            time.time() - t0))
            print(f"[{'ok' if ok else 'FAIL':4s}] {FOLDER[tut]}/tutorial.ipynb  ({time.time() - t0:.0f} s)", flush=True)
            if not ok:
                print(r.stderr[-2500:])
    print(f"\nsummary ({time.time() - t_all:.0f} s):")
    for st, name, info, dt in results:
        print(f"  {st:4s}  {name}" + (f"  [{info}]" if st == "skip" else ""))
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"kept {tmp}")
    sys.exit(1 if any(st == "FAIL" for st, *_ in results) else 0)


if __name__ == "__main__":
    main()
