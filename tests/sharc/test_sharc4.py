"""SHARC 4 interface tests: QM.in -> QM.out, units, multiplicities, finite differences,
SHARC_QMMM (FieldMACE + OpenMM) and surface-hopping dynamics with sharc.x."""
import copy
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import (CH2S, CH2S_EL, REPO, SHARC, requires_pysharc, requires_sharc, requires_sharc_x, write_pcfile, write_qm_in,
                      write_template)

HARTREE_PER_EV = 1.0 / 27.211386245988
BOHR = 0.529177210903
ALL = ["energy", "forces", "nacs", "dipoles", "socs"]


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(SHARC, "..", "lib"), SHARC, os.path.join(SHARC, "..", "pysharc"), e.get("PYTHONPATH", "")])
    return e


def run_file_mode(workdir, timeout=600):
    r = subprocess.run([sys.executable, os.path.join(SHARC, "SHARC_FIELDMACE.py"), "QM.in"], cwd=workdir,
                       capture_output=True, text=True, timeout=timeout, env=env())
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    return r


def read_qmout(path, states, natom, npc):
    from qmout import QMout

    return QMout(filepath=str(path), states=states, natom=natom, npc=npc)


def interface(workdir, qm_in="QM.in"):
    """In-process interface, set up exactly as SHARC_INTERFACE.main() does."""
    from SHARC_FIELDMACE import SHARC_FIELDMACE

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        it = SHARC_FIELDMACE(loglevel=40)
        it.setup_mol(qm_in)
        it.read_resources()
        it.read_template()
        it.setup_interface()
        it.read_requests(qm_in)
        it.set_coords(qm_in)
    finally:
        os.chdir(cwd)
    return it


def evaluate(it, workdir, coords=None, pc=None):
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        if coords is not None:
            it.set_coords(coords)
        if pc is not None:
            it.set_coords(pc, pc=True)
        it.run()
        # getQMout() returns the interface's own QMout object, which the next call overwrites
        return copy.deepcopy(it.getQMout())
    finally:
        os.chdir(cwd)


def setup_dir(tmp_path, model, properties, states, requests, pc=True):
    write_template(tmp_path / "FIELDMACE.template", model, properties)
    pcfile = None
    if pc:
        write_pcfile(tmp_path / "pc.txt")
        pcfile = str(tmp_path / "pc.txt")
    write_qm_in(tmp_path / "QM.in", CH2S, CH2S_EL, states, requests, pcfile=pcfile)
    return tmp_path


# --------------------------------------------------------------------------- #
@requires_sharc
@pytest.mark.parametrize("pc", [False, True])
def test_file_mode_qmout_matches_model(tmp_path, sharc_models, pc):
    """QM.in -> SHARC_FIELDMACE.py -> QM.out, read back with SHARC's own parser."""
    from mace.calculators import MACECalculator

    d = setup_dir(tmp_path, sharc_models["spatial"], ALL, [2, 0, 2], ["h", "soc", "dm", "grad", "nacdr"], pc=pc)
    run_file_mode(d)
    npc = 12 if pc else 0
    q = read_qmout(d / "QM.out", [2, 0, 2], 4, npc)
    M = 8
    assert q.h.shape == (M, M) and np.allclose(q.h, q.h.conj().T)
    assert q.grad.shape == (M, 4, 3) and q.nacdr.shape == (M, M, 4, 3) and q.dm.shape == (3, M, M)
    assert np.allclose(q.nacdr, -np.transpose(q.nacdr, (1, 0, 2, 3)))
    # triplet M_S components are degenerate and share gradients
    e = np.real(np.diag(q.h))
    assert np.allclose(e[2:4], e[4:6]) and np.allclose(e[2:4], e[6:8])
    assert np.allclose(q.grad[2], q.grad[4]) and np.allclose(q.grad[3], q.grad[7])
    # no NACs / transition dipoles between multiplicities or M_S components
    assert np.allclose(q.nacdr[0, 2], 0) and np.allclose(q.nacdr[2, 4], 0) and np.allclose(q.dm[:, 1, 3], 0)
    assert not np.allclose(q.nacdr[2, 3], 0) and np.allclose(q.nacdr[2, 3], q.nacdr[4, 5])

    # reference: the ASE calculator on the same geometry (eV, Angstrom) -> atomic units
    a_pc = None
    if pc:
        raw = np.loadtxt(d / "pc.txt")
        a_pc = (raw[:, :3] * BOHR, raw[:, 3])  # QM.in unit bohr -> pc file in bohr
    import ase

    atoms = ase.Atoms(CH2S_EL, positions=CH2S * BOHR)
    if pc:
        atoms.info.update(mm_positions=a_pc[0], mm_charges=a_pc[1])
    calc = MACECalculator(sharc_models["spatial"], device="cpu", default_dtype="float64")
    calc.calculate(atoms)
    r = calc.results
    sp = [0, 1, 2, 3, 2, 3, 2, 3]  # model state of every SHARC state
    assert np.allclose(e, r["energies"][sp] * HARTREE_PER_EV, atol=1e-9)
    assert np.allclose(q.grad[5], -r["forces_all"][:, 3] * HARTREE_PER_EV * BOHR, atol=1e-9)
    assert np.allclose(q.nacdr[0, 1], r["nacs"][:, 0] * BOHR, atol=1e-9) or np.allclose(
        q.nacdr[0, 1], -r["nacs"][:, 0] * BOHR, atol=1e-9)
    soc = np.zeros((8, 8))
    soc[np.triu_indices(8, 1)] = r["socs"] * HARTREE_PER_EV
    assert np.allclose(np.abs(np.triu(q.h, 1)), np.abs(soc), atol=1e-9)
    if pc:
        assert q.grad_pc.shape == (M, 12, 3)
        assert np.allclose(q.grad_pc[1], -r["mm_forces"][:, 1] * HARTREE_PER_EV * BOHR, atol=1e-9)
    assert os.path.isfile(d / "SAVE" / "FIELDMACE_previous.npz")


@requires_sharc
def test_ms_resolved_model_with_pair_socs(tmp_path, sharc_models):
    d = setup_dir(tmp_path, sharc_models["ms"], ALL, [2, 0, 2], ["h", "soc", "dm", "grad", "nacdr"], pc=False)
    it = interface(d)
    q = evaluate(it, d)
    assert it.calculator.mode == "ms"
    r = it.calculator.calc.results
    # pair (2, 5) of the 8 model states -> SHARC h[2, 5]
    k = [(i, j) for i in range(8) for j in range(i + 1, 8)].index((2, 5))
    assert np.isclose(q.h[2, 5], r["socs"][k] * HARTREE_PER_EV) or np.isclose(q.h[2, 5], -r["socs"][k] * HARTREE_PER_EV)
    assert np.isclose(q.h[5, 2], np.conj(q.h[2, 5]))
    # an M_S-resolved model couples every pair (it learned which ones vanish)
    assert not np.allclose(q.nacdr[0, 5], 0)


@requires_sharc
def test_singlets_only_and_requests(tmp_path, sharc_models):
    d = setup_dir(tmp_path, sharc_models["singlets"], ["energy", "forces", "nacs", "dipoles"], [3],
                  ["h", "soc", "dm", "grad 1 3", "nacdr"], pc=True)
    q = evaluate(interface(d), d)
    assert q.h.shape == (3, 3) and np.allclose(q.h, np.diag(np.diag(q.h)))  # SOC request: zero for singlets
    assert q.grad.shape == (3, 4, 3) and q.grad_pc.shape == (3, 12, 3)


@requires_sharc
def test_unsupported_requests_fail_clearly(tmp_path, sharc_models):
    # NACs requested but the model has none
    d = setup_dir(tmp_path / "a", sharc_models["plain"], ["energy", "forces"], [4], ["h", "grad", "nacdr"], pc=False) \
        if (tmp_path / "a").mkdir() is None else None
    with pytest.raises(ValueError):
        evaluate(interface(d), d)
    # SOCs with triplets but no SOC model
    (tmp_path / "b").mkdir()
    d = setup_dir(tmp_path / "b", sharc_models["plain"], ["energy", "forces"], [2, 0, 2], ["h", "soc", "grad"], pc=False)
    with pytest.raises(ValueError):
        evaluate(interface(d), d)
    # dipoles requested without a dipole model: zeros, no crash
    (tmp_path / "c").mkdir()
    d = setup_dir(tmp_path / "c", sharc_models["plain"], ["energy", "forces"], [4], ["h", "dm", "grad"], pc=False)
    q = evaluate(interface(d), d)
    assert np.allclose(q.dm, 0)
    # a property the model was not trained for
    (tmp_path / "e").mkdir()
    d = setup_dir(tmp_path / "e", sharc_models["plain"], ["energy", "forces", "nacs"], [4], ["h", "grad"], pc=False)
    with pytest.raises(ValueError, match="compute_nacs"):
        interface(d)
    # SHARC states that do not fit the model
    (tmp_path / "f").mkdir()
    d = setup_dir(tmp_path / "f", sharc_models["plain"], ["energy", "forces"], [2, 0, 3], ["h", "grad"], pc=False)
    with pytest.raises(ValueError, match="do not match"):
        interface(d)


@requires_sharc
def test_gradients_are_finite_differences_through_sharc(tmp_path, sharc_models):
    """grad (QM atoms) and grad_pc (point charges) == dE/dx of SHARC's Hamiltonian diagonal (au)."""
    d = setup_dir(tmp_path, sharc_models["spatial"], ["energy", "forces"], [2, 0, 2], ["h", "grad"], pc=True)
    it = interface(d)
    q0 = evaluate(it, d)
    raw = np.loadtxt(d / "pc.txt")
    pcc = raw[:, :3].copy()
    h = 1e-4  # bohr
    for state in (0, 4):
        for atom, k in ((0, 0), (1, 2), (3, 1)):
            e = []
            for s in (1, -1):
                c = CH2S.copy()
                c[atom, k] += s * h
                e.append(np.real(evaluate(it, d, coords=c, pc=pcc).h[state, state]))
            assert (e[0] - e[1]) / (2 * h) == pytest.approx(q0.grad[state, atom, k], abs=1e-7)
        for j, k in ((0, 0), (5, 1)):
            e = []
            for s in (1, -1):
                p = pcc.copy()
                p[j, k] += s * h
                e.append(np.real(evaluate(it, d, coords=CH2S, pc=p).h[state, state]))
            assert (e[0] - e[1]) / (2 * h) == pytest.approx(q0.grad_pc[state, j, k], abs=1e-7)


@requires_sharc
def test_phase_reference_is_kept_between_file_mode_calls(tmp_path, sharc_models):
    d = setup_dir(tmp_path, sharc_models["spatial"], ALL, [2, 0, 2], ["h", "soc", "dm", "grad", "nacdr"], pc=False)
    run_file_mode(d)
    q0 = read_qmout(d / "QM.out", [2, 0, 2], 4, 0)
    prev = dict(np.load(d / "SAVE" / "FIELDMACE_previous.npz"))
    prev["nacs"][:, 0] *= -1  # pretend the previous step had the other sign for pair (0, 1)
    prev["socs"] *= -1
    np.savez(d / "SAVE" / "FIELDMACE_previous.npz", **prev)
    write_qm_in(d / "QM.in", CH2S, CH2S_EL, [2, 0, 2], ["h", "soc", "dm", "grad", "nacdr"], step=1, init=False)
    run_file_mode(d)
    q1 = read_qmout(d / "QM.out", [2, 0, 2], 4, 0)
    assert np.allclose(q1.nacdr[0, 1], -q0.nacdr[0, 1]) and np.allclose(q1.nacdr[2, 3], q0.nacdr[2, 3])
    assert np.allclose(np.triu(q1.h, 1), -np.triu(q0.h, 1))
    assert np.allclose(np.diag(q1.h), np.diag(q0.h))


@requires_sharc
def test_setup_traj_hooks(tmp_path, sharc_models, monkeypatch):
    """What setup_traj.py calls: get_features before any template, get_infos (questions), prepare (copies files)."""
    from SHARC_FIELDMACE import SHARC_FIELDMACE

    src = tmp_path / "src"
    src.mkdir()
    write_template(src / "FIELDMACE.template", sharc_models["spatial"], ALL)
    (src / "FIELDMACE.resources").write_text(f"modelpath {sharc_models['spatial']}\n")
    monkeypatch.chdir(src)
    it = SHARC_FIELDMACE(loglevel=40)
    assert {"h", "soc", "dm", "grad", "nacdr", "point_charges", "phases"} <= it.get_features()  # QMMM needs point_charges
    answers = iter(["y", "y", str(src / "FIELDMACE.resources")])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    with open(tmp_path / "KEYSTROKES", "w") as ks:
        it.get_infos({}, KEYSTROKES=ks)
    for link in (False, True):
        out = tmp_path / f"QM_{link}"
        out.mkdir()
        it.prepare({"link_files": link}, str(out))
        assert (out / "FIELDMACE.template").read_text() == (src / "FIELDMACE.template").read_text()
        assert (out / "FIELDMACE.resources").exists()


# --------------------------------------------------------------------------- #
# SHARC_QMMM: FieldMACE as the QM program, OpenMM as the MM program
# --------------------------------------------------------------------------- #
QMMM_EXAMPLE = os.path.join(SHARC or "", "..", "examples", "SHARC_QMMM_LVC_OPENMM")


def make_qmmm_dir(tmp_path, model, requests, name="qmmm"):
    """SHARC's SHARC_QMMM_LVC_OPENMM example; with ``model`` the QM program is FIELDMACE, otherwise stock LVC."""
    d = tmp_path / name
    shutil.copytree(QMMM_EXAMPLE, d)
    for sub in ("MML", "MMS"):
        tmpl = glob.glob(str(d / sub / "OPENMM_*.template"))[0]
        shutil.move(tmpl, d / sub / "OPENMM.template")
    if model is not None:
        (d / "FM").mkdir()
        write_template(d / "FM" / "FIELDMACE.template", model, ALL)
        t = (d / "QMMM.template").read_text().replace("qm-program LVC", "qm-program FIELDMACE").replace(
            "qm-dir ./LVC", "qm-dir ./FM")
        (d / "QMMM.template").write_text(t)
    lines = (d / "QM.in").read_text().splitlines()
    natom = int(lines[0])
    body = [ln for ln in lines[natom + 2:] if ln.strip() and ln.split()[0].lower() not in ("soc", "dm", "grad", "h", "nacdr")]
    (d / "QM.in").write_text("\n".join(lines[:natom + 2] + body + requests) + "\n")
    return d, natom


def qmmm_interface(d):
    from SHARC_QMMM import SHARC_QMMM

    cwd = os.getcwd()
    os.chdir(d)
    try:
        it = SHARC_QMMM(loglevel=40)
        it.setup_mol("QM.in")
        it.read_resources()
        it.read_template()
        it.setup_interface()
        it.read_requests("QM.in")
        it.set_coords("QM.in")
    finally:
        os.chdir(cwd)
    return it


def qmmm_eval(it, d, coords=None):
    cwd = os.getcwd()
    os.chdir(d)
    try:
        if coords is not None:
            it.set_coords(coords)
        it.run()
        # getQMout() returns the interface's own QMout object, which the next call overwrites
        return copy.deepcopy(it.getQMout())
    finally:
        os.chdir(cwd)


@requires_sharc
def test_sharc_qmmm_fieldmace_openmm(tmp_path, sharc_models):
    pytest.importorskip("openmm")
    d, natom = make_qmmm_dir(tmp_path, sharc_models["spatial"], ["h", "soc", "dm", "grad", "nacdr"])
    # file mode: QMMM -> FIELDMACE (+ point charges from OpenMM) -> QM.out
    r = subprocess.run([sys.executable, os.path.join(SHARC, "SHARC_QMMM.py"), "QM.in"], cwd=d, capture_output=True,
                       text=True, timeout=900, env=env())
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    assert "No 'grad_pc' in QMout" not in r.stdout + r.stderr
    q = read_qmout(d / "QM.out", [2, 0, 2], natom, 0)
    assert q.grad.shape == (8, natom, 3) and np.isfinite(q.grad).all()
    assert np.abs(q.grad[:, 4:]).sum() > 0  # MM atoms feel the QM region (grad_pc) and the force field
    assert np.allclose(q.h, q.h.conj().T)


@requires_sharc
def test_sharc_qmmm_gradient_finite_differences(tmp_path, sharc_models):
    """Total QM/MM gradient (QM atoms and MM water atoms) == dE/dx through SHARC_QMMM.

    SHARC_OPENMM runs OpenMM's CPU platform in single precision, so finite differences of the force-field part
    alone are off by ~1e-5 Eh/bohr on MM atoms (also with SHARC's own LVC example). The force-field part is
    deterministic and identical for any QM program, so the FieldMACE-dependent part is checked exactly as the
    difference FIELDMACE - LVC; QM atoms (no force-field noise at this level) are also checked directly.
    """
    pytest.importorskip("openmm")
    its = {}
    for name, model in (("fm", sharc_models["spatial"]), ("lvc", None)):
        d, natom = make_qmmm_dir(tmp_path, model, ["h", "grad"], name=name)
        its[name] = (qmmm_interface(d), d)
    q0 = {k: qmmm_eval(it, d) for k, (it, d) in its.items()}
    coords = its["fm"][0].QMin.coords["coords"].copy()
    h = 1e-3  # bohr
    for atom, k in ((0, 0), (1, 1), (4, 2), (5, 0), (100, 1)):  # C, S (QM); O, H of a water, a distant atom (MM)
        fd = {}
        for name, (it, d) in its.items():
            e = []
            for s in (1, -1):
                c = coords.copy()
                c[atom, k] += s * h
                e.append(np.real(qmmm_eval(it, d, coords=c).h[0, 0]))
            fd[name] = (e[0] - e[1]) / (2 * h)
        ref = q0["fm"].grad[0, atom, k] - q0["lvc"].grad[0, atom, k]
        assert fd["fm"] - fd["lvc"] == pytest.approx(ref, abs=2e-7, rel=1e-4), (atom, k)
        if atom < 4:
            assert fd["fm"] == pytest.approx(q0["fm"].grad[0, atom, k], abs=2e-6, rel=1e-3), (atom, k)


# --------------------------------------------------------------------------- #
# surface hopping with sharc.x (file-based interface calls)
# --------------------------------------------------------------------------- #
SHARC_INPUT = """geomfile     "geom"
veloc        {veloc}
nstates      2 0 2
actstates    2 0 2
charge       0 1 0
state        2 mch
coeff        auto
rngseed      1234
tmax         {tmax}
stepsize     0.5
nsubsteps    25
surf         sharc
coupling     ddr
nogradcorrect
ekincorrect  parallel_vel
decoherence_scheme edc
{extra}
output_format ascii
"""


def make_dynamics_dir(tmp_path, model, properties, extra="", tmax=2.0, qmmm=False, veloc="random 0.1"):
    if qmmm:
        d, natom = make_qmmm_dir(tmp_path, model, [])
        qm = tmp_path / "traj" / "QM"
        shutil.copytree(d, qm)
        lines = (qm / "QM.in").read_text().splitlines()
        coords = lines[2:natom + 2]
        script = "cd QM\n$SHARC/SHARC_QMMM.py QM.in >> QM.log 2>> QM.err\nexit $?\n"
    else:
        qm = tmp_path / "traj" / "QM"
        qm.mkdir(parents=True)
        write_template(qm / "FIELDMACE.template", model, properties)
        natom = 4
        coords = [f"{el} {x:.10f} {y:.10f} {z:.10f}" for el, (x, y, z) in zip(CH2S_EL, CH2S)]
        script = "cd QM\n$SHARC/SHARC_FIELDMACE.py QM.in >> QM.log 2>> QM.err\nexit $?\n"
    traj = tmp_path / "traj"
    (qm / "runQM.sh").write_text(script)
    os.chmod(qm / "runQM.sh", 0o755)
    from ase.data import atomic_masses, atomic_numbers

    with open(traj / "geom", "w") as f:  # SHARC native geometry: symbol, Z, x, y, z (bohr), mass (amu)
        for ln in coords:
            el, x, y, z = ln.split()[:4]
            zz = atomic_numbers[el.capitalize()]
            f.write(f" {el.capitalize():2s} {zz:5.1f} {float(x):14.8f} {float(y):14.8f} {float(z):14.8f} {atomic_masses[zz]:14.8f}\n")
    (traj / "input").write_text(SHARC_INPUT.format(tmax=tmax, extra=extra, veloc=veloc))
    return traj


def run_sharc(traj, timeout=1800, pysharc=None):
    """sharc.x with file-based interface calls, or pysharc (``driver.py -i <interface> input``) in-process."""
    e = env()
    if pysharc:
        cmd = [sys.executable, os.path.join(SHARC, "driver.py"), "-i", pysharc, "input"]
    else:
        cmd = [os.path.join(SHARC, "sharc.x"), "input"]
    r = subprocess.run(cmd, cwd=traj, capture_output=True, text=True, timeout=timeout, env=e)
    log = r.stdout + r.stderr + (traj / "output.log").read_text() if (traj / "output.log").exists() else r.stdout + r.stderr
    qmerr = (traj / "QM" / "QM.err").read_text() if (traj / "QM" / "QM.err").exists() else ""
    assert r.returncode == 0, (log[-3000:] + "\nQM.err:\n" + qmerr[-3000:])
    rows = [ln.split() for ln in (traj / "output.lis").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return np.array([[float(x) for x in row[:9]] for row in rows])


@requires_sharc_x
def test_sharc_dynamics_gas_phase(tmp_path, sharc_models):
    """Surface hopping on 2 singlets + 2 triplets with NACs (ddr), SOCs, dipoles and interface phases."""
    traj = make_dynamics_dir(tmp_path, sharc_models["spatial"], ALL, tmax=2.0)
    lis = run_sharc(traj)
    assert lis.shape[0] == 5 and np.isfinite(lis).all()  # t = 0 ... 2 fs
    assert lis[0, 4] > 0.1  # it moves (0.1 eV per atom initial kinetic energy)
    assert np.ptp(lis[:, 6]) < 5e-3  # eV, total energy over 4 velocity-Verlet steps
    assert (traj / "restart" / "FIELDMACE_previous.npz").exists()  # phase reference kept between steps


@requires_sharc_x
def test_sharc_dynamics_qmmm_fieldmace_openmm(tmp_path, sharc_models):
    """sharc.x -> SHARC_QMMM -> (FIELDMACE + OpenMM).

    SHARC_QMMM (SHARC 4.0) does not pass the QM program's wave-function phases on, so sharc.x must not request
    them (``nophases_from_interface``); FieldMACE keeps NAC/SOC signs consistent between steps itself.
    """
    pytest.importorskip("openmm")
    traj = make_dynamics_dir(tmp_path, sharc_models["spatial"], ALL, tmax=1.0, qmmm=True,
                             extra="nophases_from_interface", veloc="zero")
    lis = run_sharc(traj, timeout=3600)
    assert lis.shape[0] == 3 and np.isfinite(lis).all()
    assert np.ptp(lis[:, 6]) < 0.05


# --------------------------------------------------------------------------- #
# pysharc: sharc in-process, the interface stays loaded (driver.py -i FIELDMACE)
# --------------------------------------------------------------------------- #
@requires_pysharc
@requires_sharc_x
def test_pysharc_matches_sharc_x(tmp_path, sharc_models):
    """Same trajectory (same seed) with pysharc as with sharc.x calling SHARC_FIELDMACE.py each step."""
    ref = run_sharc(make_dynamics_dir(tmp_path / "x", sharc_models["spatial"], ALL, tmax=2.0))
    traj = make_dynamics_dir(tmp_path / "py", sharc_models["spatial"], ALL, tmax=2.0)
    lis = run_sharc(traj, pysharc="FIELDMACE")
    assert lis.shape == ref.shape == (5, 9)
    assert np.allclose(lis[:, :8], ref[:, :8], atol=2e-6)  # step, time, states, energies, angular momentum
    assert (traj / "restart" / "FIELDMACE_previous.npz").exists()


@requires_pysharc
def test_pysharc_qmmm_fieldmace_openmm(tmp_path, sharc_models):
    pytest.importorskip("openmm")
    traj = make_dynamics_dir(tmp_path, sharc_models["spatial"], ALL, tmax=1.0, qmmm=True,
                             extra="nophases_from_interface", veloc="zero")
    lis = run_sharc(traj, timeout=3600, pysharc="QMMM")
    assert lis.shape[0] == 3 and np.isfinite(lis).all()
    assert np.ptp(lis[:, 6]) < 0.05
