# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""SHARC 4 with a diabatic FieldMACE model: the interface diagonalises the diabatic Hamiltonian and gives
SHARC adiabatic (MCH) energies, gradients, NACs (also on point charges / MM atoms), SOCs, dipoles and
wave-function overlaps."""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import CH2S, CH2S_EL, SHARC, requires_pysharc, requires_sharc, write_pcfile, write_qm_in
from test_sharc4 import (BOHR, HARTREE_PER_EV, SHARC_INPUT, env, evaluate, interface, make_dynamics_dir,
                         make_qmmm_dir, read_qmout, run_file_mode, run_sharc, setup_dir)

PROPS = ["energy", "forces", "dipoles", "socs"]
REQ = ["h", "soc", "dm", "grad", "nacdr"]


@requires_sharc
def test_file_mode_adiabatic_from_diabatic(tmp_path, sharc_models):
    from mace.calculators import MACECalculator
    import ase

    d = setup_dir(tmp_path, sharc_models["diabatic"], PROPS, [2, 0, 2], REQ, pc=True)
    run_file_mode(d)
    q = read_qmout(d / "QM.out", [2, 0, 2], 4, 12)
    assert np.allclose(q.h, q.h.conj().T)
    raw = np.loadtxt(d / "pc.txt")
    atoms = ase.Atoms(CH2S_EL, positions=CH2S * BOHR)
    atoms.info.update(mm_positions=raw[:, :3] * BOHR, mm_charges=raw[:, 3])
    calc = MACECalculator(sharc_models["diabatic"], device="cpu", default_dtype="float64")
    calc.calculate(atoms)
    H = calc.results["diabatic_hamiltonian"]
    e = np.real(np.diag(q.h)) / HARTREE_PER_EV
    # singlets and triplets diagonalised separately; the triplet M_S components are degenerate
    assert np.allclose(e[:2], np.linalg.eigvalsh(H[:2, :2]), atol=1e-8)
    assert np.allclose(e[2:4], np.linalg.eigvalsh(H[2:, 2:]), atol=1e-8) and np.allclose(e[2:4], e[6:8])
    # NACs only within a multiplicity and between equal M_S; antisymmetric; also on the point charges
    assert np.allclose(q.nacdr, -np.transpose(q.nacdr, (1, 0, 2, 3)))
    assert not np.allclose(q.nacdr[0, 1], 0) and np.allclose(q.nacdr[2, 3], q.nacdr[6, 7])
    assert np.allclose(q.nacdr[0, 2], 0) and np.allclose(q.nacdr[2, 5], 0)
    assert q.grad_pc.shape == (8, 12, 3)
    # next step (SHARC asks for overlaps from step 1 on): same geometry -> overlap = identity
    write_qm_in(d / "QM.in", CH2S, CH2S_EL, [2, 0, 2], REQ + ["overlap"], pcfile=str(d / "pc.txt"), step=1, init=False)
    run_file_mode(d)
    q1 = read_qmout(d / "QM.out", [2, 0, 2], 4, 12)
    assert np.allclose(np.real(q1.overlap), np.eye(8), atol=1e-10) and np.allclose(q1.nacdr, q.nacdr)
    # SOC selection rules still hold after the rotation to the adiabatic basis
    assert np.abs(q.h[0, 1]) == 0 and np.abs(q.h[2, 4]) == 0 and np.abs(q.h[0, 2]) > 0


@requires_sharc
def test_gradients_nacs_and_overlaps_are_consistent(tmp_path, sharc_models):
    """grad == dE/dx; NAC d_kl == <k(x-h)|l(x+h)> / 2h from the interface's own overlaps; nacdr_pc likewise."""
    d = setup_dir(tmp_path, sharc_models["diabatic"], PROPS, [2, 0, 2], ["h", "grad", "nacdr"], pc=True)
    it = interface(d)
    q0 = evaluate(it, d)
    it.QMin.requests["overlap"] = True  # from the second step on, as sharc.x does
    it.QMin.save["init"] = False
    pcc = np.loadtxt(d / "pc.txt")[:, :3].copy()
    h = 1e-4  # bohr
    for atom, k in ((0, 0), (1, 2)):
        res = []
        for s in (-1, 1):  # x-h then x+h: the second overlap is U(x-h)^T U(x+h)
            c = CH2S.copy()
            c[atom, k] += s * h
            res.append(evaluate(it, d, coords=c, pc=pcc))
        de = (np.real(np.diag(res[1].h)) - np.real(np.diag(res[0].h))) / (2 * h)
        assert np.allclose(de, q0.grad[:, atom, k], atol=1e-7)
        S = np.real(res[1].overlap)
        assert np.allclose(S[0, 1] / (2 * h), q0.nacdr[0, 1, atom, k], atol=1e-5, rtol=1e-3)
        assert np.allclose(S[2, 3] / (2 * h), q0.nacdr[2, 3, atom, k], atol=1e-5, rtol=1e-3)
    assert q0.nacdr_pc.shape == (8, 8, 12, 3)
    res = []
    for s in (-1, 1):
        p = pcc.copy()
        p[3, 1] += s * h
        res.append(evaluate(it, d, coords=CH2S, pc=p))
    S = np.real(res[1].overlap)
    assert np.allclose(S[0, 1] / (2 * h), q0.nacdr_pc[0, 1, 3, 1], atol=1e-5, rtol=1e-3)
    assert not np.allclose(q0.nacdr_pc[0, 1], 0)


@requires_sharc
def test_overlap_needs_diabatic_model(tmp_path, sharc_models):
    d = setup_dir(tmp_path, sharc_models["spatial"], ["energy", "forces", "nacs"], [2, 0, 2], ["h", "grad"], pc=False)
    it = interface(d)
    evaluate(it, d)
    it.QMin.requests["overlap"] = True
    it.QMin.save["init"] = False
    with pytest.raises(ValueError, match="diabatic"):
        evaluate(it, d)


@requires_sharc
def test_sharc_qmmm_passes_mm_nacs(tmp_path, sharc_models):
    """SHARC_QMMM + OpenMM: NACs on the MM atoms come from nacdr_pc (adiabatic models cannot give them)."""
    pytest.importorskip("openmm")
    d, natom = make_qmmm_dir(tmp_path, sharc_models["diabatic"], ["h", "soc", "grad", "nacdr"])
    r = subprocess.run([sys.executable, os.path.join(SHARC, "SHARC_QMMM.py"), "QM.in"], cwd=d, capture_output=True,
                       text=True, timeout=900, env=env())
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-4000:]
    assert "No 'nacdr_pc'" not in log and "No 'grad_pc'" not in log
    q = read_qmout(d / "QM.out", [2, 0, 2], natom, 0)
    assert np.isfinite(q.nacdr).all()
    assert np.abs(q.nacdr[0, 1, 4:]).sum() > 0  # MM atoms


@requires_pysharc
@pytest.mark.parametrize("coupling", ["ddr", "overlap"])
def test_pysharc_dynamics_diabatic(tmp_path, sharc_models, coupling):
    traj = make_dynamics_dir(tmp_path, sharc_models["diabatic"], PROPS, tmax=2.0)
    if coupling == "overlap":
        (traj / "input").write_text((traj / "input").read_text().replace("coupling     ddr", "coupling     overlap"))
    lis = run_sharc(traj, pysharc="FIELDMACE")
    assert lis.shape[0] == 5 and np.isfinite(lis).all()
    assert np.ptp(lis[:, 6]) < 5e-3
