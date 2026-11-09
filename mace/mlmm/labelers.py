# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Labellers: frames (ase.Atoms with mm_positions / mm_charges in info) -> the same frames with
reference labels (REF_energy, REF_forces, REF_mm_forces). Used by the active-learning loop.

A labeller is any callable ``labeller(frames, **kwargs) -> list of labelled frames``. Frames that
fail can be left out. Built in:

    pyscf             DFT with point charges in PySCF (energies, forces on atoms and charges)
    model:<path>      a FieldMACE model as the reference (distillation, tests; all states)
    external          no labelling: the loop writes the candidates and stops; label them with your
                      QM code, write <round>/labelled.xyz and run the loop again
    file.py:function  or  package.module:function   your own

Units: eV, Angstrom, e (FieldMACE's).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Callable, List, Sequence

import ase
import numpy as np

from .dataset import BOHR_TO_A, HARTREE_TO_EV, attach_reference


def pyscf_qmmm(symbols, qm_pos, mm_pos, mm_q, xc="b3lyp", basis="6-31g*", charge=0, spin=0, density_fit=False):
    """Energy (eV), QM forces and point-charge forces (eV/A) from PySCF with point charges.

    density_fit=True: density fitting, faster (p-xylene/def2-SVP: ~0.5 meV/atom in the energy, ~2 meV/A in
    the forces compared with exact integrals)."""
    from pyscf import dft, gto, qmmm

    mol = gto.M(atom=list(zip(symbols, qm_pos)), basis=basis, unit="Angstrom", charge=charge, spin=spin, verbose=0)
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-10
    if density_fit:
        mf = mf.density_fit()
    if len(mm_q):
        mf = qmmm.mm_charge(mf, mm_pos, mm_q, unit="Angstrom")
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF not converged")
    g = mf.nuc_grad_method()
    g.verbose = 0
    g_qm = g.kernel()
    g_mm = None
    if len(mm_q):
        dm = mf.make_rdm1()
        if dm.ndim == 3:  # unrestricted (spin != 0): alpha + beta density
            dm = dm[0] + dm[1]
        g_mm = g.grad_hcore_mm(dm) + g.grad_nuc_mm()
    conv = HARTREE_TO_EV / BOHR_TO_A
    return e * HARTREE_TO_EV, -g_qm * conv, None if g_mm is None else -g_mm * conv


def pyscf_labeller(frames: Sequence[ase.Atoms], xc="b3lyp", basis="6-31g*", charge=0, spin=0, density_fit=False,
                   **_) -> List[ase.Atoms]:
    out = []
    for at in frames:
        try:
            e, f, fm = pyscf_qmmm(at.get_chemical_symbols(), at.positions, at.info.get("mm_positions", np.zeros((0, 3))),
                                  at.info.get("mm_charges", np.zeros(0)), xc=xc, basis=basis, charge=charge, spin=spin,
                                  density_fit=density_fit)
        except RuntimeError as err:  # SCF failure: skip the frame
            print(f"  skipped a frame: {err}")
            continue
        out.append(attach_reference(at, e, f, fm))
    return out


def model_labeller(frames: Sequence[ase.Atoms], model: str = None, device: str = "cpu", **_) -> List[ase.Atoms]:
    """A FieldMACE model as the reference: all states, forces on atoms and point charges.

    For a diabatic model the raw diabatic outputs are the labels (train a diabatic model on them)."""
    from mace.calculators import MACECalculator

    calc = MACECalculator(model, device=device, default_dtype="float64")
    out = []
    for at in frames:
        x = at.copy()
        x.info = {k: at.info[k] for k in ("mm_positions", "mm_charges") if k in at.info}
        calc.calculate(x)
        r = calc.results
        if "diabatic_hamiltonian" in r:
            from mace.tools.diabatic import pack

            e = pack(r["diabatic_hamiltonian"])
            f = r["diabatic_forces"]
            fm = r.get("diabatic_mm_forces")
        else:
            e, f, fm = np.atleast_1d(r["energies"]), r["forces_all"], r.get("mm_forces")
        out.append(attach_reference(at, None, f, fm, energies=e))
    return out


def get_labeller(spec: str) -> Callable:
    """'pyscf' | 'model:<path>' | 'external' | 'file.py:function' | 'package.module:function'."""
    if spec == "pyscf":
        return pyscf_labeller
    if spec == "external":
        return None
    if spec.startswith("model:"):
        path = spec[len("model:"):]

        def lab(frames, **kw):
            return model_labeller(frames, model=path, **kw)

        return lab
    if ":" not in spec:
        raise ValueError(f"unknown labeller {spec!r}: pyscf, model:<path>, external, file.py:function or module:function")
    target, func = spec.rsplit(":", 1)
    if target.endswith(".py"):
        modspec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(target))[0],
                                                         os.path.abspath(target))
        mod = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, func)
