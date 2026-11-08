# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Load an OpenMM System from the common force-field input formats and select QM atoms.

    Amber      : --prmtop sys.prmtop --inpcrd sys.inpcrd
    GROMACS    : --top topol.top --gro conf.gro [--gmx-include /usr/share/gromacs/top]
                 (this is the usual route for OPLS-AA, e.g. LigParGen / GROMACS oplsaa.ff)
    CHARMM     : --psf sys.psf --crd/--pdb ... --charmm-params par.prm top.rtf ...
    OpenMM XML : --pdb sys.pdb --forcefield amber14-all.xml amber14/tip3pfb.xml
                 (any ffxml, incl. openmmforcefields GAFF/OpenFF/CHARMM36 files)
    Serialized : --system system.xml --pdb sys.pdb
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import openmm as mm
from openmm import app, unit


@dataclass
class LoadedSystem:
    system: mm.System
    topology: app.Topology
    positions: unit.Quantity
    box_vectors: Optional[tuple] = None


def load_system(
    pdb: Optional[str] = None,
    forcefield: Sequence[str] = (),
    prmtop: Optional[str] = None,
    inpcrd: Optional[str] = None,
    top: Optional[str] = None,
    gro: Optional[str] = None,
    gmx_include: Optional[str] = None,
    psf: Optional[str] = None,
    crd: Optional[str] = None,
    charmm_params: Sequence[str] = (),
    system_xml: Optional[str] = None,
    nonbonded_method=app.PME,
    cutoff_nm: float = 1.0,
    constraints=app.HBonds,
    rigid_water: bool = True,
) -> LoadedSystem:
    kw = dict(nonbondedMethod=nonbonded_method, nonbondedCutoff=cutoff_nm * unit.nanometer,
              constraints=constraints, rigidWater=rigid_water)
    if prmtop:
        prm = app.AmberPrmtopFile(prmtop)
        crd_ = app.AmberInpcrdFile(inpcrd)
        box = crd_.boxVectors
        if box is not None:
            prm.topology.setPeriodicBoxVectors(box)
        if box is None and nonbonded_method in (app.PME, app.Ewald, app.CutoffPeriodic):
            kw["nonbondedMethod"] = app.NoCutoff
        return LoadedSystem(prm.createSystem(**kw), prm.topology, crd_.positions, box)
    if top:
        g = app.GromacsGroFile(gro)
        t = app.GromacsTopFile(top, periodicBoxVectors=g.getPeriodicBoxVectors(), includeDir=gmx_include)
        return LoadedSystem(t.createSystem(**kw), t.topology, g.positions, g.getPeriodicBoxVectors())
    if psf:
        p = app.CharmmPsfFile(psf)
        if crd:
            pos = app.CharmmCrdFile(crd).positions
            box = None
        else:
            pdbf = app.PDBFile(pdb)
            pos, box = pdbf.positions, pdbf.topology.getPeriodicBoxVectors()
        if box is not None:
            p.setBox(*[box[i][i] for i in range(3)])
        params = app.CharmmParameterSet(*charmm_params)
        return LoadedSystem(p.createSystem(params, **kw), p.topology, pos, box)
    if system_xml:
        pdbf = app.PDBFile(pdb)
        with open(system_xml) as fh:
            system = mm.XmlSerializer.deserialize(fh.read())
        return LoadedSystem(system, pdbf.topology, pdbf.positions, system.getDefaultPeriodicBoxVectors())
    if pdb:
        pdbf = app.PDBFile(pdb)
        ff = app.ForceField(*forcefield)
        if pdbf.topology.getPeriodicBoxVectors() is None and nonbonded_method in (app.PME, app.Ewald, app.CutoffPeriodic):
            kw["nonbondedMethod"] = app.NoCutoff
        return LoadedSystem(ff.createSystem(pdbf.topology, **kw), pdbf.topology, pdbf.positions,
                            pdbf.topology.getPeriodicBoxVectors())
    raise ValueError("No input given")


def select_atoms(topology: app.Topology, selection: str) -> List[int]:
    """Tiny selection language, terms combined with 'or':
        'resname LIG'          residue name(s)
        'resid 12 13'          residue ids as in the input file
        'index 0-21 30 31'     0-based particle indices / ranges
        'chain B and resid 5'  is NOT supported; use index for anything fancy
    For complex selections use MDTraj (``md.Topology.from_openmm(top).select(...)``).
    """
    atoms = list(topology.atoms())
    out = set()
    for term in selection.split(" or "):
        key, *vals = term.split()
        key = key.lower()
        if key == "resname":
            out |= {a.index for a in atoms if a.residue.name in vals}
        elif key == "resid":
            out |= {a.index for a in atoms if str(a.residue.id) in vals}
        elif key == "index":
            for v in vals:
                if "-" in v:
                    lo, hi = map(int, v.split("-"))
                    out |= set(range(lo, hi + 1))
                else:
                    out.add(int(v))
        else:
            raise ValueError(f"unknown selection keyword '{key}'")
    if not out:
        raise ValueError(f"selection '{selection}' matched no atoms")
    return sorted(out)


def constrain_hbonds_except(system: mm.System, topology: app.Topology, exclude) -> int:
    """Add X-H bond constraints for every bond that does not involve an atom in ``exclude``.

    Use with a System created with ``constraints=None``: the QM region keeps flexible
    X-H bonds (with their force-field bond terms) while the MM region can use 2 fs steps.
    Returns the number of constraints added.
    """
    ex = set(exclude)
    r0 = {}
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                i, j, r, _ = f.getBondParameters(k)
                r0[frozenset((i, j))] = r
    have = {frozenset(system.getConstraintParameters(k)[:2]) for k in range(system.getNumConstraints())}
    n = 0
    for a, b in topology.bonds():
        if a.index in ex or b.index in ex:
            continue
        if (a.element is not None and a.element.symbol == "H") or (b.element is not None and b.element.symbol == "H"):
            key = frozenset((a.index, b.index))
            if key in have or key not in r0:
                continue
            system.addConstraint(a.index, b.index, r0[key])
            n += 1
    return n
