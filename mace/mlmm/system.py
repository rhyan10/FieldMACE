# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Turn any OpenMM System (Amber, CHARMM, OPLS, GAFF, ...) into the MM part of an
electrostatically embedded FieldMACE/MM system.

Energy partitioning (electrostatic embedding)
---------------------------------------------
    E_total = E_MM(MM-MM, all bonded terms touching at least one MM atom,
                   QM-MM Lennard-Jones)
            + E_FieldMACE(QM region in the field of the MM point charges)

E_FieldMACE is assumed to contain the QM energy *including* the electrostatic
interaction of the QM electrons and nuclei with the MM point charges (this is
what QM codes return for a calculation with external point charges).
Therefore this module

* removes every bonded term whose atoms are all QM atoms,
* zeroes the charges of QM atoms in the NonbondedForce (this removes QM-QM
  and QM-MM Coulomb; LJ is kept),
* excludes every QM-QM pair from all nonbonded forces (removes QM-QM LJ),
* removes constraints between QM atoms (optional, on by default),
* collects the MM point charges that FieldMACE will see.

The force field itself is never inspected beyond the standard OpenMM force
classes, so anything that OpenMM can build works.
"""
from __future__ import annotations

import itertools
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import openmm as mm
from openmm import unit

# Typical X-H bond lengths (Angstrom) used to place hydrogen link atoms.
LINK_XH_DISTANCE = {6: 1.09, 7: 1.01, 8: 0.96, 16: 1.34, 15: 1.42, 14: 1.48}

_BONDED_TYPES = {
    "HarmonicBondForce",
    "HarmonicAngleForce",
    "PeriodicTorsionForce",
    "RBTorsionForce",
    "CMAPTorsionForce",
    "CustomBondForce",
    "CustomAngleForce",
    "CustomTorsionForce",
    "CustomCompoundBondForce",
    "AmoebaBondForce",
    "AmoebaAngleForce",
    "AmoebaInPlaneAngleForce",
    "AmoebaOutOfPlaneBendForce",
    "AmoebaPiTorsionForce",
    "AmoebaStretchBendForce",
    "AmoebaTorsionTorsionForce",
    "AmoebaUreyBradleyForce",
}
_UNSUPPORTED_TYPES = {
    "GBSAOBCForce": "implicit solvent is not part of the FieldMACE embedding",
    "CustomGBForce": "implicit solvent is not part of the FieldMACE embedding",
    "AmoebaMultipoleForce": "polarisable multipoles are not supported as embedding",
    "AmoebaGeneralizedKirkwoodForce": "implicit solvent is not supported",
    "DrudeForce": "Drude polarisable force fields are not supported",
    "CustomCentroidBondForce": "cannot decide automatically which groups are QM",
    "CustomManyParticleForce": "cannot decide automatically which terms are QM",
}
_ATOM_ATTR = re.compile(r"^(p\d+|a\d+|b\d+)$")
_CHARGE_LIKE = re.compile(r"^(q|charge|chargeprod|qq|q\d)$", re.IGNORECASE)


@dataclass
class LinkAtom:
    """Hydrogen link atom placed on the QM-MM bond qm_atom -> mm_atom.

    position = r_qm + ratio * (r_mm - r_qm)
    """

    qm_atom: int
    mm_atom: int
    ratio: float


@dataclass
class QMMMSpec:
    """Everything the FieldMACE force and the data exporter need to know."""

    qm_atoms: List[int]
    qm_atomic_numbers: List[int]
    qm_masses: np.ndarray  # amu, QM atoms only (link atoms excluded)
    mm_atoms: List[int]  # MM particles that carry a non-zero embedding charge
    mm_charges: np.ndarray  # elementary charges, aligned with mm_atoms
    link_atoms: List[LinkAtom] = field(default_factory=list)
    periodic: bool = False

    @property
    def n_qm(self) -> int:
        return len(self.qm_atoms)

    @property
    def model_atomic_numbers(self) -> List[int]:
        """Atomic numbers of the atoms FieldMACE sees: QM atoms then H link atoms."""
        return list(self.qm_atomic_numbers) + [1] * len(self.link_atoms)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bond_graph(system: mm.System, topology=None) -> Dict[int, set]:
    graph: Dict[int, set] = {i: set() for i in range(system.getNumParticles())}

    def add(i, j):
        graph[i].add(j)
        graph[j].add(i)

    if topology is not None:
        for b in topology.bonds():
            add(b[0].index, b[1].index)
    else:
        for f in system.getForces():
            if isinstance(f, mm.HarmonicBondForce):
                for k in range(f.getNumBonds()):
                    i, j, *_ = f.getBondParameters(k)
                    add(i, j)
        for k in range(system.getNumConstraints()):
            i, j, _ = system.getConstraintParameters(k)
            add(i, j)
    return graph


def _equilibrium_bond_length(system: mm.System, i: int, j: int) -> Optional[float]:
    """r0 in Angstrom of the bond i-j from HarmonicBondForce or a constraint."""
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                a, b, r0, _ = f.getBondParameters(k)
                if {a, b} == {i, j}:
                    return r0.value_in_unit(unit.angstrom)
    for k in range(system.getNumConstraints()):
        a, b, r0 = system.getConstraintParameters(k)
        if {a, b} == {i, j}:
            return r0.value_in_unit(unit.angstrom)
    return None


def _atomic_numbers(system: mm.System, atoms: Sequence[int], topology=None,
                    atomic_numbers: Optional[Sequence[int]] = None) -> List[int]:
    if atomic_numbers is not None:
        return [int(z) for z in atomic_numbers]
    if topology is not None:
        topo_atoms = list(topology.atoms())
        out = []
        for i in atoms:
            el = topo_atoms[i].element
            if el is None:
                raise ValueError(f"QM atom {i} has no element (virtual site?)")
            out.append(el.atomic_number)
        return out
    # fall back to guessing from the mass
    from openmm.app import element as elem

    out = []
    for i in atoms:
        m = system.getParticleMass(i).value_in_unit(unit.dalton)
        best = min(
            (e for e in elem.Element._elements_by_symbol.values() if e.atomic_number > 0),
            key=lambda e: abs(e.mass.value_in_unit(unit.dalton) - m),
        )
        out.append(best.atomic_number)
    warnings.warn("No topology given: QM elements were guessed from particle masses "
                  "(this fails with hydrogen mass repartitioning). Pass topology= or "
                  "atomic_numbers= to be safe.")
    return out


def _filter_bonded_forces_xml(system: mm.System, qm: set, keep: set) -> mm.System:
    """Remove all bonded terms whose atoms are all QM, generically via XML."""
    root = ET.fromstring(mm.XmlSerializer.serialize(system))
    forces = root.find("Forces")
    n_removed: Dict[str, int] = {}
    for idx, f in enumerate(forces):
        ftype = f.get("type")
        if ftype not in _BONDED_TYPES or idx in keep or f.get("name") in keep:
            continue
        for container in list(f):
            for term in list(container):
                atoms = [int(v) for k, v in term.attrib.items() if _ATOM_ATTR.match(k)]
                if atoms and all(a in qm for a in atoms):
                    container.remove(term)
                    n_removed[ftype] = n_removed.get(ftype, 0) + 1
        if ftype == "CustomBondForce":
            names = [p.get("name") for p in f.iter("Parameter")]
            if any(_CHARGE_LIKE.match(n or "") for n in names):
                warnings.warn(
                    f"CustomBondForce '{f.get('name')}' has a charge-like parameter "
                    f"{names}; if it holds 1-4 Coulomb pairs between QM and MM atoms "
                    "that electrostatics is double counted. Zero it yourself if so.")
    new = mm.XmlSerializer.deserialize(ET.tostring(root, encoding="unicode"))
    return new, n_removed


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def prepare_qmmm_system(
    system: mm.System,
    qm_atoms: Iterable[int],
    topology=None,
    atomic_numbers: Optional[Sequence[int]] = None,
    link_atoms: bool | Sequence[Tuple[int, int]] = "auto",
    link_charge_scheme: str = "Z1",
    remove_qm_constraints: bool = True,
    keep_forces: Iterable = (),
    mm_charges: Optional[Sequence[float]] = None,
    charge_threshold: float = 1e-8,
) -> Tuple[mm.System, QMMMSpec]:
    """Build the MM part of a FieldMACE/MM system from an arbitrary OpenMM System.

    Parameters
    ----------
    system : openmm.System
        Built by any force field (``ForceField.createSystem``, AmberPrmtopFile,
        GromacsTopFile with OPLS, CharmmPsfFile, ParmEd, openmmforcefields ...).
        It is not modified; a copy is returned.
    qm_atoms : iterable of int
        Particle indices of the QM (FieldMACE) region.
    topology : openmm.app.Topology, optional
        Used for elements and bonds (recommended).
    atomic_numbers : sequence of int, optional
        Atomic numbers of the QM atoms (overrides topology / mass guessing).
    link_atoms : "auto", False, or list of (qm_atom, mm_atom)
        "auto" caps every covalent QM-MM bond with a hydrogen link atom.
    link_charge_scheme : "Z0" | "Z1" | "Z2" | "Z3"
        Which MM charges near a link are hidden from FieldMACE: Z1 removes the
        MM link host, Z2 also its bonded neighbours, Z3 one shell further.
        Z0 keeps all of them.
    remove_qm_constraints : bool
        Remove constraints between two QM atoms (FieldMACE describes them).
    keep_forces : iterable of int or str
        Force indices or names whose bonded terms must not be filtered
        (e.g. a restraint you added between QM atoms).
    mm_charges : sequence of float, optional
        Embedding charges for *all* particles (e). Defaults to the charges of
        the NonbondedForce (or a 'charge'/'q' parameter of a CustomNonbondedForce).
    """
    qm = sorted(set(int(i) for i in qm_atoms))
    qm_set = set(qm)
    n = system.getNumParticles()
    if not qm or qm[-1] >= n:
        raise ValueError("qm_atoms must be non-empty indices into the system")
    for i in qm:
        if system.isVirtualSite(i):
            raise ValueError(f"QM atom {i} is a virtual site")
    keep = set(keep_forces)

    # virtual sites built only from QM atoms (e.g. the M-site of a QM TIP4P water)
    # belong to the QM region: no MM charge, not part of the embedding
    qm_vsites = []
    for i in range(n):
        if system.isVirtualSite(i):
            vs = system.getVirtualSite(i)
            parents = {vs.getParticle(k) for k in range(vs.getNumParticles())}
            if parents <= qm_set:
                qm_vsites.append(i)
            elif parents & qm_set:
                raise ValueError(f"virtual site {i} has both QM and MM parent atoms {sorted(parents)}")
    qm_ext = sorted(qm_set | set(qm_vsites))
    qm_ext_set = set(qm_ext)

    for f in system.getForces():
        name = f.__class__.__name__
        if name in _UNSUPPORTED_TYPES and f.getName() not in keep:
            raise NotImplementedError(f"{name}: {_UNSUPPORTED_TYPES[name]}")

    # ---- embedding charges of all particles (before we zero anything) ------
    charges = np.zeros(n)
    if mm_charges is not None:
        charges = np.asarray(mm_charges, dtype=float)
        if charges.shape != (n,):
            raise ValueError("mm_charges must have one entry per particle")
    else:
        found = False
        for f in system.getForces():
            if isinstance(f, mm.NonbondedForce):
                for i in range(n):
                    charges[i] += f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                found = True
        if not found:
            for f in system.getForces():
                if isinstance(f, mm.CustomNonbondedForce):
                    names = [f.getPerParticleParameterName(k) for k in range(f.getNumPerParticleParameters())]
                    for k, nm_ in enumerate(names):
                        if nm_.lower() in ("q", "charge"):
                            charges[:] = [f.getParticleParameters(i)[k] for i in range(n)]
                            found = True
        if not found:
            warnings.warn("No point charges found in the system: FieldMACE will see an empty field.")

    # ---- link atoms ----------------------------------------------------------
    graph = _bond_graph(system, topology)
    qm_z = _atomic_numbers(system, qm, topology, atomic_numbers)
    z_of = dict(zip(qm, qm_z))
    if link_atoms == "auto" or link_atoms is True:
        pairs = sorted({(i, j) for i in qm for j in graph[i] if j not in qm_set})
    elif not link_atoms:
        pairs = []
    else:
        pairs = [(int(i), int(j)) for i, j in link_atoms]
    if str(link_charge_scheme).upper() not in ("Z0", "Z1", "Z2", "Z3"):
        raise ValueError(f"link_charge_scheme '{link_charge_scheme}': use Z0, Z1 (default), Z2 or Z3")
    links = []
    mm_z = dict(zip([j for _, j in pairs], _atomic_numbers(system, [j for _, j in pairs], topology, None))) \
        if pairs else {}
    odd_cuts = []
    for i, j in pairs:
        if i not in qm_set or j in qm_set:
            raise ValueError(f"link pair ({i}, {j}) must be (QM atom, MM atom)")
        if mm_z.get(j) == 1:
            raise ValueError(f"the QM region cuts the bond {i}-{j} to a hydrogen ({j} is MM): put that hydrogen "
                             "in the QM region (link atoms replace heavy MM atoms, never hydrogens)")
        if z_of[i] == 1:
            raise ValueError(f"QM hydrogen {i} is bonded to MM atom {j}: the QM region cuts an X-H bond; include "
                             f"atom {j} in the QM region, or remove hydrogen {i} from it")
        # good boundaries: the QM side an sp3 carbon (4 bonds), the MM side C or N (e.g. CA-CB, CA-C, CA-N)
        if z_of[i] != 6 or len(graph[i]) != 4 or mm_z.get(j) not in (6, 7):
            odd_cuts.append((i, j))
        r0 = _equilibrium_bond_length(system, i, j) or 1.53
        d_xh = LINK_XH_DISTANCE.get(z_of[i], 1.09)
        links.append(LinkAtom(i, j, d_xh / r0))
    if pairs and link_atoms == "auto":
        warnings.warn(f"{len(pairs)} covalent QM-MM bond(s) found; capping with H link atoms: {pairs}. "
                      "Your training data must be generated with the same QMMMSpec.")
    if odd_cuts:
        warnings.warn(f"QM-MM boundary through bond(s) {odd_cuts} (QM atom, MM atom): hydrogen link atoms are "
                      "reliable when the QM atom is an sp3 carbon bonded to an MM C or N; cutting at heteroatoms, "
                      "double, aromatic or amide bonds distorts the QM region. Move the boundary if you can.")
    if links and str(link_charge_scheme).upper() == "Z0":
        warnings.warn("link_charge_scheme Z0 keeps the MM link host's charge, ~0.4 A from the link hydrogen: "
                      "this strongly over-polarises the QM region. Z1 (default) or Z2 is usually better.")

    # MM charges hidden from FieldMACE next to links
    hidden = set()
    depth = {"Z0": 0, "Z1": 1, "Z2": 2, "Z3": 3}[link_charge_scheme.upper()]
    frontier = {lk.mm_atom for lk in links}
    for _ in range(depth):
        hidden |= frontier
        frontier = {k for a in frontier for k in graph[a]} - qm_set - hidden

    mm_list = [i for i in range(n)
               if i not in qm_ext_set and i not in hidden and abs(charges[i]) > charge_threshold]

    # ---- MM system -------------------------------------------------------------
    new, removed = _filter_bonded_forces_xml(system, qm_set, keep)

    if remove_qm_constraints:
        for k in reversed(range(new.getNumConstraints())):
            i, j, _ = new.getConstraintParameters(k)
            if i in qm_set and j in qm_set:
                new.removeConstraint(k)

    qm_pairs = list(itertools.combinations(qm_ext, 2))
    for f in new.getForces():
        if isinstance(f, mm.NonbondedForce):
            for i in qm_ext:
                q, sig, eps = f.getParticleParameters(i)
                f.setParticleParameters(i, 0.0, sig, eps)
            for k in range(f.getNumParticleParameterOffsets()):
                param, i, qs, sigs, epss = f.getParticleParameterOffset(k)
                if i in qm_ext_set and qs != 0:
                    f.setParticleParameterOffset(k, param, i, 0.0, sigs, epss)
            for k in range(f.getNumExceptions()):
                i, j, qq, sig, eps = f.getExceptionParameters(k)
                if i in qm_ext_set or j in qm_ext_set:
                    if i in qm_ext_set and j in qm_ext_set:
                        eps = 0.0
                    f.setExceptionParameters(k, i, j, 0.0, sig, eps)
            for k in range(f.getNumExceptionParameterOffsets()):
                param, idx, qqs, sigs, epss = f.getExceptionParameterOffset(k)
                i, j, *_ = f.getExceptionParameters(idx)
                if i in qm_ext_set or j in qm_ext_set:
                    f.setExceptionParameterOffset(k, param, idx, 0.0, sigs, 0.0 if (i in qm_ext_set and j in qm_ext_set) else epss)
            for i, j in qm_pairs:
                f.addException(i, j, 0.0, 1.0, 0.0, replace=True)
        elif isinstance(f, mm.CustomNonbondedForce):
            names = [f.getPerParticleParameterName(k) for k in range(f.getNumPerParticleParameters())]
            if any(_CHARGE_LIKE.match(x) for x in names):
                warnings.warn(f"CustomNonbondedForce '{f.getName()}' has charge-like parameters {names}; "
                              "zeroing them for QM atoms.")
                for i in qm_ext:
                    p = list(f.getParticleParameters(i))
                    for k, x in enumerate(names):
                        if _CHARGE_LIKE.match(x):
                            p[k] = 0.0
                    f.setParticleParameters(i, p)
            excluded = {tuple(sorted(f.getExclusionParticles(k))) for k in range(f.getNumExclusions())}
            for i, j in qm_pairs:
                if (i, j) not in excluded:
                    f.addExclusion(i, j)
        elif isinstance(f, mm.CustomBondForce):
            pass  # handled (QM-only terms removed) in the XML pass

    spec = QMMMSpec(
        qm_atoms=qm,
        qm_atomic_numbers=qm_z,
        qm_masses=np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in qm]),
        mm_atoms=mm_list,
        mm_charges=charges[mm_list].copy(),
        link_atoms=links,
        periodic=system.usesPeriodicBoundaryConditions(),
    )
    return new, spec
