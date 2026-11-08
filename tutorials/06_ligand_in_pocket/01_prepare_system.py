#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Step 1: p-xylene in the L99A pocket of T4 lysozyme, in explicit water.

The complex (protein + ligand, Amber/GAFF, built with tleap) ships with openmmtools
(data/T4-lysozyme-L99A-implicit). The protein is described with Amber14, the water with TIP3P,
and the ligand with its GAFF parameters from ligand.prmtop:

  * a small force-field XML for the ligand (one atom type per atom, its GAFF charge and LJ) lets
    OpenMM's ForceField and Modeller handle it like any residue (solvation, the system),
  * the ligand's bonds, angles and torsions (incl. impropers) are then copied atom by atom from
    ligand.prmtop. The script checks that the ligand's energy equals the prmtop energy.

The same route works for any ligand you have Amber parameters for (antechamber/tleap prmtop).
With openmmforcefields (GAFF/OpenFF template generators), build the system directly instead.

QM region: the ligand (18 atoms, no covalent bond to the protein, so no link atoms).
Embedding: every protein, water and ion charge within 12 A of the ligand centre, switched off
smoothly between 10 and 12 A.

Outputs (work/):
  system_classical.xml   OpenMM System (protein Amber14, water TIP3P, ligand GAFF), PME
  equilibrated.pdb       solvated, equilibrated complex (truncated octahedron)
  setup.json             QM atoms and embedding (for the data and for ML/MM MD)
"""
import argparse
import os
import tempfile

import numpy as np
import openmm as mm
from openmm import app, unit

from mace.mlmm import EmbeddingSettings, prepare_qmmm_system
from mace.mlmm.dataset import save_setup

HERE = os.path.dirname(os.path.abspath(__file__))
LIG = "TMP"


def openmmtools_data():
    import importlib.util

    spec = importlib.util.find_spec("openmmtools")  # (without importing it: no pymbar banner)
    if spec is None:
        raise SystemExit("openmmtools not found (pip install openmmtools): it ships the T4 lysozyme complex")
    d = os.path.join(spec.submodule_search_locations[0], "data", "T4-lysozyme-L99A-implicit")
    if not os.path.isdir(d):
        raise SystemExit("openmmtools' T4 lysozyme data not found (pip install openmmtools)")
    return d


def ligand_ffxml(prmtop, path, resname=LIG):
    """Nonbonded-only ForceField XML for the ligand: one atom type per atom, GAFF charge and LJ."""
    prm = app.AmberPrmtopFile(prmtop)
    sys_ = prm.createSystem(nonbondedMethod=app.NoCutoff)
    nb = [f for f in sys_.getForces() if isinstance(f, mm.NonbondedForce)][0]
    atoms = list(prm.topology.atoms())
    lines = ["<ForceField>", " <AtomTypes>"]
    for a in atoms:
        lines.append(f'  <Type name="{resname}-{a.name}" class="{resname}-{a.name}" element="{a.element.symbol}" '
                     f'mass="{a.element.mass.value_in_unit(unit.dalton)}"/>')
    lines += [" </AtomTypes>", " <Residues>", f'  <Residue name="{resname}">']
    for a in atoms:
        q = nb.getParticleParameters(a.index)[0].value_in_unit(unit.elementary_charge)
        lines.append(f'   <Atom name="{a.name}" type="{resname}-{a.name}" charge="{q:.6f}"/>')
    for b in prm.topology.bonds():
        lines.append(f'   <Bond atomName1="{b[0].name}" atomName2="{b[1].name}"/>')
    lines += ["  </Residue>", " </Residues>",
              ' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">',
              '  <UseAttributeFromResidue name="charge"/>']
    for a in atoms:
        _, sig, eps = nb.getParticleParameters(a.index)
        lines.append(f'  <Atom type="{resname}-{a.name}" sigma="{sig.value_in_unit(unit.nanometer):.8f}" '
                     f'epsilon="{eps.value_in_unit(unit.kilojoule_per_mole):.8f}"/>')
    lines += [" </NonbondedForce>", "</ForceField>"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return prm, sys_


def add_ligand_bonded(system, topology, lig_prm, lig_sys, resname=LIG):
    """Copy the ligand's bonds, angles and torsions from its prmtop System (matched by atom name)."""
    lig_atoms = [a for a in topology.atoms() if a.residue.name == resname]
    by_name = {a.name: a.index for a in lig_atoms}
    idx = {a.index: by_name[a.name] for a in lig_prm.topology.atoms()}
    bonds = mm.HarmonicBondForce()
    angles = mm.HarmonicAngleForce()
    tors = mm.PeriodicTorsionForce()
    for f in lig_sys.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                i, j, r0, kk = f.getBondParameters(k)
                bonds.addBond(idx[i], idx[j], r0, kk)
        elif isinstance(f, mm.HarmonicAngleForce):
            for k in range(f.getNumAngles()):
                i, j, l, a0, kk = f.getAngleParameters(k)
                angles.addAngle(idx[i], idx[j], idx[l], a0, kk)
        elif isinstance(f, mm.PeriodicTorsionForce):
            for k in range(f.getNumTorsions()):
                i, j, l, m, n, ph, kk = f.getTorsionParameters(k)
                tors.addTorsion(idx[i], idx[j], idx[l], idx[m], n, ph, kk)
    for f in (bonds, angles, tors):
        f.setName(f"{resname}_bonded")
        system.addForce(f)
    return [a.index for a in lig_atoms]


def check_ligand_energy(ff_files, lig_prm, lig_sys, lig_xml, crd):
    """Ligand alone: energy from ForceField + copied bonded terms == energy from ligand.prmtop."""
    ff = app.ForceField(*ff_files, lig_xml)
    pos = app.AmberInpcrdFile(crd).positions
    s = ff.createSystem(lig_prm.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    add_ligand_bonded(s, lig_prm.topology, lig_prm, lig_sys)

    def energy(system):
        c = mm.Context(system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName("Reference"))
        c.setPositions(pos)
        return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    return energy(s), energy(lig_sys)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--padding-nm", type=float, default=1.0)
    p.add_argument("--equil-ps", type=float, default=50.0)
    p.add_argument("--cutoff", type=float, default=12.0, help="embedding cutoff (A) around the ligand centre")
    p.add_argument("--switch", type=float, default=2.0, help="switching width (A)")
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)
    d = openmmtools_data()
    ff_files = ["amber14-all.xml", "amber14/tip3p.xml"]
    lig_xml = os.path.join(a.work, "ligand.xml")
    lig_prm, lig_sys = ligand_ffxml(os.path.join(d, "ligand.prmtop"), lig_xml)
    e_ff, e_prm = check_ligand_energy(ff_files, lig_prm, lig_sys, lig_xml, os.path.join(d, "ligand-minimized.crd"))
    print(f"ligand energy: ForceField + copied bonded terms {e_ff:.4f}, ligand.prmtop {e_prm:.4f} kJ/mol")
    if abs(e_ff - e_prm) > 1e-3:
        raise SystemExit("ligand parameters were not transferred correctly")

    pdb = app.PDBFile(os.path.join(d, "complex.pdb"))
    ff = app.ForceField(*ff_files, lig_xml)
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addSolvent(ff, padding=a.padding_nm * unit.nanometer, boxShape="octahedron", model="tip3p",
                   neutralize=True)
    n_wat = sum(r.name == "HOH" for r in mod.topology.residues())
    ions = sum(r.name in ("NA", "CL") for r in mod.topology.residues())
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    lig = add_ligand_bonded(system, mod.topology, lig_prm, lig_sys)
    print(f"solvated: {mod.topology.getNumAtoms()} atoms ({n_wat} waters, {ions} ions), ligand atoms {lig[0]}-{lig[-1]}")

    # equilibrate: minimise, then NVT with the protein heavy atoms restrained, then without
    restr = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restr.addGlobalParameter("k", 1000.0)
    for name in ("x0", "y0", "z0"):
        restr.addPerParticleParameter(name)
    pos0 = mod.positions.value_in_unit(unit.nanometer)
    for atom in mod.topology.atoms():
        if atom.residue.name not in ("HOH", "NA", "CL") and atom.element.symbol != "H":
            restr.addParticle(atom.index, pos0[atom.index])
    k_restr = system.addForce(restr)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    sim = app.Simulation(mod.topology, system, integ)
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    n = int(a.equil_ps * 500)
    sim.step(n // 2)
    for k in (100.0, 10.0, 0.0):
        sim.context.setParameter("k", k)
        sim.step(n // 6)
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
    system.removeForce(k_restr)
    with open(os.path.join(a.work, "system_classical.xml"), "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    top = mod.topology
    top.setPeriodicBoxVectors(st.getPeriodicBoxVectors())
    with open(os.path.join(a.work, "equilibrated.pdb"), "w") as fh:
        app.PDBFile.writeFile(top, st.getPositions(), fh, keepIds=True)

    settings = EmbeddingSettings(origin="centroid", cutoff=a.cutoff, switch_width=a.switch)
    _, spec = prepare_qmmm_system(system, lig, topology=top)
    save_setup(os.path.join(a.work, "setup.json"), spec, settings)
    pos = np.array(st.getPositions().value_in_unit(unit.angstrom))
    c = pos[lig].mean(0)
    box = np.array(st.getPeriodicBoxVectors().value_in_unit(unit.angstrom))
    from mace.mlmm.embedding import _min_image
    import torch

    dvec = _min_image(torch.tensor(pos - c), torch.tensor(box)).numpy()
    n_in = int(np.sum(np.linalg.norm(dvec, axis=1) < a.cutoff)) - len(lig)
    print(f"equilibrated {a.equil_ps} ps; {n_in} MM atoms within {a.cutoff} A of the ligand centre; "
          f"wrote {a.work}/system_classical.xml, equilibrated.pdb, setup.json")


if __name__ == "__main__":
    main()
