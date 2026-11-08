#!/usr/bin/env python3
"""
Step 1: build the classical OpenMM system, choose the QM region, equilibrate.

Any force field OpenMM can read works. The default is alanine dipeptide in TIP3P water with
Amber14, built from a PDB file and OpenMM's force-field XML files. Other inputs:

  Amber        --prmtop sys.prmtop --inpcrd sys.inpcrd
  GROMACS/OPLS --top topol.top --gro conf.gro [--gmx-include /path/to/gromacs/top]
  CHARMM       --psf sys.psf --pdb sys.pdb --charmm-params par.prm top.rtf
  OpenMM XML   --pdb sys.pdb --forcefield amber14-all.xml amber14/tip3pfb.xml   (or charmm36.xml, GAFF/OpenFF ffxml)
  serialized   --system-xml system.xml --pdb sys.pdb

The QM region is chosen with --qm (selection syntax of mace.mlmm.io.select_atoms):
  "index 8-13"      0-based atom indices / ranges    (default: the alanine side chain CA HA CB HB1-3)
  "resname LIG"     whole residues by name
  "resid 2"         residues by id
Covalent bonds between QM and MM atoms are capped with hydrogen link atoms automatically.

Outputs (work/):
  system_classical.xml   the classical OpenMM System (used for sampling)
  setup.json             QM atoms, link atoms and embedding settings (used for data AND for ML/MM MD)
  equilibrated.pdb/.xml  equilibrated coordinates / OpenMM state
"""
import argparse
import os
import warnings

import numpy as np
import openmm as mm
from openmm import app, unit

from mace.mlmm import EmbeddingSettings, prepare_qmmm_system
from mace.mlmm.dataset import save_setup
from mace.mlmm.io import constrain_hbonds_except, load_system, select_atoms

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("input (any force field)")
    g.add_argument("--pdb", default=os.path.join(HERE, "inputs", "aladip_solvated.pdb"))
    g.add_argument("--forcefield", nargs="*", default=["amber14-all.xml", "amber14/tip3p.xml"])
    g.add_argument("--prmtop")
    g.add_argument("--inpcrd")
    g.add_argument("--top")
    g.add_argument("--gro")
    g.add_argument("--gmx-include")
    g.add_argument("--psf")
    g.add_argument("--crd")
    g.add_argument("--charmm-params", nargs="*", default=[])
    g.add_argument("--system-xml")
    p.add_argument("--qm", default="index 8-13", help="QM selection (default: alanine side chain)")
    p.add_argument("--cutoff", type=float, default=8.0, help="embedding cutoff in Angstrom (none: all charges)")
    p.add_argument("--switch-width", type=float, default=2.0, help="charges are switched off smoothly over this width")
    p.add_argument("--link-charge-scheme", default="Z1", choices=["Z0", "Z1", "Z2", "Z3"])
    p.add_argument("--equil-ps", type=float, default=20.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--platform", default=None, help="OpenMM platform (CUDA, CPU, Reference); default: fastest")
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)

    # ---- 1. classical system, built by any force field ----------------------------------
    # Flexible bonds first; X-H constraints are then added only in the MM region, because the
    # QM region will be described by FieldMACE (which sees all its vibrations).
    use_pdb = a.pdb if not (a.prmtop or a.top or (a.psf and a.crd)) else None
    sysload = load_system(pdb=use_pdb, forcefield=a.forcefield, prmtop=a.prmtop, inpcrd=a.inpcrd, top=a.top,
                          gro=a.gro, gmx_include=a.gmx_include, psf=a.psf, crd=a.crd, charmm_params=a.charmm_params,
                          system_xml=a.system_xml, constraints=None, cutoff_nm=0.9)
    system, top, pos = sysload.system, sysload.topology, sysload.positions
    qm = select_atoms(top, a.qm)
    atoms = list(top.atoms())
    print(f"system: {system.getNumParticles()} particles, periodic={system.usesPeriodicBoundaryConditions()}")
    print("QM atoms:", ", ".join(f"{i}:{atoms[i].residue.name}{atoms[i].residue.id}-{atoms[i].name}" for i in qm))
    n_con = constrain_hbonds_except(system, top, qm)
    print(f"added {n_con} X-H constraints outside the QM region")

    # ---- 2. QM/MM bookkeeping: link atoms, embedding charges ----------------------------
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, spec = prepare_qmmm_system(system, qm, topology=top, link_atoms="auto",
                                      link_charge_scheme=a.link_charge_scheme)
    for msg in w:
        print("note:", msg.message)
    for lk in spec.link_atoms:
        print(f"link atom: H on bond {atoms[lk.qm_atom].name}({lk.qm_atom}) - {atoms[lk.mm_atom].name}({lk.mm_atom}), "
              f"ratio {lk.ratio:.3f}")
    print(f"FieldMACE sees {len(spec.model_atomic_numbers)} atoms (Z = {spec.model_atomic_numbers}) "
          f"and up to {len(spec.mm_atoms)} MM point charges")
    settings = EmbeddingSettings(origin="centroid", cutoff=a.cutoff if a.cutoff > 0 else None,
                                 switch_width=a.switch_width, cutoff_reference="centroid")
    save_setup(os.path.join(a.work, "setup.json"), spec, settings, link_charge_scheme=a.link_charge_scheme)
    with open(os.path.join(a.work, "system_classical.xml"), "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))

    # ---- 3. minimise and equilibrate classically -----------------------------------------
    integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
    if system.usesPeriodicBoundaryConditions():
        system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, a.temperature * unit.kelvin))
    plat = [mm.Platform.getPlatformByName(a.platform)] if a.platform else []
    sim = app.Simulation(top, system, integ, *plat)
    sim.context.setPositions(pos)
    if sysload.box_vectors is not None:
        sim.context.setPeriodicBoxVectors(*sysload.box_vectors)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin, 1)
    n = int(a.equil_ps * 1000)
    sim.step(n)
    st = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
    print(f"equilibrated {a.equil_ps} ps on {sim.context.getPlatform().getName()}: "
          f"E_pot = {st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
    with open(os.path.join(a.work, "equilibrated.pdb"), "w") as fh:
        if st.getPeriodicBoxVectors() is not None and system.usesPeriodicBoundaryConditions():
            top.setPeriodicBoxVectors(st.getPeriodicBoxVectors())
        app.PDBFile.writeFile(top, st.getPositions(), fh, keepIds=True)
    with open(os.path.join(a.work, "equilibrated.xml"), "w") as fh:  # positions, velocities, box
        fh.write(mm.XmlSerializer.serialize(st))
    print(f"wrote {a.work}/setup.json, system_classical.xml, equilibrated.pdb/.xml")


if __name__ == "__main__":
    main()
