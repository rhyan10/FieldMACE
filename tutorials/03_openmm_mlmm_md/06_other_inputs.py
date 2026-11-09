#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Part 6: the same ML/MM set-up from other force-field inputs.

FieldMACE only sees an openmm.System, so it does not care where the System came from. This script
builds the alanine side-chain ML/MM system from several input routes and runs a few MD steps with each:

  A  PDB + OpenMM force-field XML     Amber14 / TIP3P   (ForceField.createSystem; tutorial 01's route)
  B  Amber prmtop / inpcrd            ff96 / TIP3P      (AmberPrmtopFile; from tleap)
  C  GROMACS top / gro                same as B         (GromacsTopFile; converted from B with ParmEd here,
                                                         normally written by pdb2gmx / LigParGen / CHARMM-GUI)
  D  serialized System XML + PDB      same as A         (XmlSerializer; any program that writes OpenMM XML)

and shows the QM-selection language (mace.mlmm.io.select_atoms). Other routes, not run here:

  CHARMM psf   load_system(psf="sys.psf", pdb="sys.pdb", charmm_params=["par.prm", "top.rtf", "toppar.str"])
               (or charmm36.xml with ForceField; capped peptides need residueTemplates=...)
  GAFF/OpenFF  openmmforcefields: GAFFTemplateGenerator / SMIRNOFFTemplateGenerator registered on a
               ForceField, then as A. Without it: tutorial 06 turns a ligand prmtop into ffxml.
  OPLS-AA      GROMACS top/gro (LigParGen, oplsaa.ff), as C.

Important: the model was trained with Amber14/TIP3P embedding charges (tutorial 01). Routes B and C
use ff96 charges - the plumbing works, but the model then sees point charges it was not trained on.
Train with the force field you will run with (the embedding charges are part of the data).

Output: a table of the routes, results/other_inputs.txt
"""
import argparse
import importlib.util
import os
import tempfile
import time

import numpy as np
import openmm as mm
import torch
from openmm import app, unit

from mace.mlmm import create_fieldmace_mm_system
from mace.mlmm.dataset import load_setup
from mace.mlmm.io import constrain_hbonds_except, load_system, select_atoms

HERE = os.path.dirname(os.path.abspath(__file__))
T01 = os.path.join(HERE, "..", "01_data_and_training")
SIDECHAIN = "resid 2 and name CA HA CB HB1 HB2 HB3"   # the same 6 atoms in every route


def amber_files():
    spec = importlib.util.find_spec("openmmtools")
    if spec is None:
        return None
    d = os.path.join(spec.submodule_search_locations[0], "data", "alanine-dipeptide-explicit")
    return os.path.join(d, "alanine-dipeptide.prmtop"), os.path.join(d, "alanine-dipeptide.crd")


def routes(tmp):
    """(label, kwargs for load_system) - built lazily so a missing optional tool only skips its route."""
    pdb = os.path.join(T01, "inputs", "aladip_solvated.pdb")
    yield "A  PDB + ffxml (Amber14/TIP3P)", dict(pdb=pdb, forcefield=["amber14-all.xml", "amber14/tip3p.xml"])
    amb = amber_files()
    if amb is None:
        yield "B  Amber prmtop/inpcrd", "skipped: openmmtools (ships the example prmtop) not installed"
    else:
        yield "B  Amber prmtop/inpcrd (ff96)", dict(prmtop=amb[0], inpcrd=amb[1])
        try:
            import parmed

            struct = parmed.load_file(amb[0], xyz=amb[1])
            top, gro = os.path.join(tmp, "topol.top"), os.path.join(tmp, "conf.gro")
            struct.save(top, format="gromacs", overwrite=True)
            struct.save(gro, format="gro", overwrite=True)
            yield "C  GROMACS top/gro (ff96, via ParmEd)", dict(top=top, gro=gro)
        except ImportError:
            yield "C  GROMACS top/gro", "skipped: ParmEd not installed (pip install parmed) to make the example files"
    # D: serialize route A and read it back
    sA = load_system(pdb=pdb, forcefield=["amber14-all.xml", "amber14/tip3p.xml"], constraints=None, cutoff_nm=0.9)
    xml = os.path.join(tmp, "system.xml")
    with open(xml, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(sA.system))
    yield "D  serialized system.xml + PDB", dict(system_xml=xml, pdb=pdb)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(T01, "models", "aladip_PerAtomFieldEMACE.model"))
    p.add_argument("--setup", default=os.path.join(T01, "data", "setup.json"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--platform", default="CPU")
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    _, settings, prep_kw = load_setup(a.setup)
    prep_kw = {k: v for k, v in prep_kw.items() if k != "link_atoms"}  # found again from the bonds of each topology
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("QM selection examples (mace.mlmm.io.select_atoms), on tutorial 01's topology:")
    top = app.PDBFile(os.path.join(T01, "inputs", "aladip_solvated.pdb")).topology
    for sel in ("index 8-13", SIDECHAIN, "resid 2", "resname ACE or resid 2 and name CB HB1 HB2 HB3"):
        idx = select_atoms(top, sel)
        out(f"  {sel!r:52s} -> {len(idx):3d} atoms {idx if len(idx) < 12 else str(idx[:12])[:-1] + ', ...]'}")
    out()

    out(f"{'route':40s} {'atoms':>6s} {'QM':>3s} {'links':>5s} {'charges':>7s} {'E_FieldMACE (eV)':>17s} {'ms/step':>8s}")
    with tempfile.TemporaryDirectory() as tmp:
        for label, kw in routes(tmp):
            if isinstance(kw, str):
                out(f"{label:40s} {kw}")
                continue
            try:
                s = load_system(**kw, constraints=None, cutoff_nm=0.9) if "system_xml" not in kw else load_system(**kw)
            except Exception as err:  # noqa: BLE001  report and go on
                out(f"{label:40s} failed to load: {err}")
                continue
            qm = select_atoms(s.topology, SIDECHAIN)
            if "system_xml" not in kw:
                constrain_hbonds_except(s.system, s.topology, qm)
            system, spec, comp = create_fieldmace_mm_system(
                s.system, qm, a.model, topology=s.topology, settings=settings, **prep_kw,
                model_kwargs=dict(device="cpu", dtype=torch.float64, subtract_e0=True))
            integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtosecond)
            sim = app.Simulation(s.topology, system, integ, mm.Platform.getPlatformByName(a.platform))
            sim.context.setPositions(s.positions)
            if s.box_vectors is not None:
                sim.context.setPeriodicBoxVectors(*s.box_vectors)
            sim.context.getState(getEnergy=True)  # one evaluation at the starting coordinates
            e0, n_mm = comp.last["energies"][0], comp.last["n_mm_charges"]
            sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
            sim.step(1)
            t0 = time.time()
            sim.step(a.steps)
            ms = 1000 * (time.time() - t0) / a.steps
            out(f"{label:40s} {system.getNumParticles():6d} {len(qm):3d} {len(spec.link_atoms):5d} "
                f"{n_mm:7d} {e0:17.4f} {ms:8.1f}")
    out()
    out("E_FieldMACE is the interaction-shifted QM energy without atomic reference energies (subtract_e0=True).")
    out("A and D are the same System and agree exactly. B and C are the same ff96 System in two file formats; C")
    out("differs slightly because .gro files store coordinates to 0.001 nm. All routes start from the same")
    out("coordinates, so A vs B is the effect of the embedding charges alone (Amber14 vs ff96).")
    with open(os.path.join(HERE, "results", "other_inputs.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
