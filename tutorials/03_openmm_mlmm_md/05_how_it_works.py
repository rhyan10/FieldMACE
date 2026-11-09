#!/usr/bin/env python3
# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Part 5: how the OpenMM interface works - a look inside create_fieldmace_mm_system.

Step by step, on the alanine-dipeptide system of tutorial 01 (QM region: the alanine side chain,
cut from the backbone at two C-C bonds):

  1. what prepare_qmmm_system changes in the force field: every force, before and after
  2. link atoms: where they sit, and which MM charges each charge scheme (Z0-Z3) hides
  3. the embedding frame: which point charges the model sees, after cutoff and switching
  4. energy bookkeeping: total = force field + FieldMACE, separated with force groups
  5. forces through link atoms, periodic images and switching are exact derivatives
     (finite differences on a QM atom, a link host and a water in the switching zone)

Nothing here runs dynamics; it prints what the interface does. Output: results/how_it_works.txt
"""
import argparse
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import openmm as mm
from openmm import app, unit

from mace.mlmm import EmbeddingFrame, create_fieldmace_mm_system, prepare_qmmm_system
from mace.mlmm.dataset import load_setup
from mace.mlmm.force import EV_TO_KJMOL
from mace.mlmm.io import constrain_hbonds_except, load_system
from common import default_pdb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T01 = os.path.join(HERE, "..", "01_data_and_training")
FIELDMACE_GROUP = 31  # the ML force gets its own force group, the force field keeps group 0


def count_terms(f):
    for name in ("Bonds", "Angles", "Torsions", "Exceptions", "Particles"):
        if hasattr(f, f"getNum{name}"):
            return getattr(f, f"getNum{name}")(), name.lower()
    return None, ""


def qm_terms(f, qm):
    """Number of bonded terms whose atoms are all QM atoms."""
    qm = set(qm)
    if isinstance(f, mm.HarmonicBondForce):
        return sum(set(f.getBondParameters(k)[:2]) <= qm for k in range(f.getNumBonds()))
    if isinstance(f, mm.HarmonicAngleForce):
        return sum(set(f.getAngleParameters(k)[:3]) <= qm for k in range(f.getNumAngles()))
    if isinstance(f, mm.PeriodicTorsionForce):
        return sum(set(f.getTorsionParameters(k)[:4]) <= qm for k in range(f.getNumTorsions()))
    return 0


def group_energy(ctx, groups):
    return ctx.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(T01, "models", "aladip_PerAtomFieldEMACE.model"))
    p.add_argument("--setup", default=os.path.join(T01, "data", "setup.json"))
    p.add_argument("--pdb", default=None, help="start structure (default: tutorial 01's equilibrated.pdb)")
    a = p.parse_args()
    a.pdb = a.pdb or default_pdb()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        run(a)
    with open(os.path.join(HERE, "results", "how_it_works.txt"), "w") as fh:
        fh.write(buf.getvalue())


def run(a):
    s = load_system(pdb=a.pdb, forcefield=["amber14-all.xml", "amber14/tip3p.xml"], constraints=None, cutoff_nm=0.9)
    qm, settings, prep_kw = load_setup(a.setup)
    constrain_hbonds_except(s.system, s.topology, qm)
    atoms = list(s.topology.atoms())
    name = {i: f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}" for i in range(len(atoms))}
    print(f"System: {s.system.getNumParticles()} particles; QM region {[name[i] for i in qm]}")
    print(f"setup.json: embedding {settings}, {prep_kw}\n")

    # ------------------------------------------------------------------ 1
    print("1. What prepare_qmmm_system changes (the original System is not modified; a copy is returned)")
    new, spec = prepare_qmmm_system(s.system, qm, topology=s.topology, **prep_kw)
    print(f"   {'force':24s} {'before':>8s} {'after':>8s}  note")
    after = {f.__class__.__name__: f for f in new.getForces()}
    for f in s.system.getForces():
        cls = f.__class__.__name__
        n0, what = count_terms(f)
        n1, _ = count_terms(after[cls]) if cls in after else (None, "")
        if isinstance(f, mm.NonbondedForce):
            npair = len(qm) * (len(qm) - 1) // 2
            note = (f"QM charges set to 0 (they were {sum(f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in qm):+.3f} e in total); "
                    f"all {npair} QM-QM pairs excluded (no Coulomb, no LJ); QM-MM LJ kept")
            what = "exceptions"
            n0, n1 = f.getNumExceptions(), after[cls].getNumExceptions()
        elif cls in ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"):
            note = f"{qm_terms(f, qm)} terms with only QM atoms removed (FieldMACE describes them)"
        else:
            note = "unchanged"
        print(f"   {cls:24s} {str(n0):>8s} {str(n1):>8s}  {what}: {note}")
    nc0, nc1 = s.system.getNumConstraints(), new.getNumConstraints()
    print(f"   {'constraints':24s} {nc0:>8d} {nc1:>8d}  QM-QM constraints removed (here none: QM X-H bonds were "
          "left flexible by constrain_hbonds_except)")
    print("   Terms that mix QM and MM atoms (e.g. the CA-N bond, angles and torsions across the boundary) stay in "
          "the force field.\n")

    # ------------------------------------------------------------------ 2
    print("2. Link atoms: a hydrogen on every cut QM-MM bond, placed at r_QM + g (r_MM - r_QM) every step")
    pos = np.array(s.positions.value_in_unit(unit.angstrom))
    for lk in spec.link_atoms:
        d = np.linalg.norm(pos[lk.mm_atom] - pos[lk.qm_atom])
        print(f"   {name[lk.qm_atom]:>12s} - {name[lk.mm_atom]:<12s} bond {d:.3f} A, g = {lk.ratio:.4f} -> H at "
              f"{lk.ratio * d:.3f} A from the QM atom")
    print("   Their forces are passed to the QM atom and the MM host by the chain rule (part of the autograd graph).")
    print("   Charge schemes - which MM charges near a link are hidden from the model (they stay in the force field):")
    full = set(range(s.system.getNumParticles())) - set(qm)
    for scheme in ("Z0", "Z1", "Z2", "Z3"):
        _, sp = prepare_qmmm_system(s.system, qm, topology=s.topology, link_charge_scheme=scheme)
        hidden = sorted(i for i in full if i not in set(sp.mm_atoms) and abs(
            [f for f in s.system.getForces() if isinstance(f, mm.NonbondedForce)][0]
            .getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)) > 1e-8)
        q_hidden = sum([f for f in s.system.getForces() if isinstance(f, mm.NonbondedForce)][0]
                       .getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in hidden)
        mark = "   <- setup.json" if scheme == prep_kw.get("link_charge_scheme", "Z1") else ""
        print(f"   {scheme}: {len(sp.mm_atoms)} embedding charges, hidden {[name[i] for i in hidden]} "
              f"(total {q_hidden:+.3f} e){mark}")
    print("   Use the scheme the training data were made with (it is stored in setup.json).\n")

    # ------------------------------------------------------------------ 3
    print("3. The embedding frame: what the model sees in one configuration")
    frame = EmbeddingFrame(spec, settings)
    box = np.array([v.value_in_unit(unit.angstrom) for v in s.box_vectors])
    qm_pos, mm_pos, mm_q, sel = frame.numpy(pos, box)
    r = np.linalg.norm(mm_pos, axis=1) if settings.origin != "none" else None
    q_full = spec.mm_charges[sel]
    scale = np.divide(mm_q, q_full, out=np.ones_like(mm_q), where=np.abs(q_full) > 1e-12)
    print(f"   origin '{settings.origin}': QM centroid at (0, 0, 0); QM atoms + {len(spec.link_atoms)} link H -> "
          f"{len(qm_pos)} model atoms")
    print(f"   MM charges: {len(spec.mm_atoms)} candidates -> {len(mm_q)} within {settings.cutoff} A of the centroid "
          f"(minimum image in the {box[0, 0]:.1f} A box)")
    print(f"   switching between {settings.cutoff - settings.switch_width:.1f} and {settings.cutoff:.1f} A: "
          f"{int(np.sum(scale > 0.999))} charges at full strength, {int(np.sum((scale > 1e-6) & (scale < 0.999)))} "
          f"scaled down; total charge seen {mm_q.sum():+.3f} e")
    if r is not None:
        print(f"   nearest charge {r.min():.2f} A, farthest {r.max():.2f} A from the centroid")
    print("   The same frame wrote the training data (frame_to_atoms), so the model is used as it was trained.\n")

    # ------------------------------------------------------------------ 4
    print("4. Energy bookkeeping: E_total = E_forcefield + E_FieldMACE")
    system, spec2, comp = create_fieldmace_mm_system(s.system, qm, a.model, topology=s.topology, settings=settings,
                                                     force_group=FIELDMACE_GROUP, model_kwargs=dict(device="cpu"),
                                                     **prep_kw)
    ctx = mm.Context(system, mm.VerletIntegrator(0.5 * unit.femtosecond), mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(s.positions)
    ctx.setPeriodicBoxVectors(*s.box_vectors)
    e_tot = group_energy(ctx, set(range(32)))
    e_ml = group_energy(ctx, {FIELDMACE_GROUP})
    e_mm = e_tot - e_ml
    print(f"   total {e_tot:14.2f} kJ/mol = force field {e_mm:12.2f} + FieldMACE {e_ml:12.2f} kJ/mol")
    print(f"   FieldMACE: {e_ml / EV_TO_KJMOL:.4f} eV; comp.last says {comp.last['energies'][0]:.4f} eV with "
          f"{comp.last['n_mm_charges']} point charges")
    print("   The FieldMACE energy is the QM energy *including* the interaction with the point charges, so it is")
    print("   large (sum of atomic energies). Only differences matter; subtract_e0=True in model_kwargs removes the")
    print("   atomic reference energies if you want readable numbers - forces are unchanged.")
    ff_groups = {}
    for f in system.getForces():
        ff_groups.setdefault(f.getForceGroup(), []).append(f.getName())
    print(f"   force groups: { {g: n for g, n in sorted(ff_groups.items())} }")
    print("   (set other groups with force.setForceGroup to decompose further, e.g. bonded vs nonbonded)\n")

    # ------------------------------------------------------------------ 5
    print("5. Forces are exact derivatives (finite differences of the FieldMACE energy, kJ/mol/nm)")
    print("   (model in float64 and without atomic reference energies, so the finite differences are precise)")
    import torch

    system, _, _ = create_fieldmace_mm_system(s.system, qm, a.model, topology=s.topology, settings=settings,
                                              force_group=FIELDMACE_GROUP, **prep_kw,
                                              model_kwargs=dict(device="cpu", dtype=torch.float64, subtract_e0=True))
    ctx = mm.Context(system, mm.VerletIntegrator(0.5 * unit.femtosecond), mm.Platform.getPlatformByName("Reference"))
    ctx.setPeriodicBoxVectors(*s.box_vectors)
    ctx.setPositions(s.positions)
    sub = list(comp.frame.particles)
    pos_nm = np.array(s.positions.value_in_unit(unit.nanometer))
    f_all = ctx.getState(getForces=True, groups={FIELDMACE_GROUP}).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    host = spec.link_atoms[0].mm_atom
    # a water oxygen in the switching zone
    in_switch = [int(spec.mm_atoms[sel[k]]) for k in range(len(sel)) if 0.05 < scale[k] < 0.95]
    picks = [(qm[0], "QM atom"), (host, "MM link host"), (in_switch[0] if in_switch else sub[-1], "MM charge in the switching zone")]
    h = 1e-4  # nm
    for idx, label in picks:
        fd = []
        for k in range(3):
            e = []
            for sgn in (1, -1):
                x = pos_nm.copy()
                x[idx, k] += sgn * h
                ctx.setPositions(x)
                e.append(group_energy(ctx, {FIELDMACE_GROUP}))
            fd.append(-(e[0] - e[1]) / (2 * h))
        print(f"   {label:32s} {name[idx]:>12s}: force {np.round(f_all[idx], 3)}  finite diff. {np.round(fd, 3)}")
    print("   The link host gets force through the link atom; the switched charge through its distance-dependent scale.")


if __name__ == "__main__":
    main()
