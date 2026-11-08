#!/usr/bin/env python3
"""
Step 1: sample thioformaldehyde (CH2S) in a droplet of 100 water molecules.

The system is SHARC 4's QM/MM example (examples/SHARC_QMMM_LVC_OPENMM): an Amber prmtop for
CH2S + 100 TIP3P waters. The QM region is CH2S (4 atoms, no link atoms); FieldMACE sees all
300 water charges (no cutoff, the droplet is small).

Classical sampling details
  * The prmtop has zero charges on CH2S (it was made for QM/MM). For sampling we give CH2S
    the ground-state charges of SHARC's LVC model (Multipolar Density Fit, monopoles), so the
    water orients around it as it should.
  * Flat-bottom restraints keep the droplet together and CH2S inside it (no periodic box).
  * Half of the frames get extra random displacements of the QM atoms (sigma 0.04, 0.08 or 0.12 A),
    so the data cover geometries away from the classical ground-state distribution. Excited-state
    dynamics stretches the C=S bond to ~1.9 A, where T2 and S1 cross; a model that has not seen
    such geometries extrapolates exactly where the hops happen.

Outputs (work/):
  frames.xyz       QM atoms + point charges in FieldMACE's frame (unlabelled)
  setup.json       QM atoms and embedding settings (used again for ML/MM MD in tutorial 03)
  system_mm.xml    the MM system (force field + droplet restraint) for ML/MM MD in tutorial 03
  droplet.pdb      last snapshot (topology and starting coordinates for tutorial 03)
  initconds.npz    full-system positions (bohr) and velocities (bohr/atu) of a few snapshots,
                   initial conditions for the SHARC dynamics in tutorial 04
"""
import argparse
import os

import numpy as np
import openmm as mm
from openmm import app, unit

from mace.mlmm import EmbeddingFrame, EmbeddingSettings, prepare_qmmm_system
from mace.mlmm.dataset import frame_to_atoms, save_setup, write_extxyz

HERE = os.path.dirname(os.path.abspath(__file__))
BOHR_TO_A = 0.529177210903
LVC_S0_CHARGES = [-0.08109983, -0.13886037, 0.10998011, 0.10998010]  # C S H H, from LVC.template
AU_TIME_PS = 2.4188843265857e-5  # 1 atomic time unit in ps


def example_dir(path=None):
    if path:
        return path
    sharc = os.environ.get("SHARC")
    if not sharc:
        raise SystemExit("Set $SHARC (SHARC 4 bin directory) or pass --example-dir "
                         "(SHARC's examples/SHARC_QMMM_LVC_OPENMM)")
    return os.path.join(sharc, "..", "examples", "SHARC_QMMM_LVC_OPENMM")


def read_qm_in(path):
    """Elements and coordinates (bohr) from SHARC's QM.in."""
    lines = open(path).read().splitlines()
    n = int(lines[0])
    el = [ln.split()[0] for ln in lines[2:n + 2]]
    xyz = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:n + 2]])
    return el, xyz


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example-dir", default=None, help="default: $SHARC/../examples/SHARC_QMMM_LVC_OPENMM")
    p.add_argument("--work", default=os.path.join(HERE, "work"))
    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--stride-ps", type=float, default=0.2)
    p.add_argument("--equil-ps", type=float, default=20.0)
    p.add_argument("--noise", type=float, nargs="+", default=[0.04, 0.08, 0.12],
                   help="Angstrom; every second frame gets random QM displacements of these sizes, in turn")
    p.add_argument("--n-initconds", type=int, default=10)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)
    ex = example_dir(a.example_dir)
    rng = np.random.default_rng(a.seed)

    prmtop = app.AmberPrmtopFile(os.path.join(ex, "MML", "system_100_chrg_0.prmtop"))
    elements, xyz_bohr = read_qm_in(os.path.join(ex, "QM.in"))
    positions = xyz_bohr * BOHR_TO_A * 0.1  # nm
    top = prmtop.topology
    top.setPeriodicBoxVectors(None)
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=True)
    qm = [a_.index for a_ in top.atoms() if a_.residue.name == "MOL"]
    print(f"{system.getNumParticles()} particles, QM atoms {qm} ({[elements[i] for i in qm]})")

    # FieldMACE bookkeeping (on the original system: CH2S charges are irrelevant there, the QM
    # charges are removed anyway) - saved for tutorials 03 and 04
    settings = EmbeddingSettings(origin="centroid", cutoff=None)
    _, spec = prepare_qmmm_system(system, qm, topology=top, link_atoms=False)
    save_setup(os.path.join(a.work, "setup.json"), spec, settings, link_atoms=False)
    frame = EmbeddingFrame(spec, settings)
    print(f"FieldMACE sees {len(spec.model_atomic_numbers)} QM atoms and {len(spec.mm_atoms)} point charges")

    # ---- classical sampling system: LVC S0 charges on CH2S, droplet restraint ----------------
    sampling = mm.XmlSerializer.clone(system)
    nb = [f for f in sampling.getForces() if isinstance(f, mm.NonbondedForce)][0]
    for i, qi in zip(qm, LVC_S0_CHARGES):
        _, sig, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, qi, sig, eps)
    for k in range(nb.getNumExceptions()):  # keep 1-4 scaling consistent with the new charges
        i, j, qq, sig, eps = nb.getExceptionParameters(k)
        if i in qm and j in qm:
            nb.setExceptionParameters(k, i, j, 0.0, sig, eps)
    oxygens = [a_.index for a_ in top.atoms() if a_.element is not None and a_.element.symbol == "O"]
    centre = positions[qm].mean(0)
    r_max = np.linalg.norm(positions[oxygens] - centre, axis=1).max() + 0.1  # nm

    def droplet_restraint():
        """Flat-bottom restraints: water oxygens within the droplet radius, and CH2S within 3 A of the
        droplet centre. Without the second one, the weakly polar CH2S drifts out of the droplet."""
        f = mm.CustomExternalForce("0.5*k*step(r-r0)*(r-r0)^2; r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        f.addPerParticleParameter("r0")
        for name, val in (("k", 1000.0), ("x0", centre[0]), ("y0", centre[1]), ("z0", centre[2])):
            f.addGlobalParameter(name, val)
        for i in oxygens:
            f.addParticle(i, [r_max])
        for i in qm:
            f.addParticle(i, [0.3])
        f.setName("droplet_restraint")
        return f

    rest = droplet_restraint()
    sampling.addForce(rest)
    sampling.addForce(mm.CMMotionRemover(100))  # stop the droplet from drifting away from the restraint centre

    # the MM part for ML/MM MD (tutorial 03): original force field (CH2S charges are removed by
    # prepare_qmmm_system anyway) plus the droplet restraint
    mlmm_base = mm.XmlSerializer.clone(system)
    rest2 = droplet_restraint()
    mlmm_base.addForce(rest2)
    with open(os.path.join(a.work, "system_mm.xml"), "w") as fh:
        fh.write(mm.XmlSerializer.serialize(mlmm_base))

    dt = 0.5 * unit.femtosecond
    integ = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, 2.0 / unit.picosecond, dt)
    integ.setRandomNumberSeed(a.seed)
    sim = app.Simulation(top, sampling, integ, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin, a.seed)
    sim.step(int(a.equil_ps * 1000 / 0.5))
    print(f"equilibrated {a.equil_ps} ps, droplet radius {r_max * 10:.1f} A")

    stride = int(round(a.stride_ps * 1000 / 0.5))
    frames, ic_pos, ic_vel = [], [], []
    ic_every = max(1, a.n_frames // max(1, a.n_initconds))
    for k in range(a.n_frames):
        sim.step(stride)
        st = sim.context.getState(getPositions=True, getVelocities=True)
        pos_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if k % ic_every == 0 and len(ic_pos) < a.n_initconds:
            ic_pos.append(pos_nm * 10.0 / BOHR_TO_A)  # bohr
            ic_vel.append(st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
                          * 10.0 / BOHR_TO_A * AU_TIME_PS)  # bohr / atomic time unit
        noisy = pos_nm.copy()
        sigma = a.noise[(k // 2) % len(a.noise)] if k % 2 == 1 else 0.0
        if sigma > 0:
            noisy[qm] += rng.normal(0.0, sigma * 0.1, (len(qm), 3))
        at = frame_to_atoms(frame, noisy, extra_info={"md_frame": k, "qm_noise": sigma})
        frames.append(at)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{a.n_frames} frames")
    write_extxyz(os.path.join(a.work, "frames.xyz"), frames)
    with open(os.path.join(a.work, "droplet.pdb"), "w") as fh:  # last snapshot, starting point for ML/MM MD
        app.PDBFile.writeFile(top, st.getPositions(), fh, keepIds=True)
    np.savez(os.path.join(a.work, "initconds.npz"), positions_bohr=np.array(ic_pos),
             velocities_au=np.array(ic_vel), elements=np.array([x.element.symbol for x in top.atoms()]),
             masses_amu=np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                                  for i in range(system.getNumParticles())]), qm_atoms=np.array(qm))
    print(f"wrote {len(frames)} frames to work/frames.xyz and {len(ic_pos)} initial conditions to work/initconds.npz")


if __name__ == "__main__":
    main()
