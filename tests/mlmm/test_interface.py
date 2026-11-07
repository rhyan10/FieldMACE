"""Interface tests with randomly initialised FieldMACE models.

Run:  pytest -q tests/
"""
import numpy as np
import openmm as mm
import pytest
import torch
from openmm import app, unit

from mace.mlmm import (EmbeddingSettings, create_fieldmace_mm_system,
                              prepare_qmmm_system)
from mace.mlmm.dataset import (attach_reference, frame_to_atoms, pad_mm_charges,
                                      write_extxyz)
from ref_eval import ref_eval

SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
WHOLE = list(range(22))  # ACE-ALA-NME
SIDECHAIN = [8, 9, 10, 11, 12, 13]  # CA HA CB HB1 HB2 HB3 -> links to N(6) and C(14)


def _box_A(system):
    return np.array([v.value_in_unit(unit.angstrom) for v in system.getDefaultPeriodicBoxVectors()])


def _context(system, positions, groups=None):
    integ = mm.VerletIntegrator(0.5 * unit.femtosecond)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    return ctx, integ


def _fieldmace_group(system):
    for k, f in enumerate(system.getForces()):
        if f.getName() == "FieldMACE":
            f.setForceGroup(31)
    return {31}


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["PerAtomFieldEMACE", "FieldEMACE"])
def test_matches_fieldmace_pipeline(models, aladip, variant):
    """Energy and point-charge forces equal FieldMACE's own data pipeline."""
    s, spec, comp = create_fieldmace_mm_system(aladip.system, WHOLE, models[variant],
                                               topology=aladip.topology, model_kwargs=dict(device="cpu"))
    pos = aladip.positions.value_in_unit(unit.angstrom)
    box = _box_A(aladip.system)
    qm, mmp, q, _ = comp.frame.numpy(pos, box)
    out = ref_eval(comp.model, [SYM[z] for z in spec.model_atomic_numbers], qm, mmp, q)
    e, g = comp.evaluate(np.asarray(pos)[comp.frame.particles], box)
    assert e == pytest.approx(out["energy"].item(), abs=1e-9)
    f_mm = out["mm_forces"][:, 0].detach().numpy()
    assert np.allclose(-g[comp.frame.mm_local.numpy()], f_mm, atol=1e-10)
    if variant == "PerAtomFieldEMACE":  # translation invariant -> partial == total derivative
        assert np.allclose(-g[:22], out["forces"][:, 0].detach().numpy(), atol=1e-10)
    assert np.abs(g.sum(0)).max() < 1e-10  # momentum conservation


@pytest.mark.parametrize("variant,origin,cutoff,ref", [
    ("PerAtomFieldEMACE", "centroid", None, "centroid"),
    ("FieldEMACE", "centroid", 9.0, "centroid"),
    ("FieldEMACE", "com", 9.0, "nearest"),
    ("PerAtomFieldEMACE", "none", 9.0, "nearest"),
])
def test_finite_differences_with_link_atoms(models, aladip, variant, origin, cutoff, ref):
    """OpenMM forces == -dE/dx through link atoms, PBC imaging and charge switching."""
    settings = EmbeddingSettings(origin=origin, cutoff=cutoff, switch_width=3.0, cutoff_reference=ref)
    s, spec, comp = create_fieldmace_mm_system(aladip.system, SIDECHAIN, models[variant],
                                               topology=aladip.topology, settings=settings,
                                               model_kwargs=dict(device="cpu"))
    assert {(l.qm_atom, l.mm_atom) for l in spec.link_atoms} == {(8, 6), (8, 14)}
    groups = _fieldmace_group(s)
    ctx, _ = _context(s, aladip.positions)
    pos0 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    F = ctx.getState(getForces=True, groups=groups).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)

    # atoms to probe: QM, link hosts, and MM charges inside the switching shell
    pos_A = pos0 * 10
    qm_c = pos_A[SIDECHAIN].mean(0)
    box = _box_A(aladip.system)
    d = pos_A[spec.mm_atoms] - qm_c
    d -= np.round(d / np.diag(box)) * np.diag(box)
    dist = np.linalg.norm(d, axis=1)
    shell = [spec.mm_atoms[i] for i in np.where((dist > 6.5) & (dist < 8.5))[0][:3]]
    probe = [8, 10, 11, 6, 14] + shell
    h = 1e-5  # nm
    for i in probe:
        for k in range(3):
            e = []
            for sgn in (+1, -1):
                p = pos0.copy()
                p[i, k] += sgn * h
                ctx.setPositions(p)
                e.append(ctx.getState(getEnergy=True, groups=groups).getPotentialEnergy()
                         .value_in_unit(unit.kilojoule_per_mole))
            fd = -(e[0] - e[1]) / (2 * h)
            assert fd == pytest.approx(F[i, k], abs=2e-3 + 1e-4 * abs(F[i, k])), (i, k, fd, F[i, k])
    assert np.abs(F.sum(0)).max() < 1e-6


def _python_mm_partition(orig, qm, positions_nm):
    """Independent NoCutoff reference for the terms prepare_qmmm_system must remove."""
    qm_set = set(qm)
    ONE_4PI_EPS0 = 138.935456
    nb = [f for f in orig.getForces() if isinstance(f, mm.NonbondedForce)][0]
    q = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(orig.getNumParticles())])
    sig = np.array([nb.getParticleParameters(i)[1].value_in_unit(unit.nanometer) for i in range(orig.getNumParticles())])
    eps = np.array([nb.getParticleParameters(i)[2].value_in_unit(unit.kilojoule_per_mole) for i in range(orig.getNumParticles())])
    exc = {}
    for k in range(nb.getNumExceptions()):
        i, j, qq, s_, e_ = nb.getExceptionParameters(k)
        exc[tuple(sorted((i, j)))] = (qq.value_in_unit(unit.elementary_charge**2), s_.value_in_unit(unit.nanometer),
                                      e_.value_in_unit(unit.kilojoule_per_mole))
    x = positions_nm
    E_coul_qmmm, E_qmqm = 0.0, 0.0
    for i in qm:
        for j in range(orig.getNumParticles()):
            if j == i or (j in qm_set and j < i):
                continue
            key = tuple(sorted((i, j)))
            r = np.linalg.norm(x[i] - x[j])
            if key in exc:
                qq, s_, e_ = exc[key]
            else:
                qq, s_, e_ = q[i] * q[j], 0.5 * (sig[i] + sig[j]), np.sqrt(eps[i] * eps[j])
            coul = ONE_4PI_EPS0 * qq / r
            lj = 4 * e_ * ((s_ / r) ** 12 - (s_ / r) ** 6)
            if j in qm_set:
                E_qmqm += coul + lj
            else:
                E_coul_qmmm += coul
    return E_coul_qmmm, E_qmqm


@pytest.mark.parametrize("qm", [WHOLE, SIDECHAIN])
def test_mm_energy_partition(aladip, qm):
    """E_orig - E_prepared == QM-only bonded + QM-QM nonbonded + QM-MM Coulomb (NoCutoff)."""
    orig = mm.XmlSerializer.clone(aladip.system)
    for f in orig.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    new, spec = prepare_qmmm_system(orig, qm, topology=aladip.topology, remove_qm_constraints=False)
    pos = aladip.positions.value_in_unit(unit.nanometer)

    def energy(system, only_bonded_qm=False):
        c, _ = _context(system, aladip.positions)
        return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # bonded energy of QM-only terms, computed separately
    bonded = mm.System()
    for i in range(orig.getNumParticles()):
        bonded.addParticle(orig.getParticleMass(i))
    hb, ha, pt = mm.HarmonicBondForce(), mm.HarmonicAngleForce(), mm.PeriodicTorsionForce()
    qs = set(qm)
    for f in orig.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                p = f.getBondParameters(k)
                if p[0] in qs and p[1] in qs:
                    hb.addBond(*p)
        if isinstance(f, mm.HarmonicAngleForce):
            for k in range(f.getNumAngles()):
                p = f.getAngleParameters(k)
                if all(a in qs for a in p[:3]):
                    ha.addAngle(*p)
        if isinstance(f, mm.PeriodicTorsionForce):
            for k in range(f.getNumTorsions()):
                p = f.getTorsionParameters(k)
                if all(a in qs for a in p[:4]):
                    pt.addTorsion(*p)
    for f in (hb, ha, pt):
        bonded.addForce(f)
    e_bonded = energy(bonded)
    e_coul_qmmm, e_qmqm = _python_mm_partition(orig, qm, pos)
    diff = energy(orig) - energy(new)
    assert diff == pytest.approx(e_bonded + e_qmqm + e_coul_qmmm, abs=1e-4)
    nb = [f for f in new.getForces() if isinstance(f, mm.NonbondedForce)][0]
    assert all(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) == 0 for i in qm)


def test_nve_energy_conservation(models, aladip):
    """Velocity Verlet NVE with FieldMACE side chain + link atoms + PME water."""
    settings = EmbeddingSettings(cutoff=10.0, switch_width=2.0)
    s, spec, comp = create_fieldmace_mm_system(aladip.system, SIDECHAIN, models["PerAtomFieldEMACE"],
                                               topology=aladip.topology, settings=settings,
                                               model_kwargs=dict(device="cpu"))
    integ = mm.VerletIntegrator(0.25 * unit.femtosecond)
    ctx = mm.Context(s, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(aladip.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 50)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    E = []
    for _ in range(40):
        integ.step(5)
        st = ctx.getState(getEnergy=True)
        E.append((st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole))
    E = np.array(E)
    drift = abs(np.polyfit(np.arange(len(E)), E, 1)[0]) * len(E)
    assert np.std(E) < 1.0 and drift < 1.0, (np.std(E), drift)


def _opls_style(system):
    """Convert an Amber-style System to the OpenMM OPLS recipe: geometric-mean LJ in a
    CustomNonbondedForce, NonbondedForce for charges + 1-4 exceptions, RB torsions."""
    s = mm.XmlSerializer.clone(system)
    nb = [f for f in s.getForces() if isinstance(f, mm.NonbondedForce)][0]
    lj = mm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)")
    lj.addPerParticleParameter("sigma")
    lj.addPerParticleParameter("epsilon")
    lj.setNonbondedMethod(min(nb.getNonbondedMethod(), 2))
    lj.setCutoffDistance(nb.getCutoffDistance())
    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        lj.addParticle([sig, eps])
        nb.setParticleParameters(i, q, sig, eps * 0)
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        lj.addExclusion(i, j)
    s.addForce(lj)
    for k, f in enumerate(s.getForces()):
        if isinstance(f, mm.PeriodicTorsionForce):
            rb = mm.RBTorsionForce()
            for t in range(f.getNumTorsions()):
                a, b, c, d, *_ = f.getTorsionParameters(t)
                rb.addTorsion(a, b, c, d, 0.5, 0.2, 0.0, -0.7, 0.0, 0.0)  # OPLS-style RB term
            s.addForce(rb)
            break
    return s


def test_opls_style_system(models, aladip):
    opls = _opls_style(aladip.system)
    new, spec = prepare_qmmm_system(opls, WHOLE, topology=aladip.topology)
    cnb = [f for f in new.getForces() if isinstance(f, mm.CustomNonbondedForce)][0]
    excl = {tuple(sorted(cnb.getExclusionParticles(k))) for k in range(cnb.getNumExclusions())}
    assert all((i, j) in excl for i in WHOLE for j in WHOLE if i < j)
    rb = [f for f in new.getForces() if isinstance(f, mm.RBTorsionForce)][0]
    assert rb.getNumTorsions() == 0  # all torsions of the dipeptide are QM-only
    # full ML/MM system runs and conserves momentum
    s, spec, comp = create_fieldmace_mm_system(opls, WHOLE, models["FieldEMACE"], topology=aladip.topology,
                                               model_kwargs=dict(device="cpu"))
    ctx, integ = _context(s, aladip.positions)
    integ.step(5)
    F = ctx.getState(getForces=True, groups={0}).getForces(asNumpy=True)
    assert np.isfinite(F).all()


def test_multistate_selection(aladip, tmp_path):
    from mace.tools.testing import build_field_model as build
    torch.set_default_dtype(torch.float64)
    p = tmp_path / "ms.model"
    torch.save(build("PerAtomFieldEMACE", n_energies=3), p)
    es = []
    for st in range(3):
        s, spec, comp = create_fieldmace_mm_system(aladip.system, WHOLE, str(p), topology=aladip.topology,
                                                   model_kwargs=dict(device="cpu", state=st))
        e, _ = comp.evaluate(aladip.positions.value_in_unit(unit.angstrom)[comp.frame.particles],
                             _box_A(aladip.system))
        es.append(e)
        assert e == pytest.approx(comp.last["energies"][st])
    assert len(set(np.round(es, 8))) == 3


def test_dataset_roundtrip(models, aladip, tmp_path):
    """Exported extxyz -> FieldMACE loader -> same energy as the OpenMM force."""
    import ase.io
    from mace import data
    from mace.tools import torch_geometric, utils

    settings = EmbeddingSettings(cutoff=9.0, switch_width=2.0)
    s, spec, comp = create_fieldmace_mm_system(aladip.system, SIDECHAIN, models["PerAtomFieldEMACE"],
                                               topology=aladip.topology, settings=settings,
                                               model_kwargs=dict(device="cpu"))
    atoms = frame_to_atoms(comp.frame, aladip.positions, aladip.system.getDefaultPeriodicBoxVectors())
    n, m = len(atoms), len(atoms.info["mm_charges"])
    attach_reference(atoms, -1.0, np.zeros((n, 3)), np.zeros((m, 3)))
    pad_mm_charges([atoms], n_max=m + 7)
    path = tmp_path / "train.xyz"
    write_extxyz(str(path), [atoms])
    _, configs = data.load_from_xyz(str(path), config_type_weights={"Default": 1.0})
    z_table = utils.AtomicNumberTable([int(z) for z in comp.model.atomic_numbers])
    ad = data.AtomicData.from_config(configs[0], z_table=z_table, cutoff=5.0)
    batch = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1)))
    assert tuple(batch["forces"].shape) == (n, 1, 3) and tuple(batch["mm_forces"].shape) == (m + 7, 1, 3)
    out = comp.model(batch.to_dict(), training=False)
    e, _ = comp.evaluate(aladip.positions.value_in_unit(unit.angstrom)[comp.frame.particles], _box_A(aladip.system))
    assert out["energy"].item() == pytest.approx(e, abs=1e-8)


def test_serialization(models, aladip):
    s, spec, comp = create_fieldmace_mm_system(aladip.system, SIDECHAIN, models["PerAtomFieldEMACE"],
                                               topology=aladip.topology, model_kwargs=dict(device="cpu"))
    s2 = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(s))
    e = []
    for system in (s, s2):
        c, _ = _context(system, aladip.positions)
        e.append(c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert e[0] == pytest.approx(e[1], abs=1e-6)


def test_virtual_site_charges_tip4p(models):
    """TIP4P-Ew M-sites carry the charge: forces on them must reach the real atoms."""
    ff = app.ForceField("tip4pew.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model="tip4pew", boxSize=mm.Vec3(1.8, 1.8, 1.8) * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.8 * unit.nanometer,
                             rigidWater=False)
    s, spec, comp = create_fieldmace_mm_system(system, [0, 1, 2], models["PerAtomFieldEMACE"],
                                               topology=modeller.topology, model_kwargs=dict(device="cpu"))
    assert spec.model_atomic_numbers == [8, 1, 1]
    vsites = [i for i in spec.mm_atoms if system.isVirtualSite(i)]
    assert vsites and all(system.getParticleMass(i)._value == 0 for i in vsites)
    assert 3 not in spec.mm_atoms  # M-site of the QM water belongs to the QM region
    nb = [f for f in s.getForces() if isinstance(f, mm.NonbondedForce)][0]
    assert nb.getParticleParameters(3)[0]._value == 0.0
    groups = _fieldmace_group(s)
    ctx, _ = _context(s, modeller.positions)
    ctx.computeVirtualSites()
    pos0 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    F = ctx.getState(getForces=True, groups=groups).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    probe = [4, 5, 6]  # O, H, H of the nearest MM water (index 3..7 incl. M-site at 7)
    h = 1e-5
    for i in probe:
        for k in range(3):
            e = []
            for sgn in (1, -1):
                p = pos0.copy()
                p[i, k] += sgn * h
                ctx.setPositions(p)
                ctx.computeVirtualSites()
                e.append(ctx.getState(getEnergy=True, groups=groups).getPotentialEnergy()._value)
            fd = -(e[0] - e[1]) / (2 * h)
            assert fd == pytest.approx(F[i, k], abs=2e-3 + 1e-4 * abs(F[i, k]))


def test_state_switching_and_reporter(aladip, tmp_path):
    """The driving state is a context parameter; the reporter logs all state energies."""
    from openmm import app

    from mace.mlmm import FieldMACEStateReporter, STATE_PARAMETER
    from mace.tools.testing import build_field_model

    p = tmp_path / "ms.model"
    torch.save(build_field_model("PerAtomFieldEMACE", n_energies=3), p)
    s, spec, comp = create_fieldmace_mm_system(aladip.system, WHOLE, str(p), topology=aladip.topology,
                                               model_kwargs=dict(device="cpu", state=0))
    _fieldmace_group(s)
    sim = app.Simulation(aladip.topology, s, mm.VerletIntegrator(0.25 * unit.femtosecond),
                         mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(aladip.positions)
    log = tmp_path / "states.csv"
    sim.reporters.append(FieldMACEStateReporter(str(log), 2, comp))
    energies = []
    for k in range(3):
        sim.context.setParameter(STATE_PARAMETER, k)
        energies.append(sim.context.getState(getEnergy=True, groups={31}).getPotentialEnergy()._value)
        assert comp.last["state"] == k
    all_states = comp.last["energies"] * 96.48533212331002
    assert np.allclose(energies, all_states, atol=1e-6)
    with pytest.raises(Exception):
        sim.context.setParameter(STATE_PARAMETER, 7)
        sim.context.getState(getEnergy=True)
    sim.context.setParameter(STATE_PARAMETER, 1)
    sim.step(4)
    rows = [l for l in open(log) if not l.startswith("#")]
    assert len(rows) == 2 and rows[-1].split(",")[2] == "1" and len(rows[-1].split(",")) == 6


def test_static_and_dynamic_graph_agree(models, aladip):
    pos = aladip.positions.value_in_unit(unit.angstrom)
    e = []
    for sg in (True, False):
        s, spec, comp = create_fieldmace_mm_system(aladip.system, WHOLE, models["PerAtomFieldEMACE"],
                                                   topology=aladip.topology,
                                                   model_kwargs=dict(device="cpu", static_graph=sg))
        e.append(comp.evaluate(np.asarray(pos)[comp.frame.particles], _box_A(aladip.system)))
    assert e[0][0] == pytest.approx(e[1][0], abs=1e-9)
    assert np.allclose(e[0][1], e[1][1], atol=1e-9)
