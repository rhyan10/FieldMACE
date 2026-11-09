# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
Automated active learning for ML/MM: train a committee -> ML/MM MD in OpenMM -> committee
disagreement -> label the most uncertain frames -> add them -> repeat.

    fieldmace_active_learning --config al.yaml          # runs or resumes the loop
    fieldmace_active_learning --config al.yaml --status

Every round lives in <work_dir>/round_NN/:
    models/model_K.model    committee (K = 0 .. committee_size-1, different seeds)
    frames.xyz              all MD frames of this round, with the committee's disagreement
    candidates.xyz          the selected frames (to be labelled)
    labelled.xyz            the labelled frames (written by the labeller, or by you with "external")
    train.xyz               training set of the next round = this round's train.xyz + labelled.xyz
and <work_dir>/state.json records how far the loop got, so it resumes after a crash, a time limit or
external labelling.

Disagreement (per frame): force_std = max over atoms and states of the RMS deviation of the
committee's forces from their mean (eV/A); energy_std = std of the energies after removing each
model's mean over the frames (eV). Frames with force_std in [force_std_min, force_std_max] are
candidates; above force_std_max the model is far outside its data (the frame is unphysical) and the
frame is not labelled. The most uncertain candidates are taken, at least min_separation frames apart
along a trajectory. The loop stops after `rounds` rounds or when fewer than `converged_fraction` of
the frames are uncertain.

Configuration (YAML; paths relative to the YAML file):

    work_dir: al_run
    system: {pdb: system.pdb, forcefield: [amber14-all.xml, amber14/tip3p.xml], cutoff_nm: 0.9}
                                   # anything mace.mlmm.io.load_system accepts (prmtop/inpcrd, top/gro, psf, system_xml)
    setup: setup.json              # QM region, link atoms, embedding (mace.mlmm.dataset.save_setup)
    initial_data: train.xyz
    test_file: test.xyz            # optional, passed to training
    initial_models: []             # optional: a committee for round 0 (skips the first training)
    committee_size: 3
    train_args: {model: PerAtomFieldEMACE, r_max: 5.0, max_num_epochs: 200, ...}   # mace_run_train flags
    sampling: {temperatures: [300, 400], n_frames: 50, every: 100, dt_fs: 0.5, state: 0,
               backend: python, platform: null, minimize: true}
    selection: {n_select: 20, force_std_min: 0.05, force_std_max: 1.0, min_separation: 2}
    labeller: pyscf                # pyscf | model:<path> | external | file.py:function | module:function
    labeller_args: {xc: b3lyp, basis: 6-31g*}
    rounds: 5
    converged_fraction: 0.02
    device: cpu
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ase.io
import numpy as np

from .dataset import write_extxyz

PHASES = ["train", "sample", "select", "label", "merge"]
SAMPLING_KEYS = ("temperatures", "n_frames", "every", "dt_fs", "state", "backend", "platform", "minimize",
                 "friction_per_ps")
SELECTION_KEYS = ("n_select", "force_std_min", "force_std_max", "min_separation")


@dataclass
class ALConfig:
    work_dir: str
    setup: str
    initial_data: str
    system: Dict = field(default_factory=dict)
    test_file: Optional[str] = None
    initial_models: List[str] = field(default_factory=list)
    committee_size: int = 3
    train_args: Dict = field(default_factory=dict)
    sampling: Dict = field(default_factory=dict)
    selection: Dict = field(default_factory=dict)
    labeller: str = "external"
    labeller_args: Dict = field(default_factory=dict)
    rounds: int = 5
    converged_fraction: float = 0.02
    device: str = "cpu"

    @classmethod
    def from_file(cls, path: str) -> "ALConfig":
        import yaml

        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        base = os.path.dirname(os.path.abspath(path))

        def rel(p):
            return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

        for k in ("work_dir", "setup", "initial_data", "test_file"):
            if d.get(k) is not None:
                d[k] = rel(d[k])
        d["initial_models"] = [rel(p) for p in d.get("initial_models", []) or []]
        sysd = dict(d.get("system", {}) or {})
        for k in ("pdb", "prmtop", "inpcrd", "top", "gro", "gmx_include", "psf", "crd", "system_xml"):
            if sysd.get(k):
                sysd[k] = rel(sysd[k])
        sysd["charmm_params"] = [rel(p) for p in sysd.get("charmm_params", []) or []]
        sysd["forcefield"] = [p if not os.path.exists(rel(p)) else rel(p) for p in sysd.get("forcefield", []) or []]
        d["system"] = sysd
        lab = d.get("labeller", "external")
        if lab.startswith("model:"):
            d["labeller"] = "model:" + rel(lab[len("model:"):])
        elif ".py:" in lab:
            f, fn = lab.rsplit(":", 1)
            d["labeller"] = rel(f) + ":" + fn
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)} (known: {sorted(cls.__dataclass_fields__)})")
        missing_req = [k for k in ("work_dir", "setup", "initial_data") if not d.get(k)]
        if missing_req:
            raise ValueError(f"{path}: required key(s) {missing_req} missing (fieldmace_active_learning --example)")
        checks = [("setup", d.get("setup")), ("initial_data", d.get("initial_data")), ("test_file", d.get("test_file"))]
        checks += [("initial_models", m) for m in d["initial_models"]]
        checks += [(f"system.{k}", v) for k, v in sysd.items()
                   if k in ("pdb", "prmtop", "inpcrd", "top", "gro", "psf", "crd", "system_xml") and v]
        if str(d.get("labeller", "")).startswith("model:"):
            checks.append(("labeller", d["labeller"][len("model:"):]))
        for key, p_ in checks:
            if p_ and not os.path.exists(p_):
                raise ValueError(f"{path}: {key} = {p_} does not exist (paths are relative to the YAML file)")
        for sub, allowed in (("sampling", SAMPLING_KEYS), ("selection", SELECTION_KEYS)):
            bad = set(d.get(sub) or {}) - set(allowed)
            if bad:
                raise ValueError(f"{path}: unknown key(s) {sorted(bad)} in '{sub}' (known: {sorted(allowed)})")
        if int(d.get("committee_size", 3)) < 2 and len(d["initial_models"]) < 2:
            raise ValueError(f"{path}: committee_size must be at least 2 (the disagreement needs a committee)")
        return cls(**d)


def train_flags(args: Dict) -> List[str]:
    """{key: value} -> mace_run_train flags; True for an on/off flag (e.g. ema: true) gives just --ema."""
    from mace.tools import build_default_arg_parser

    switches = {a.dest for a in build_default_arg_parser()._actions if a.nargs == 0}
    out = []
    for k, v in args.items():
        if v is None:
            continue
        if k in switches:
            if v is True or str(v).lower() == "true":
                out.append(f"--{k}")
            continue
        out.append(f"--{k}={v}")
    return out


# --------------------------------------------------------------------------- #
# committee disagreement and selection
# --------------------------------------------------------------------------- #
def committee_disagreement(models: List[str], frames: List[ase.Atoms], device: str = "cpu") -> Dict[str, np.ndarray]:
    """energy_std [F] (eV, all states: max), force_std [F] (eV/A, max over atoms and states)."""
    from mace.calculators import MACECalculator

    E, Fo = [], []
    for path in models:
        calc = MACECalculator(path, device=device, default_dtype="float64")
        e, f = [], []
        for at in frames:
            x = at.copy()
            x.info = {k: at.info[k] for k in ("mm_positions", "mm_charges") if k in at.info}
            calc.calculate(x)
            e.append(np.atleast_1d(calc.results["energies"]).copy())
            fa = calc.results["forces_all"] if "forces_all" in calc.results else calc.results["forces"][:, None]
            f.append(np.asarray(fa).copy())
        e = np.array(e)  # [F, S]
        E.append(e - e.mean(0, keepdims=True))
        Fo.append(np.array(f))  # [F, N, S, 3]
    E, Fo = np.array(E), np.array(Fo)
    e_std = E.std(0).max(-1)
    f_dev = np.sqrt(((Fo - Fo.mean(0)) ** 2).sum(-1).mean(0))  # [F, N, S]
    return {"energy_std": e_std, "force_std": f_dev.reshape(len(frames), -1).max(-1)}


def select_frames(force_std: np.ndarray, traj_id: np.ndarray, n_select: int, force_std_min: float = 0.05,
                  force_std_max: float = 1.0, min_separation: int = 2) -> List[int]:
    """Most uncertain frames inside [min, max], at least min_separation frames apart within a trajectory."""
    order = np.argsort(force_std)[::-1]
    picked: List[int] = []
    for i in order:
        if len(picked) >= n_select:
            break
        if not force_std_min <= force_std[i] <= force_std_max:
            continue
        if any(traj_id[j] == traj_id[i] and abs(j - i) < min_separation for j in picked):
            continue
        picked.append(int(i))
    return sorted(picked)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
class ActiveLearningLoop:
    def __init__(self, config: ALConfig, log=print):
        self.c = config
        self.log = log
        os.makedirs(self.c.work_dir, exist_ok=True)
        self.state_file = os.path.join(self.c.work_dir, "state.json")
        self.state = {"round": 0, "done": [], "history": [], "finished": False}
        if os.path.exists(self.state_file):
            with open(self.state_file) as fh:
                self.state = json.load(fh)

    # ------------------------------------------------------------- #
    def rdir(self, r: Optional[int] = None) -> str:
        return os.path.join(self.c.work_dir, f"round_{self.state['round'] if r is None else r:02d}")

    def save(self):
        with open(self.state_file, "w") as fh:
            json.dump(self.state, fh, indent=2)

    def train_file(self, r: int) -> str:
        return self.c.initial_data if r == 0 else os.path.join(self.rdir(r - 1), "train.xyz")

    def committee(self, r: int) -> List[str]:
        if r == 0 and self.c.initial_models:
            return list(self.c.initial_models)
        return [os.path.join(self.rdir(r), "models", f"model_{k}.model") for k in range(self.c.committee_size)]

    # ------------------------------------------------------------- #
    def run(self, max_rounds: Optional[int] = None) -> Dict:
        rounds_run = 0
        while not self.state["finished"]:
            r = self.state["round"]
            if r >= self.c.rounds:
                self.state["finished"] = True
                self.log(f"reached {self.c.rounds} rounds")
                break
            os.makedirs(self.rdir(), exist_ok=True)
            for phase in PHASES:
                if phase in self.state["done"]:
                    continue
                ok = getattr(self, f"_{phase}")(r)
                if ok is False:  # waiting for external labels, or converged
                    self.save()
                    return self.state
                self.state["done"].append(phase)
                self.save()
            self.state["round"] += 1
            self.state["done"] = []
            self.save()
            rounds_run += 1
            if max_rounds is not None and rounds_run >= max_rounds:
                break
        self.save()
        return self.state

    # ------------------------------------------------------------- #
    def _train(self, r: int):
        if r == 0 and self.c.initial_models:
            self.log(f"round {r}: using the initial committee {self.c.initial_models}")
            return True
        mdir = os.path.join(self.rdir(r), "models")
        os.makedirs(mdir, exist_ok=True)
        for k in range(self.c.committee_size):
            out = os.path.join(mdir, f"model_{k}.model")
            if os.path.exists(out):
                continue
            args = dict(self.c.train_args)
            args.update(name=f"model_{k}", train_file=self.train_file(r), seed=int(args.get("seed", 1)) + k,
                        work_dir=mdir)
            args.setdefault("device", self.c.device)
            if self.c.test_file:
                args.setdefault("test_file", self.c.test_file)
            cmd = [sys.executable, "-m", "mace.cli.run_train"] + train_flags(args)
            self.log(f"round {r}: training committee member {k} on {self.train_file(r)}")
            with open(os.path.join(mdir, f"train_{k}.log"), "w") as fh:
                p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            if p.returncode != 0 or not os.path.exists(out):
                raise RuntimeError(f"training failed, see {os.path.join(mdir, f'train_{k}.log')}")
        return True

    def _sample(self, r: int):
        import openmm as mm
        from openmm import app, unit

        from .dataset import frame_to_atoms, load_setup
        from .force import create_fieldmace_mm_system
        from .io import constrain_hbonds_except, load_system

        s = dict(temperatures=[300.0], n_frames=50, every=100, dt_fs=0.5, state=0, backend="python", platform=None,
                 minimize=True, friction_per_ps=1.0)
        s.update(self.c.sampling)
        sysd = dict(self.c.system)
        sysd.setdefault("constraints", None)
        for key in ("constraints", "nonbonded_method"):  # names in the YAML -> OpenMM objects
            if isinstance(sysd.get(key), str):
                sysd[key] = None if sysd[key].lower() == "none" else getattr(app, sysd[key])
        loaded = load_system(**sysd)
        qm_atoms, settings, prep_kw = load_setup(self.c.setup)
        if sysd["constraints"] is None and not sysd.get("system_xml"):  # X-H constraints, except in the QM region
            constrain_hbonds_except(loaded.system, loaded.topology, qm_atoms)
        model = self.committee(r)[0]
        system, spec, comp = create_fieldmace_mm_system(
            loaded.system, qm_atoms, model, topology=loaded.topology, settings=settings, backend=s["backend"],
            model_kwargs=dict(device=self.c.device, state=int(s["state"])), **prep_kw)
        from .embedding import EmbeddingFrame

        frame = EmbeddingFrame(spec, settings)
        frames = []
        plat = [mm.Platform.getPlatformByName(s["platform"])] if s.get("platform") else []
        for ti, T in enumerate(s["temperatures"]):
            integ = mm.LangevinMiddleIntegrator(float(T) * unit.kelvin, float(s["friction_per_ps"]) / unit.picosecond,
                                                float(s["dt_fs"]) * unit.femtosecond)
            sim = app.Simulation(loaded.topology, system, integ, *plat)
            sim.context.setPositions(loaded.positions)
            if loaded.box_vectors is not None:
                sim.context.setPeriodicBoxVectors(*loaded.box_vectors)
            if s["minimize"]:
                sim.minimizeEnergy(maxIterations=200)
            sim.context.setVelocitiesToTemperature(float(T) * unit.kelvin, 1000 * r + ti + 1)
            n_ok = 0
            for k in range(int(s["n_frames"])):
                try:
                    sim.step(int(s["every"]))
                    st = sim.context.getState(getPositions=True, getEnergy=True)
                    if not np.isfinite(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                        raise ValueError("NaN energy")
                except Exception as err:  # noqa: BLE001  the model left its domain: keep what we have
                    self.log(f"  {T} K: MD stopped after {k} frames ({type(err).__name__}: {err})")
                    break
                box = st.getPeriodicBoxVectors(asNumpy=True) if spec.periodic else None
                frames.append(frame_to_atoms(frame, st.getPositions(asNumpy=True), box, extra_info={
                    "al_round": r, "al_traj": ti, "al_temperature": float(T), "md_step": sim.currentStep,
                    "config_type": "active_learning"}))
                n_ok += 1
            self.log(f"round {r}: ML/MM MD at {T} K with {os.path.basename(model)}: {n_ok} frames")
            del sim, integ
        if not frames:
            raise RuntimeError("no MD frames: the first model is unstable from the start structure")
        write_extxyz(os.path.join(self.rdir(r), "frames.xyz"), frames)
        return True

    def _select(self, r: int):
        sel = dict(n_select=20, force_std_min=0.05, force_std_max=1.0, min_separation=2)
        sel.update(self.c.selection)
        frames = ase.io.read(os.path.join(self.rdir(r), "frames.xyz"), index=":")
        models = self.committee(r)
        if len(models) < 2:
            raise ValueError("active learning needs a committee of at least 2 models")
        d = committee_disagreement(models, frames, self.c.device)
        for at, e, f in zip(frames, d["energy_std"], d["force_std"]):
            at.info["committee_energy_std"] = float(e)
            at.info["committee_force_std"] = float(f)
        write_extxyz(os.path.join(self.rdir(r), "frames.xyz"), frames)
        fs = d["force_std"]
        traj = np.array([at.info.get("al_traj", 0) for at in frames])
        pick = select_frames(fs, traj, int(sel["n_select"]), float(sel["force_std_min"]), float(sel["force_std_max"]),
                             int(sel["min_separation"]))
        frac = float(np.mean(fs > float(sel["force_std_min"])))
        n_wild = int(np.sum(fs > float(sel["force_std_max"])))
        info = {"round": r, "n_frames": len(frames), "force_std_median": float(np.median(fs)),
                "force_std_max": float(fs.max()), "uncertain_fraction": frac, "above_max": n_wild,
                "n_selected": len(pick)}
        self.state["history"].append(info)
        self.log(f"round {r}: committee force std median {1000 * np.median(fs):.0f} meV/A, max {1000 * fs.max():.0f}; "
                 f"{100 * frac:.1f}% of frames uncertain, {n_wild} beyond force_std_max; selected {len(pick)}")
        write_extxyz(os.path.join(self.rdir(r), "candidates.xyz"), [frames[i] for i in pick])
        if frac < self.c.converged_fraction or not pick:
            self.log(f"round {r}: converged (uncertain fraction {frac:.3f} < {self.c.converged_fraction})")
            self.state["finished"] = True
            return False
        return True

    def _label(self, r: int):
        from .labelers import get_labeller

        out = os.path.join(self.rdir(r), "labelled.xyz")
        if os.path.exists(out):
            self.log(f"round {r}: using {out}")
            return True
        lab = get_labeller(self.c.labeller)
        cand = os.path.join(self.rdir(r), "candidates.xyz")
        if lab is None:
            self.log(f"round {r}: label {cand} with your QM code, write the result to {out} "
                     "(REF_energy, REF_forces, REF_mm_forces) and run again")
            return False
        frames = ase.io.read(cand, index=":")
        labelled = lab(frames, **self.c.labeller_args)
        if not labelled:
            raise RuntimeError("the labeller returned no frames")
        write_extxyz(out, labelled)
        self.log(f"round {r}: labelled {len(labelled)}/{len(frames)} frames")
        return True

    def _merge(self, r: int):
        dst = os.path.join(self.rdir(r), "train.xyz")
        shutil.copyfile(self.train_file(r), dst)
        new = ase.io.read(os.path.join(self.rdir(r), "labelled.xyz"), index=":")
        for at in new:  # bookkeeping keys the loader does not need
            for k in ("committee_energy_std", "committee_force_std"):
                at.info.pop(k, None)
        write_extxyz(dst, new, append=True)
        n = len(ase.io.read(dst, index=":"))
        self.log(f"round {r}: training set for round {r + 1}: {n} frames ({dst})")
        return True

    def status(self) -> str:
        lines = [f"work_dir {self.c.work_dir}: round {self.state['round']}, done {self.state['done']}, "
                 f"finished {self.state['finished']}"]
        for h in self.state["history"]:
            lines.append(f"  round {h['round']}: {h['n_frames']} frames, force std median "
                         f"{1000 * h['force_std_median']:.0f} meV/A, uncertain {100 * h['uncertain_fraction']:.1f}%, "
                         f"selected {h['n_selected']}")
        return "\n".join(lines)
