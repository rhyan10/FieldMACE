# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""fieldmace_active_learning: automated ML/MM active learning (see mace.mlmm.active_learning).

    fieldmace_active_learning --config al.yaml            # run, or resume where it stopped
    fieldmace_active_learning --config al.yaml --rounds 1 # at most one more round now
    fieldmace_active_learning --config al.yaml --status
    fieldmace_active_learning --example > al.yaml         # an annotated configuration
"""
import argparse
import sys

EXAMPLE = """\
# FieldMACE active learning (paths relative to this file)
work_dir: al_run
system:                       # the MM system, as mace.mlmm.io.load_system takes it
  pdb: system.pdb
  forcefield: [amber14-all.xml, amber14/tip3p.xml]
  cutoff_nm: 0.9
setup: setup.json             # QM atoms, link atoms, embedding (written with the training data)
initial_data: train.xyz       # labelled frames to start from
# test_file: test.xyz
# initial_models: [model_a.model, model_b.model]   # skip the first training
committee_size: 3             # models per round (different seeds)
train_args:                   # mace_run_train flags (train_file, seed, name, work_dir are set by the loop)
  model: PerAtomFieldEMACE
  E0s: average
  r_max: 5.0
  num_interactions: 2
  hidden_irreps: 32x0e + 32x1o
  MLP_irreps: 16x0e
  multipole_max_ell: 2
  valid_fraction: 0.1
  batch_size: 10
  max_num_epochs: 200
  energy_weight: 1
  forces_weight: 100
  mm_forces_weight: 10
  default_dtype: float32
sampling:
  temperatures: [300, 400]    # one trajectory each
  n_frames: 50                # frames per trajectory
  every: 100                  # MD steps between frames
  dt_fs: 0.5
  state: 0                    # electronic state that drives the MD
  backend: python             # or torch (openmm-torch)
selection:
  n_select: 20                # frames labelled per round
  force_std_min: 0.05         # eV/A: below, the committee agrees
  force_std_max: 1.0          # eV/A: above, the frame is outside anything sensible
  min_separation: 2           # frames apart along a trajectory
labeller: external            # pyscf | model:<path> | external | file.py:function | module:function
# labeller_args: {xc: b3lyp, basis: 6-31g*}
rounds: 5
converged_fraction: 0.02      # stop when fewer frames are uncertain
device: cpu
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--rounds", type=int, default=None, help="run at most this many rounds now")
    p.add_argument("--status", action="store_true")
    p.add_argument("--example", action="store_true", help="print an annotated configuration")
    a = p.parse_args(argv)
    if a.example:
        sys.stdout.write(EXAMPLE)
        return 0
    if not a.config:
        p.error("--config is required")
    from mace.mlmm.active_learning import ActiveLearningLoop, ALConfig

    try:
        import yaml  # noqa: F401
    except ImportError:
        print("ERROR: fieldmace_active_learning reads YAML: pip install pyyaml", file=sys.stderr)
        return 1
    try:
        cfg = ALConfig.from_file(a.config)
    except (ValueError, FileNotFoundError) as err:
        print(f"ERROR in the configuration: {err}", file=sys.stderr)
        return 1
    loop = ActiveLearningLoop(cfg, log=lambda m: print(m, flush=True))
    if a.status:
        print(loop.status())
        return 0
    try:
        loop.run(max_rounds=a.rounds)
    except (RuntimeError, ValueError, FileNotFoundError) as err:  # a failed step: say which, resume later
        print(f"ERROR: {err}", file=sys.stderr)
        print(loop.status())
        print("Fix the problem and run the same command again: finished steps are not repeated.", file=sys.stderr)
        return 1
    print(loop.status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
