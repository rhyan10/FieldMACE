"""End-to-end training through the command line (small, CPU, a few epochs)."""
import os
import subprocess
import sys

import ase
import ase.io
import numpy as np
import pytest
import torch

from mace.tools.testing import build_field_model, make_batch, methane_in_charges

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_dataset(path, n_states=1, n=24):
    """Labels from a random 'teacher' FieldMACE; mixed numbers of charges incl. gas phase."""
    teacher = build_field_model("PerAtomFieldEMACE", seed=99, n_energies=n_states)
    frames = []
    rng = np.random.default_rng(0)
    for i in range(n):
        n_mm = [0, 5, 17, 30][i % 4]
        sym, qm, mm, q = methane_in_charges(n_mm, seed=i)
        qm = qm + rng.normal(scale=0.05, size=qm.shape)
        out = teacher(make_batch(teacher, [(sym, qm, mm, q)]).to_dict())
        a = ase.Atoms(sym, positions=qm)
        if n_mm:
            a.info["mm_positions"] = mm
            a.info["mm_charges"] = q
            if i % 8 != 1:  # some frames without MM-force labels
                a.info["REF_mm_forces"] = out["mm_forces"].detach().numpy()
        a.info["REF_energy"] = out["energy"].detach().numpy()
        a.info["REF_forces"] = out["forces"].detach().numpy()
        frames.append(a)
    ase.io.write(path, frames, format="extxyz")


def train(tmp_path, model, extra=(), n_states=1, epochs=3):
    data = tmp_path / "train.xyz"
    make_dataset(data, n_states=n_states)
    cmd = [
        sys.executable, os.path.join(REPO, "scripts", "run_train.py"),
        "--name=t", f"--train_file={data}", "--valid_fraction=0.25", "--seed=1", "--E0s=average",
        f"--model={model}", "--r_max=5.0", "--batch_size=4", "--lr=0.01", f"--n_energies={n_states}",
        "--correlation=2", f"--max_num_epochs={epochs}", "--default_dtype=float64", "--device=cpu",
        "--hidden_irreps=8x0e + 8x1o", "--MLP_irreps=8x0e", "--num_radial_basis=6", "--num_interactions=2",
        "--multipole_max_ell=1", "--error_table=EnergyNacsDipoleMAE", "--scheduler=ReduceLROnPlateau",
        f"--work_dir={tmp_path}", *extra,
    ]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=900)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert os.path.exists(tmp_path / "t.model"), log[-3000:]
    return log


@pytest.mark.parametrize("model,n_states,extra", [
    ("PerAtomFieldEMACE", 1, []),
    ("FieldEMACE", 2, []),
    ("AttentionPerAtomFieldEMACE", 1, []),
    ("PerAtomFieldEMACE", 1, ["--multipole_embedding=legacy", "--mm_forces_weight=0"]),
    ("FieldEMACE", 1, ["--expansion_centre=origin", "--multipole_embedding=legacy"]),
])
def test_cli_training(tmp_path, model, n_states, extra):
    log = train(tmp_path, model, extra, n_states)
    assert "Could not save a TorchScript-compiled model" not in log
    m = torch.load(tmp_path / "t.model", weights_only=False)
    assert type(m).__name__ == model and m.n_energies == n_states
    assert m.legacy == ("--multipole_embedding=legacy" in extra)
    # TorchScript export works and agrees with the eager model
    sm = torch.jit.load(str(tmp_path / "t_compiled.model"))
    cfg = [methane_in_charges(11, seed=5)]
    e1 = m(make_batch(m, cfg).to_dict())["energy"]
    e2 = sm(make_batch(m, cfg).to_dict())["energy"]
    assert torch.allclose(e1, e2, atol=1e-10)
    if "--mm_forces_weight=0" in extra:
        assert "MAE_mm_forces" not in log.split("Started training")[-1].split("\n")[1]


def test_eval_configs_cli(tmp_path):
    train(tmp_path, "PerAtomFieldEMACE", epochs=1, n_states=2)
    out = tmp_path / "pred.xyz"
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "eval_configs.py"), f"--configs={tmp_path / 'train.xyz'}",
         f"--model={tmp_path / 't.model'}", f"--output={out}", "--batch_size=5"],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    frames = ase.io.read(out, ":")
    assert len(frames) == 24
    assert frames[2].info["MACE_energies"].shape == (2,)
    assert frames[2].info["MACE_mm_forces"].shape == (17, 2, 3)
    assert frames[0].arrays["MACE_forces"].shape == (5, 3)


def test_hdf5_preprocess_and_train(tmp_path):
    make_dataset(tmp_path / "train.xyz")
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "preprocess_data.py"), "--train_file=train.xyz",
         "--h5_prefix=h5/", "--r_max=5.0", "--valid_fraction=0.25", "--E0s=average", "--seed=1",
         "--compute_statistics", "--num_process=1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "run_train.py"), "--name=h",
         "--train_file=h5/train/train_0.h5", "--valid_file=h5/val/val_0.h5", "--statistics_file=h5/statistics.json",
         "--test_dir=h5/test", "--model=PerAtomFieldEMACE", "--batch_size=4", "--max_num_epochs=1",
         "--default_dtype=float64", "--device=cpu", "--hidden_irreps=8x0e + 8x1o", "--MLP_irreps=8x0e",
         "--num_interactions=2", "--error_table=EnergyNacsDipoleMAE"],
        cwd=tmp_path, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert "MAE_mm_forces" in r.stdout + r.stderr


def test_transfer_learning_from_bundled_foundation(tmp_path):
    foundation = os.path.join(REPO, "mace", "calculators", "foundations_models", "2023-12-03-mace-mp.model")
    log = train(tmp_path, "PerAtomFieldEMACE", epochs=1,
                extra=[f"--foundation_model={foundation}", "--MLP_irreps=16x0e"])
    assert "to match the foundation model" in log
    m = torch.load(tmp_path / "t.model", weights_only=False)
    assert float(m.r_max) == 6.0
