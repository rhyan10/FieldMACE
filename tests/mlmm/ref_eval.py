# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""Reference evaluation through FieldMACE's own data pipeline (used to validate the OpenMM path)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, ase
from mace import data
from mace.tools import torch_geometric, utils

def ref_eval(model, symbols, qm_pos, mm_pos, mm_q):
    atoms = ase.Atoms(symbols, positions=qm_pos)
    atoms.info = {"mm_charges": np.asarray(mm_q), "mm_positions": np.asarray(mm_pos)}
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cfg = data.config_from_atoms(atoms)
    ad = data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))
    batch = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1)))
    out = model(batch.to_dict(), training=False, compute_force=True)
    return out
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    qm = np.array([[0,0,0],[1.09,0,0],[-0.36,1.03,0],[-0.36,-0.51,0.89],[-0.36,-0.51,-0.89]],float)
    mm = rng.uniform(-8,8,(20,3)); q = rng.uniform(-0.8,0.8,20)
    for name in ["PerAtomFieldEMACE","FieldEMACE"]:
        m = torch.load(f"{name}.model", weights_only=False)
        out = ref_eval(m, "CHHHH", qm, mm, q)
        print(name, {k:(tuple(v.shape) if torch.is_tensor(v) else v) for k,v in out.items() if k in ("energy","forces","mm_forces")})
        print(" E=", out["energy"].detach().numpy())
