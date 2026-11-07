"""Reference data for test_original_pickles_reproduce_original_numbers.

Run with the ORIGINAL FieldMACE (commit 5a50538) importable, in an empty directory:

    git worktree add /tmp/fieldmace-v1 5a50538
    mkdir /tmp/legacy && cd /tmp/legacy
    PYTHONPATH=/tmp/fieldmace-v1 python /path/to/tests/legacy/make_legacy_reference.py
    cd /path/to/FieldMACE && FIELDMACE_LEGACY_DIR=/tmp/legacy pytest tests/test_field_models.py -k original

It builds models with the original code, evaluates them and pickles them; the test loads the
pickles with the current code and checks that energies and forces are unchanged.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, ase, mace
from mace.modules import models as _m
assert not hasattr(_m, "_FieldMACEBase") and "field" not in dir(mace.modules), "run this with the ORIGINAL FieldMACE"
from e3nn import o3
from mace import modules, data
from mace.tools import torch_geometric, utils
torch.set_default_dtype(torch.float64)
ZS=[1,6,7,8]
def build(cls, n_e=1):
    torch.manual_seed(0)
    return getattr(modules, cls)(r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2, num_elements=4, hidden_irreps=o3.Irreps("16x0e+16x1o"), MLP_irreps=o3.Irreps("16x0e"),
        field_irreps=o3.Irreps("4x0e+4x1o+4x2e+4x3o"), atomic_energies=np.array([-1.,-2.,-3.,-4.]), avg_num_neighbors=3.0, atomic_numbers=ZS,
        pair_repulsion=False, n_energies=n_e, distance_transform="None", multipole_max_ell=1,
        correlation=3, gate=torch.nn.functional.silu, radial_MLP=[32,32], radial_type="bessel",
        compute_nacs=False, compute_dipoles=False)
rng=np.random.default_rng(5)
qm=np.array([[0,0,0],[1.09,0,0],[-0.36,1.03,0],[-0.36,-0.51,0.89],[-0.36,-0.51,-0.89]],float)+np.array([0.7,-0.4,1.1])
mm=rng.uniform(-8,8,(60,3)); mm=mm[np.linalg.norm(mm-qm.mean(0),axis=1)>3.0][:40]; q=rng.uniform(-.8,.8,len(mm))
a=ase.Atoms("CHHHH", positions=qm); a.info={"mm_positions":mm,"mm_charges":q}
z=utils.AtomicNumberTable(ZS)
ad=data.AtomicData.from_config(data.config_from_atoms(a), z_table=z, cutoff=5.0)
b=next(iter(torch_geometric.dataloader.DataLoader([ad],batch_size=1)))
res={"qm":qm,"mm":mm,"q":q}
for cls,n_e in [("FieldEMACE",1),("PerAtomFieldEMACE",1),("AttentionPerAtomFieldEMACE",1),("PerAtomFieldEMACE",3)]:
    m=build(cls,n_e); key=f"{cls}_{n_e}"
    out=m(b.to_dict(), training=False)
    res[key+"_E"]=out["energy"].detach().numpy(); res[key+"_F"]=out["forces"].detach().numpy(); res[key+"_Fmm"]=out["mm_forces"].detach().numpy() if out.get("mm_forces") is not None else np.zeros(0)
    torch.save(m, f"{key}.model")
np.savez("legacy_ref.npz", **res); print("ok")
