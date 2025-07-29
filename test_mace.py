from tqdm import tqdm
import ase.io
from mace.calculators import MACECalculator
import numpy as np


# Load molecules
mols = ase.io.read("dmabn_test.xyz", ":")

# Initialize calculator once (assumes model can be reused)
calc = MACECalculator(model_paths="dmabn_L1.model", n_energies=4, device="cuda")

# Containers for accumulating errors
energy_errors = []
forces_errors = []
multipolar_fit_errors = []

# Loop over all molecules

for mol in tqdm(mols):
    calc.calculate(mol)
    results = calc.results

    # Energies
    pred_energy = results["REF_energy"]
    ref_energy = mol.info["REF_energy"]
    energy_errors.append(np.abs(pred_energy - ref_energy))

    # Forces
    pred_forces = results["REF_forces"]
    ref_forces = mol.info["REF_forces"]
    forces_errors.append(np.abs(pred_forces - ref_forces).mean())

    # Multipolar fit
    pred_mfit = results.get("REF_multipolar_fit", 0.0)  
    ref_mfit = mol.info.get("REF_scalars", 0.0)
    neg = np.square(ref_mfit - pred_mfit)[..., np.newaxis]
    pos = np.square(ref_mfit + pred_mfit)[..., np.newaxis]

    # stack them along the last axis
    vec = np.concatenate((pos, neg), axis=-1)

    # take the min over that last axis and append
    multipolar_fit_errors.append(np.min(vec, axis=-1))


# Compute MAEs
mae_energy = np.mean(energy_errors)
mae_forces = np.mean(forces_errors)
mae_mfit = np.mean(multipolar_fit_errors)

# Print results
print(f"MAE energy: {mae_energy:.6f}")
print(f"MAE forces: {mae_forces:.6f}")
print(f"MAE multipolar fit: {mae_mfit:.6f}")

