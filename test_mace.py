import ase.io
from mace.calculators import MACECalculator
import numpy as np

mols = ase.io.read("correct_esystems2.xyz", ":2")
properties = ['energy', 'forces']
for mol in mols:
    calc = MACECalculator(model_paths="esystem_notransfer_0.8_3.model", n_energies=3, device="cuda")
    calc.calculate(mol)
    results = calc.results
    print(results)
