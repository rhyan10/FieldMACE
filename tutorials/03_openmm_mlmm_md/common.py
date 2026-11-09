# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
"""Shared defaults of the tutorial-03 scripts."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T01 = os.path.join(HERE, "..", "01_data_and_training")
EQUILIBRATED = os.path.join(T01, "work", "equilibrated.pdb")
INPUT = os.path.join(T01, "inputs", "aladip_solvated.pdb")


def default_pdb():
    """Start structure: tutorial 01's NPT-equilibrated system if it exists.

    inputs/aladip_solvated.pdb is the raw solvated input: its box is too large (water density
    0.66 g/mL). Tutorial 01 step 1 equilibrates it at 1 bar (~1.0 g/mL) before sampling the training data,
    so MD should start from work/equilibrated.pdb too - otherwise the model sees a sparser solvent than in
    training. Without step 1, the raw input is used (with a warning).
    """
    if os.path.exists(EQUILIBRATED):
        return EQUILIBRATED
    print("NOTE: ../01_data_and_training/work/equilibrated.pdb not found (tutorial 01 step 1); starting from the "
          "raw input, whose water density is only 0.66 g/mL. Run 01_prepare_system.py or equilibrate with NPT first.")
    return INPUT
