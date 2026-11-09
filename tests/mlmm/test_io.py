# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""QM-region selection language."""
import os

import pytest
from openmm import app

from mace.mlmm.io import select_atoms

PDB = os.path.join(os.path.dirname(__file__), "..", "..", "tutorials", "01_data_and_training", "inputs",
                   "aladip_solvated.pdb")


def test_select_atoms():
    top = app.PDBFile(PDB).topology
    side = [8, 9, 10, 11, 12, 13]
    assert select_atoms(top, "index 8-13") == side
    assert select_atoms(top, "resid 2 and name CA HA CB HB1 HB2 HB3") == side
    assert select_atoms(top, "resid 2") == list(range(6, 16))
    assert select_atoms(top, "resname ACE or resid 2 and name CB") == [0, 1, 2, 3, 4, 5, 10]  # 'and' binds first
    assert len(select_atoms(top, "resname HOH and element O")) == (top.getNumAtoms() - 22) // 3
    with pytest.raises(ValueError):
        select_atoms(top, "resname XYZ")
    with pytest.raises(ValueError):
        select_atoms(top, "within 5 of resname ALA")
