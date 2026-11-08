# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
FieldMACE / MM simulations in OpenMM with any force field, and training-data export.

    from mace.mlmm import create_fieldmace_mm_system, EmbeddingSettings
    system, spec, comp = create_fieldmace_mm_system(mm_system, qm_atoms, "model.model", topology=top)

Requires OpenMM >= 8.3 (openmm.PythonForce). With backend="torch" the model runs inside OpenMM through
openmm-torch (TorchForce); mace.mlmm.torchforce.export_torchscript writes the ML/MM potential as a
TorchScript file for other programs. The rest of FieldMACE does not need OpenMM.
"""
from .embedding import EmbeddingFrame, EmbeddingSettings
from .force import (EV_TO_KJMOL, STATE_PARAMETER, FieldMACEComputation, FieldMACEStateReporter,
                    create_fieldmace_force, create_fieldmace_mm_system, load_fieldmace_model)
from .system import LinkAtom, QMMMSpec, prepare_qmmm_system
from .torchforce import FieldMACETorchModule, build_torch_module, create_fieldmace_torchforce, export_torchscript

__all__ = [
    "EmbeddingFrame", "EmbeddingSettings", "EV_TO_KJMOL", "STATE_PARAMETER", "FieldMACEComputation",
    "FieldMACEStateReporter", "create_fieldmace_force", "create_fieldmace_mm_system", "load_fieldmace_model",
    "LinkAtom", "QMMMSpec", "prepare_qmmm_system", "FieldMACETorchModule", "build_torch_module",
    "create_fieldmace_torchforce", "export_torchscript",
]
