# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

import os, sys, warnings
import pytest
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    """Small randomly initialised FieldMACE models (float64) of both variants."""
    import torch
    torch.set_default_dtype(torch.float64)
    from mace.tools.testing import build_field_model as build
    d = tmp_path_factory.mktemp("models")
    out = {}
    for cls in ["PerAtomFieldEMACE", "FieldEMACE"]:
        p = d / f"{cls}.model"
        torch.save(build(cls), p)
        out[cls] = str(p)
    return out


@pytest.fixture(scope="session")
def aladip():
    import openmmtools.testsystems as T
    return T.AlanineDipeptideExplicit()
