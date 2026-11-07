import os
import sys
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def float64():
    """Tests run in float64 unless they say otherwise."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)



