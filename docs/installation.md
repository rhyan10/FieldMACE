# Installation

*Author: Rhyan Barrett*

## pip (PythonForce backend)

```bash
git clone https://github.com/rhyan10/FieldMACE.git
cd FieldMACE
python -m venv venv && source venv/bin/activate        # Python >= 3.10
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
pip install -e ".[mlmm,test]"                           # OpenMM >= 8.3, pytest, openmmtools
python scripts/check_env.py                             # --gpu inside a GPU job
pytest -q tests/
```

From PyPI, once released: `pip install "fieldmace[mlmm]"`.

## conda-forge (adds the TorchForce backend)

openmm-torch is only on conda-forge and needs PyTorch from conda-forge too:

```bash
mamba env create -f ci/environment-openmm-torch.yml     # CPU; edit pytorch=*=cuda* for GPUs
mamba activate fieldmace-omt
pip install -e . --no-deps
```

Then `create_fieldmace_mm_system(..., backend="torch")` runs the model inside OpenMM as a TorchScript
`TorchForce`. `mace.mlmm.export_torchscript` writes the ML/MM potential as a file that other programs
can run.

## SHARC 4

```bash
bash sharc/install_sharc4.sh --env venv --prefix ~/software     # or --env conda
source ~/software/sharc4/sharc_env.sh
```

On ZIH (TU Dresden), use `sharc/install_sharc4_zih.sh`, which loads the right modules.

## Checking everything

```bash
pytest -q tests/                                  # SHARC tests run when $SHARC is set
python tutorials/check_tutorials.py               # every tutorial step with tiny settings
python tutorials/check_tutorials.py --notebooks   # and the notebooks
```

## Documentation site

```bash
pip install mkdocs-material
mkdocs serve          # http://127.0.0.1:8000
```
