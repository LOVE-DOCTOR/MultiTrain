# Installation

MultiTrain supports Python 3.8 through Python 3.13 on Windows, Ubuntu, and macOS.

## Install from PyPI

Create and activate a virtual environment before installing the package:

::::{tab-set}

:::{tab-item} Windows
```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install MultiTrain
```
:::

:::{tab-item} Ubuntu
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install MultiTrain
```
:::

:::{tab-item} macOS
```bash
brew install libomp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install MultiTrain
```
:::

::::

The OpenMP runtime is installed first on macOS because LightGBM and XGBoost depend on it.

## Notebook tools

Install the optional notebook dependencies when you want to run the repository's Jupyter examples:

```bash
python -m pip install "MultiTrain[notebook]"
```

## Documentation tools

The documentation extra belongs to a source checkout because the published
site is built from the repository's `docs` directory. From the repository root,
install the compatible Sphinx toolchain with:

```bash
python -m pip install -e ".[docs]" -c requirements.txt
```

This command installs Sphinx, MyST-NB, the PyData Sphinx theme,
Sphinx-Design, Sphinx-Gallery, and the copy button extension. The constraints
file selects versions compatible with the active Python runtime.

## Install a development checkout

```bash
git clone https://github.com/LOVE-DOCTOR/MultiTrain.git
cd MultiTrain
python -m pip install -e ".[dev,notebook]" -c requirements.txt
```

The constraints file selects the tested dependency versions for the Python runtime in use. The package metadata still allows compatible releases within its supported ranges.

## Confirm the installation

```python
import MultiTrain

print(MultiTrain.__version__)
```

If importing fails on macOS, start with the {ref}`macOS troubleshooting section
<macos-openmp-errors>`.
