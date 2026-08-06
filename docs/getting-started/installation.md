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

If importing fails on macOS, start with the [macOS troubleshooting section](../troubleshooting/index.md#macos-openmp-errors).
