# Full example notebooks

The repository includes two longer tutorials with saved outputs from real datasets:

- [Classification on the Titanic dataset](https://github.com/LOVE-DOCTOR/MultiTrain/blob/main/classification.ipynb)
- [Regression on the housing dataset](https://github.com/LOVE-DOCTOR/MultiTrain/blob/main/regression.ipynb)

Both notebooks are executed from beginning to end by the documentation workflow before the website is built. They demonstrate the complete built-in catalog, retained artifacts, custom estimator names, pipeline parameter overrides, and diagnostic tables.

## Run them locally

```bash
python -m pip install -e ".[notebook]" -c requirements.txt
jupyter notebook
```

Open `classification.ipynb` or `regression.ipynb` from the repository root. Dataset paths in the notebooks are relative to that directory.
