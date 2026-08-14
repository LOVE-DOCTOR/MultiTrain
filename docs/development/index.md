# Development guide

## Set up the repository

```bash
git clone https://github.com/LOVE-DOCTOR/MultiTrain.git
cd MultiTrain
python3.10 -m venv .venv
```

Activate the environment, then install the tested development set:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]" -c requirements.txt
```

On macOS, install `libomp` before installing the project.

## Run project checks

```bash
python -m ruff check .
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

## Build the documentation

Documentation supports Python 3.10 through Python 3.13. Create a separate
environment so its build tools do not affect the environment used for model
development:

```bash
python3.10 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -e ".[docs]" -c requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

On Windows, activate with `.venv-docs\Scripts\activate`.

The `-W` option promotes Sphinx warnings to build failures. Fix broken references, malformed directives, missing API imports, and gallery errors rather than suppressing them without a specific reason.

## Documentation CI

The documentation workflow runs on pull requests and pushes to `v2-test` and
`main`. It installs and builds the documentation on Python 3.10, 3.11, 3.12,
and 3.13, executes both repository notebooks, and treats HTML build warnings as
errors. The Python 3.12 build supplies the Pages artifact, and only a successful
matrix build from `main` is deployed to GitHub Pages.

## Publish with GitHub Pages

Configure the first deployment as follows:

1. Open **Settings → Pages** in the GitHub repository.
2. Select **GitHub Actions** as the publishing source.
3. Push documentation work to `v2-test` and confirm every documentation matrix
   job succeeds.
4. Download the Python 3.12 Pages artifact if you want to inspect the exact site
   before merging.
5. Merge the verified branch into `main` to deploy the same build to GitHub
   Pages.

No branch folder or ruleset is required because the workflow uploads the built
HTML directly. The published site is available at
`https://love-doctor.github.io/MultiTrain/` after the `main` deployment finishes.

## Release checklist

Before publishing a release:

1. Update `MultiTrain.__version__` and the changelog.
2. Run tests and documentation builds.
3. Build and inspect the wheel and source distribution.
4. Confirm the release tag matches the package version.
5. Push the version change to `main` to trigger trusted PyPI publishing. Publishing a GitHub release also triggers it.
6. Confirm the stable documentation describes the released API.

See the repository's [contribution guidance](https://github.com/LOVE-DOCTOR/MultiTrain#contributing) before opening a large pull request.
