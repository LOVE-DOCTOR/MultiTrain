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

Documentation uses Python 3.12 independently of the package's broader runtime matrix:

```bash
python3.12 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -e ".[docs]" -c requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

On Windows, activate with `.venv-docs\Scripts\activate`.

The `-W` option promotes Sphinx warnings to build failures. Fix broken references, malformed directives, missing API imports, and gallery errors rather than suppressing them without a specific reason.

## Documentation writing rules

- Put user goals and expected outcomes before implementation detail.
- Keep complete examples runnable from a clean checkout.
- Use public imports in examples.
- Explain parameter interactions and failure behavior, not only valid types.
- Add API details to source docstrings so generated reference pages stay synchronized.
- Add longer workflows to the user guide or example gallery.
- Preserve the README's introductory tutorial and link deeper explanations to this site.

## Documentation CI

The documentation workflow runs on pull requests and pushes to `v2-test` and `main`. It executes both repository notebooks and builds the HTML with warnings treated as errors. Only a successful build from `main` is deployed to GitHub Pages.

## Publish with GitHub Pages

Before the first deployment, open **Settings → Pages** in the GitHub repository
and select **GitHub Actions** as the publishing source. No branch folder or
ruleset is required because the workflow uploads the built HTML directly.

Push documentation work to `v2-test` first. GitHub Actions will execute the
notebooks, run the gallery examples, build the complete site, and retain the
Pages artifact without publishing it. After that check succeeds and the branch
is merged into `main`, the same workflow deploys the verified artifact to
`https://love-doctor.github.io/MultiTrain/`.

## Release checklist

Before publishing a release:

1. Update `MultiTrain.__version__` and the changelog.
2. Run tests and documentation builds.
3. Build and inspect the wheel and source distribution.
4. Confirm the release tag matches the package version.
5. Publish the GitHub release to trigger trusted PyPI publishing.
6. Confirm the stable documentation describes the released API.

See the repository's [contribution guidance](https://github.com/LOVE-DOCTOR/MultiTrain#contributing) before opening a large pull request.
