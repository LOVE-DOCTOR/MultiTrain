"""Sphinx configuration for the MultiTrain documentation website."""

from pathlib import Path
import os
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "docs" / "_build" / "matplotlib"),
)
sys.path.insert(0, str(REPOSITORY_ROOT))

from MultiTrain import __version__  # noqa: E402


project = "MultiTrain"
author = "Shittu Samson"
copyright = "2026, Shittu Samson"
version = __version__
release = __version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]
# Notebook files in the repository are executed by the docs workflow. If a
# notebook is added under docs later, MyST-NB renders its saved, verified output.
nb_execution_mode = "off"

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    # Match both POSIX and Windows paths so every example runs in local and CI builds.
    "filename_pattern": r".*",
    "download_all_examples": False,
    "notebook_extensions": set(),
    "remove_config_comments": True,
    "run_stale_examples": True,
    "abort_on_example_error": True,
    "show_memory": False,
}

html_theme = "pydata_sphinx_theme"
html_title = f"MultiTrain {release} documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_theme_options = {
    "github_url": "https://github.com/LOVE-DOCTOR/MultiTrain",
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {
    "github_user": "LOVE-DOCTOR",
    "github_repo": "MultiTrain",
    "github_version": "main",
    "doc_path": "docs",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
