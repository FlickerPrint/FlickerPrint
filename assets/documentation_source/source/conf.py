# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys

# sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "FlickerPrint"
copyright = "FlickerPrint 2020-2025"
author = "FlickerPrint"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon",
              'sphinx_copybutton',
              'sphinx_tabs.tabs',
              'sphinx_design',
              'pydata_sphinx_theme']

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "haiku"
# html_theme = "nature"
# html_theme = "bizstyle"
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# html_theme_options = {"logo": None, "github_banch": None}

# -- LaTeX configuration
# https://www.sphinx-doc.org/en/master/latex.html

latex_elements = dict(textgreek=r"\usepackage{textalpha,alphabeta}")


# Required for GitHub Pages when repo is not at user/org root
html_baseurl = "https://flickerprint.github.io/FlickerPrint/"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/FlickerPrint/FlickerPrint",
            "icon": "fab fa-github",
        },
    ],
}


# """Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# """

# # -- Path setup --------------------------------------------------------------
# import os
# import sys
# from pathlib import Path
# from typing import Any, Dict

# from sphinx.application import Sphinx
# from sphinx.locale import _

# import pydata_sphinx_theme

# sys.path.append(str(Path(".").resolve()))

# # -- Project information -----------------------------------------------------

# project = "FlickerPrint"
# copyright = "FlickerPrint 2020-2024"
# author = "FlickerPrint"

# # -- General configuration ---------------------------------------------------

# extensions = [
#     "sphinx.ext.napoleon",
#     "sphinx.ext.autodoc",
#     "sphinx.ext.autosummary",
#     "sphinx.ext.todo",
#     "sphinx.ext.viewcode",
#     "sphinx.ext.intersphinx",
#     "sphinx.ext.graphviz",
#     # "sphinxext.rediraffe",
#     # "sphinx_design",
#     # "sphinx_copybutton",
#     # "autoapi.extension",
#     # # custom extentions
#     # "_extension.gallery_directive",
#     # "_extension.component_directive",
#     # # For extension examples and demos
#     # "myst_parser",
#     # "ablog",
#     # "jupyter_sphinx",
#     # "sphinxcontrib.youtube",
#     # "nbsphinx",
#     # "numpydoc",
#     # "sphinx_togglebutton",
#     # "jupyterlite_sphinx",
#     # "sphinx_favicon",
# ]

# jupyterlite_config = "jupyterlite_config.json"

# # Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# # List of patterns, relative to source directory, that match files and
# # directories to ignore when looking for source files.
# # This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# intersphinx_mapping = {"sphinx": ("https://www.sphinx-doc.org/en/master", None)}

# # -- Sitemap -----------------------------------------------------------------

# # ReadTheDocs has its own way of generating sitemaps, etc.
# # if not os.environ.get("READTHEDOCS"):
# #     extensions += ["sphinx_sitemap"]

# #     html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
# #     sitemap_locales = [None]
# #     sitemap_url_scheme = "{link}"

# # -- MyST options ------------------------------------------------------------

# # This allows us to use ::: to denote directives, useful for admonitions
# # myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
# # myst_heading_anchors = 2
# # myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# # -- Internationalization ----------------------------------------------------

# # specifying the natural language populates some key tags
# language = "en"

# # -- sphinx_ext_graphviz options ---------------------------------------------

# graphviz_output_format = "svg"
# inheritance_graph_attrs = dict(
#     rankdir="LR",
#     fontsize=14,
#     ratio="compress",
# )

# # -- sphinx_togglebutton options ---------------------------------------------
# togglebutton_hint = str(_("Click to expand"))
# togglebutton_hint_hide = str(_("Click to collapse"))

# # -- Sphinx-copybutton options ---------------------------------------------
# # Exclude copy button from appearing over notebook cell numbers by using :not()
# # The default copybutton selector is `div.highlight pre`
# # https://github.com/executablebooks/sphinx-copybutton/blob/master/sphinx_copybutton/__init__.py#L82
# copybutton_selector = ":not(.prompt) > div.highlight pre"

# # -- Options for HTML output -------------------------------------------------

# html_theme = "pydata_sphinx_theme"
# # html_logo = "_static/logo.svg"
# # html_favicon = "_static/logo.svg"
# # html_sourcelink_suffix = ""
# html_last_updated_fmt = ""  # to reveal the build date in the pages meta




# # -- Set Up the Navbar -----------------------------------------------------

# html_theme_options = {
#     # "navbar_start": ["navbar-logo"],
#     "navbar_center": ["navbar-nav"],
#     "navbar_end": ["navbar-icon-links"],

#     "header_links": [
#         {"name": "Installation", "url": "installation.html"},
#         {"name": "Getting Started", "url": "getting_started.html"},
#         {"name": "Source Code", "url": "module_source_code.html"},
#         {
#             "name": "Find Out More",
#             "children": [
#                 {"name": "Flicker Spectroscopy Paper", "url": "https://doi.org/10.1126/sciadv.adg0432", "external": True},
#             ]
#         }
#     ],
# }

