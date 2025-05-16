# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
# So you can import your package:
sys.path.insert(0, os.path.abspath('../..'))

project = 'green-dcc'
copyright = '2025, Hewlett Packard Enterprise (HPE)'
author = 'Hewlett Packard Enterprise (HPE)'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
master_doc = 'index' 
html_static_path = ['_static']

# -- Global substitutions ------------------------------------------------
rst_epilog = """
.. |F| replace:: Green-DCC
"""
