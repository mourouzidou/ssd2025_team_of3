# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
sys.path.insert(0, os.path.abspath('../../src/'))
project = 'ssd_course_team3'
copyright = '2025, Eleni Mourouzidou, Matheus Bernini-Peron, Carolin Lindow'
author = 'Eleni Mourouzidou, Matheus Bernini-Peron, Carolin Lindow'
release = '[0.0.1]'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'myst_parser', 
              'sphinx.ext.napoleon',]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
