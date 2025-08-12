# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'RFI Pipeline'
copyright = '2025, Breakthrough Listen'
author = 'Breakthrough Listen'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Autodoc settings
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Don't include type annotations in signature for constants
autodoc_preserve_defaults = True

# Custom processing for module constants
def skip_member(app, what, name, obj, skip, options):
    """Skip certain members that cause documentation issues."""
    # Skip string constants that show built-in str() documentation
    if what == "data" and isinstance(obj, str) and name.startswith('__'):
        return True
    
    # Skip private attributes and methods (those starting with single underscore)
    # but keep special methods (those starting and ending with double underscores)
    if name.startswith('_') and not (name.startswith('__') and name.endswith('__')):
        return True
    
    # Skip most built-in methods but keep useful ones
    useful_special_methods = {'__init__', '__call__'}
    if name.startswith('__') and name.endswith('__') and name not in useful_special_methods:
        return True
    
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_member)

# -- Options for napoleon ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx ------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
}

# -- Other options -----------------------------------------------------------
autosummary_generate = True
add_module_names = False
