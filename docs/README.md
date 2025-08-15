RFI Pipeline Documentation
=========================

This directory contains the Sphinx documentation for the RFI Pipeline package.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

Or install the package with documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Building HTML Documentation

To build the HTML documentation:

```bash
make html
```

The generated documentation will be available in `_build/html/index.html`.

### Live Development Server

For live development with automatic rebuilding:

```bash
sphinx-autobuild . _build/html
```

This will start a development server at `http://localhost:8000` that automatically rebuilds the documentation when files change.

### Other Formats

Build other formats:

```bash
make latexpdf  # PDF via LaTeX
make epub      # EPUB format
make man       # Manual pages
```

## Documentation Structure

- `index.rst` - Main documentation index
- `installation.rst` - Installation instructions
- `usage.rst` - Usage guide and examples
- `api.rst` - API reference documentation
- `examples.rst` - Detailed examples and tutorials
- `changelog.rst` - Version history and changes
- `conf.py` - Sphinx configuration
- `_static/` - Static files (CSS, images, etc.)
- `_templates/` - Custom templates (if needed)

## Contributing to Documentation

When adding new features or making changes:

1. Update the relevant `.rst` files
2. Add docstrings to new functions/classes following Google/NumPy style
3. Include examples in the appropriate sections
4. Update the changelog
5. Build and review the documentation locally
6. Test that all cross-references work correctly

## Documentation Style Guide

- Use reStructuredText format
- Follow Google or NumPy docstring conventions
- Include practical examples for all major features
- Keep language clear and concise
- Use proper cross-references for internal links
- Include type hints in function signatures
