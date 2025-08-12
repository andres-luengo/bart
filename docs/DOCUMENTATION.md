# RFI Pipeline Documentation

This directory contains comprehensive Sphinx documentation for the RFI Pipeline package.

## What's Included

### 📖 Documentation Pages
- **Installation Guide** - Complete setup instructions
- **Usage Guide** - Command-line tools and Python API
- **API Reference** - Detailed class and function documentation
- **Examples** - Practical usage examples and tutorials
- **Changelog** - Version history and feature updates

### 🔧 Documentation Tools
- **Sphinx Configuration** - Professional documentation framework
- **Read the Docs Theme** - Clean, responsive design
- **Auto-generated API docs** - Synchronized with code
- **GitHub Actions** - Automated documentation deployment
- **Cross-references** - Linked documentation ecosystem

## Quick Start

### Building Locally
```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# View at docs/_build/html/index.html
```

### Development Server
```bash
# Auto-rebuilding development server
sphinx-autobuild . _build/html
# View at http://localhost:8000
```

### Other Formats
```bash
make latexpdf  # PDF via LaTeX
make epub      # EPUB e-book
make man       # Manual pages
```

## Features

### 🎯 Complete Coverage
- All public classes and functions documented
- Command-line interface documentation
- Practical examples and tutorials
- Performance guidelines
- Troubleshooting guides

### 🔗 Professional Standards
- Google/NumPy docstring format
- Cross-referenced API documentation
- Type hints and parameter descriptions
- Return value documentation
- Exception handling notes

### 🚀 Modern Tooling
- Sphinx with RTD theme
- Automated builds via GitHub Actions
- Live development server
- Multiple output formats
- Mobile-responsive design

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation instructions
├── usage.rst           # Usage guide and CLI reference
├── api.rst             # API reference (auto-generated)
├── examples.rst        # Detailed examples and tutorials
├── changelog.rst       # Version history
├── requirements.txt    # Documentation dependencies
├── Makefile           # Build commands (Unix)
├── make.bat           # Build commands (Windows)
├── _static/           # Static files (CSS, images)
├── _templates/        # Custom templates
└── _build/            # Generated output (gitignored)
```

## Maintenance

### Adding New Content
1. Update relevant `.rst` files
2. Add docstrings to new code following Google/NumPy style
3. Include practical examples
4. Update changelog for new features
5. Test documentation builds locally

### Keeping Documentation Current
- Docstrings automatically sync with API changes
- Manual pages require updates for new features
- Examples should be tested with new releases
- Links should be verified periodically

### Publishing
Documentation is automatically built and deployed via GitHub Actions when changes are pushed to the main branch.

## Quality Features

✅ **Auto-generated API documentation** from docstrings  
✅ **Cross-referenced links** between sections  
✅ **Search functionality** built into the documentation  
✅ **Mobile-responsive design** with RTD theme  
✅ **Multiple output formats** (HTML, PDF, EPUB)  
✅ **Automated testing** of documentation builds  
✅ **Version control integration** with GitHub  
✅ **Professional styling** with consistent formatting  

This documentation system provides a comprehensive, maintainable, and professional resource for users and developers of the RFI Pipeline package.
