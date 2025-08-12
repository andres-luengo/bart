#!/bin/bash
# Build documentation script

set -e

echo "Building RFI Pipeline Documentation"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: This script must be run from the project root directory"
    exit 1
fi

# Check if virtual environment should be created
if [ ! -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
fi

# Activate virtual environment if it exists and we're not already in one
if [ -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[docs]"

# Build documentation
echo "Building documentation..."
cd docs
make clean
make html

echo ""
echo "Documentation built successfully!"
echo "Open docs/_build/html/index.html in your browser to view the documentation."
echo ""

# Optionally open in browser (uncomment if desired)
# if command -v xdg-open > /dev/null; then
#     xdg-open _build/html/index.html
# elif command -v open > /dev/null; then
#     open _build/html/index.html
# fi
