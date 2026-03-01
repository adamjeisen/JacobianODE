#!/bin/bash
set -e

echo "Creating Python environment with uv..."

# Create virtual environment with Python 3.12
uv venv --python 3.12

echo "Installing the project in editable mode..."
# Install the project in editable mode
uv pip install -e .

echo "Installing ipykernel and creating a Jupyter kernel bound to this venv..."
# Add ipykernel as a dev dependency (persists to pyproject/uv.lock) and install the kernel
uv add --dev ipykernel
# uv run ipython kernel install --user --env VIRTUAL_ENV "$(pwd)/.venv" --name=JacobianODE
uv run python -m ipykernel install \
  --user \
  --name jacobianode \
  --display-name "JacobianODE"

echo "Environment created successfully!"
echo "To start Jupyter in Cursor/VSCode, click the kernel selecter in the top right corner."
echo "Then select the 'Select Another Kernel...' --> 'Jupyter Kernel...' --> 'JacobianODE'"
echo "In the 'Jupyter Kernel...' selector, you may need to click the refresh button in the top right corner to see the new kernel."
echo "To activate the environment, from inside this directory, run: source .venv/bin/activate"

# Changelog:
# - Changed Python version from 3.11 to 3.12