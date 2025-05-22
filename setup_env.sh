#!/bin/bash

# Exit on error
set -e

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create conda environment from yml file
echo "Creating conda environment from jacobianode.yml..."
conda env create -f jacobianode.yml

# Activate the environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jacobianode

# Install the package in editable mode
echo "Installing JacobianODE in editable mode..."
pip install -e .

echo "Setup complete! The environment 'jacobianode' has been created and JacobianODE has been installed in editable mode."
echo "To activate the environment, run: conda activate jacobianode" 