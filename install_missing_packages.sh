#!/bin/bash
# Install missing packages
# Run this in WSL2: bash install_missing_packages.sh

echo "Installing missing packages..."
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rppg

# Install wandb (required for main.py but can be disabled in config)
echo "Installing wandb..."
pip install wandb

# Install any other missing packages from the original setup
echo ""
echo "Installing other required packages..."
pip install self-attention-cv

echo ""
echo "✓ Installation complete!"
echo ""
echo "Test the installation:"
echo "  python -c \"import wandb; print('wandb:', wandb.__version__)\""
