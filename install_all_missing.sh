#!/bin/bash
# Install all missing packages
# Run this in WSL2: bash install_all_missing.sh

echo "Installing all required packages..."
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rppg

# Install packages that should have been installed
echo "Installing h5py..."
pip install h5py

echo "Installing wandb..."
pip install wandb

echo "Installing self-attention-cv..."
pip install self-attention-cv

echo ""
echo "✓ All packages installed!"
echo ""
echo "Verifying installation:"
python -c "import h5py; print('✓ h5py:', h5py.__version__)" 2>/dev/null || echo "✗ h5py failed"
python -c "import wandb; print('✓ wandb:', wandb.__version__)" 2>/dev/null || echo "✗ wandb failed"
python -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>/dev/null || echo "✗ PyTorch failed"

