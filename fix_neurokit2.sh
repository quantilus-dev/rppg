#!/bin/bash
# Fix neurokit2 Python 3.9 compatibility issue
# Run this in WSL2: bash fix_neurokit2.sh

echo "Fixing neurokit2 compatibility issue..."
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rppg

# Check current version
echo "Current neurokit2 version:"
pip show neurokit2 | grep Version

# Downgrade to Python 3.9 compatible version
echo ""
echo "Downgrading neurokit2 to Python 3.9 compatible version..."
pip install 'neurokit2<0.3.0' --force-reinstall

echo ""
echo "Testing import..."
python -c "import neurokit2 as nk; print('✓ neurokit2 imported successfully')" && echo "✓ Fix successful!" || echo "✗ Still having issues"

