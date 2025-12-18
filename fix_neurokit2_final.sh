#!/bin/bash
# Final fix for neurokit2 compatibility - try older version or patch approach
# Run this in WSL2: bash fix_neurokit2_final.sh

echo "Attempting final fix for neurokit2 compatibility..."
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rppg

# Try an even older version that definitely works with Python 3.9
echo "Trying neurokit2 0.2.10 (older version)..."
pip uninstall neurokit2 -y
pip install 'neurokit2==0.2.10'

echo ""
echo "Testing import..."
python -c "import neurokit2 as nk; print('✓ neurokit2 0.2.10 imported successfully')" 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "Version 0.2.10 still has issues. Trying 0.2.9..."
    pip uninstall neurokit2 -y
    pip install 'neurokit2==0.2.9'
    
    python -c "import neurokit2 as nk; print('✓ neurokit2 0.2.9 imported successfully')" 2>&1
fi

echo ""
echo "Testing rPPG model import..."
python -c "from rppg.models import get_model; print('✓ rPPG models imported successfully')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS! All imports working!"
else
    echo ""
    echo "Still having issues. Will try patch approach..."
    echo "The issue is in neurokit2's internal files using Python 3.10+ syntax."
    echo "We'll need to patch the problematic file or make the import lazy."
fi

