#!/bin/bash
# Fix NumPy and neurokit2 compatibility issues
# Run this in WSL2: bash fix_dependencies.sh

echo "Fixing dependency compatibility issues..."
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rppg

# Check current versions
echo "Current versions:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: not found"
pip show neurokit2 | grep Version || echo "neurokit2: not found"
echo ""

# Fix NumPy - downgrade to 1.x (compatible with packages in rppg.yaml)
echo "Downgrading NumPy to 1.x (required for OpenCV and other packages)..."
pip install 'numpy<2.0' --force-reinstall

# Fix neurokit2 - ensure it's downgraded properly
echo ""
echo "Ensuring neurokit2 is Python 3.9 compatible..."
pip uninstall neurokit2 -y
pip install 'neurokit2==0.2.12'

# Reinstall packages that might have been affected
echo ""
echo "Reinstalling affected packages..."
pip install --force-reinstall --no-deps opencv-python

echo ""
echo "Testing imports..."
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" && \
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')" && \
python -c "import neurokit2; print('✓ neurokit2')" && \
python -c "from rppg.models import get_model; print('✓ rPPG models')" && \
echo "" && \
echo "✓ All dependencies fixed successfully!" || \
echo "✗ Some issues remain - please check the errors above"

