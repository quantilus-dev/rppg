#!/bin/bash
# Quick installation status checker
# Usage: bash check_install.sh

echo "=== rPPG Installation Status Check ==="
echo ""

# Check conda
echo "1. Conda installation:"
if command -v conda &> /dev/null; then
    echo "   ✓ Conda is installed: $(conda --version)"
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "Not found")
    echo "   Location: $CONDA_BASE"
else
    echo "   ✗ Conda is NOT installed"
fi
echo ""

# Check rppg environment
echo "2. rppg conda environment:"
if command -v conda &> /dev/null; then
    if conda env list | grep -q "^rppg "; then
        echo "   ✓ rppg environment exists"
        ENV_SIZE=$(du -sh $(conda info --envs | grep rppg | awk '{print $NF}') 2>/dev/null | awk '{print $1}' || echo "unknown")
        echo "   Size: $ENV_SIZE"
        
        # Try to activate and test
        echo ""
        echo "3. Testing environment:"
        eval "$(conda shell.bash hook 2>/dev/null)"
        if conda activate rppg 2>/dev/null; then
            echo "   ✓ Environment can be activated"
            
            # Test key imports
            python -c "import torch" 2>/dev/null && echo "   ✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')" || echo "   ✗ PyTorch import failed"
            python -c "import numpy" 2>/dev/null && echo "   ✓ NumPy: $(python -c 'import numpy; print(numpy.__version__)')" || echo "   ✗ NumPy import failed"
            python -c "import cv2" 2>/dev/null && echo "   ✓ OpenCV: $(python -c 'import cv2; print(cv2.__version__)')" || echo "   ✗ OpenCV import failed"
            
            # Check CUDA
            python -c "import torch; print('   CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "   (Cannot check CUDA)"
        else
            echo "   ✗ Cannot activate environment"
        fi
    else
        echo "   ✗ rppg environment does NOT exist yet"
        echo "   Run: bash setup_rppg_wsl2.sh to create it"
    fi
else
    echo "   (Cannot check - conda not installed)"
fi
echo ""

# Check disk space
echo "4. Disk space:"
df -h ~ | tail -1 | awk '{print "   Available: " $4 " of " $2 " (" $5 " used)"}'
echo ""

# Check if setup script is running
echo "5. Setup script status:"
if pgrep -f setup_rppg_wsl2.sh > /dev/null; then
    echo "   ⏳ Setup script is currently running"
    echo "   PID: $(pgrep -f setup_rppg_wsl2.sh)"
else
    echo "   ✓ Setup script is not running (or completed)"
fi
echo ""

# Check conda processes
echo "6. Active conda processes:"
CONDA_PROCS=$(pgrep -f conda | wc -l)
if [ "$CONDA_PROCS" -gt 0 ]; then
    echo "   ⏳ $CONDA_PROCS conda process(es) running (installation may be in progress)"
else
    echo "   ✓ No conda processes running"
fi
echo ""

echo "=== End of Status Check ==="

