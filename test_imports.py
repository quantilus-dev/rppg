#!/usr/bin/env python
# Quick test script to verify all imports work correctly

print("Testing rPPG project imports...")
print()

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV import failed: {e}")

# neurokit2 is now a lazy import (only imported when get_hrv() is called)
# Testing lazy import functionality
try:
    from rppg.utils.funcs import _get_nk
    nk = _get_nk()  # This will trigger the actual import
    print(f"✓ neurokit2 (lazy import) works")
except Exception as e:
    print(f"⚠ neurokit2 lazy import issue: {e} (may not affect basic usage)")

try:
    from rppg.models import get_model
    print(f"✓ rPPG models imported successfully")
except Exception as e:
    print(f"✗ rPPG models import failed: {e}")

print()
print("Import test complete!")

