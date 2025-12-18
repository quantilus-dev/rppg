#!/usr/bin/env python
"""
Minimal test script to verify rPPG setup is working
This tests imports and model creation without requiring datasets
"""

import torch
import numpy as np

print("=" * 60)
print("Testing rPPG Setup")
print("=" * 60)
print()

# Test 1: Basic imports
print("1. Testing basic imports...")
try:
    from rppg.models import get_model
    from rppg.config import get_config
    from rppg.loss import loss_fn
    from rppg.optim import optimizer
    print("   ✓ All core modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Model creation (without dataset)
print("\n2. Testing model creation...")
try:
    # Create a dummy config for model creation
    class DummyConfig:
        def __init__(self):
            self.model = "EfficientPhys"
            self.img_size = 72
            self.time_length = 180
    
    dummy_cfg = DummyConfig()
    model = get_model(dummy_cfg)
    print(f"   ✓ Model '{dummy_cfg.model}' created successfully")
    print(f"   ✓ Model type: {type(model).__name__}")
    
    # Test forward pass with dummy data
    batch_size = 2
    # Validating dimension requirements: EfficientPhys expects 4D input (N*T, C, H, W)
    # The TSM module inside will handle temporal segmentation based on frame_depth
    dummy_input = torch.randn(batch_size * dummy_cfg.time_length, 3, dummy_cfg.img_size, dummy_cfg.img_size)
    # Force CPU to avoid CUDA compatibility crashes (RTX 5070 requires newer PyTorch)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Input shape: {dummy_input.shape}")
    print(f"   ✓ Output shape: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")
    
except Exception as e:
    print(f"   ✗ Model creation/forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Check CUDA availability
print("\n3. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available")
    print(f"   ✓ Device: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
else:
    print("   ⚠ CUDA not available (CPU mode)")

print("\n" + "=" * 60)
print("✓ All tests passed! Setup is working correctly.")
print("=" * 60)
print("\nNext steps:")
print("  - Download a dataset (e.g., UBFC-rPPG) to data/raw/")
print("  - Run preprocessing if needed")
print("  - Use examples/ scripts with proper config files")

