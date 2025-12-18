import torch
import sys
import os
sys.path.append(os.getcwd())

from rppg.models import get_model
from rppg.nets.PPNet import PPNet

class DummyConfig:
    def __init__(self):
        self.model = "PPNet"
        self.time_length = 32  # Frames
        self.img_size = 72     # Height/Width

def test_ppnet():
    print("Testing PPNet verification...")
    cfg = DummyConfig()
    
    # 1. Test Instantiation via get_model
    try:
        model = get_model(cfg).cpu() # Force CPU
        print("✓ Model instantiated successfully via get_model")
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        return

    # 2. Test Forward Pass
    batch_size = 2
    # Input shape: (Batch, Channel, Time, Height, Width)
    dummy_input = torch.randn(batch_size, 3, cfg.time_length, cfg.img_size, cfg.img_size)
    
    try:
        output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        # Expect (Batch, Time, 1) -> (2, 32, 1) or (Batch, 2) depending on implementation
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

if __name__ == "__main__":
    test_ppnet()
