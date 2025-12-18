
import sys
import os
import argparse
import torch
import numpy as np
import coremltools as ct
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rppg.nets.EfficientPhys import EfficientPhys
from rppg.nets.PPNet import PPNet

def verify_efficientphys(model_path, time_length=32, img_size=72):
    print(f"--- Verifying EfficientPhys ---")
    
    # 1. Load PyTorch Model
    pt_model = EfficientPhys(frame_depth=time_length, img_size=img_size)
    pt_model.eval()
    
    # 2. Load CoreML Model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    ml_model = ct.models.MLModel(model_path)
    
    # 3. Create Dummy Input
    # (Time, 3, H, W)
    dummy_input = torch.randn(time_length, 3, img_size, img_size)
    
    # 4. Run PyTorch Inference
    # PyTorch model expects (Batch, C, Time, H, W) for EfficientPhys TSM usually?
    # Wait, EfficientPhys forward expects (N, C, T, H, W).
    # Forward: nt, c, h, w = x.size() -> so it expects 4D? 
    # Let's check EfficientPhys.py again.
    # TSM forward: nt, c, h, w = x.size(). It takes flattened (Batch*Time) or just (Time) if Batch=1.
    # The forward pass: inputs = self.batch_norm(inputs) -> ...
    # So it takes (N*T, C, H, W).
    
    with torch.no_grad():
        pt_out = pt_model(dummy_input)
    
    # 5. Run CoreML Inference
    # CoreML input is dictionary
    # Assuming the conversion script named inputs "input_frames"
    coreml_input = {"input_frames": dummy_input.numpy()}
    prediction = ml_model.predict(coreml_input)
    
    # 6. Compare
    # Output name "rppg_output"
    cm_out = prediction["rppg_output"]
    
    # Calculate Error
    pt_val = pt_out.numpy().flatten()
    cm_val = cm_out.flatten() if isinstance(cm_out, np.ndarray) else np.array([cm_out]).flatten()
    
    mae = np.mean(np.abs(pt_val - cm_val))
    print(f"PyTorch Output: {pt_val[:5]}")
    print(f"CoreML Output:  {cm_val[:5]}")
    print(f"MAE: {mae}")
    
    if mae < 1e-3:
        print("✅ Conversion Successful!")
    else:
        print("⚠️ High Error - Check input shapes or preprocessing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../Models")
    args = parser.parse_args()
    
    model_dir = os.path.join(os.path.dirname(__file__), args.model_dir)
    
    eff_path = os.path.join(model_dir, "EfficientPhys.mlpackage")
    try:
        verify_efficientphys(eff_path)
    except Exception as e:
        print(f"Verification failed: {e}")
