
import sys
import os
import argparse
import torch
import torch.nn as nn
import coremltools as ct
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rppg.nets.EfficientPhys import EfficientPhys
from rppg.nets.PPNet import PPNet

def convert_efficientphys(output_path, time_length=32, img_size=72):
    print(f"Converting EfficientPhys (Time={time_length}, Size={img_size})...")
    
    # Initialize model
    model = EfficientPhys(frame_depth=time_length, img_size=img_size)
    model.eval()
    
    # Trace expects (Batch*Time, C, H, W)
    # For inference, we usually process one clip at a time, so Batch=1
    # Input shape: (Time, 3, H, W)
    dummy_input = torch.randn(time_length, 3, img_size, img_size)
    
    # Trace
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert
    # We treat the input as an Image for easy CoreML integration if possible, 
    # but since it's a sequence of frames, MultiArray is safer to start.
    # The SDK will handle the buffer management.
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_frames", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="rppg_output")],
        minimum_deployment_target=ct.target.iOS15
    )
    
    save_path = os.path.join(output_path, "EfficientPhys.mlpackage")
    mlmodel.save(save_path)
    print(f"Saved to {save_path}")

def convert_ppnet(output_path, time_length=32, img_size=72):
    print(f"Converting PPNet (Time={time_length}, Size={img_size})...")
    
    # Initialize model
    model = PPNet(frame_depth=time_length)
    model.eval()
    
    # PPNet expects (Batch, Channel, Time, Height, Width)
    # Batch=1
    dummy_input = torch.randn(1, 3, time_length, img_size, img_size)
    
    # Trace
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_video_tensor", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="bp_output")],
        minimum_deployment_target=ct.target.iOS15
    )
    
    save_path = os.path.join(output_path, "PPNet.mlpackage")
    mlmodel.save(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../Models", help="Output directory for CoreML models")
    parser.add_argument("--img_size", type=int, default=72)
    parser.add_argument("--time_length", type=int, default=32)
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        convert_efficientphys(output_dir, args.time_length, args.img_size)
    except Exception as e:
        print(f"Failed to convert EfficientPhys: {e}")
        
    try:
        convert_ppnet(output_dir, args.time_length, args.img_size)
    except Exception as e:
        print(f"Failed to convert PPNet: {e}")
