import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from rppg.utils.funcs import calculate_spo2

def test_spo2_logic():
    print("Testing SpO2 Calculation (Ratio of Ratios)...")
    
    # 1. Create Synthetic Signal
    # Video Shape: (T, H, W, C)
    T = 300
    H, W = 10, 10
    fs = 30
    t = np.linspace(0, T/fs, T)
    
    # Simulate Pulse: 1 Hz (60 BPM)
    pulse = np.sin(2 * np.pi * 1 * t)
    
    # Red Channel: Low AC/DC ratio (Higher SpO2)
    # DC = 200, AC = 1
    red_signal = 200 + 1 * pulse
    
    # Green Channel: High AC/DC ratio
    # DC = 200, AC = 3 (Green absorbs more pulse?)
    # Ratio = (1/200) / (3/200) = 0.33
    # SpO2 = 110 - 25 * 0.33 = 101.75 -> clipped to 100
    green_signal = 200 + 3 * pulse
    
    # Blue Channel: Noise
    blue_signal = 200 + 0 * pulse
    
    # Construct Video Tensor
    video = np.zeros((T, H, W, 3))
    for i in range(T):
        video[i, :, :, 0] = red_signal[i]
        video[i, :, :, 1] = green_signal[i]
        video[i, :, :, 2] = blue_signal[i]
        
    # 2. Run Calculation
    spo2 = calculate_spo2(video)
    print(f"Calculated SpO2: {spo2}%")
    
    # 3. Test with Hypoxic Signal (Low SpO2)
    # Ratio needs to be higher. Say R/G ratio = 1.0 (AC_r/DC_r == AC_g/DC_g)
    # SpO2 = 110 - 25 * 1 = 85%
    red_signal_low = 200 + 10 * pulse
    green_signal_low = 200 + 10 * pulse
    
    video_low = np.zeros((T, H, W, 3))
    for i in range(T):
        video_low[i, :, :, 0] = red_signal_low[i]
        video_low[i, :, :, 1] = green_signal_low[i]
        video_low[i, :, :, 2] = blue_signal[i]
        
    spo2_low = calculate_spo2(video_low)
    print(f"Calculated Hypoxic SpO2: {spo2_low}%")

if __name__ == "__main__":
    test_spo2_logic()
