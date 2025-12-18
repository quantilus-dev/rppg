# rPPG iOS SDK

This SDK provides on-device remote photoplethysmography (rPPG) capabilities for iOS using CoreML.

## Directory Structure
- **Models/**: Place your converted `.mlpackage` files here.
- **Source/**: Swift source files to include in your Xcode project.
  - `RppgManager.swift`: Main interface for running inference.
  - `FramePreprocessor.swift`: Utilities for converting buffers.
- **Scripts/**: Python scripts to convert PyTorch models to CoreML.

## Setup

1. **Convert Models**:
   Run the conversion script to generate CoreML models from your existing PyTorch checkpoints.
   ```bash
   python rppg_ios_sdk/Scripts/convert_to_coreml.py --time_length 32 --img_size 72
   ```
   This will create `EfficientPhys.mlpackage` and `PPNet.mlpackage` in the `Models/` directory.

2. **Integrate into iOS App**:
   - Drag the `rppg_ios_sdk` folder (or just the `Source` contents and your generated `.mlpackage` models) into your Xcode project.
   - Ensure the models are added to the "Compile Sources" or "Copy Bundle Resources" as appropriate (CoreML models are compiled).
   - Add `CoreML`, `Vision`, `Accelerate` frameworks if not already present.

## Usage

```swift
// 1. Initialize
let rppg = try RppgManager(modelName: "EfficientPhys", timeLength: 32)

// 2. Process frames (CVPixelBuffer) from camera delegate
func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    
    rppg.processFrame(pixelBuffer) { result in
        switch result {
        case .success(let value):
            print("Estimated rPPG Signal: \(value)")
        case .failure(let error):
            print("Error: \(error)")
        }
    }
}
```

## Requirements
- iOS 15.0+
- Swift 5.0+
