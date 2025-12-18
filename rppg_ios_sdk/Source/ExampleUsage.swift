
import Foundation
import CoreVideo

// Example Usage of RppgManager
// Place this in your App Delegate, ViewController, or Playground

func runExample() {
    print("Initializing rPPG Manager...")
    
    do {
        // Ensure "EfficientPhys" model is in your bundle
        let manager = try RppgManager(modelName: "EfficientPhys", timeLength: 32)
        
        print("Creating dummy frames...")
        // Simulate 32 frames of video
        let frames = createDummyFrames(count: 32)
        
        print("Processing frames...")
        for (i, frame) in frames.enumerated() {
            manager.processFrame(frame) { result in
                switch result {
                case .success(let value):
                    print("Frame \(i): Output = \(value)")
                case .failure(let error):
                    // Errors expected until buffer is full (32 frames)
                    // Or if model is missing
                    print("Frame \(i): Status = \(error)")
                }
            }
        }
        
    } catch {
        print("Failed to initialize manager: \(error)")
    }
}

func createDummyFrames(count: Int) -> [CVPixelBuffer] {
    var frames: [CVPixelBuffer] = []
    
    for _ in 0..<count {
        var pixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        
        // standard 640x480 frame
        CVPixelBufferCreate(kCFAllocatorDefault, 640, 480, kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)
        
        if let buffer = pixelBuffer {
            // Fill with random noise or color
            CVPixelBufferLockBaseAddress(buffer, [])
            let data = CVPixelBufferGetBaseAddress(buffer)!
            let height = CVPixelBufferGetHeight(buffer)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            
            // Just fill with memset for speed (gray)
            memset(data, 128, height * bytesPerRow)
            
            CVPixelBufferUnlockBaseAddress(buffer, [])
            frames.append(buffer)
        }
    }
    
    return frames
}

// Run the example
// runExample()
