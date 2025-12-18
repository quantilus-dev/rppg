
import Foundation
import CoreVideo
import CoreImage
import Accelerate

public class FramePreprocessor {
    
    private static let context = CIContext()
    
    public static func preprocess(_ pixelBuffer: CVPixelBuffer, targetSize: CGSize) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // 1. Center Crop
        // Assume portrait or landscape, verify aspect ratio.
        let extent = ciImage.extent
        let side = min(extent.width, extent.height)
        let xOffset = (extent.width - side) / 2
        let yOffset = (extent.height - side) / 2
        let cropRect = CGRect(x: xOffset, y: yOffset, width: side, height: side)
        let cropped = ciImage.cropped(to: cropRect)
        
        // 2. Resize
        // Using CoreImage scale
        let scaleX = targetSize.width / side
        let scaleY = targetSize.height / side
        let scaled = cropped.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // 3. Render back to CVPixelBuffer
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       Int(targetSize.width),
                                       Int(targetSize.height),
                                       kCVPixelFormatType_32BGRA, // CoreML usually likes 32BGRA or 32ARGB
                                       attrs,
                                       &outputBuffer)
        
        guard status == kCVReturnSuccess, let buffer = outputBuffer else {
            return nil
        }
        
        context.render(scaled, to: buffer)
        return buffer
    }
}

public class MLMultiArrayUtils {
    /// Converts a list of CVPixelBuffers into a (T, 3, H, W) MLMultiArray
    public static func createMultiArray(from frames: [CVPixelBuffer], size: Int) throws -> MLMultiArray {
        let count = frames.count
        let shape = [NSNumber(value: count), NSNumber(value: 3), NSNumber(value: size), NSNumber(value: size)]
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        for (t, buffer) in frames.enumerated() {
            // Lock base address
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            
            guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { continue }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let width = CVPixelBufferGetWidth(buffer)
            let height = CVPixelBufferGetHeight(buffer)
            
            // Iterate pixels and normalize
            // Assume BGRA
            let pointer = baseAddress.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
            
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * bytesPerRow + x * 4
                    let b = Float(pointer[offset]) / 255.0
                    let g = Float(pointer[offset + 1]) / 255.0
                    let r = Float(pointer[offset + 2]) / 255.0
                    
                    // Assign to multiarray: (t, 0, y, x) = r, (1) = g, (2) = b
                    let idxBase = [NSNumber(value: t), 0, NSNumber(value: y), NSNumber(value: x)] as [NSNumber]
                    
                    // R
                    multiArray[idxBase] = NSNumber(value: r)
                    
                    // G
                    var idxG = idxBase
                    idxG[1] = 1
                    multiArray[idxG] = NSNumber(value: g)
                    
                    // B
                    var idxB = idxBase
                    idxB[1] = 2
                    multiArray[idxB] = NSNumber(value: b)
                }
            }
        }
        
        return multiArray
    }
}
