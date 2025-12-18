
import Foundation
import CoreML
import Vision
import UIKit

public enum RppgError: Error {
    case modelNotFound
    case preprocessingFailed
    case inferenceFailed
}

public class RppgManager {
    
    private var model: MLModel?
    private var inputTimeLength: Int
    private var inputSize: Int
    
    /// Buffer to store frames until we have enough for a batch (e.g. 32 frames)
    private var frameBuffer: [CVPixelBuffer] = []
    
    public init(modelName: String = "EfficientPhys", timeLength: Int = 32, inputSize: Int = 72) throws {
        self.inputTimeLength = timeLength
        self.inputSize = inputSize
        try loadModel(name: modelName)
    }
    
    private func loadModel(name: String) throws {
        // Load the model from the bundle
        // Note: The caller must ensure the model is compiled and in the bundle resource path
        guard let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw RppgError.modelNotFound
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use NPU/GPU/CPU
        
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    public func processFrame(_ pixelBuffer: CVPixelBuffer, completion: @escaping (Result<Float, Error>) -> Void) {
        // 1. Preprocess (Center crop & Resize)
        guard let resized = FramePreprocessor.preprocess(pixelBuffer, targetSize: CGSize(width: inputSize, height: inputSize)) else {
            completion(.failure(RppgError.preprocessingFailed))
            return
        }
        
        // 2. Add to buffer
        frameBuffer.append(resized)
        
        // 3. Check if buffer is full
        if frameBuffer.count >= inputTimeLength {
            let chunk = Array(frameBuffer.prefix(inputTimeLength))
            frameBuffer.removeFirst(1) // Sliding window: remove oldest, keep overlap?
            // Or sliding stride = 1 (very expensive).
            // Usually rPPG uses a stride. Let's assume stride=1 for smoothness but performance might hit.
            // For now, let's just infer on the chunk.
            
            runInference(on: chunk, completion: completion)
        }
    }
    
    private func runInference(on frames: [CVPixelBuffer], completion: @escaping (Result<Float, Error>) -> Void) {
        guard let model = model else {
            completion(.failure(RppgError.modelNotFound))
            return
        }
        
        // 4. Prepare Input
        // EfficientPhys expects input (T, 3, H, W) or specific multiarray.
        // We'll assume the conversion script output "input_frames" as a MultiArray of shape (T, 3, H, W).
        // Creating that MultiArray in Swift from CVPixelBuffers is non-trivial.
        // Ideally, we'd use VNCoreMLRequest if the model took an Image input, but it takes a sequence.
        
        do {
            let inputMultiArray = try MLMultiArrayUtils.createMultiArray(from: frames, size: inputSize)
            let input = try MLDictionaryFeatureProvider(dictionary: ["input_frames": inputMultiArray])
            
            // 5. Predict
            let output = try model.prediction(from: input)
            
            // 6. Parse Output
            // Assuming output "rppg_output" is a MultiArray or float
            if let resultFeature = output.featureValue(for: "rppg_output"),
               let value = resultFeature.multiArrayValue {
                // Return average or last value
                let scalar = value[0].floatValue
                completion(.success(scalar))
            } else {
                completion(.failure(RppgError.inferenceFailed))
            }
            
        } catch {
            completion(.failure(error))
        }
    }
    
    /// Reset buffer (e.g. on finger lift or new session)
    public func reset() {
        frameBuffer.removeAll()
    }
}
