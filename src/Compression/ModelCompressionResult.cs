namespace AiDotNet.Compression;

using AiDotNet.Enums;
using System;
using System.Collections.Generic;

/// <summary>
/// Contains the results and metrics from a model compression operation.
/// </summary>
public class ModelCompressionResult
{
    /// <summary>
    /// Gets or sets the compression technique that was applied.
    /// </summary>
    public CompressionTechnique Technique { get; set; }
    
    /// <summary>
    /// Gets or sets a detailed name or description of the compression technique.
    /// </summary>
    public string CompressionTechniqueName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the size of the original model in bytes.
    /// </summary>
    public long OriginalModelSizeBytes { get; set; }

    /// <summary>
    /// Gets or sets the size of the compressed model in bytes.
    /// </summary>
    public long CompressedModelSizeBytes { get; set; }

    /// <summary>
    /// Gets or sets the compression ratio achieved.
    /// </summary>
    public double CompressionRatio { get; set; }

    /// <summary>
    /// Gets or sets the accuracy of the original model.
    /// </summary>
    public double OriginalAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the accuracy of the compressed model.
    /// </summary>
    public double CompressedAccuracy { get; set; }
    
    /// <summary>
    /// Gets or sets the impact on accuracy due to compression (can be positive or negative).
    /// </summary>
    public double AccuracyImpact { get; set; }

    /// <summary>
    /// Gets the absolute decrease in accuracy due to compression.
    /// </summary>
    public double AccuracyDecrease => 
        Math.Max(0, OriginalAccuracy - CompressedAccuracy);

    /// <summary>
    /// Gets or sets the speedup factor for inference after compression.
    /// </summary>
    public double InferenceSpeedupFactor { get; set; }
    
    /// <summary>
    /// Gets or sets the reduction in memory usage during inference.
    /// </summary>
    public double MemoryReduction { get; set; }
    
    /// <summary>
    /// Gets or sets the compression time in milliseconds.
    /// </summary>
    public long CompressionTimeMs { get; set; }
    
    /// <summary>
    /// Gets or sets the device where compression was performed.
    /// </summary>
    public string CompressionDevice { get; set; } = "CPU";
    
    /// <summary>
    /// Gets or sets the average inference time for the original model in milliseconds.
    /// </summary>
    public double OriginalInferenceTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the average inference time for the compressed model in milliseconds.
    /// </summary>
    public double CompressedInferenceTimeMs { get; set; }

    /// <summary>
    /// Gets the speedup ratio for inference due to compression.
    /// </summary>
    public double InferenceSpeedupRatio =>
        OriginalInferenceTimeMs / CompressedInferenceTimeMs;

    /// <summary>
    /// Gets or sets additional metrics and statistics about the compression.
    /// </summary>
    public Dictionary<string, object> AdditionalMetrics { get; set; } = 
        new Dictionary<string, object>();

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelCompressionResult"/> class.
    /// </summary>
    public ModelCompressionResult()
    {
    }

    /// <summary>
    /// Creates a human-readable summary of the compression results.
    /// </summary>
    /// <returns>A string containing a summary of the compression results.</returns>
    public override string ToString()
    {
        return $"Model Compression Summary ({CompressionTechniqueName}):\n" +
               $"- Size: {OriginalModelSizeBytes / 1024.0:F2} KB → {CompressedModelSizeBytes / 1024.0:F2} KB" +
               $" (Ratio: {CompressionRatio:F2}x)\n" +
               $"- Accuracy: {OriginalAccuracy:P2} → {CompressedAccuracy:P2}" + 
               $" (Impact: {AccuracyImpact:P2})\n" +
               $"- Inference Time: {OriginalInferenceTimeMs:F2} ms → {CompressedInferenceTimeMs:F2} ms" +
               $" (Speedup: {InferenceSpeedupFactor:F2}x)\n" +
               $"- Memory Reduction: {MemoryReduction:P2}";
    }
}