namespace AiDotNet.ModelCompression;

/// <summary>
/// Provides metrics and statistics for model compression operations.
/// </summary>
/// <remarks>
/// <para>
/// CompressionMetrics tracks important statistics about the compression process, including
/// compression ratio, model size reduction, inference speed impact, and accuracy preservation.
/// These metrics help evaluate the effectiveness of different compression strategies.
/// </para>
/// <para><b>For Beginners:</b> CompressionMetrics helps you measure how well compression worked.
///
/// When you compress a model, you want to know:
/// - How much smaller did it get? (compression ratio)
/// - How much memory did we save? (size reduction)
/// - Did it get faster? (inference speed)
/// - Is it still accurate? (accuracy preservation)
///
/// This class tracks all these important measurements so you can:
/// - Compare different compression techniques
/// - Decide if the compression is worth it
/// - Find the best balance between size and accuracy
///
/// Example:
/// - Original model: 100 MB, 95% accuracy, 10ms inference
/// - Compressed model: 10 MB, 94% accuracy, 5ms inference
/// - Metrics show: 10x compression, 1% accuracy loss, 2x speedup
/// - Conclusion: Great compression! The small accuracy loss is worth the huge size reduction.
/// </para>
/// </remarks>
public class CompressionMetrics
{
    /// <summary>
    /// Gets or sets the original model size in bytes.
    /// </summary>
    public long OriginalSize { get; set; }

    /// <summary>
    /// Gets or sets the compressed model size in bytes.
    /// </summary>
    public long CompressedSize { get; set; }

    /// <summary>
    /// Gets or sets the compression ratio (original size / compressed size).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The compression ratio shows how much smaller the model became.
    ///
    /// Examples:
    /// - Ratio of 2.0 = model is half the size (50% reduction)
    /// - Ratio of 10.0 = model is 1/10th the size (90% reduction)
    /// - Ratio of 50.0 = model is 1/50th the size (98% reduction)
    ///
    /// Higher is better! A ratio of 20 means you reduced the model to 5% of its original size.
    /// </para>
    /// </remarks>
    public double CompressionRatio { get; set; }

    /// <summary>
    /// Gets or sets the percentage of size reduction.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows the size reduction as a percentage.
    ///
    /// Formula: (1 - compressed size / original size) × 100%
    ///
    /// Examples:
    /// - 50% = model is half the original size
    /// - 90% = model is 1/10th the original size
    /// - 98% = model is 1/50th the original size
    ///
    /// This is just another way to express the compression ratio, often easier to understand.
    /// </para>
    /// </remarks>
    public double SizeReductionPercentage { get; set; }

    /// <summary>
    /// Gets or sets the original inference time in milliseconds.
    /// </summary>
    public double OriginalInferenceTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the compressed model inference time in milliseconds.
    /// </summary>
    public double CompressedInferenceTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the inference speedup factor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows how much faster (or slower) the compressed model is.
    ///
    /// Formula: original time / compressed time
    ///
    /// Examples:
    /// - 1.0 = same speed
    /// - 2.0 = twice as fast
    /// - 0.5 = half as fast (slower due to decompression overhead)
    ///
    /// Compression usually makes models faster because there's less data to move around,
    /// but sometimes decompression adds overhead.
    /// </para>
    /// </remarks>
    public double InferenceSpeedup { get; set; }

    /// <summary>
    /// Gets or sets the original model accuracy (before compression).
    /// </summary>
    public double OriginalAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the compressed model accuracy (after compression).
    /// </summary>
    public double CompressedAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the accuracy loss percentage.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows how much accuracy was lost due to compression.
    ///
    /// Formula: original accuracy - compressed accuracy
    ///
    /// Examples:
    /// - 0% = no accuracy loss (perfect!)
    /// - 1% = slight loss (usually acceptable)
    /// - 5% = significant loss (might be too much)
    ///
    /// The goal is to keep this under 2% for most applications. If you lose more than that,
    /// you might need to use less aggressive compression or a different technique.
    /// </para>
    /// </remarks>
    public double AccuracyLoss { get; set; }

    /// <summary>
    /// Gets or sets the number of parameters in the original model.
    /// </summary>
    public long OriginalParameterCount { get; set; }

    /// <summary>
    /// Gets or sets the effective number of unique parameters after compression.
    /// </summary>
    public long EffectiveParameterCount { get; set; }

    /// <summary>
    /// Gets or sets the compression technique used.
    /// </summary>
    public string CompressionTechnique { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the time taken to perform compression in milliseconds.
    /// </summary>
    public double CompressionTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the time taken to decompress in milliseconds.
    /// </summary>
    public double DecompressionTimeMs { get; set; }

    /// <summary>
    /// Calculates all derived metrics from the base measurements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates all the ratios and percentages automatically.
    ///
    /// Call this after setting the base values:
    /// - OriginalSize and CompressedSize
    /// - OriginalInferenceTimeMs and CompressedInferenceTimeMs
    /// - OriginalAccuracy and CompressedAccuracy
    ///
    /// It will then calculate:
    /// - CompressionRatio
    /// - SizeReductionPercentage
    /// - InferenceSpeedup
    /// - AccuracyLoss
    /// </para>
    /// </remarks>
    public void CalculateDerivedMetrics()
    {
        if (CompressedSize > 0)
        {
            CompressionRatio = (double)OriginalSize / CompressedSize;
            SizeReductionPercentage = (1.0 - (double)CompressedSize / OriginalSize) * 100.0;
        }

        if (CompressedInferenceTimeMs > 0)
        {
            InferenceSpeedup = OriginalInferenceTimeMs / CompressedInferenceTimeMs;
        }

        AccuracyLoss = OriginalAccuracy - CompressedAccuracy;
    }

    /// <summary>
    /// Gets a human-readable summary of the compression metrics.
    /// </summary>
    /// <returns>A formatted string containing all metrics.</returns>
    public override string ToString()
    {
        return $@"Compression Metrics Summary
===========================
Technique: {CompressionTechnique}

Size Metrics:
  Original Size: {FormatBytes(OriginalSize)}
  Compressed Size: {FormatBytes(CompressedSize)}
  Compression Ratio: {CompressionRatio:F2}x
  Size Reduction: {SizeReductionPercentage:F2}%

Parameter Metrics:
  Original Parameters: {OriginalParameterCount:N0}
  Effective Parameters: {EffectiveParameterCount:N0}
  Parameter Reduction: {(1.0 - (double)EffectiveParameterCount / OriginalParameterCount) * 100:F2}%

Performance Metrics:
  Original Inference Time: {OriginalInferenceTimeMs:F2}ms
  Compressed Inference Time: {CompressedInferenceTimeMs:F2}ms
  Inference Speedup: {InferenceSpeedup:F2}x

Accuracy Metrics:
  Original Accuracy: {OriginalAccuracy:F4}
  Compressed Accuracy: {CompressedAccuracy:F4}
  Accuracy Loss: {AccuracyLoss:F4} ({AccuracyLoss * 100:F2}%)

Timing:
  Compression Time: {CompressionTimeMs:F2}ms
  Decompression Time: {DecompressionTimeMs:F2}ms";
    }

    /// <summary>
    /// Formats a byte count into a human-readable string.
    /// </summary>
    private static string FormatBytes(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB", "TB" };
        double len = bytes;
        int order = 0;
        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len /= 1024;
        }
        return $"{len:F2} {sizes[order]}";
    }

    /// <summary>
    /// Determines if the compression meets the specified quality threshold.
    /// </summary>
    /// <param name="maxAccuracyLossPercentage">Maximum acceptable accuracy loss (default: 2%).</param>
    /// <param name="minCompressionRatio">Minimum acceptable compression ratio (default: 2x).</param>
    /// <returns>True if compression meets the quality criteria, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method checks if the compression is "good enough".
    ///
    /// It verifies two things:
    /// 1. Accuracy loss is acceptable (not too much)
    /// 2. Compression is significant enough (worthwhile)
    ///
    /// Example thresholds:
    /// - maxAccuracyLossPercentage = 2% means we accept up to 2% accuracy loss
    /// - minCompressionRatio = 2x means we want at least 50% size reduction
    ///
    /// If both conditions are met, the compression is considered successful.
    /// </para>
    /// </remarks>
    public bool MeetsQualityThreshold(double maxAccuracyLossPercentage = 2.0, double minCompressionRatio = 2.0)
    {
        double accuracyLossPercentage = AccuracyLoss * 100.0;
        return accuracyLossPercentage <= maxAccuracyLossPercentage && CompressionRatio >= minCompressionRatio;
    }
}
