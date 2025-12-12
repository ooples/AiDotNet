using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Provides metrics and statistics for model compression operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
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
public class CompressionMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
    public T CompressionRatio { get; set; } = NumOps.Zero;

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
    public T SizeReductionPercentage { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the original inference time in milliseconds.
    /// </summary>
    public T OriginalInferenceTimeMs { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the compressed model inference time in milliseconds.
    /// </summary>
    public T CompressedInferenceTimeMs { get; set; } = NumOps.Zero;

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
    public T InferenceSpeedup { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the original model accuracy (before compression).
    /// </summary>
    public T OriginalAccuracy { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the compressed model accuracy (after compression).
    /// </summary>
    public T CompressedAccuracy { get; set; } = NumOps.Zero;

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
    public T AccuracyLoss { get; set; } = NumOps.Zero;

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
    public T CompressionTimeMs { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the time taken to decompress in milliseconds.
    /// </summary>
    public T DecompressionTimeMs { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the sparsity level achieved (fraction of zero weights).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sparsity shows what fraction of weights are zero after pruning.
    ///
    /// Examples:
    /// - 0.0 = no zeros (dense model)
    /// - 0.9 = 90% zeros (very sparse)
    /// - 0.99 = 99% zeros (extremely sparse)
    ///
    /// Higher sparsity means better compression potential but may affect accuracy.
    /// </para>
    /// </remarks>
    public T Sparsity { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the number of bits per weight after quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows how many bits are used to represent each weight.
    ///
    /// Examples:
    /// - 32 bits = full precision (float)
    /// - 16 bits = half precision
    /// - 8 bits = int8 quantization
    /// - 5 bits = aggressive quantization (32 clusters)
    ///
    /// Lower bits = more compression but potentially less accuracy.
    /// </para>
    /// </remarks>
    public T BitsPerWeight { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the memory bandwidth savings ratio.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows how much memory bandwidth is saved during inference.
    ///
    /// Memory bandwidth is often the bottleneck for neural network inference.
    /// Smaller models need less data moved from memory, making inference faster.
    /// </para>
    /// </remarks>
    public T MemoryBandwidthSavings { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the reconstruction error (for lossy compression).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows the average error when decompressing weights.
    ///
    /// For lossy compression techniques (like quantization), the decompressed weights
    /// are approximations of the original. This metric measures that approximation error.
    /// Lower is better.
    /// </para>
    /// </remarks>
    public T ReconstructionError { get; set; } = NumOps.Zero;

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
    /// - MemoryBandwidthSavings
    /// </para>
    /// </remarks>
    public void CalculateDerivedMetrics()
    {
        if (CompressedSize > 0)
        {
            CompressionRatio = NumOps.FromDouble((double)OriginalSize / CompressedSize);
            SizeReductionPercentage = NumOps.FromDouble((1.0 - (double)CompressedSize / OriginalSize) * 100.0);
            MemoryBandwidthSavings = CompressionRatio;
        }

        var compressedInferenceTime = NumOps.ToDouble(CompressedInferenceTimeMs);
        if (compressedInferenceTime > 0)
        {
            InferenceSpeedup = NumOps.FromDouble(NumOps.ToDouble(OriginalInferenceTimeMs) / compressedInferenceTime);
        }

        AccuracyLoss = NumOps.Subtract(OriginalAccuracy, CompressedAccuracy);
    }

    /// <summary>
    /// Creates a CompressionMetrics instance from a DeepCompressionStats object.
    /// </summary>
    /// <param name="stats">The DeepCompressionStats containing compression statistics.</param>
    /// <param name="technique">The name of the compression technique used.</param>
    /// <returns>A populated CompressionMetrics instance.</returns>
    public static CompressionMetrics<T> FromDeepCompressionStats(DeepCompressionStats stats, string technique = "Deep Compression")
    {
        var metrics = new CompressionMetrics<T>
        {
            OriginalSize = stats.OriginalSizeBytes,
            CompressedSize = stats.CompressedSizeBytes,
            CompressionRatio = NumOps.FromDouble(stats.CompressionRatio),
            Sparsity = NumOps.FromDouble(stats.Sparsity),
            BitsPerWeight = NumOps.FromDouble(stats.BitsPerWeight),
            CompressionTechnique = technique
        };

        if (stats.CompressedSizeBytes > 0)
        {
            metrics.SizeReductionPercentage = NumOps.FromDouble(
                (1.0 - (double)stats.CompressedSizeBytes / stats.OriginalSizeBytes) * 100.0);
        }

        return metrics;
    }

    /// <summary>
    /// Gets a human-readable summary of the compression metrics.
    /// </summary>
    /// <returns>A formatted string containing all metrics.</returns>
    public override string ToString()
    {
        var paramReduction = OriginalParameterCount > 0
            ? (1.0 - (double)EffectiveParameterCount / OriginalParameterCount) * 100
            : 0;

        return $@"Compression Metrics Summary
===========================
Technique: {CompressionTechnique}

Size Metrics:
  Original Size: {FormatBytes(OriginalSize)}
  Compressed Size: {FormatBytes(CompressedSize)}
  Compression Ratio: {NumOps.ToDouble(CompressionRatio):F2}x
  Size Reduction: {NumOps.ToDouble(SizeReductionPercentage):F2}%

Parameter Metrics:
  Original Parameters: {OriginalParameterCount:N0}
  Effective Parameters: {EffectiveParameterCount:N0}
  Parameter Reduction: {paramReduction:F2}%
  Sparsity: {NumOps.ToDouble(Sparsity) * 100:F2}%
  Bits Per Weight: {NumOps.ToDouble(BitsPerWeight):F2}

Performance Metrics:
  Original Inference Time: {NumOps.ToDouble(OriginalInferenceTimeMs):F2}ms
  Compressed Inference Time: {NumOps.ToDouble(CompressedInferenceTimeMs):F2}ms
  Inference Speedup: {NumOps.ToDouble(InferenceSpeedup):F2}x
  Memory Bandwidth Savings: {NumOps.ToDouble(MemoryBandwidthSavings):F2}x

Accuracy Metrics:
  Original Accuracy: {NumOps.ToDouble(OriginalAccuracy):F4}
  Compressed Accuracy: {NumOps.ToDouble(CompressedAccuracy):F4}
  Accuracy Loss: {NumOps.ToDouble(AccuracyLoss):F4} ({NumOps.ToDouble(AccuracyLoss) * 100:F2}%)
  Reconstruction Error: {NumOps.ToDouble(ReconstructionError):F6}

Timing:
  Compression Time: {NumOps.ToDouble(CompressionTimeMs):F2}ms
  Decompression Time: {NumOps.ToDouble(DecompressionTimeMs):F2}ms";
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
    public bool MeetsQualityThreshold(T maxAccuracyLossPercentage, T minCompressionRatio)
    {
        var accuracyLossPercentage = NumOps.Multiply(AccuracyLoss, NumOps.FromDouble(100.0));
        var maxLoss = maxAccuracyLossPercentage;
        var minRatio = minCompressionRatio;

        return NumOps.ToDouble(accuracyLossPercentage) <= NumOps.ToDouble(maxLoss) &&
               NumOps.ToDouble(CompressionRatio) >= NumOps.ToDouble(minRatio);
    }

    /// <summary>
    /// Determines if the compression meets the specified quality threshold using default values.
    /// </summary>
    /// <param name="maxAccuracyLossPercentage">Maximum acceptable accuracy loss percentage (default: 2.0).</param>
    /// <param name="minCompressionRatio">Minimum acceptable compression ratio (default: 2.0).</param>
    /// <returns>True if compression meets the quality criteria, false otherwise.</returns>
    public bool MeetsQualityThreshold(double maxAccuracyLossPercentage = 2.0, double minCompressionRatio = 2.0)
    {
        return MeetsQualityThreshold(
            NumOps.FromDouble(maxAccuracyLossPercentage),
            NumOps.FromDouble(minCompressionRatio));
    }

    /// <summary>
    /// Calculates a composite fitness score for multi-objective optimization.
    /// </summary>
    /// <param name="accuracyWeight">Weight for accuracy preservation (default: 0.5).</param>
    /// <param name="compressionWeight">Weight for compression ratio (default: 0.3).</param>
    /// <param name="speedWeight">Weight for inference speedup (default: 0.2).</param>
    /// <returns>A composite fitness score where higher is better.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates a single score that balances multiple objectives.
    ///
    /// When optimizing compression, we care about multiple things:
    /// - High accuracy preservation (less accuracy loss = good)
    /// - High compression ratio (smaller model = good)
    /// - Fast inference (more speedup = good)
    ///
    /// The weights control how much each factor matters. Default weights prioritize:
    /// - 50% accuracy preservation
    /// - 30% compression ratio
    /// - 20% inference speed
    ///
    /// This is useful for AutoML and genetic algorithms that need a single fitness value.
    /// </para>
    /// </remarks>
    public T CalculateCompositeFitness(
        double accuracyWeight = 0.5,
        double compressionWeight = 0.3,
        double speedWeight = 0.2)
    {
        // Normalize accuracy component (1.0 - accuracy_loss_fraction)
        // Higher is better - we want to preserve accuracy
        var accuracyPreservation = 1.0 - NumOps.ToDouble(AccuracyLoss);
        accuracyPreservation = Math.Max(0, Math.Min(1, accuracyPreservation)); // Clamp to [0, 1]

        // Normalize compression ratio to [0, 1] range using sigmoid-like function
        // 1x ratio = 0, 10x ratio ≈ 0.9, 50x ratio ≈ 0.98
        var compressionScore = 1.0 - 1.0 / (1.0 + NumOps.ToDouble(CompressionRatio) / 10.0);

        // Normalize speedup to [0, 1] range
        // 1x speedup = 0.5, 2x speedup ≈ 0.67, 10x speedup ≈ 0.91
        var speedupValue = NumOps.ToDouble(InferenceSpeedup);
        var speedScore = speedupValue > 0 ? 1.0 - 1.0 / (1.0 + speedupValue) : 0;

        // Calculate weighted sum
        var totalWeight = accuracyWeight + compressionWeight + speedWeight;
        var fitness = (accuracyPreservation * accuracyWeight +
                      compressionScore * compressionWeight +
                      speedScore * speedWeight) / totalWeight;

        return NumOps.FromDouble(fitness);
    }

    /// <summary>
    /// Compares this compression result to another and determines which is better.
    /// </summary>
    /// <param name="other">The other compression metrics to compare against.</param>
    /// <param name="accuracyWeight">Weight for accuracy preservation.</param>
    /// <param name="compressionWeight">Weight for compression ratio.</param>
    /// <param name="speedWeight">Weight for inference speedup.</param>
    /// <returns>True if this compression is better than the other.</returns>
    public bool IsBetterThan(
        CompressionMetrics<T> other,
        double accuracyWeight = 0.5,
        double compressionWeight = 0.3,
        double speedWeight = 0.2)
    {
        if (other == null) return true;

        var thisFitness = CalculateCompositeFitness(accuracyWeight, compressionWeight, speedWeight);
        var otherFitness = other.CalculateCompositeFitness(accuracyWeight, compressionWeight, speedWeight);

        return NumOps.ToDouble(thisFitness) > NumOps.ToDouble(otherFitness);
    }
}
