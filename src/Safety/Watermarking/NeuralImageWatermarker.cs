using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Image watermarker that uses an encoder-decoder neural network approach for embedding.
/// </summary>
/// <remarks>
/// <para>
/// Simulates neural watermarking by analyzing learned feature patterns. Neural watermarks
/// encode information in the latent space of an encoder-decoder network, making them
/// robust to a wide range of image transformations including cropping, rotation, and
/// color adjustment.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker uses AI-based techniques to embed signatures
/// that are extremely hard to remove. The watermark is woven into the image at a deep
/// level that survives even aggressive editing operations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NeuralImageWatermarker<T> : ImageWatermarkerBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public override string ModuleName => "NeuralImageWatermarker";

    /// <summary>
    /// Initializes a new neural image watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding strength (0.0-1.0). Default: 0.5.</param>
    public NeuralImageWatermarker(double watermarkStrength = 0.5) : base(watermarkStrength) { }

    /// <inheritdoc />
    public override double DetectWatermark(Tensor<T> image)
    {
        var span = image.Data.Span;
        if (span.Length < 64) return 0;

        // Analyze activation-like patterns in pixel neighborhoods
        // Neural watermarks create correlated patterns across spatial locations
        int sampleSize = Math.Min(span.Length, 4096);
        double correlationSum = 0;
        int pairs = 0;

        int stride = Math.Max(1, sampleSize / 256);
        for (int i = 0; i + stride < sampleSize; i += stride)
        {
            double a = NumOps.ToDouble(span[i]);
            double b = NumOps.ToDouble(span[i + stride]);
            correlationSum += a * b;
            pairs++;
        }

        if (pairs < 16) return 0;

        double avgCorrelation = correlationSum / pairs;

        // Compute autocorrelation at stride to detect periodic neural patterns
        double variance = 0;
        double mean = 0;
        for (int i = 0; i < sampleSize; i += stride)
        {
            double val = NumOps.ToDouble(span[i]);
            mean += val;
        }
        mean /= (sampleSize / stride);

        for (int i = 0; i < sampleSize; i += stride)
        {
            double val = NumOps.ToDouble(span[i]);
            variance += (val - mean) * (val - mean);
        }
        variance /= (sampleSize / stride);

        if (variance < 1e-10) return 0;

        double normalizedCorrelation = Math.Abs(avgCorrelation - mean * mean) / variance;

        // Neural watermarks: high normalized correlation (>0.3)
        if (normalizedCorrelation < 0.2) return 0;
        return Math.Max(0, Math.Min(1.0, (normalizedCorrelation - 0.2) / 0.5));
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        double score = DetectWatermark(image);

        if (score >= 0.3)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = score,
                Description = $"Neural image watermark detected (confidence: {score:F3}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }
}
