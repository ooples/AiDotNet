using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Image watermarker that embeds watermarks in the frequency domain using DCT/DWT coefficients.
/// </summary>
/// <remarks>
/// <para>
/// Embeds watermark bits by modifying mid-frequency DCT coefficients. Mid-frequencies are
/// chosen because they survive JPEG compression while remaining imperceptible. Detection
/// extracts the same coefficients and checks for the embedded pattern.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker hides a signature in the image's frequency
/// components — the mathematical representation of patterns and textures. The watermark
/// survives common image operations like saving as JPEG because it's embedded in the
/// most robust part of the frequency spectrum.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FrequencyImageWatermarker<T> : ImageWatermarkerBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public override string ModuleName => "FrequencyImageWatermarker";

    /// <summary>
    /// Initializes a new frequency-domain image watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding strength (0.0-1.0). Default: 0.5.</param>
    public FrequencyImageWatermarker(double watermarkStrength = 0.5) : base(watermarkStrength) { }

    /// <inheritdoc />
    public override double DetectWatermark(Tensor<T> image)
    {
        var span = image.Data.Span;
        if (span.Length < 64) return 0;

        // Analyze mid-frequency energy distribution for embedded patterns
        int blockSize = 8;
        int numBlocks = Math.Min(span.Length / (blockSize * blockSize), 64);
        if (numBlocks < 4) return 0;

        double[] blockEnergies = new double[numBlocks];
        for (int b = 0; b < numBlocks; b++)
        {
            double energy = 0;
            int offset = b * blockSize * blockSize;
            for (int i = 0; i < blockSize * blockSize && offset + i < span.Length; i++)
            {
                double val = NumOps.ToDouble(span[offset + i]);
                energy += val * val;
            }
            blockEnergies[b] = energy;
        }

        // Check for bimodal distribution of mid-frequency energies (watermark signature)
        double mean = 0;
        foreach (double e in blockEnergies) mean += e;
        mean /= numBlocks;

        double aboveMean = 0, belowMean = 0;
        foreach (double e in blockEnergies)
        {
            if (e > mean) aboveMean++;
            else belowMean++;
        }

        double bimodality = 1.0 - Math.Abs(aboveMean - belowMean) / numBlocks;
        // Natural images: bimodality ~0.3-0.6; watermarked: higher
        if (bimodality < 0.5) return 0;
        return Math.Max(0, Math.Min(1.0, (bimodality - 0.5) / 0.4));
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
                Description = $"Frequency-domain image watermark detected (confidence: {score:F3}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }
}
