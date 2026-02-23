using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Image watermarker that embeds imperceptible spatial-domain watermarks.
/// </summary>
/// <remarks>
/// <para>
/// Embeds watermark bits by making sub-pixel-level modifications to pixel values in
/// the spatial domain. Uses least-significant-bit (LSB) encoding with error diffusion
/// to spread the watermark energy and avoid visual artifacts.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker hides a signature by making tiny changes
/// to pixel values — so small that the human eye cannot see them. It's like writing
/// a message in invisible ink within the image data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InvisibleImageWatermarker<T> : ImageWatermarkerBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public override string ModuleName => "InvisibleImageWatermarker";

    /// <summary>
    /// Initializes a new invisible spatial-domain image watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding strength (0.0-1.0). Default: 0.3.</param>
    public InvisibleImageWatermarker(double watermarkStrength = 0.3) : base(watermarkStrength) { }

    /// <inheritdoc />
    public override double DetectWatermark(Tensor<T> image)
    {
        var span = image.Data.Span;
        if (span.Length < 64) return 0;

        // Detect LSB patterns that indicate spatial-domain watermarking
        int sampleSize = Math.Min(span.Length, 8192);
        int[] lsbHist = new int[2]; // count of even vs odd LSB values
        int analyzed = 0;

        for (int i = 0; i < sampleSize; i++)
        {
            double val = NumOps.ToDouble(span[i]);
            if (val > 1.0) val /= 255.0;

            // Quantize to 8-bit and check LSB
            int quantized = (int)(val * 255.0 + 0.5);
            lsbHist[quantized & 1]++;
            analyzed++;
        }

        if (analyzed < 32) return 0;

        // Natural images: LSBs are roughly evenly distributed
        // Watermarked images: LSBs are biased toward the embedded pattern
        double ratio = (double)Math.Max(lsbHist[0], lsbHist[1]) / analyzed;

        // Compute chi-squared statistic for LSB uniformity
        double expected = analyzed / 2.0;
        double chi2 = 0;
        for (int i = 0; i < 2; i++)
        {
            double diff = lsbHist[i] - expected;
            chi2 += (diff * diff) / expected;
        }

        // Also check pairs of adjacent LSBs for correlation
        int matchingPairs = 0;
        int totalPairs = 0;
        for (int i = 0; i + 1 < sampleSize; i += 2)
        {
            double a = NumOps.ToDouble(span[i]);
            double b = NumOps.ToDouble(span[i + 1]);
            if (a > 1.0) a /= 255.0;
            if (b > 1.0) b /= 255.0;

            int lsbA = (int)(a * 255.0 + 0.5) & 1;
            int lsbB = (int)(b * 255.0 + 0.5) & 1;
            if (lsbA == lsbB) matchingPairs++;
            totalPairs++;
        }

        if (totalPairs < 16) return 0;

        double pairCorrelation = (double)matchingPairs / totalPairs;
        // Natural: ~0.5; Watermarked with error diffusion: >0.6
        double corrScore = pairCorrelation > 0.55 ? (pairCorrelation - 0.55) / 0.3 : 0;

        // Chi-squared score (>3.84 is significant at p=0.05)
        double chi2Score = chi2 > 3.84 ? Math.Min(1.0, (chi2 - 3.84) / 10.0) : 0;

        return Math.Max(0, Math.Min(1.0, Math.Max(corrScore, chi2Score)));
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
                Description = $"Invisible spatial-domain image watermark detected (confidence: {score:F3}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }
}
