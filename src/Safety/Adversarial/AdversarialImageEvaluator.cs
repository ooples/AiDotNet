using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Adversarial;

/// <summary>
/// Detects adversarial perturbations in images designed to evade image safety classifiers.
/// </summary>
/// <remarks>
/// <para>
/// Adversarial attacks on images add imperceptible perturbations that cause classifiers to
/// misclassify content. This module detects such perturbations by analyzing statistical
/// properties that differ between natural and adversarially perturbed images: high-frequency
/// energy anomalies, pixel distribution irregularities, and JPEG artifact inconsistencies.
/// </para>
/// <para>
/// <b>For Beginners:</b> An adversarial image looks normal to humans but tricks AI classifiers.
/// Someone might add invisible noise to a harmful image so that safety classifiers think it's
/// safe. This module detects that invisible noise by looking for patterns that don't occur
/// in natural photographs.
/// </para>
/// <para>
/// <b>References:</b>
/// - Feature squeezing for adversarial example detection (Xu et al., NDSS 2018)
/// - Adversarial examples in physical world (Kurakin et al., 2017)
/// - Detecting adversarial examples via neural fingerprinting (2024)
/// - Robust adversarial perturbation detection survey (2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AdversarialImageEvaluator<T> : IImageSafetyModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;

    /// <inheritdoc />
    public string ModuleName => "AdversarialImageEvaluator";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new adversarial image evaluator.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    public AdversarialImageEvaluator(double threshold = 0.5)
    {
        _threshold = threshold;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var span = image.Data.Span;

        if (span.Length < 64) return findings;

        // 1. High-frequency energy ratio (adversarial perturbations have unusual HF content)
        double hfScore = ComputeHighFrequencyAnomalyScore(span, image.Shape);

        // 2. Pixel distribution analysis (adversarial images have non-natural histogram shapes)
        double histScore = ComputeHistogramAnomalyScore(span);

        // 3. Feature squeezing detection (compare original with bit-depth-reduced version)
        double squeezingScore = ComputeFeeSqueezingScore(span);

        double combinedScore = 0.40 * hfScore + 0.30 * histScore + 0.30 * squeezingScore;

        if (combinedScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Manipulated,
                Severity = combinedScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, combinedScore),
                Description = $"Adversarial perturbation detected (score: {combinedScore:F3}). " +
                              $"HF energy: {hfScore:F3}, histogram: {histScore:F3}, " +
                              $"feature squeezing: {squeezingScore:F3}. " +
                              $"Image may have been adversarially modified to evade classifiers.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        var tensor = new Tensor<T>(content.ToArray(), new[] { content.Length });
        return EvaluateImage(tensor);
    }

    private static double ComputeHighFrequencyAnomalyScore(ReadOnlySpan<T> span, int[] shape)
    {
        // Compute energy in high-frequency components using Laplacian approximation
        // Natural images: most energy in low-frequency; adversarial: elevated HF
        int width = shape.Length >= 2 ? shape[shape.Length - 1] : (int)Math.Sqrt(span.Length);
        int height = span.Length / Math.Max(width, 1);

        if (width < 4 || height < 4) return 0;

        double lfEnergy = 0, hfEnergy = 0;
        int count = 0;

        // Simple gradient-based HF estimation: pixel differences
        int stride = width;
        int maxRows = Math.Min(height - 1, 128);
        int maxCols = Math.Min(width - 1, 128);

        for (int y = 1; y < maxRows; y++)
        {
            for (int x = 1; x < maxCols; x++)
            {
                int idx = y * stride + x;
                if (idx >= span.Length || idx + 1 >= span.Length || idx - 1 < 0 ||
                    idx - stride < 0 || idx + stride >= span.Length) continue;

                double center = NumOps.ToDouble(span[idx]);
                double right = NumOps.ToDouble(span[idx + 1]);
                double left = NumOps.ToDouble(span[idx - 1]);
                double up = NumOps.ToDouble(span[idx - stride]);
                double down = NumOps.ToDouble(span[idx + stride]);

                // Laplacian = 4*center - left - right - up - down
                double laplacian = 4 * center - left - right - up - down;
                hfEnergy += laplacian * laplacian;
                lfEnergy += center * center;
                count++;
            }
        }

        if (count == 0 || lfEnergy < 1e-10) return 0;

        double hfRatio = hfEnergy / (lfEnergy + hfEnergy);

        // Natural images: HF ratio typically 0.01-0.15
        // Adversarial: HF ratio typically 0.2-0.8
        if (hfRatio < 0.15) return 0;
        return Math.Min(1.0, (hfRatio - 0.15) / 0.5);
    }

    private static double ComputeHistogramAnomalyScore(ReadOnlySpan<T> span)
    {
        // Compute pixel value histogram and check for non-natural patterns
        // Adversarial perturbations create unusual histogram shapes (gaps, spikes)
        int bins = 64;
        int[] histogram = new int[bins];
        int totalPixels = 0;

        for (int i = 0; i < span.Length && i < 65536; i++)
        {
            double val = NumOps.ToDouble(span[i]);
            // Normalize to [0, 1] if needed
            if (val > 1.0) val /= 255.0;
            val = Math.Max(0, Math.Min(1.0 - 1e-10, val));
            int bin = (int)(val * bins);
            bin = Math.Max(0, Math.Min(bins - 1, bin));
            histogram[bin]++;
            totalPixels++;
        }

        if (totalPixels < 100) return 0;

        // Count empty bins (gaps) — natural images rarely have many empty bins
        int emptyBins = 0;
        for (int i = 0; i < bins; i++)
        {
            if (histogram[i] == 0) emptyBins++;
        }

        // Compute histogram smoothness (Laplacian of histogram)
        double smoothnessViolation = 0;
        for (int i = 1; i < bins - 1; i++)
        {
            double laplacian = histogram[i - 1] + histogram[i + 1] - 2.0 * histogram[i];
            smoothnessViolation += Math.Abs(laplacian);
        }
        double avgSmoothness = smoothnessViolation / (bins - 2) / Math.Max(totalPixels / bins, 1);

        // Natural: avg smoothness < 0.5, adversarial: > 1.0
        double smoothnessScore = Math.Min(1.0, Math.Max(0, (avgSmoothness - 0.5) / 1.0));

        // Empty bins score
        double emptyBinRatio = (double)emptyBins / bins;
        double gapScore = Math.Min(1.0, Math.Max(0, (emptyBinRatio - 0.1) / 0.4));

        return 0.6 * smoothnessScore + 0.4 * gapScore;
    }

    private static double ComputeFeeSqueezingScore(ReadOnlySpan<T> span)
    {
        // Feature squeezing: reduce bit depth and measure L2 distance
        // Adversarial perturbations are removed by bit-depth reduction,
        // causing large changes in adversarial images but small changes in natural ones
        int numPixels = Math.Min(span.Length, 16384);
        double l2Sum = 0;
        double maxVal = 0;

        for (int i = 0; i < numPixels; i++)
        {
            double val = NumOps.ToDouble(span[i]);
            if (val > 1.0) val /= 255.0;

            // Squeeze to 4-bit depth
            double squeezed = Math.Round(val * 15.0) / 15.0;
            double diff = val - squeezed;
            l2Sum += diff * diff;
            if (val > maxVal) maxVal = val;
        }

        double rms = Math.Sqrt(l2Sum / numPixels);

        // Natural images: RMS change from squeezing < 0.01
        // Adversarial: RMS change > 0.02
        if (rms < 0.01) return 0;
        return Math.Min(1.0, (rms - 0.01) / 0.03);
    }
}
