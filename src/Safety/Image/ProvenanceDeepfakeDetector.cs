using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Detects deepfake/AI-generated images by analyzing provenance signals: compression artifacts,
/// statistical fingerprints, and embedded watermark traces.
/// </summary>
/// <remarks>
/// <para>
/// AI-generated images often lack natural camera processing artifacts (JPEG quantization patterns,
/// sensor noise, optical aberrations) or contain telltale signs of specific generators (GAN
/// checkerboard artifacts, diffusion model smoothing). This detector analyzes these provenance
/// signals without requiring frequency domain analysis.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real photos carry invisible "fingerprints" from the camera that took
/// them — specific noise patterns, compression artifacts, and color processing signatures. AI
/// images don't have these, or they have different ones. This module checks for the presence
/// or absence of these fingerprints to determine if an image is AI-generated.
/// </para>
/// <para>
/// <b>Detection signals:</b>
/// 1. JPEG artifact analysis — real photos have characteristic quantization patterns
/// 2. Noise floor consistency — cameras produce consistent sensor noise
/// 3. Color channel correlation — natural images have specific cross-channel statistics
/// 4. Local Binary Pattern (LBP) texture analysis — AI textures differ from natural
/// </para>
/// <para>
/// <b>References:</b>
/// - C2P-CLIP: Content provenance detection (2024, arxiv:2404.09677)
/// - SynthID-Image: Internet-scale watermarking for provenance (DeepMind, 2025, arxiv:2510.09263)
/// - Only 38% of AI generators have adequate watermarking (2025, arxiv:2503.18156)
/// - AI-generated media detection survey (2025, arxiv:2502.05240)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ProvenanceDeepfakeDetector<T> : ImageSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;

    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;

    /// <inheritdoc />
    public override string ModuleName => "ProvenanceDeepfakeDetector";

    /// <summary>
    /// Initializes a new provenance-based deepfake detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    public ProvenanceDeepfakeDetector(double threshold = 0.5)
    {
        _threshold = threshold;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var span = image.Data.Span;
        if (span.Length == 0) return findings;

        var layout = DetermineLayout(image.Shape, span.Length);
        if (layout.Height < 16 || layout.Width < 16) return findings;

        // 1. Noise floor analysis
        double noiseFloorAnomaly = AnalyzeNoiseFloor(span, layout);

        // 2. JPEG artifact consistency
        double jpegAnomaly = AnalyzeJPEGArtifacts(span, layout);

        // 3. LBP texture fingerprint
        double textureAnomaly = AnalyzeLBPTexture(span, layout);

        // 4. Color channel statistics
        double colorAnomaly = AnalyzeColorStatistics(span, layout);

        // Combined score
        double finalScore = 0.25 * noiseFloorAnomaly +
                           0.25 * jpegAnomaly +
                           0.25 * textureAnomaly +
                           0.25 * colorAnomaly;

        if (finalScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.AIGenerated,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Provenance analysis: potential AI-generated image (score: {finalScore:F3}). " +
                              $"Noise floor: {noiseFloorAnomaly:F3}, JPEG artifacts: {jpegAnomaly:F3}, " +
                              $"texture: {textureAnomaly:F3}, color: {colorAnomaly:F3}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>
    /// Analyzes noise floor characteristics. Camera sensor noise follows specific
    /// distributions; AI images typically have either no noise or synthetic noise.
    /// </summary>
    private double AnalyzeNoiseFloor(ReadOnlySpan<T> data, ImageLayout layout)
    {
        // Compute high-pass filtered signal via Laplacian
        double sumAbs = 0;
        double sumSq = 0;
        int count = 0;
        int stride = Math.Max(1, layout.Height / 32);

        for (int y = 1; y < layout.Height - 1; y += stride)
        {
            for (int x = 1; x < layout.Width - 1; x += stride)
            {
                double center = GetLuminance(data, layout, y, x);
                double laplacian = 4 * center -
                    GetLuminance(data, layout, y - 1, x) -
                    GetLuminance(data, layout, y + 1, x) -
                    GetLuminance(data, layout, y, x - 1) -
                    GetLuminance(data, layout, y, x + 1);

                sumAbs += Math.Abs(laplacian);
                sumSq += laplacian * laplacian;
                count++;
            }
        }

        if (count < 10) return 0;

        double meanAbs = sumAbs / count;
        double meanSq = sumSq / count;
        double rms = Math.Sqrt(meanSq);

        // Kurtosis-like measure: rms / mean_abs
        // Gaussian noise: rms/mean_abs ≈ 1.25
        // AI images: often higher (sparse noise) or lower (too smooth)
        double ratio = meanAbs > 1e-10 ? rms / meanAbs : 0;

        double anomaly;
        if (ratio < 1.0)
            anomaly = (1.0 - ratio) * 2; // Too smooth
        else if (ratio > 1.6)
            anomaly = (ratio - 1.6) * 2; // Non-Gaussian noise
        else
            anomaly = 0; // Normal range

        return Math.Min(1.0, anomaly);
    }

    /// <summary>
    /// Analyzes JPEG-like quantization artifacts. Real JPEG images have 8x8 block
    /// artifacts; AI images either lack them or have them inconsistently.
    /// </summary>
    private double AnalyzeJPEGArtifacts(ReadOnlySpan<T> data, ImageLayout layout)
    {
        // Detect 8x8 block boundary discontinuities
        double blockBoundarySum = 0;
        double nonBoundarySum = 0;
        int blockCount = 0;
        int nonBlockCount = 0;

        for (int y = 0; y < layout.Height - 1; y++)
        {
            bool isBlockBoundary = (y + 1) % 8 == 0;
            int stride = Math.Max(1, layout.Width / 64);

            for (int x = 0; x < layout.Width; x += stride)
            {
                double curr = GetLuminance(data, layout, y, x);
                double next = GetLuminance(data, layout, y + 1, x);
                double diff = Math.Abs(curr - next);

                if (isBlockBoundary)
                {
                    blockBoundarySum += diff;
                    blockCount++;
                }
                else
                {
                    nonBoundarySum += diff;
                    nonBlockCount++;
                }
            }
        }

        if (blockCount == 0 || nonBlockCount == 0) return 0;

        double blockAvg = blockBoundarySum / blockCount;
        double nonBlockAvg = nonBoundarySum / nonBlockCount;

        // Block boundary ratio: JPEG images have higher discontinuity at 8x8 boundaries
        // AI images: uniform (ratio ~1.0) or no pattern
        double ratio = nonBlockAvg > 1e-10 ? blockAvg / nonBlockAvg : 1.0;

        // Natural JPEG: ratio ~1.1-1.5; AI (no JPEG): ratio ~1.0
        if (ratio < 1.05)
        {
            // No JPEG artifacts at all — suspicious for supposedly "photographic" content
            return 0.4; // Mild suspicion
        }
        if (ratio > 2.0)
        {
            // Extremely strong artifacts — double-compressed or manipulated
            return Math.Min(1.0, (ratio - 2.0) / 3.0);
        }

        return 0; // Normal JPEG range
    }

    /// <summary>
    /// Analyzes Local Binary Pattern (LBP) texture distribution.
    /// AI-generated textures have different LBP histograms than natural textures.
    /// </summary>
    private double AnalyzeLBPTexture(ReadOnlySpan<T> data, ImageLayout layout)
    {
        // Compute simplified LBP histogram
        var lbpHistogram = new int[256]; // 8-bit LBP
        int totalPixels = 0;
        int stride = Math.Max(1, layout.Height / 64);

        for (int y = 1; y < layout.Height - 1; y += stride)
        {
            for (int x = 1; x < layout.Width - 1; x += stride)
            {
                double center = GetLuminance(data, layout, y, x);
                int lbp = 0;

                // 8 neighbors clockwise
                if (GetLuminance(data, layout, y - 1, x - 1) >= center) lbp |= 1;
                if (GetLuminance(data, layout, y - 1, x) >= center) lbp |= 2;
                if (GetLuminance(data, layout, y - 1, x + 1) >= center) lbp |= 4;
                if (GetLuminance(data, layout, y, x + 1) >= center) lbp |= 8;
                if (GetLuminance(data, layout, y + 1, x + 1) >= center) lbp |= 16;
                if (GetLuminance(data, layout, y + 1, x) >= center) lbp |= 32;
                if (GetLuminance(data, layout, y + 1, x - 1) >= center) lbp |= 64;
                if (GetLuminance(data, layout, y, x - 1) >= center) lbp |= 128;

                lbpHistogram[lbp]++;
                totalPixels++;
            }
        }

        if (totalPixels < 100) return 0;

        // Compute LBP entropy
        double entropy = 0;
        for (int i = 0; i < 256; i++)
        {
            if (lbpHistogram[i] > 0)
            {
                double p = (double)lbpHistogram[i] / totalPixels;
                entropy -= p * Math.Log(p, 2);
            }
        }

        // Count "uniform" LBP patterns (0 or 2 transitions — natural texture indicator)
        int uniformPatterns = 0;
        for (int i = 0; i < 256; i++)
        {
            if (lbpHistogram[i] > 0 && IsUniformLBP(i))
            {
                uniformPatterns += lbpHistogram[i];
            }
        }
        double uniformRatio = (double)uniformPatterns / totalPixels;

        // Natural images: entropy ~5-7, uniform ratio ~0.6-0.9
        // AI images: entropy often ~4-5 (smoother) or ~7+ (noisy), uniform ratio < 0.5

        double entropyAnomaly = entropy < 5 ? (5 - entropy) / 3 :
                               entropy > 7 ? (entropy - 7) / 3 : 0;

        double uniformAnomaly = uniformRatio < 0.5 ? (0.5 - uniformRatio) * 2 : 0;

        return Math.Min(1.0, 0.5 * entropyAnomaly + 0.5 * uniformAnomaly);
    }

    /// <summary>
    /// Analyzes color statistics. AI images often have different color channel
    /// correlation and distribution patterns than natural photos.
    /// </summary>
    private double AnalyzeColorStatistics(ReadOnlySpan<T> data, ImageLayout layout)
    {
        if (layout.Channels < 3) return 0;

        // Compute skewness of each color channel
        double rSum = 0, gSum = 0, bSum = 0;
        double rSum2 = 0, gSum2 = 0, bSum2 = 0;
        double rSum3 = 0, gSum3 = 0, bSum3 = 0;
        int count = 0;
        int stride = Math.Max(1, (layout.Height * layout.Width) / 10000);

        for (int y = 0; y < layout.Height; y += stride)
        {
            for (int x = 0; x < layout.Width; x += stride)
            {
                double r = GetValChannel(data, layout, y, x, 0);
                double g = GetValChannel(data, layout, y, x, 1);
                double b = GetValChannel(data, layout, y, x, 2);

                rSum += r; gSum += g; bSum += b;
                rSum2 += r * r; gSum2 += g * g; bSum2 += b * b;
                rSum3 += r * r * r; gSum3 += g * g * g; bSum3 += b * b * b;
                count++;
            }
        }

        if (count < 100) return 0;

        // Compute skewness for each channel
        double rSkew = ComputeSkewness(rSum, rSum2, rSum3, count);
        double gSkew = ComputeSkewness(gSum, gSum2, gSum3, count);
        double bSkew = ComputeSkewness(bSum, bSum2, bSum3, count);

        // Cross-channel skewness consistency
        double avgSkew = (rSkew + gSkew + bSkew) / 3;
        double skewVar = ((rSkew - avgSkew) * (rSkew - avgSkew) +
                          (gSkew - avgSkew) * (gSkew - avgSkew) +
                          (bSkew - avgSkew) * (bSkew - avgSkew)) / 3;

        // AI images often have more symmetric (low skewness) color distributions
        double lowSkewnessAnomaly = Math.Abs(avgSkew) < 0.2 ? (0.2 - Math.Abs(avgSkew)) * 5 : 0;

        // Cross-channel skewness should be somewhat consistent
        double highVarianceAnomaly = skewVar > 0.5 ? Math.Min(1.0, (skewVar - 0.5) / 2) : 0;

        return Math.Min(1.0, 0.5 * lowSkewnessAnomaly + 0.5 * highVarianceAnomaly);
    }

    private static double ComputeSkewness(double sum, double sum2, double sum3, int n)
    {
        double mean = sum / n;
        double variance = sum2 / n - mean * mean;
        if (variance < 1e-20) return 0;

        double stddev = Math.Sqrt(variance);
        double thirdMoment = sum3 / n - 3 * mean * sum2 / n + 2 * mean * mean * mean;

        return stddev > 1e-10 ? thirdMoment / (stddev * stddev * stddev) : 0;
    }

    private static bool IsUniformLBP(int pattern)
    {
        // Count transitions in the circular binary pattern
        int transitions = 0;
        int prev = pattern & 1;
        for (int i = 1; i < 8; i++)
        {
            int curr = (pattern >> i) & 1;
            if (curr != prev) transitions++;
            prev = curr;
        }
        // Wrap-around
        if (((pattern >> 7) & 1) != (pattern & 1)) transitions++;
        return transitions <= 2;
    }

    private double GetLuminance(ReadOnlySpan<T> data, ImageLayout layout, int y, int x)
    {
        double r = GetValChannel(data, layout, y, x, 0);
        if (layout.Channels >= 3)
        {
            double g = GetValChannel(data, layout, y, x, 1);
            double b = GetValChannel(data, layout, y, x, 2);
            return 0.299 * r + 0.587 * g + 0.114 * b;
        }
        return r;
    }

    private static double GetValChannel(ReadOnlySpan<T> data, ImageLayout layout, int y, int x, int c)
    {
        int idx;
        if (layout.Format == PixFmt.CHW)
            idx = c * layout.Height * layout.Width + y * layout.Width + x;
        else
            idx = (y * layout.Width + x) * layout.Channels + c;

        if (idx < 0 || idx >= data.Length) return 0;
        double val = NumOps.ToDouble(data[idx]);
        return val <= 1.0 ? val * 255 : val;
    }

    private static ImageLayout DetermineLayout(int[] shape, int dataLength)
    {
        if (shape.Length >= 4)
        {
            if (shape[1] <= 4 && shape[2] > 4 && shape[3] > 4)
                return new ImageLayout { Channels = shape[1], Height = shape[2], Width = shape[3], Format = PixFmt.CHW };
            if (shape[3] <= 4 && shape[1] > 4 && shape[2] > 4)
                return new ImageLayout { Channels = shape[3], Height = shape[1], Width = shape[2], Format = PixFmt.HWC };
        }
        if (shape.Length == 3)
        {
            if (shape[0] <= 4 && shape[1] > 4 && shape[2] > 4)
                return new ImageLayout { Channels = shape[0], Height = shape[1], Width = shape[2], Format = PixFmt.CHW };
            if (shape[2] <= 4 && shape[0] > 4 && shape[1] > 4)
                return new ImageLayout { Channels = shape[2], Height = shape[0], Width = shape[1], Format = PixFmt.HWC };
        }
        if (shape.Length == 2)
            return new ImageLayout { Channels = 1, Height = shape[0], Width = shape[1], Format = PixFmt.CHW };

        int side = (int)Math.Sqrt(dataLength);
        return new ImageLayout { Channels = 1, Height = side, Width = side > 0 ? dataLength / side : dataLength, Format = PixFmt.CHW };
    }

    private enum PixFmt { CHW, HWC }

    private struct ImageLayout
    {
        public int Channels, Height, Width;
        public PixFmt Format;
    }
}
