using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Detects deepfake/AI-generated images by checking spatial consistency and natural image
/// statistics violations.
/// </summary>
/// <remarks>
/// <para>
/// Real images follow natural image statistics (NIS) — predictable relationships between
/// neighboring pixels, consistent noise patterns, and regular edge profiles. AI-generated
/// images often violate these statistics: inconsistent noise levels across regions, unnatural
/// symmetry patterns, and edge artifacts. This detector measures multiple NIS features and
/// flags images with anomalous combinations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real photos have certain statistical patterns — for example, noise
/// is usually consistent across the image, and edges follow natural gradients. AI-generated
/// images often have subtly wrong noise patterns or edges that look slightly "off". This
/// detector checks for those inconsistencies.
/// </para>
/// <para>
/// <b>References:</b>
/// - NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024, arxiv:2407.10550)
/// - Spatio-temporal consistency exploitation for deepfake detection (2025, arxiv:2502.08216)
/// - Rich and poor texture analysis for deepfake detection (2023)
/// - Generalizable deepfake detection (CVPR 2025, arxiv:2508.06248)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ConsistencyDeepfakeDetector<T> : ImageSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;
    private readonly int _gridSize;

    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;
    private static readonly T TwoFiftyFive = NumOps.FromDouble(255.0);

    /// <inheritdoc />
    public override string ModuleName => "ConsistencyDeepfakeDetector";

    /// <summary>
    /// Initializes a new consistency-based deepfake detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    /// <param name="gridSize">Grid size for region-level analysis. Default: 4.</param>
    public ConsistencyDeepfakeDetector(double threshold = 0.5, int gridSize = 4)
    {
        _threshold = threshold;
        _gridSize = gridSize;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var span = image.Data.Span;
        if (span.Length == 0) return findings;

        var layout = DetermineLayout(image.Shape, span.Length);
        if (layout.Height < 16 || layout.Width < 16) return findings;

        // 1. Noise inconsistency: different regions should have similar noise levels
        double noiseInconsistency = ComputeNoiseInconsistency(span, layout);

        // 2. Edge coherence: edges should have consistent profiles
        double edgeAnomaly = ComputeEdgeAnomaly(span, layout);

        // 3. Symmetry anomaly: AI images often have unnatural symmetry
        double symmetryAnomaly = ComputeSymmetryAnomaly(span, layout);

        // 4. Color consistency: smooth color transitions vs. abrupt changes
        double colorAnomaly = ComputeColorConsistencyAnomaly(span, layout);

        // Combined score
        double finalScore = 0.30 * noiseInconsistency +
                           0.25 * edgeAnomaly +
                           0.20 * symmetryAnomaly +
                           0.25 * colorAnomaly;

        if (finalScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Spatial consistency analysis: potential deepfake/AI-generated image " +
                              $"(score: {finalScore:F3}). Noise inconsistency: {noiseInconsistency:F3}, " +
                              $"edge anomaly: {edgeAnomaly:F3}, symmetry: {symmetryAnomaly:F3}, " +
                              $"color consistency: {colorAnomaly:F3}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>
    /// Measures noise level inconsistency across different image regions.
    /// Real images have relatively uniform noise; deepfakes often have regions with
    /// very different noise levels due to blending artifacts.
    /// </summary>
    private double ComputeNoiseInconsistency(ReadOnlySpan<T> data, ImageLayout layout)
    {
        int rows = _gridSize;
        int cols = _gridSize;
        int cellH = layout.Height / rows;
        int cellW = layout.Width / cols;

        if (cellH < 4 || cellW < 4) return 0;

        var noiseLevels = new List<double>();

        for (int gr = 0; gr < rows; gr++)
        {
            for (int gc = 0; gc < cols; gc++)
            {
                int startY = gr * cellH;
                int startX = gc * cellW;

                // Estimate noise as standard deviation of Laplacian (high-pass filter)
                double laplacianSum = 0;
                double laplacianSumSq = 0;
                int count = 0;

                for (int y = startY + 1; y < startY + cellH - 1 && y < layout.Height - 1; y++)
                {
                    for (int x = startX + 1; x < startX + cellW - 1 && x < layout.Width - 1; x++)
                    {
                        double center = GetVal(data, layout, y, x);
                        double up = GetVal(data, layout, y - 1, x);
                        double down = GetVal(data, layout, y + 1, x);
                        double left = GetVal(data, layout, y, x - 1);
                        double right = GetVal(data, layout, y, x + 1);

                        double laplacian = 4 * center - up - down - left - right;
                        laplacianSum += laplacian;
                        laplacianSumSq += laplacian * laplacian;
                        count++;
                    }
                }

                if (count > 1)
                {
                    double mean = laplacianSum / count;
                    double variance = (laplacianSumSq / count) - (mean * mean);
                    noiseLevels.Add(variance > 0 ? Math.Sqrt(variance) : 0);
                }
            }
        }

        if (noiseLevels.Count < 2) return 0;

        // Coefficient of variation of noise levels
        double noiseMean = 0;
        foreach (var n in noiseLevels) noiseMean += n;
        noiseMean /= noiseLevels.Count;

        double noiseVar = 0;
        foreach (var n in noiseLevels)
        {
            double diff = n - noiseMean;
            noiseVar += diff * diff;
        }
        noiseVar /= noiseLevels.Count;

        double cv = noiseMean > 1e-10 ? Math.Sqrt(noiseVar) / noiseMean : 0;

        // Natural images: CV < 0.3; deepfakes: CV > 0.5 typical
        return Math.Min(1.0, Math.Max(0, (cv - 0.3) / 0.5));
    }

    /// <summary>
    /// Detects unnatural edge profiles. AI-generated edges often have
    /// different gradient distributions than natural edges.
    /// </summary>
    private double ComputeEdgeAnomaly(ReadOnlySpan<T> data, ImageLayout layout)
    {
        // Compute gradient magnitude distribution
        var gradMagnitudes = new List<double>();
        int stride = Math.Max(1, layout.Height / 64); // Sample for efficiency

        for (int y = 1; y < layout.Height - 1; y += stride)
        {
            for (int x = 1; x < layout.Width - 1; x += stride)
            {
                double gx = GetVal(data, layout, y, x + 1) - GetVal(data, layout, y, x - 1);
                double gy = GetVal(data, layout, y + 1, x) - GetVal(data, layout, y - 1, x);
                double mag = Math.Sqrt(gx * gx + gy * gy);
                gradMagnitudes.Add(mag);
            }
        }

        if (gradMagnitudes.Count < 10) return 0;

        // Sort for percentile analysis
        gradMagnitudes.Sort();
        int n = gradMagnitudes.Count;

        // Natural images follow a heavy-tailed gradient distribution
        // AI images often have more uniform gradient magnitudes
        double p50 = gradMagnitudes[n / 2];
        double p90 = gradMagnitudes[n * 9 / 10];
        double p99 = gradMagnitudes[Math.Min(n - 1, n * 99 / 100)];

        // Kurtosis-like measure: ratio of extreme to median gradients
        double tailRatio = p50 > 1e-10 ? p99 / p50 : 0;

        // Natural images: tail ratio > 5; AI: often 2-4
        // Lower tail ratio = more uniform = more suspicious
        double anomaly = tailRatio < 5 ? (5 - tailRatio) / 5 : 0;

        // Also check gradient distribution smoothness
        double p10 = gradMagnitudes[n / 10];
        double spread = p50 > 1e-10 ? (p90 - p10) / p50 : 0;

        // Narrow spread is suspicious
        double spreadAnomaly = spread < 2 ? (2 - spread) / 2 : 0;

        return Math.Min(1.0, 0.6 * anomaly + 0.4 * spreadAnomaly);
    }

    /// <summary>
    /// Detects unnatural bilateral symmetry, which is common in AI-generated faces.
    /// </summary>
    private double ComputeSymmetryAnomaly(ReadOnlySpan<T> data, ImageLayout layout)
    {
        int centerX = layout.Width / 2;
        int maxOffset = Math.Min(centerX, layout.Width - centerX);
        int stride = Math.Max(1, layout.Height / 32);

        double symmetrySum = 0;
        int count = 0;

        for (int y = 0; y < layout.Height; y += stride)
        {
            for (int offset = 1; offset < maxOffset; offset += 2)
            {
                double left = GetVal(data, layout, y, centerX - offset);
                double right = GetVal(data, layout, y, centerX + offset);
                double diff = Math.Abs(left - right);
                double avg = (Math.Abs(left) + Math.Abs(right)) / 2;

                symmetrySum += avg > 1e-10 ? 1.0 - (diff / avg) : 1.0;
                count++;
            }
        }

        if (count == 0) return 0;

        double symmetryScore = symmetrySum / count;

        // Natural images: symmetry ~0.5-0.7; AI faces: symmetry > 0.85
        return Math.Min(1.0, Math.Max(0, (symmetryScore - 0.7) / 0.25));
    }

    /// <summary>
    /// Detects color transition anomalies. AI images sometimes have subtle banding or
    /// unnaturally smooth gradients.
    /// </summary>
    private double ComputeColorConsistencyAnomaly(ReadOnlySpan<T> data, ImageLayout layout)
    {
        if (layout.Channels < 3) return 0;

        // Check cross-channel correlation consistency across regions
        int rows = _gridSize;
        int cols = _gridSize;
        int cellH = layout.Height / rows;
        int cellW = layout.Width / cols;

        if (cellH < 4 || cellW < 4) return 0;

        var rgCorrelations = new List<double>();

        for (int gr = 0; gr < rows; gr++)
        {
            for (int gc = 0; gc < cols; gc++)
            {
                int startY = gr * cellH;
                int startX = gc * cellW;

                double rSum = 0, gSum = 0, rgSum = 0, r2Sum = 0, g2Sum = 0;
                int count = 0;

                for (int y = startY; y < startY + cellH && y < layout.Height; y += 2)
                {
                    for (int x = startX; x < startX + cellW && x < layout.Width; x += 2)
                    {
                        double r = GetValChannel(data, layout, y, x, 0);
                        double g = GetValChannel(data, layout, y, x, 1);
                        rSum += r; gSum += g;
                        rgSum += r * g;
                        r2Sum += r * r; g2Sum += g * g;
                        count++;
                    }
                }

                if (count > 2)
                {
                    double rMean = rSum / count;
                    double gMean = gSum / count;
                    double cov = rgSum / count - rMean * gMean;
                    double rVar = r2Sum / count - rMean * rMean;
                    double gVar = g2Sum / count - gMean * gMean;
                    double denom = Math.Sqrt(rVar * gVar);
                    double corr = denom > 1e-10 ? cov / denom : 0;
                    rgCorrelations.Add(corr);
                }
            }
        }

        if (rgCorrelations.Count < 2) return 0;

        // Coefficient of variation of cross-channel correlations
        double corrMean = 0;
        foreach (var c in rgCorrelations) corrMean += c;
        corrMean /= rgCorrelations.Count;

        double corrVar = 0;
        foreach (var c in rgCorrelations)
        {
            double diff = c - corrMean;
            corrVar += diff * diff;
        }
        corrVar /= rgCorrelations.Count;

        double corrCV = Math.Abs(corrMean) > 1e-10 ? Math.Sqrt(corrVar) / Math.Abs(corrMean) : 0;

        // High variation in cross-channel correlation = potential manipulation
        return Math.Min(1.0, Math.Max(0, (corrCV - 0.2) / 0.6));
    }

    private double GetVal(ReadOnlySpan<T> data, ImageLayout layout, int y, int x)
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
