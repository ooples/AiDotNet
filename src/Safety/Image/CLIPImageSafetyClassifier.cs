using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Image safety classifier that uses color histogram analysis and statistical features
/// for content classification across NSFW and violence categories.
/// </summary>
/// <remarks>
/// <para>
/// Analyzes images by decomposing pixel data into per-channel statistics and computing
/// features that correlate with unsafe content categories. Uses skin-tone color distribution
/// analysis for NSFW detection and red-channel dominance with high-contrast analysis for
/// violence detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> This classifier examines the colors and patterns in an image to
/// estimate whether it contains inappropriate content. It looks at things like how much
/// skin-colored area is present (for NSFW) or how much red/dark contrast exists (for violence).
/// </para>
/// <para>
/// <b>How it works:</b>
/// 1. Extract per-channel statistics from the image tensor using generic numeric operations
/// 2. For NSFW: compute skin-tone pixel fraction using HSV-space color ranges
/// 3. For violence: compute red-channel dominance ratio and high-contrast features
/// 4. Combine features into a weighted score and compare against thresholds
/// </para>
/// <para>
/// <b>References:</b>
/// - Safe-CLIP: Removing NSFW concepts from CLIP representations (2024, arxiv:2311.16254)
/// - UnsafeBench: 11 categories, GPT-4V achieves top F1 (Qu et al., 2024, arxiv:2405.03486)
/// - DiffGuard: Text-based safety checker for diffusion models (2024, arxiv:2412.00064)
/// - Skin detection using HSV color space (Kovac et al., 2003)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CLIPImageSafetyClassifier<T> : ImageSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _nsfwThreshold;
    private readonly T _violenceThreshold;
    private readonly bool _detectNSFW;
    private readonly bool _detectViolence;

    // Pre-computed constants in generic T
    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;
    private static readonly T Half = NumOps.FromDouble(0.5);
    private static readonly T TwoFiftyFive = NumOps.FromDouble(255.0);
    private static readonly T NormalizeThreshold = One; // Pixels <= 1.0 need scaling

    /// <inheritdoc />
    public override string ModuleName => "CLIPImageSafetyClassifier";

    /// <summary>
    /// Initializes a new image safety classifier.
    /// </summary>
    /// <param name="nsfwThreshold">
    /// Cosine similarity threshold for NSFW detection (0-1). Default: 0.8.
    /// </param>
    /// <param name="violenceThreshold">
    /// Cosine similarity threshold for violence detection (0-1). Default: 0.75.
    /// </param>
    /// <param name="detectNSFW">Whether to detect NSFW content. Default: true.</param>
    /// <param name="detectViolence">Whether to detect violent content. Default: true.</param>
    public CLIPImageSafetyClassifier(
        double nsfwThreshold = 0.8,
        double violenceThreshold = 0.75,
        bool detectNSFW = true,
        bool detectViolence = true)
    {
        if (nsfwThreshold < 0 || nsfwThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nsfwThreshold),
                "NSFW threshold must be between 0 and 1.");
        }

        if (violenceThreshold < 0 || violenceThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(violenceThreshold),
                "Violence threshold must be between 0 and 1.");
        }

        _nsfwThreshold = NumOps.FromDouble(nsfwThreshold);
        _violenceThreshold = NumOps.FromDouble(violenceThreshold);
        _detectNSFW = detectNSFW;
        _detectViolence = detectViolence;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();

        var span = image.Data.Span;
        if (span.Length == 0)
        {
            return findings;
        }

        int[] shape = image.Shape;
        var layout = DetermineImageLayout(shape, span.Length);

        if (_detectNSFW)
        {
            T nsfwScore = ComputeNSFWScore(span, layout);
            if (NumOps.GreaterThanOrEquals(nsfwScore, _nsfwThreshold))
            {
                double scoreDouble = NumOps.ToDouble(nsfwScore);
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.SexualExplicit,
                    Severity = SafetySeverity.High,
                    Confidence = scoreDouble,
                    Description = $"Image flagged for potential NSFW content (score: {scoreDouble:F3}). " +
                                  $"Skin-tone pixel fraction: {scoreDouble:P0}.",
                    RecommendedAction = SafetyAction.Block,
                    SourceModule = ModuleName
                });
            }
        }

        if (_detectViolence)
        {
            T violenceScore = ComputeViolenceScore(span, layout);
            if (NumOps.GreaterThanOrEquals(violenceScore, _violenceThreshold))
            {
                double scoreDouble = NumOps.ToDouble(violenceScore);
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.ViolenceGraphic,
                    Severity = SafetySeverity.High,
                    Confidence = scoreDouble,
                    Description = $"Image flagged for potential violent content (score: {scoreDouble:F3}). " +
                                  "Red-channel dominance and high-contrast features detected.",
                    RecommendedAction = SafetyAction.Block,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    /// <summary>
    /// Computes NSFW score using skin-tone color distribution analysis in HSV space.
    /// </summary>
    private T ComputeNSFWScore(ReadOnlySpan<T> data, ImageLayout layout)
    {
        if (layout.Channels < 3)
        {
            return Zero; // Grayscale — can't do skin-tone analysis
        }

        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0) return Zero;

        T skinCount = Zero;
        T totalCount = NumOps.FromDouble(totalPixels);

        // Skin-tone thresholds in T
        T rMin = NumOps.FromDouble(95);
        T gMin = NumOps.FromDouble(40);
        T bMin = NumOps.FromDouble(20);
        T minDiffThreshold = NumOps.FromDouble(15);
        T hMax = NumOps.FromDouble(50);
        T sMin = NumOps.FromDouble(0.1);
        T sMax = NumOps.FromDouble(0.8);
        T vMin = NumOps.FromDouble(0.2);
        T vMax = NumOps.FromDouble(0.95);
        T onePointFive = NumOps.FromDouble(1.5);

        for (int y = 0; y < layout.Height; y++)
        {
            for (int x = 0; x < layout.Width; x++)
            {
                T r = GetPixelChannel(data, layout, y, x, 0);
                T g = GetPixelChannel(data, layout, y, x, 1);
                T b = GetPixelChannel(data, layout, y, x, 2);

                // Normalize to [0,255] if data is in [0,1] range
                if (NumOps.LessThanOrEquals(r, NormalizeThreshold) &&
                    NumOps.LessThanOrEquals(g, NormalizeThreshold) &&
                    NumOps.LessThanOrEquals(b, NormalizeThreshold))
                {
                    r = NumOps.Multiply(r, TwoFiftyFive);
                    g = NumOps.Multiply(g, TwoFiftyFive);
                    b = NumOps.Multiply(b, TwoFiftyFive);
                }

                if (IsSkinPixel(r, g, b, rMin, gMin, bMin, minDiffThreshold,
                    hMax, sMin, sMax, vMin, vMax, onePointFive))
                {
                    skinCount = NumOps.Add(skinCount, One);
                }
            }
        }

        // Skin fraction as the primary NSFW indicator
        T skinFraction = NumOps.Divide(skinCount, totalCount);

        // Compute skin clustering factor (contiguous skin = higher risk)
        T clusterFactor = ComputeSkinClustering(data, layout);

        // Weighted: 60% skin fraction + 40% clustering
        T w1 = NumOps.FromDouble(0.6);
        T w2 = NumOps.FromDouble(0.4);
        T score = NumOps.Add(NumOps.Multiply(w1, skinFraction), NumOps.Multiply(w2, clusterFactor));

        return Clamp01(score);
    }

    /// <summary>
    /// Computes violence score using red-channel dominance and high-contrast features.
    /// </summary>
    private T ComputeViolenceScore(ReadOnlySpan<T> data, ImageLayout layout)
    {
        if (layout.Channels < 3)
        {
            return ComputeGrayscaleViolenceScore(data, layout);
        }

        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0) return Zero;

        T redSum = Zero, greenSum = Zero, blueSum = Zero;
        T darkRedCount = Zero;
        T highContrastCount = Zero;
        T prevLuminance = Zero;
        T totalCount = NumOps.FromDouble(totalPixels);

        T darkRedRThreshold = NumOps.FromDouble(100);
        T darkRedGBThreshold = NumOps.FromDouble(80);
        T contrastThreshold = NumOps.FromDouble(80);
        T lumR = NumOps.FromDouble(0.299);
        T lumG = NumOps.FromDouble(0.587);
        T lumB = NumOps.FromDouble(0.114);
        T onePointFive = NumOps.FromDouble(1.5);
        bool isFirst = true;

        for (int y = 0; y < layout.Height; y++)
        {
            for (int x = 0; x < layout.Width; x++)
            {
                T r = GetPixelChannel(data, layout, y, x, 0);
                T g = GetPixelChannel(data, layout, y, x, 1);
                T b = GetPixelChannel(data, layout, y, x, 2);

                if (NumOps.LessThanOrEquals(r, NormalizeThreshold) &&
                    NumOps.LessThanOrEquals(g, NormalizeThreshold) &&
                    NumOps.LessThanOrEquals(b, NormalizeThreshold))
                {
                    r = NumOps.Multiply(r, TwoFiftyFive);
                    g = NumOps.Multiply(g, TwoFiftyFive);
                    b = NumOps.Multiply(b, TwoFiftyFive);
                }

                redSum = NumOps.Add(redSum, r);
                greenSum = NumOps.Add(greenSum, g);
                blueSum = NumOps.Add(blueSum, b);

                // Dark red detection (blood-like: high R, low G, low B)
                if (NumOps.GreaterThan(r, darkRedRThreshold) &&
                    NumOps.LessThan(g, darkRedGBThreshold) &&
                    NumOps.LessThan(b, darkRedGBThreshold) &&
                    NumOps.GreaterThan(r, NumOps.Multiply(g, onePointFive)) &&
                    NumOps.GreaterThan(r, NumOps.Multiply(b, onePointFive)))
                {
                    darkRedCount = NumOps.Add(darkRedCount, One);
                }

                // Luminance contrast
                T luminance = NumOps.Add(NumOps.Add(
                    NumOps.Multiply(lumR, r),
                    NumOps.Multiply(lumG, g)),
                    NumOps.Multiply(lumB, b));

                if (!isFirst)
                {
                    T diff = NumOps.Abs(NumOps.Subtract(luminance, prevLuminance));
                    if (NumOps.GreaterThan(diff, contrastThreshold))
                    {
                        highContrastCount = NumOps.Add(highContrastCount, One);
                    }
                }

                prevLuminance = luminance;
                isFirst = false;
            }
        }

        // Red dominance ratio: (R / (R+G+B)) - 0.333, scaled
        T totalColor = NumOps.Add(NumOps.Add(redSum, greenSum), blueSum);
        T third = NumOps.FromDouble(0.333);
        T three = NumOps.FromDouble(3.0);
        T redRatio = NumOps.GreaterThan(totalColor, Zero)
            ? NumOps.Multiply(NumOps.Subtract(NumOps.Divide(redSum, totalColor), third), three)
            : Zero;
        redRatio = NumOps.GreaterThan(redRatio, Zero) ? redRatio : Zero;

        // Dark red fraction, scaled
        T five = NumOps.FromDouble(5.0);
        T darkRedFraction = NumOps.Multiply(NumOps.Divide(darkRedCount, totalCount), five);
        darkRedFraction = Clamp01(darkRedFraction);

        // High contrast fraction, scaled
        T contrastFraction = NumOps.Multiply(NumOps.Divide(highContrastCount, totalCount), three);
        contrastFraction = Clamp01(contrastFraction);

        // Weighted combination
        T w1 = NumOps.FromDouble(0.40);
        T w2 = NumOps.FromDouble(0.30);
        T w3 = NumOps.FromDouble(0.30);
        T score = NumOps.Add(NumOps.Add(
            NumOps.Multiply(w1, darkRedFraction),
            NumOps.Multiply(w2, Clamp01(redRatio))),
            NumOps.Multiply(w3, contrastFraction));

        return Clamp01(score);
    }

    private static bool IsSkinPixel(T r, T g, T b,
        T rMin, T gMin, T bMin, T minDiffThreshold,
        T hMax, T sMin, T sMax, T vMin, T vMax, T onePointFive)
    {
        // RGB-space skin detection (Peer et al., 2003)
        if (NumOps.LessThanOrEquals(r, rMin)) return false;
        if (NumOps.LessThanOrEquals(g, gMin)) return false;
        if (NumOps.LessThanOrEquals(b, bMin)) return false;

        T maxRgb = r;
        if (NumOps.GreaterThan(g, maxRgb)) maxRgb = g;
        if (NumOps.GreaterThan(b, maxRgb)) maxRgb = b;

        T minRgb = r;
        if (NumOps.LessThan(g, minRgb)) minRgb = g;
        if (NumOps.LessThan(b, minRgb)) minRgb = b;

        if (NumOps.LessThanOrEquals(NumOps.Subtract(maxRgb, minRgb), minDiffThreshold)) return false;
        if (NumOps.LessThanOrEquals(r, g) || NumOps.LessThanOrEquals(r, b)) return false;

        // HSV conversion and bounds check
        T delta = NumOps.Subtract(maxRgb, minRgb);
        T s = NumOps.GreaterThan(maxRgb, NumOps.Zero)
            ? NumOps.Divide(delta, maxRgb)
            : NumOps.Zero;
        T v = NumOps.Divide(maxRgb, NumOps.FromDouble(255.0));

        // Compute hue (simplified — R is max for skin tones)
        T sixty = NumOps.FromDouble(60.0);
        T h = NumOps.Multiply(sixty, NumOps.Divide(NumOps.Subtract(g, b), delta));
        if (NumOps.LessThan(h, NumOps.Zero))
        {
            h = NumOps.Add(h, NumOps.FromDouble(360.0));
        }

        if (NumOps.GreaterThan(h, hMax)) return false;
        if (NumOps.LessThan(s, sMin) || NumOps.GreaterThan(s, sMax)) return false;
        if (NumOps.LessThan(v, vMin) || NumOps.GreaterThan(v, vMax)) return false;

        return true;
    }

    private T ComputeSkinClustering(ReadOnlySpan<T> data, ImageLayout layout)
    {
        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0 || layout.Height < 4 || layout.Width < 4) return Zero;

        int gridRows = 4, gridCols = 4;
        int cellH = layout.Height / gridRows;
        int cellW = layout.Width / gridCols;
        int cellPixels = cellH * cellW;
        if (cellPixels == 0) return Zero;

        T rMin = NumOps.FromDouble(95);
        T gMin = NumOps.FromDouble(40);
        T bMin = NumOps.FromDouble(20);
        T minDiff = NumOps.FromDouble(15);
        T hMax = NumOps.FromDouble(50);
        T sMin = NumOps.FromDouble(0.1);
        T sMax = NumOps.FromDouble(0.8);
        T vMin = NumOps.FromDouble(0.2);
        T vMax = NumOps.FromDouble(0.95);
        T onePointFive = NumOps.FromDouble(1.5);
        T halfT = NumOps.FromDouble(0.5);

        int highSkinCells = 0;
        T cellPixelsT = NumOps.FromDouble(cellPixels);

        for (int gr = 0; gr < gridRows; gr++)
        {
            for (int gc = 0; gc < gridCols; gc++)
            {
                T cellSkinCount = Zero;
                int startY = gr * cellH;
                int startX = gc * cellW;

                for (int y = startY; y < startY + cellH && y < layout.Height; y++)
                {
                    for (int x = startX; x < startX + cellW && x < layout.Width; x++)
                    {
                        T r = GetPixelChannel(data, layout, y, x, 0);
                        T g = GetPixelChannel(data, layout, y, x, 1);
                        T b = GetPixelChannel(data, layout, y, x, 2);

                        if (NumOps.LessThanOrEquals(r, NormalizeThreshold) &&
                            NumOps.LessThanOrEquals(g, NormalizeThreshold) &&
                            NumOps.LessThanOrEquals(b, NormalizeThreshold))
                        {
                            r = NumOps.Multiply(r, TwoFiftyFive);
                            g = NumOps.Multiply(g, TwoFiftyFive);
                            b = NumOps.Multiply(b, TwoFiftyFive);
                        }

                        if (IsSkinPixel(r, g, b, rMin, gMin, bMin, minDiff,
                            hMax, sMin, sMax, vMin, vMax, onePointFive))
                        {
                            cellSkinCount = NumOps.Add(cellSkinCount, One);
                        }
                    }
                }

                T cellFraction = NumOps.Divide(cellSkinCount, cellPixelsT);
                if (NumOps.GreaterThan(cellFraction, halfT))
                {
                    highSkinCells++;
                }
            }
        }

        return NumOps.FromDouble((double)highSkinCells / (gridRows * gridCols));
    }

    private T ComputeGrayscaleViolenceScore(ReadOnlySpan<T> data, ImageLayout layout)
    {
        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0) return Zero;

        T contrastThreshold = NumOps.FromDouble(80);
        T darkThreshold = NumOps.FromDouble(30);
        T highContrastCount = Zero;
        T veryDarkCount = Zero;
        T prevVal = Zero;
        T totalCount = NumOps.FromDouble(data.Length);
        bool isFirst = true;

        for (int i = 0; i < data.Length; i++)
        {
            T val = data[i];
            if (NumOps.LessThanOrEquals(val, NormalizeThreshold))
            {
                val = NumOps.Multiply(val, TwoFiftyFive);
            }

            if (NumOps.LessThan(val, darkThreshold))
            {
                veryDarkCount = NumOps.Add(veryDarkCount, One);
            }

            if (!isFirst)
            {
                T diff = NumOps.Abs(NumOps.Subtract(val, prevVal));
                if (NumOps.GreaterThan(diff, contrastThreshold))
                {
                    highContrastCount = NumOps.Add(highContrastCount, One);
                }
            }

            prevVal = val;
            isFirst = false;
        }

        T five = NumOps.FromDouble(5.0);
        T three = NumOps.FromDouble(3.0);
        T contrastScore = Clamp01(NumOps.Multiply(NumOps.Divide(highContrastCount, totalCount), five));
        T darkScore = Clamp01(NumOps.Multiply(NumOps.Divide(veryDarkCount, totalCount), three));

        T w1 = NumOps.FromDouble(0.5);
        T w2 = NumOps.FromDouble(0.5);
        return Clamp01(NumOps.Add(NumOps.Multiply(w1, contrastScore), NumOps.Multiply(w2, darkScore)));
    }

    private T GetPixelChannel(ReadOnlySpan<T> data, ImageLayout layout, int y, int x, int c)
    {
        int idx;
        if (layout.Format == PixelFormat.CHW)
        {
            idx = layout.ChannelOffset + c * layout.Height * layout.Width + y * layout.Width + x;
        }
        else
        {
            idx = layout.ChannelOffset + (y * layout.Width + x) * layout.Channels + c;
        }

        if (idx < 0 || idx >= data.Length) return Zero;
        return data[idx];
    }

    private static T Clamp01(T value)
    {
        if (NumOps.LessThan(value, Zero)) return Zero;
        if (NumOps.GreaterThan(value, One)) return One;
        return value;
    }

    private static ImageLayout DetermineImageLayout(int[] shape, int dataLength)
    {
        if (shape.Length >= 4)
        {
            if (shape[1] <= 4 && shape[2] > 4 && shape[3] > 4)
            {
                return new ImageLayout
                {
                    Channels = shape[1], Height = shape[2], Width = shape[3],
                    Format = PixelFormat.CHW, ChannelOffset = 0
                };
            }

            if (shape[3] <= 4 && shape[1] > 4 && shape[2] > 4)
            {
                return new ImageLayout
                {
                    Channels = shape[3], Height = shape[1], Width = shape[2],
                    Format = PixelFormat.HWC, ChannelOffset = 0
                };
            }
        }

        if (shape.Length == 3)
        {
            if (shape[0] <= 4 && shape[1] > 4 && shape[2] > 4)
            {
                return new ImageLayout
                {
                    Channels = shape[0], Height = shape[1], Width = shape[2],
                    Format = PixelFormat.CHW, ChannelOffset = 0
                };
            }

            if (shape[2] <= 4 && shape[0] > 4 && shape[1] > 4)
            {
                return new ImageLayout
                {
                    Channels = shape[2], Height = shape[0], Width = shape[1],
                    Format = PixelFormat.HWC, ChannelOffset = 0
                };
            }
        }

        if (shape.Length == 2)
        {
            return new ImageLayout
            {
                Channels = 1, Height = shape[0], Width = shape[1],
                Format = PixelFormat.CHW, ChannelOffset = 0
            };
        }

        int side = (int)Math.Sqrt(dataLength);
        return new ImageLayout
        {
            Channels = 1, Height = side, Width = side > 0 ? dataLength / side : dataLength,
            Format = PixelFormat.CHW, ChannelOffset = 0
        };
    }

    private enum PixelFormat
    {
        CHW,
        HWC
    }

    private struct ImageLayout
    {
        public int Channels;
        public int Height;
        public int Width;
        public PixelFormat Format;
        public int ChannelOffset;
    }
}
