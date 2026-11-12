using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Colour-heuristic image safety classifier: flags possible NSFW content from the fraction of skin-coloured pixels and
/// possible violence from red-channel dominance and dark/bright contrast. Despite its name it does not use CLIP.
/// </summary>
/// <remarks>
/// <para>
/// The NSFW score combines the fraction of pixels inside the RGB skin-colour cluster of Kovac, Peer and Solina (2003)
/// (r &gt; 95, g &gt; 40, b &gt; 20, r &gt; g, r &gt; b, |r - g| &gt; 15, here with extra HSV bounds) with how
/// contiguous those pixels are; the violence score combines the fraction of red-dominant pixels with high-contrast
/// features. These are fixed hand-set rules, not a learned model: skin tones,
/// sunsets, wood and red objects trigger them, and they cannot tell nudity from portraits. Treat a finding as a weak
/// signal for review, not a classification. A CLIP-based check that compares image and concept embeddings is
/// available through <c>TextImageAlignmentChecker</c> when a CLIP model (for example <c>ClipNeuralNetwork</c>) is
/// supplied.
/// </para>
/// <para>
/// <b>For Beginners:</b> This classifier examines the colors in an image to estimate whether it might contain
/// inappropriate content. It looks at how much skin-colored area is present (for NSFW) and how much red/dark contrast
/// exists (for violence). It is a quick colour check, not an AI model that understands the picture.
/// </para>
/// <para>
/// <b>How it works:</b>
/// 1. Extract per-channel statistics from the image tensor using generic numeric operations
/// 2. For NSFW: compute the skin-tone pixel fraction (RGB skin rule with HSV bounds) and its clustering
/// 3. For violence: compute red-channel dominance ratio and high-contrast features
/// 4. Combine features into a weighted score and compare against thresholds
/// </para>
/// <para>
/// <b>References:</b>
/// - Kovac, Peer, Solina. Human Skin Colour Clustering for Face Detection (EUROCON 2003) — the skin rule
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Human Skin Colour Clustering for Face Detection",
    "https://doi.org/10.1109/EURCON.2003.1248169",
    Year = 2003,
    Authors = "Jure Kovac, Peter Peer, Franc Solina")]
internal class CLIPImageSafetyClassifier<T> : ImageSafetyModuleBase<T>
{

    private readonly T _nsfwThreshold;
    private readonly T _violenceThreshold;
    private readonly bool _detectNSFW;
    private readonly bool _detectViolence;

    // Pre-computed constants in generic T
    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;
    private static readonly T TwoFiftyFive = NumOps.FromDouble(255.0);

    /// <inheritdoc />
    public override string ModuleName => "CLIPImageSafetyClassifier";

    /// <summary>
    /// Initializes a new image safety classifier.
    /// </summary>
    /// <param name="nsfwThreshold">
    /// Skin-colour score threshold for NSFW findings (0-1). Default: 0.8.
    /// </param>
    /// <param name="violenceThreshold">
    /// Red-dominance and contrast score threshold for violence findings (0-1). Default: 0.75.
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

        if (image.Length == 0)
        {
            return findings;
        }

        using var noGrad = new NoGradScope<T>();
        int[] shape = image._shape;
        var layout = DetermineImageLayout(shape, image.Length);
        var data = Engine.Reshape(image.Contiguous(), new[] { image.Length });

        if (_detectNSFW)
        {
            T nsfwScore = ComputeNSFWScore(data, layout);
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
            T violenceScore = ComputeViolenceScore(data, layout);
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
    private T ComputeNSFWScore(Tensor<T> data, ImageLayout layout)
    {
        if (layout.Channels < 3)
        {
            return Zero; // Grayscale — can't do skin-tone analysis
        }

        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0) return Zero;

        T totalCount = NumOps.FromDouble(totalPixels);
        var (r, g, b) = GetRgbPlanes(data, layout);
        var skin = SkinMask(r, g, b);

        // Skin fraction as the primary NSFW indicator
        T skinFraction = NumOps.Divide(Engine.TensorSum(skin), totalCount);

        // Compute skin clustering factor (contiguous skin = higher risk)
        T clusterFactor = ComputeSkinClustering(skin, layout);

        // Weighted: 60% skin fraction + 40% clustering
        T w1 = NumOps.FromDouble(0.6);
        T w2 = NumOps.FromDouble(0.4);
        T score = NumOps.Add(NumOps.Multiply(w1, skinFraction), NumOps.Multiply(w2, clusterFactor));

        return Clamp01(score);
    }

    /// <summary>
    /// Computes violence score using red-channel dominance and high-contrast features.
    /// </summary>
    private T ComputeViolenceScore(Tensor<T> data, ImageLayout layout)
    {
        if (layout.Channels < 3)
        {
            return ComputeGrayscaleViolenceScore(data, layout);
        }

        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0) return Zero;

        var engine = Engine;
        T totalCount = NumOps.FromDouble(totalPixels);
        var (r, g, b) = GetRgbPlanes(data, layout);

        T redSum = engine.TensorSum(r);
        T greenSum = engine.TensorSum(g);
        T blueSum = engine.TensorSum(b);

        // Dark red detection (blood-like: high R, low G, low B)
        T onePointFive = NumOps.FromDouble(1.5);
        var darkRed = SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.And(
                SafetyTensorOps<T>.And(SafetyTensorOps<T>.Gt(r, 100), SafetyTensorOps<T>.Lt(g, 80)),
                SafetyTensorOps<T>.Lt(b, 80)),
            SafetyTensorOps<T>.And(
                engine.TensorGreaterThan(r, engine.TensorMultiplyScalar(g, onePointFive)),
                engine.TensorGreaterThan(r, engine.TensorMultiplyScalar(b, onePointFive))));
        T darkRedCount = engine.TensorSum(darkRed);

        // Luminance contrast between consecutive pixels in raster order
        var luminance = engine.TensorAdd(engine.TensorAdd(
            engine.TensorMultiplyScalar(r, NumOps.FromDouble(0.299)),
            engine.TensorMultiplyScalar(g, NumOps.FromDouble(0.587))),
            engine.TensorMultiplyScalar(b, NumOps.FromDouble(0.114)));
        T highContrastCount = CountContrastSteps(engine.Reshape(luminance, new[] { totalPixels }), 80);

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

    /// <summary>
    /// RGB-space skin detection (Peer et al., 2003) with an HSV bounds check, per pixel: 1 for
    /// skin. Each rejection test is applied exactly as the scalar test is written, so a NaN
    /// channel fails the same comparisons it failed there.
    /// </summary>
    private Tensor<T> SkinMask(Tensor<T> r, Tensor<T> g, Tensor<T> b)
    {
        var engine = Engine;

        var maxRgb = SafetyTensorOps<T>.Where(engine.TensorGreaterThan(g, r), g, r);
        maxRgb = SafetyTensorOps<T>.Where(engine.TensorGreaterThan(b, maxRgb), b, maxRgb);
        var minRgb = SafetyTensorOps<T>.Where(engine.TensorLessThan(g, r), g, r);
        minRgb = SafetyTensorOps<T>.Where(engine.TensorLessThan(b, minRgb), b, minRgb);

        // HSV conversion
        var delta = engine.TensorSubtract(maxRgb, minRgb);
        var positiveMax = SafetyTensorOps<T>.Gt(maxRgb, 0.0);
        var s = SafetyTensorOps<T>.Where(positiveMax,
            engine.TensorDivide(delta, SafetyTensorOps<T>.Where(positiveMax, maxRgb, 1.0)), 0.0);
        var v = engine.TensorDivideScalar(maxRgb, NumOps.FromDouble(255.0));

        // Compute hue (simplified — R is max for skin tones)
        var h = engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(g, b), delta),
            NumOps.FromDouble(60.0));
        h = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Lt(h, 0.0),
            engine.TensorAddScalar(h, NumOps.FromDouble(360.0)), h);

        var rejected = SafetyTensorOps<T>.Le(r, 95);
        rejected = SafetyTensorOps<T>.Or(rejected, SafetyTensorOps<T>.Le(g, 40));
        rejected = SafetyTensorOps<T>.Or(rejected, SafetyTensorOps<T>.Le(b, 20));
        rejected = SafetyTensorOps<T>.Or(rejected, SafetyTensorOps<T>.Le(delta, 15));
        rejected = SafetyTensorOps<T>.Or(rejected,
            SafetyTensorOps<T>.Or(engine.TensorLessEqual(r, g), engine.TensorLessEqual(r, b)));
        rejected = SafetyTensorOps<T>.Or(rejected, SafetyTensorOps<T>.Gt(h, 50));
        rejected = SafetyTensorOps<T>.Or(rejected,
            SafetyTensorOps<T>.Or(SafetyTensorOps<T>.Lt(s, 0.1), SafetyTensorOps<T>.Gt(s, 0.8)));
        rejected = SafetyTensorOps<T>.Or(rejected,
            SafetyTensorOps<T>.Or(SafetyTensorOps<T>.Lt(v, 0.2), SafetyTensorOps<T>.Gt(v, 0.95)));

        return SafetyTensorOps<T>.Not(rejected);
    }

    private T ComputeSkinClustering(Tensor<T> skin, ImageLayout layout)
    {
        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0 || layout.Height < 4 || layout.Width < 4) return Zero;

        int gridRows = 4, gridCols = 4;
        int cellH = layout.Height / gridRows;
        int cellW = layout.Width / gridCols;
        int cellPixels = cellH * cellW;
        if (cellPixels == 0) return Zero;

        // Skin count of every grid cell: [gridRows, cellH, gridCols, cellW] summed over the cell axes.
        var engine = Engine;
        var grid = engine.TensorSlice(skin, new[] { 0, 0 }, new[] { gridRows * cellH, gridCols * cellW });
        var cells = engine.ReduceSum(
            engine.Reshape(grid.Contiguous(), new[] { gridRows, cellH, gridCols, cellW }), new[] { 1, 3 }, false);
        var cellFraction = engine.TensorDivideScalar(cells, NumOps.FromDouble(cellPixels));
        T highSkinCells = engine.TensorSum(SafetyTensorOps<T>.Gt(cellFraction, 0.5));

        return NumOps.Divide(highSkinCells, NumOps.FromDouble(gridRows * gridCols));
    }

    private T ComputeGrayscaleViolenceScore(Tensor<T> data, ImageLayout layout)
    {
        int totalPixels = layout.Height * layout.Width;
        if (totalPixels == 0) return Zero;

        var engine = Engine;
        T totalCount = NumOps.FromDouble(data.Length);

        var val = SafetyTensorOps<T>.Where(engine.TensorLessEqual(data, SafetyTensorOps<T>.FullLike(data, 1.0)),
            engine.TensorMultiplyScalar(data, TwoFiftyFive), data);
        T veryDarkCount = engine.TensorSum(SafetyTensorOps<T>.Lt(val, 30));
        T highContrastCount = CountContrastSteps(val, 80);

        T five = NumOps.FromDouble(5.0);
        T three = NumOps.FromDouble(3.0);
        T contrastScore = Clamp01(NumOps.Multiply(NumOps.Divide(highContrastCount, totalCount), five));
        T darkScore = Clamp01(NumOps.Multiply(NumOps.Divide(veryDarkCount, totalCount), three));

        T w1 = NumOps.FromDouble(0.5);
        T w2 = NumOps.FromDouble(0.5);
        return Clamp01(NumOps.Add(NumOps.Multiply(w1, contrastScore), NumOps.Multiply(w2, darkScore)));
    }

    /// <summary>Number of consecutive-element steps <c>|x[i] - x[i-1]| &gt; threshold</c>.</summary>
    private T CountContrastSteps(Tensor<T> values, double threshold)
    {
        if (values.Length < 2) return Zero;
        var engine = Engine;
        return engine.TensorSum(SafetyTensorOps<T>.Gt(engine.TensorAbs(engine.TensorDiff(values, 1, 0)), threshold));
    }

    /// <summary>
    /// The R, G and B planes (<c>[height, width]</c>) of the image, each scaled from [0, 1] to
    /// [0, 255] where all three channels of a pixel are at most 1.
    /// </summary>
    private (Tensor<T> R, Tensor<T> G, Tensor<T> B) GetRgbPlanes(Tensor<T> data, ImageLayout layout)
    {
        var engine = Engine;
        int pixels = layout.Height * layout.Width;
        var planeShape = new[] { layout.Height, layout.Width };
        var planes = new Tensor<T>[3];
        if (layout.Format == PixelFormat.CHW)
        {
            for (int c = 0; c < 3; c++)
                planes[c] = engine.Reshape(engine.TensorNarrow(data, 0, layout.ChannelOffset + c * pixels, pixels), planeShape);
        }
        else
        {
            var interleaved = engine.Reshape(
                engine.TensorNarrow(data, 0, layout.ChannelOffset, pixels * layout.Channels),
                new[] { pixels, layout.Channels });
            for (int c = 0; c < 3; c++)
                planes[c] = engine.Reshape(engine.TensorNarrow(interleaved, 1, c, 1).Contiguous(), planeShape);
        }

        var r = planes[0];
        var g = planes[1];
        var b = planes[2];

        // Normalize to [0,255] if data is in [0,1] range
        var normalized = SafetyTensorOps<T>.And(SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.Le(r, 1.0), SafetyTensorOps<T>.Le(g, 1.0)), SafetyTensorOps<T>.Le(b, 1.0));
        r = SafetyTensorOps<T>.Where(normalized, engine.TensorMultiplyScalar(r, TwoFiftyFive), r);
        g = SafetyTensorOps<T>.Where(normalized, engine.TensorMultiplyScalar(g, TwoFiftyFive), g);
        b = SafetyTensorOps<T>.Where(normalized, engine.TensorMultiplyScalar(b, TwoFiftyFive), b);
        return (r, g, b);
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
