using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Patch-statistics image safety classifier: hand-set colour and edge rules evaluated per image patch and pooled with
/// fixed similarity weights. Despite its name it is not a Vision Transformer and has no learned weights.
/// </summary>
/// <remarks>
/// <para>
/// The image is cut into square patches, as a ViT tokenizes an image, but each patch is then summarised by fixed
/// statistics (channel means and spreads, the fraction of pixels passing an RGB skin-colour rule after Kovac, Peer and
/// Solina 2003, dark, bright and red-dominant fractions, and gradient energy). The patch summaries are pooled with
/// softmax weights given by their similarity to the mean patch, and each safety category is scored by a hand-written
/// rule on the pooled summary. Nothing is trained: the scores are heuristics that colour and texture alone can trigger,
/// and should be treated as weak signals for review rather than classifications.
/// </para>
/// <para>
/// <b>For Beginners:</b> this classifier cuts the image into small squares, measures simple colour and edge
/// statistics in each, and combines them with fixed rules. It catches some obviously skin-heavy or red-heavy images,
/// but it does not understand what the image shows the way a trained neural network would.
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
public class ViTImageSafetyClassifier<T> : ImageSafetyModuleBase<T>
{

    private readonly int _patchSize;
    private readonly int _featureDim;
    private readonly double _threshold;
    private readonly SafetyCategory[] _categories;

    private static readonly T TwoFiftyFive = NumOps.FromDouble(255.0);

    /// <inheritdoc />
    public override string ModuleName => "ViTImageSafetyClassifier";

    /// <summary>
    /// Initializes a new patch-statistics image safety classifier.
    /// </summary>
    /// <param name="patchSize">Patch size in pixels. Default: 16.</param>
    /// <param name="threshold">Classification threshold (0-1). Default: 0.7.</param>
    public ViTImageSafetyClassifier(int patchSize = 16, double threshold = 0.7)
    {
        _patchSize = patchSize;
        _featureDim = 32; // Feature dimension per patch
        _threshold = threshold;
        _categories = new[]
        {
            SafetyCategory.SexualExplicit,
            SafetyCategory.SexualSuggestive,
            SafetyCategory.ViolenceGraphic,
            SafetyCategory.ViolenceWeapons,
            SafetyCategory.ViolenceSelfHarm
        };
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        if (image.Length == 0) return findings;

        var layout = DetermineLayout(image._shape, image.Length);
        if (layout.Height < _patchSize || layout.Width < _patchSize) return findings;

        using var noGrad = new NoGradScope<T>();
        var engine = Engine;

        // Extract patch features: [numPatches, featureDim], patches in row-major grid order
        int patchRows = layout.Height / _patchSize;
        int patchCols = layout.Width / _patchSize;
        int numPatches = patchRows * patchCols;
        var patchFeatures = ExtractPatchFeatures(image, layout, patchRows, patchCols);

        // Compute attention weights (based on feature magnitude — "how interesting is this patch?")
        var attentionWeights = ComputeAttentionWeights(patchFeatures);

        // Attention-weighted pooling → single feature vector
        var globalFeature = engine.Reshape(
            engine.TensorMatMul(engine.Reshape(attentionWeights, new[] { 1, numPatches }), patchFeatures),
            new[] { _featureDim }).ToVector();

        // Classify against each safety category
        for (int c = 0; c < _categories.Length; c++)
        {
            double score = ComputeCategoryScore(globalFeature, _categories[c]);

            if (score >= _threshold)
            {
                findings.Add(new SafetyFinding
                {
                    Category = _categories[c],
                    Severity = score >= 0.9 ? SafetySeverity.High : SafetySeverity.Medium,
                    Confidence = score,
                    Description = $"Patch colour/edge heuristic: possible {_categories[c]} (score: {score:F3}). " +
                                  $"Analyzed {numPatches} patches of size {_patchSize}x{_patchSize}.",
                    RecommendedAction = score >= 0.9 ? SafetyAction.Block : SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    /// <summary>
    /// Features of every patch, <c>[patchRows * patchCols, featureDim]</c>: channel means and
    /// standard deviations (0-5), skin/dark/bright/red-dominant fractions (6-9) and edge density
    /// and strength (10-11); the remaining features are zero.
    /// </summary>
    private Tensor<T> ExtractPatchFeatures(Tensor<T> image, ImageLayout layout, int patchRows, int patchCols)
    {
        var engine = Engine;
        int p = _patchSize;
        int numPatches = patchRows * patchCols;
        int channels = Math.Min(layout.Channels, 3);
        T pixelCount = NumOps.FromDouble(_patchSize * _patchSize);

        // Patch-blocked planes [patchRows, p, patchCols, p] (channel c, cropped to whole patches)
        var flat = engine.Reshape(image.Contiguous(), new[] { image.Length });
        var blocked = new[] { patchRows, p, patchCols, p };
        var raw0 = engine.Reshape(CropPlane(flat, layout, 0, patchRows * p, patchCols * p), blocked);
        var r = raw0;
        var g = channels >= 2 ? engine.Reshape(CropPlane(flat, layout, 1, patchRows * p, patchCols * p), blocked) : raw0;
        var b = channels >= 3 ? engine.Reshape(CropPlane(flat, layout, 2, patchRows * p, patchCols * p), blocked) : raw0;

        // Normalize to [0,255] (decided by the red channel)
        var scaled = SafetyTensorOps<T>.Le(r, 1.0);
        r = SafetyTensorOps<T>.Where(scaled, engine.TensorMultiplyScalar(r, TwoFiftyFive), r);
        g = SafetyTensorOps<T>.Where(scaled, engine.TensorMultiplyScalar(g, TwoFiftyFive), g);
        b = SafetyTensorOps<T>.Where(scaled, engine.TensorMultiplyScalar(b, TwoFiftyFive), b);

        var patchAxes = new[] { 1, 3 };
        var featureList = new List<Tensor<T>>();

        // Features 0-5: channel means and stdevs (normalized to [0,1])
        foreach (var plane in new[] { r, g, b })
        {
            var mean = engine.TensorDivideScalar(engine.ReduceSum(plane, patchAxes, false), pixelCount);
            var meanSq = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(plane), patchAxes, false), pixelCount);
            var variance = engine.TensorSubtract(meanSq, engine.TensorSquare(mean));
            var positive = SafetyTensorOps<T>.Gt(variance, 0.0);
            var stddev = SafetyTensorOps<T>.Where(positive,
                engine.TensorSqrt(SafetyTensorOps<T>.Where(positive, variance, 0.0)), 0.0);
            featureList.Add(engine.TensorDivideScalar(mean, TwoFiftyFive));
            featureList.Add(engine.TensorDivideScalar(stddev, TwoFiftyFive));
        }

        // Features 6-9: semantic ratios
        var skin = SafetyTensorOps<T>.And(SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.Gt(r, 95), engine.TensorGreaterThan(r, g)), engine.TensorGreaterThan(g, b));
        var lum = engine.TensorAdd(engine.TensorAdd(
            engine.TensorMultiplyScalar(r, NumOps.FromDouble(0.299)),
            engine.TensorMultiplyScalar(g, NumOps.FromDouble(0.587))),
            engine.TensorMultiplyScalar(b, NumOps.FromDouble(0.114)));
        T onePointFive = NumOps.FromDouble(1.5);
        var redDominant = SafetyTensorOps<T>.And(
            engine.TensorGreaterThan(r, engine.TensorMultiplyScalar(g, onePointFive)),
            engine.TensorGreaterThan(r, engine.TensorMultiplyScalar(b, onePointFive)));
        foreach (var mask in new[] { skin, SafetyTensorOps<T>.Lt(lum, 40), SafetyTensorOps<T>.Gt(lum, 220), redDominant })
            featureList.Add(engine.TensorDivideScalar(engine.ReduceSum(mask, patchAxes, false), pixelCount));

        // Features 10-11: edge statistics via horizontal/vertical gradients of the raw first
        // channel, over the patch interior positions that have a right and a lower neighbour
        var inner = new[] { patchRows, p - 1, patchCols, p - 1 };
        var curr = engine.TensorSlice(raw0, new[] { 0, 0, 0, 0 }, inner);
        var right = engine.TensorSlice(raw0, new[] { 0, 0, 0, 1 }, inner);
        var below = engine.TensorSlice(raw0, new[] { 0, 1, 0, 0 }, inner);
        var grad = engine.TensorAdd(
            engine.TensorAbs(engine.TensorSubtract(right, curr)),
            engine.TensorAbs(engine.TensorSubtract(below, curr)));
        featureList.Add(engine.TensorDivideScalar(
            engine.ReduceSum(SafetyTensorOps<T>.Gt(grad, 0.1), patchAxes, false), pixelCount));
        featureList.Add(engine.TensorDivideScalar(engine.ReduceSum(grad, patchAxes, false), pixelCount));

        var computed = engine.Reshape(engine.TensorStack(featureList.ToArray(), 2), new[] { numPatches, featureList.Count });
        return engine.TensorConcatenate(new[]
        {
            computed, Tensor<T>.CreateZeros(numPatches, _featureDim - featureList.Count)
        }, 1);
    }

    /// <summary>Channel <paramref name="c"/> of the image cropped to <c>[rows, cols]</c>.</summary>
    private static Tensor<T> CropPlane(Tensor<T> flat, ImageLayout layout, int c, int rows, int cols)
    {
        var engine = AiDotNetEngine.Current;
        int pixels = layout.Height * layout.Width;
        Tensor<T> plane;
        if (layout.Format == PixFmt.CHW)
        {
            plane = engine.Reshape(engine.TensorNarrow(flat, 0, c * pixels, pixels), new[] { layout.Height, layout.Width });
        }
        else
        {
            var interleaved = engine.Reshape(engine.TensorNarrow(flat, 0, 0, pixels * layout.Channels),
                new[] { pixels, layout.Channels });
            plane = engine.Reshape(engine.TensorNarrow(interleaved, 1, c, 1).Contiguous(),
                new[] { layout.Height, layout.Width });
        }

        return engine.TensorSlice(plane, new[] { 0, 0 }, new[] { rows, cols }).Contiguous();
    }

    private Tensor<T> ComputeAttentionWeights(Tensor<T> patchFeatures)
    {
        var engine = Engine;
        int n = patchFeatures.Shape[0];

        // Simple attention: dot product of each patch with the mean patch
        var meanPatch = engine.TensorDivideScalar(
            engine.ReduceSum(patchFeatures, new[] { 0 }, false), NumOps.FromDouble(n));
        var dot = engine.Reshape(
            engine.TensorMatMul(patchFeatures, engine.Reshape(meanPatch, new[] { _featureDim, 1 })), new[] { n });

        // exp(dot) for softmax, argument clamped to [-20, 20]
        var clamped = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Gt(dot, 20), 20.0, dot);
        clamped = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Lt(clamped, -20), -20.0, clamped);
        var weights = engine.TensorExp(clamped);
        T sum = engine.TensorSum(weights);

        // Normalize
        if (NumOps.GreaterThan(sum, NumOps.FromDouble(1e-10)))
        {
            weights = engine.TensorDivideScalar(weights, sum);
        }

        return weights;
    }

    private double ComputeCategoryScore(Vector<T> features, SafetyCategory category)
    {
        // Category-specific feature weighting based on which features are relevant
        double score;

        switch (category)
        {
            case SafetyCategory.SexualExplicit:
            case SafetyCategory.SexualSuggestive:
                // Skin fraction (feat 6), brightness (feat 8), low edge density (feat 10)
                double skin = NumOps.ToDouble(features[6]);
                double bright = NumOps.ToDouble(features[8]);
                double edgeLow = 1.0 - NumOps.ToDouble(features[10]);
                score = 0.5 * skin + 0.2 * bright + 0.3 * Math.Max(0, edgeLow);
                if (category == SafetyCategory.SexualSuggestive) score *= 0.8;
                break;

            case SafetyCategory.ViolenceGraphic:
                // Red dominance (feat 9), dark areas (feat 7), high edges (feat 10)
                double redDom = NumOps.ToDouble(features[9]);
                double dark = NumOps.ToDouble(features[7]);
                double edges = NumOps.ToDouble(features[10]);
                score = 0.4 * redDom + 0.3 * dark + 0.3 * edges;
                break;

            case SafetyCategory.ViolenceWeapons:
                // High edge density, metallic grey tones
                double edgeD = NumOps.ToDouble(features[10]);
                double rMean = NumOps.ToDouble(features[0]);
                double gMean = NumOps.ToDouble(features[2]);
                double bMean = NumOps.ToDouble(features[4]);
                double greyness = 1.0 - Math.Abs(rMean - gMean) - Math.Abs(gMean - bMean);
                score = 0.5 * edgeD + 0.5 * Math.Max(0, greyness);
                break;

            case SafetyCategory.ViolenceSelfHarm:
                // Red dominance with skin tones
                double redSH = NumOps.ToDouble(features[9]);
                double skinSH = NumOps.ToDouble(features[6]);
                score = 0.5 * redSH + 0.5 * skinSH;
                break;

            default:
                score = 0;
                break;
        }

        return Math.Max(0, Math.Min(1.0, score));
    }

    private static ImageLayout DetermineLayout(int[] shape, int dataLength)
    {
        if (shape.Length >= 3)
        {
            if (shape.Length >= 4)
            {
                if (shape[1] <= 4 && shape[2] > 4 && shape[3] > 4)
                    return new ImageLayout { Channels = shape[1], Height = shape[2], Width = shape[3], Format = PixFmt.CHW };
                if (shape[3] <= 4 && shape[1] > 4 && shape[2] > 4)
                    return new ImageLayout { Channels = shape[3], Height = shape[1], Width = shape[2], Format = PixFmt.HWC };
            }
            else
            {
                if (shape[0] <= 4 && shape[1] > 4 && shape[2] > 4)
                    return new ImageLayout { Channels = shape[0], Height = shape[1], Width = shape[2], Format = PixFmt.CHW };
                if (shape[2] <= 4 && shape[0] > 4 && shape[1] > 4)
                    return new ImageLayout { Channels = shape[2], Height = shape[0], Width = shape[1], Format = PixFmt.HWC };
            }
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
