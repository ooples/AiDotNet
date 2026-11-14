using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// kMaX-DeepLab: k-means Mask Transformer for panoptic segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Panoptic segmentation (stuff + things). Scene understanding for autonomous driving.
///
/// Common use cases:
/// - Panoptic segmentation (stuff + things)
/// - Scene understanding for autonomous driving
/// - Robotics environment parsing
/// - Dense scene labeling
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - k-means cross-attention replacing standard cross-attention: every pixel is assigned to its argmax cluster and
///   the assigned pixel values update the cluster centers
/// - Each decoder stage: k-means cross-attention, self-attention among the centers and an FFN, each with learnable
///   projections, residual connections and BatchNorm (deeplab2's kMaXTransformerLayer)
/// - Trained with the PQ-style loss, the mask-ID cross-entropy, the pixel-wise instance discrimination loss and an
///   auxiliary semantic loss, every decoder stage supervised (weights in <see cref="KMaXDeepLabOptions"/>)
/// </para>
/// <para>
/// <b>Reference:</b> Yu et al., "k-means Mask Transformer", ECCV 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a kMaX-DeepLab model for panoptic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 133);
/// var model = new KMaXDeepLab&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new KMaXDeepLab&lt;double&gt;(architecture,
///     options: new KMaXDeepLabOptions { NumClasses = 133, DropRate = 0.1, ModelSize = KMaXDeepLabModelSize.R50 });
///
/// // Or load a pre-trained ONNX model for autonomous driving scene parsing
/// var onnxModel = new KMaXDeepLab&lt;double&gt;(architecture, "kmaxdeeplab.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("k-means Mask Transformer", "https://arxiv.org/abs/2207.04044", Year = 2022, Authors = "Yu et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, WeightDecay = 0.05,
                WarmupSteps = 5000, Schedule = LearningRateSchedulerType.LinearWarmup, ReferenceBatchSize = 64,
                Source = "Yu et al. 2022, Sec. 4 (Training and testing; COCO dataset): AdamW with weight decay "
                        + "0.05, batch size 64, and a learning rate that rises linearly from 0 to 5e-4 over the "
                        + "first 5k steps. The segmentation default (AdamW at 1e-3, decay 0.01) raised the "
                        + "loss from 1.98 to 2.63 in two steps on the nightly heavy lane (#2087).")]
public partial class KMaXDeepLab<T> : Common.PanopticSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// The stride-4 cluster masks are bilinearly upsampled to the input, so the output keeps the input's spatial size.
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 1);

    private readonly KMaXDeepLabOptions _options;

    #region Fields
    // Only KMaXDeepLab's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly KMaXDeepLabModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    // Mask-classification head (Yu et al. 2022): N cluster centers compared with a D-wide pixel embedding,
    // refined by NumKMeansIterations k-means cross-attention updates.
    private readonly int _numClusters;
    private readonly int _pixelDim;
    private readonly int _kMeansIterations;
    private readonly Common.KMaXDecoderShape _decoderShape;
    private readonly Common.SemanticDecoderShape _semanticShape;
    private readonly Common.KMaXLossSettings _lossSettings;
    // Draws the auxiliary losses' pixel samples (Gumbel top-k); seeded from Options.Seed when one is set.
    private readonly Random _lossRandom;
    // True when no loss function was supplied: training then uses the paper's set-prediction loss.
    private readonly bool _usesSetPredictionLoss;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, NumStuffClasses and
    // NumThingClasses are all inherited and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal KMaXDeepLabModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes KMaXDeepLab in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function. When null, training uses kMaX-DeepLab's loss (Hungarian matching,
    /// PQ-style loss, mask-ID cross-entropy, pixel-wise instance discrimination and the auxiliary semantic loss, with
    /// the weights of <see cref="KMaXDeepLabOptions"/>); a supplied loss is applied to the semantic scores.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable KMaXDeepLab model.
    /// </para>
    /// </remarks>
    public KMaXDeepLab(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        KMaXDeepLabOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer(), so null is passed straight through.
        // The stuff/thing split is the same one/two-thirds rule the explicit interface members used.
        : base(architecture, optimizer, lossFunction, (options ??= new KMaXDeepLabOptions()).NumClasses,
            Math.Max(1, options.NumClasses / 3), options.NumClasses - Math.Max(1, options.NumClasses / 3))
    {
        _options = options; Options = _options;
        // KMaXDeepLab's own fallback input geometry is 640x640, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 640;
        if (architecture.InputWidth <= 0) _width = 640;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_numClusters, _pixelDim, _kMeansIterations) = ValidateHead(_options);
        (_decoderShape, _semanticShape, _lossSettings, _lossRandom) = DecoderSettings(_options, _pixelDim);
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        _usesSetPredictionLoss = lossFunction is null;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes KMaXDeepLab in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained KMaXDeepLab from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public KMaXDeepLab(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        KMaXDeepLabOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, (options ??= new KMaXDeepLabOptions()).NumClasses,
            Math.Max(1, options.NumClasses / 3), options.NumClasses - Math.Max(1, options.NumClasses / 3))
    {
        _options = options; Options = _options;
        // KMaXDeepLab's own fallback input geometry is 640x640, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 640;
        if (architecture.InputWidth <= 0) _width = 640;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_numClusters, _pixelDim, _kMeansIterations) = ValidateHead(_options);
        (_decoderShape, _semanticShape, _lossSettings, _lossRandom) = DecoderSettings(_options, _pixelDim);
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Runs a forward pass to produce segmentation logits.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get a per-pixel class prediction map.
    /// </para>
    /// <para>
    /// The output is MaskFormer's semantic inference over the mask-classification head,
    /// <c>log(sum_n p(class | cluster n) * p(pixel in cluster n) + 1e-8)</c>, shaped <c>[K, H, W]</c> (or
    /// <c>[B, K, H, W]</c>) at the input resolution; its argmax is the semantic map. In ONNX mode the graph must
    /// emit two outputs, in order: the class logits <c>[B, N, K + 1]</c> and the cluster (mask) logits
    /// <c>[B, N, H, W]</c>.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input) => Forward(input);

    /// <summary>
    /// Trains with kMaX-DeepLab's loss (Hungarian matching, PQ-style loss, mask-ID cross-entropy, pixel-wise instance
    /// discrimination and the auxiliary semantic loss) when no loss function was given to the constructor.
    /// </summary>
    /// <remarks>
    /// The target is the semantic segmentation <c>[B, K, H, W]</c> (or a <c>[B, H, W]</c> class-index map); every class
    /// present in an image is one ground-truth segment. A caller-supplied loss function is applied to the semantic
    /// scores instead, through the standard step.
    /// </remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        if (!_usesSetPredictionLoss) return false;
        if (!_useNativeMode) throw new NotSupportedException("Training is not supported in ONNX inference mode.");
        TrainWithCustomObjective(input, expected, SetPredictionLoss);
        return true;
    }

    /// <summary>
    /// kMaX-DeepLab's training loss of one batch, recomputed from the input: for every decoder stage's prediction and
    /// the final one, <c>w_PQ * L_PQ + w_ID * L_maskID + w_insdis * L_insdis</c> (each matched on its own), plus
    /// <c>w_sem * L_semantic</c> of the auxiliary semantic head.
    /// </summary>
    /// <remarks>
    /// The cluster logits and pixel features of every prediction are bilinearly upsampled to the input (the pixel
    /// features re-normalized) before they meet the ground truth. The Hungarian matching of the final prediction
    /// assigns the supervision of every prediction (Yu et al. 2022, Sec. 4.1; deeplab2's MaXDeepLabLoss), and the
    /// instance discrimination loss reads the same sampled pixels for every prediction of an image.
    /// </remarks>
    internal Tensor<T> SetPredictionLoss(Tensor<T> input, Tensor<T> expected)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        bool hasBatch = input.Rank == 4;
        var batched = hasBatch ? input : AddBatchDimension(input);
        var target = hasBatch ? expected : Engine.Reshape(expected, new[] { 1 }.Concat(expected.Shape.ToArray()).ToArray());
        var decoded = Decode(batched, includeSemantic: _lossSettings.SemanticWeight > 0);
        int b = batched.Shape[0], h = decoded.Height, w = decoded.Width;
        var segments = Common.MaskSetPredictionLoss.SemanticSegments(target, _numClasses, b, h, w);
        var settings = _lossSettings;

        int[][]? instanceSamples = settings.InstanceDiscriminationWeight > 0
            ? segments.Select(s => Common.MaskSetPredictionLoss.SamplePixels(
                s, settings.InstanceDiscriminationSampleCount, settings.InstanceDiscriminationSampleTemperature, _lossRandom)).ToArray()
            : null;
        var discriminationWeight = NumOps.FromDouble(settings.InstanceDiscriminationWeight / b);
        var final = decoded.Predictions[decoded.Predictions.Count - 1];
        var matchings = Common.MaskSetPredictionLoss.KMaXMatching(
            final.ClassLogits, ToInputResolution(final.MaskLogits, final.Height, final.Width, h, w), segments);
        Tensor<T>? total = null;
        foreach (var prediction in decoded.Predictions)
        {
            var masks = ToInputResolution(prediction.MaskLogits, prediction.Height, prediction.Width, h, w);
            var term = Common.MaskSetPredictionLoss.KMaX(prediction.ClassLogits, masks, segments, settings, matchings);
            if (instanceSamples is not null && prediction.PixelFeatures is not null)
            {
                var features = Common.MaskClassificationOps.NormalizeAxis(
                    ToInputResolution(prediction.PixelFeatures, prediction.Height, prediction.Width, h, w), 1);
                int d = features.Shape[1];
                Tensor<T>? discrimination = null;
                for (int i = 0; i < b; i++)
                {
                    var image = Engine.Reshape(Engine.TensorSlice(features, new[] { i, 0, 0 }, new[] { 1, d, h * w }), new[] { d, h * w });
                    var loss = Common.MaskSetPredictionLoss.InstanceDiscrimination(
                        image, segments[i], instanceSamples[i], settings.InstanceDiscriminationTemperature);
                    discrimination = discrimination is null ? loss : Engine.TensorAdd(discrimination, loss);
                }

                if (discrimination is not null)
                    term = Engine.TensorAdd(term, Engine.TensorMultiplyScalar(discrimination, discriminationWeight));
            }

            total = total is null ? term : Engine.TensorAdd(total, term);
        }

        if (total is null) throw new InvalidOperationException("kMaX-DeepLab's decoder produced no prediction.");
        if (decoded.SemanticLogits is not null && settings.SemanticWeight > 0)
        {
            var semantic = decoded.SemanticLogits;
            int k = semantic.Shape[1], sh = semantic.Shape[2], sw = semantic.Shape[3];
            var logits = ToInputResolution(Engine.Reshape(semantic, new[] { b, k, sh * sw }), sh, sw, h, w);
            Tensor<T>? semanticLoss = null;
            for (int i = 0; i < b; i++)
            {
                var sampled = Common.MaskSetPredictionLoss.SamplePixels(
                    segments[i], settings.SemanticSampleCount, settings.SemanticSampleTemperature, _lossRandom);
                var image = Engine.Reshape(Engine.TensorSlice(logits, new[] { i, 0, 0 }, new[] { 1, k, h * w }), new[] { k, h * w });
                var loss = Common.MaskSetPredictionLoss.SemanticCrossEntropy(image, segments[i], sampled);
                semanticLoss = semanticLoss is null ? loss : Engine.TensorAdd(semanticLoss, loss);
            }

            if (semanticLoss is not null)
                total = Engine.TensorAdd(total, Engine.TensorMultiplyScalar(semanticLoss, NumOps.FromDouble(settings.SemanticWeight / b)));
        }

        return total;
    }

    /// <summary>The decoder's layers (the initial centers, every stage and the final heads), in model order.</summary>
    /// <remarks>A method, not a property: the layers already live in <c>Layers</c>, and a layer-collection property
    /// would be surfaced a second time by the parameter generator.</remarks>
    internal IReadOnlyList<ILayer<T>> GetDecoderLayers()
        => _useNativeMode
            ? Layers.Skip(DecoderStart).Take(Common.KMaXTransformerDecoder.LayerCount(_kMeansIterations)).ToList()
            : new List<ILayer<T>>();

    #endregion

    #region Private Methods
    /// <summary>Backbone stages the pixel decoder reads: strides 4, 8, 16 and 32 (Yu et al. 2022, Fig. 2).</summary>
    private const int PyramidLevels = 4;

    /// <summary>Index of the auxiliary semantic decoder's first layer, after the backbone and the pixel decoder.</summary>
    private int SemanticDecoderStart => _encoderLayerEnd + 2 * PyramidLevels;

    /// <summary>Index of the kMaX decoder's first layer (the initial cluster centers).</summary>
    private int DecoderStart => SemanticDecoderStart + Common.PanopticDeepLabSemanticDecoder.LayerCount(_semanticShape);

    /// <summary>The auxiliary semantic decoder's and head's layers, in model order.</summary>
    /// <remarks>A method, not a property, for the reason <see cref="GetDecoderLayers"/> gives.</remarks>
    internal IReadOnlyList<ILayer<T>> GetSemanticDecoderLayers()
        => _useNativeMode
            ? Layers.Skip(SemanticDecoderStart).Take(Common.PanopticDeepLabSemanticDecoder.LayerCount(_semanticShape)).ToList()
            : new List<ILayer<T>>();

    private static (Common.KMaXDecoderShape Shape, Common.SemanticDecoderShape Semantic, Common.KMaXLossSettings Loss, Random Random)
        DecoderSettings(KMaXDeepLabOptions options, int pixelDim)
    {
        if (options.QueryDimension < 2 || options.QueryDimension % 2 != 0)
            throw new ArgumentException("QueryDimension must be a positive even number (the keys take half of it).", nameof(options));
        if (options.NumAttentionHeads < 1 || (options.QueryDimension / 2) % options.NumAttentionHeads != 0)
            throw new ArgumentException("NumAttentionHeads must divide QueryDimension / 2.", nameof(options));
        if (options.FeedForwardDimension < 1)
            throw new ArgumentException("FeedForwardDimension must be positive.", nameof(options));
        if (options.PixelHeadDimension < 1)
            throw new ArgumentException("PixelHeadDimension must be positive.", nameof(options));
        if (!(options.VoidPriorProbability > 0 && options.VoidPriorProbability < 1))
            throw new ArgumentException("VoidPriorProbability must lie in (0, 1).", nameof(options));
        if (options.SemanticAtrousRates is null || options.SemanticLowLevelProjections is null)
            throw new ArgumentException("SemanticAtrousRates and SemanticLowLevelProjections must be set.", nameof(options));
        if (options.SemanticLowLevelProjections.Length > PyramidLevels)
            throw new ArgumentException(
                $"SemanticLowLevelProjections can fuse at most {PyramidLevels} backbone stages.", nameof(options));
        if (!(options.PanopticQualityLossWeight >= 0) || !(options.MaskIdLossWeight >= 0)
            || !(options.InstanceDiscriminationLossWeight >= 0) || !(options.SemanticLossWeight >= 0))
            throw new ArgumentException("The loss weights must be non-negative.", nameof(options));
        if (!(options.InstanceDiscriminationTemperature > 0) || !(options.InstanceDiscriminationSampleTemperature >= 0)
            || !(options.SemanticSampleTemperature >= 0))
            throw new ArgumentException(
                "The instance discrimination temperature must be positive and the sampling temperatures non-negative.", nameof(options));
        if (options.InstanceDiscriminationSampleCount < 1 || options.SemanticSampleCount < 1)
            throw new ArgumentException("The auxiliary losses must sample at least one pixel.", nameof(options));
        var shape = new Common.KMaXDecoderShape(pixelDim, options.QueryDimension, options.NumAttentionHeads,
            options.FeedForwardDimension, options.NumClasses + 1, options.PixelHeadDimension, options.VoidPriorProbability);
        // The shape validates the semantic decoder widths, rates and dropout.
        var semantic = new Common.SemanticDecoderShape(options.SemanticAsppDimension, options.SemanticAtrousRates,
            options.SemanticDecoderDimension, options.SemanticLowLevelProjections, options.SemanticHeadDimension,
            options.NumClasses, options.SemanticAsppDropRate);

        var loss = new Common.KMaXLossSettings
        {
            PanopticQualityWeight = options.PanopticQualityLossWeight,
            MaskIdWeight = options.MaskIdLossWeight,
            InstanceDiscriminationWeight = options.InstanceDiscriminationLossWeight,
            InstanceDiscriminationTemperature = options.InstanceDiscriminationTemperature,
            InstanceDiscriminationSampleCount = options.InstanceDiscriminationSampleCount,
            InstanceDiscriminationSampleTemperature = options.InstanceDiscriminationSampleTemperature,
            SemanticWeight = options.SemanticLossWeight,
            SemanticSampleCount = options.SemanticSampleCount,
            SemanticSampleTemperature = options.SemanticSampleTemperature,
        };
        var random = options.Seed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(options.Seed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        return (shape, semantic, loss, random);
    }

    private static (int NumClusters, int PixelDim, int Iterations) ValidateHead(KMaXDeepLabOptions options)
    {
        if (options.NumClusterCenters < 1)
            throw new ArgumentException("NumClusterCenters must be positive.", nameof(options));
        if (options.PixelEmbeddingDimension < 1)
            throw new ArgumentException("PixelEmbeddingDimension must be positive.", nameof(options));
        if (options.NumKMeansIterations < 1)
            throw new ArgumentException("NumKMeansIterations must be positive.", nameof(options));
        return (options.NumClusterCenters, options.PixelEmbeddingDimension, options.NumKMeansIterations);
    }

    /// <summary>
    /// The final mask-classification heads of a batched image: class logits <c>[B, N, K + 1]</c> and cluster logits
    /// <c>[B, N, H*W]</c> at the input resolution.
    /// </summary>
    private (Tensor<T> ClassLogits, Tensor<T> MaskLogits, int Height, int Width) Heads(Tensor<T> batched)
    {
        var decoded = Decode(batched, includeSemantic: false); // inference never reads the auxiliary semantic head
        var final = decoded.Predictions[decoded.Predictions.Count - 1];
        return (final.ClassLogits,
            ToInputResolution(final.MaskLogits, final.Height, final.Width, decoded.Height, decoded.Width),
            decoded.Height, decoded.Width);
    }

    /// <summary>The decoder's predictions and the auxiliary semantic logits of a batched image.</summary>
    private sealed class DecodedImage
    {
        internal DecodedImage(IReadOnlyList<Common.KMaXPrediction<T>> predictions, Tensor<T>? semanticLogits, int height, int width)
        {
            Predictions = predictions;
            SemanticLogits = semanticLogits;
            Height = height;
            Width = width;
        }

        internal IReadOnlyList<Common.KMaXPrediction<T>> Predictions { get; }
        internal Tensor<T>? SemanticLogits { get; }
        internal int Height { get; }
        internal int Width { get; }
    }

    /// <summary>
    /// Runs the backbone, the pixel decoder and the kMaX decoder on a batched image.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The pixel decoder is a top-down pathway over the backbone's stride-4, 8, 16 and 32 stages. The kMaX decoder's
    /// stages read its stride-32, 16 and 8 features, coarse to fine (with the default six stages, two per scale), and
    /// the final predictor reads the stride-4 features (Yu et al. 2022, Sec. 3.3). The auxiliary semantic decoder
    /// (<see cref="Common.PanopticDeepLabSemanticDecoder"/>) runs an ASPP on the stride-32 backbone stage and fuses the
    /// stride-8 and stride-4 backbone stages, as deeplab2's kMaX-DeepLab does (its <c>feature_semantic</c> is the
    /// backbone's <c>stage5</c>). In ONNX mode the graph's final heads are the only prediction.
    /// </para>
    /// <para>
    /// The decoded image's size is always the input's: a graph whose masks are at a stride keeps that size on its
    /// prediction, so <see cref="Heads"/> upsamples the masks to the stride-1 output the contract declares.
    /// </para>
    /// </remarks>
    /// <param name="batched">The batched image <c>[B, C, H, W]</c>.</param>
    /// <param name="includeSemantic">Whether to run the auxiliary semantic decoder; only its loss reads it.</param>
    private DecodedImage Decode(Tensor<T> batched, bool includeSemantic)
    {
        if (!_useNativeMode)
        {
            var (classLogits, maskLogits, h, w) = OnnxHeads(batched);
            return new DecodedImage(
                new[] { new Common.KMaXPrediction<T>(classLogits, maskLogits, null, h, w) }, null,
                batched.Shape[2], batched.Shape[3]);
        }

        int b = batched.Shape[0], height = batched.Shape[2], width = batched.Shape[3];
        var (_, taps) = Common.FeaturePyramidOps.RunAndTapStages(Layers, 0, _encoderLayerEnd, batched);
        int levelCount = Math.Min(PyramidLevels, taps.Count);
        if (levelCount == 0)
            throw new InvalidOperationException("kMaX-DeepLab's backbone produced no [B, C, H, W] feature map.");
        var stages = taps.Skip(taps.Count - levelCount).ToList();
        var lateral = new List<Func<Tensor<T>, Tensor<T>>>(levelCount);
        var output = new List<Func<Tensor<T>, Tensor<T>>>(levelCount);
        for (int i = 0; i < levelCount; i++)
        {
            lateral.Add(Layers[_encoderLayerEnd + PyramidLevels - levelCount + i].Forward);
            output.Add(Layers[_encoderLayerEnd + 2 * PyramidLevels - levelCount + i].Forward);
        }

        var levels = Common.FeaturePyramidOps.TopDown(stages, lateral, output);                      // finest first
        var flat = levels.Select(level => (
            Features: Engine.Reshape(level, new[] { b, _pixelDim, level.Shape[2] * level.Shape[3] }),
            Height: level.Shape[2], Width: level.Shape[3])).ToList();
        // The decoder stages read the coarser levels, coarsest first; the final predictor reads the finest level.
        var decoderScales = levelCount == 1 ? flat : Enumerable.Reverse(flat.Skip(1)).ToList();
        var finest = flat[0];
        var predictions = Common.KMaXTransformerDecoder.Run(
            Layers, DecoderStart, _decoderShape, decoderScales, _kMeansIterations,
            finest.Features, finest.Height, finest.Width, _numClusters);
        var semanticLogits = includeSemantic
            ? Common.PanopticDeepLabSemanticDecoder.Run(Layers, SemanticDecoderStart, _semanticShape, stages) // [B, K, h4, w4]
            : null;
        return new DecodedImage(predictions, semanticLogits, height, width);
    }

    /// <summary>Bilinearly upsamples <c>[B, C, h * w]</c> to <c>[B, C, H * W]</c> (unchanged when the sizes match).</summary>
    private Tensor<T> ToInputResolution(Tensor<T> flat, int h, int w, int height, int width)
    {
        if (h == height && w == width) return flat;
        int b = flat.Shape[0], c = flat.Shape[1];
        var upsampled = Engine.Interpolate(
            Engine.Reshape(flat, new[] { b, c, h, w }), new[] { height, width }, InterpolateMode.Bilinear, false);
        return Engine.Reshape(upsampled, new[] { b, c, height * width });
    }

    /// <summary>The two heads read from the ONNX graph's outputs (class logits, then cluster logits).</summary>
    private (Tensor<T> ClassLogits, Tensor<T> MaskLogits, int Height, int Width) OnnxHeads(Tensor<T> batched)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        var inputData = new float[batched.Length];
        NumOps.ToFloatSpan(batched.Data.Span.Slice(0, batched.Length), inputData);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, batched._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputs = results.Select(result => result.AsTensor<float>()).ToList();
        if (outputs.Count < 2)
            throw new InvalidOperationException(
                "A kMaX-DeepLab ONNX graph must emit the class logits [B, N, K + 1] and the cluster logits [B, N, H, W].");

        var classLogits = ReadOnnx(outputs[0]);
        var masks = ReadOnnx(outputs[1]);
        if (classLogits.Rank != 3 || masks.Rank != 4 || masks.Shape[1] != classLogits.Shape[1] || classLogits.Shape[2] != _numClasses + 1)
            throw new InvalidOperationException(
                $"The ONNX outputs must be [B, N, {_numClasses + 1}] class logits and [B, N, H, W] cluster logits.");
        int b = masks.Shape[0], n = masks.Shape[1], h = masks.Shape[2], w = masks.Shape[3];
        return (classLogits, Engine.Reshape(masks, new[] { b, n, h * w }), h, w);
    }

    private Tensor<T> ReadOnnx(OnnxTensors.Tensor<float> tensor)
    {
        var data = new T[tensor.Length];
        NumOps.FromFloatSpan(tensor.ToDenseTensor().Buffer.Span, data);
        return new Tensor<T>(tensor.Dimensions.ToArray(), new Vector<T>(data));
    }

    /// <summary>MaskFormer semantic inference as log scores <c>[B, K, H, W]</c>.</summary>
    private Tensor<T> SemanticScores(Tensor<T> classLogits, Tensor<T> maskLogits, int h, int w)
    {
        int b = classLogits.Shape[0], n = classLogits.Shape[1];
        var classProbabilities = Engine.TensorSlice(
            Engine.TensorSoftmax(classLogits, 2), new[] { 0, 0, 0 }, new[] { b, n, _numClasses });
        var maskProbabilities = Engine.TensorSoftmax(maskLogits, 1);
        return Common.MaskClassificationOps.SemanticLogScores(classProbabilities, maskProbabilities, h, w);
    }

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(KMaXDeepLabModelSize modelSize) => modelSize switch
    {
        KMaXDeepLabModelSize.R50 => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256),
        KMaXDeepLabModelSize.ConvNeXtLarge => ([192, 384, 768, 1536], [3, 3, 27, 3], 256),
        _ => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256)
    };

    /// <summary>
    /// Forward pass: backbone, multi-scale pixel decoder, kMaX decoder and MaskFormer semantic inference.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var (classLogits, maskLogits, h, w) = Heads(input);
        var scores = SemanticScores(classLogits, maskLogits, h, w);
        if (!hasBatch) scores = RemoveBatchDimension(scores); return scores;
    }

    /// <summary>
    /// The training forward for a caller-supplied loss: the same semantic scores <see cref="Forward"/> returns (the
    /// layers are not a sequential chain, so the base layer walk would train a different tensor). The layer seeds
    /// are wired first, as the base does. The set-prediction objective reads the decoder directly instead.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        EnsureLayerRandomSeedsWired();
        return Forward(input);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input) => Forward(input);
    #endregion

    #region Abstract Implementation
    /// <summary>
    /// Initializes the backbone, the pixel decoder and the kMaX decoder's learnable parts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the neural network layers.
    /// In ONNX mode, no layers are created.
    /// </para>
    /// <para>
    /// After the backbone (or the architecture's own layers, which then act as the backbone) come the pixel
    /// decoder's four lateral 1x1 convolutions and four output 3x3 convolutions (to the D-wide pixel features), the
    /// auxiliary semantic decoder and head (<see cref="Common.PanopticDeepLabSemanticDecoder"/>: ASPP, two low-level
    /// fusions and the depthwise-separable head) and the kMaX transformer decoder
    /// (<see cref="Common.KMaXTransformerDecoder"/>: the learnable initial cluster centers, every decoder stage's
    /// projections and BatchNorms, and the final class and mask heads).
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Architecture.Layers.Count; }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateKMaXDeepLabEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
        }

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        for (int i = 0; i < PyramidLevels; i++) Layers.Add(new ConvolutionalLayer<T>(_decoderDim, 1, 1, 0, identity));
        for (int i = 0; i < PyramidLevels; i++) Layers.Add(new ConvolutionalLayer<T>(_pixelDim, 3, 1, 1, identity));
        Layers.AddRange(Common.PanopticDeepLabSemanticDecoder.CreateLayers<T>(_semanticShape));
        Layers.AddRange(Common.KMaXTransformerDecoder.CreateLayers<T>(_decoderShape, _kMeansIterations));
    }

    private bool _lazyShapesResolved;

    /// <summary>
    /// Resolves the lazy layers through the real multi-scale forward: the pixel decoder reads backbone stages, not
    /// its sequential predecessor, so the base class's sequential shape walk would size it wrongly.
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesResolved) return;
        if (Layers is null || Layers.Count == 0) return;
        if (!_useNativeMode) { _lazyShapesResolved = true; return; }
        using (InferenceMode.Enter())
        {
            Forward(new Tensor<T>([1, _channels, _height, _width]));
        }
        _lazyShapesResolved = true;
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "KMaXDeepLab" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };
    #endregion

    #region IPanopticSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, NumStuffClasses and NumThingClasses
    // all come from the base; only the model-specific override remains.

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// kMaX-DeepLab's panoptic post-processing on the input-resolution cluster masks (softmax over the clusters):
    /// clusters are visited by class score times mask score, become segments when their class score clears the
    /// thing or stuff threshold and most of their mask is still free, and stuff of one class merges.
    /// </para>
    /// </remarks>
    public override PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var batched = image.Rank == 4 ? image : AddBatchDimension(image);
        if (batched.Shape[0] != 1) throw new ArgumentException("SegmentPanoptic takes a single image.", nameof(image));
        using var noGrad = new NoGradScope<T>();
        var (classLogits, maskLogits, h, w) = Heads(batched);
        int clusters = classLogits.Shape[1]; // an ONNX graph may decode a different query count than the options
        var classProbabilities = Engine.Reshape(Engine.TensorSoftmax(classLogits, 2), new[] { clusters, _numClasses + 1 });
        var maskProbabilities = Engine.Reshape(Engine.TensorSoftmax(maskLogits, 1), new[] { clusters, h, w });
        var settings = new Common.KMaXPanopticSettings(
            _options.ClassThresholdThing, _options.ClassThresholdStuff, _options.PixelConfidenceThreshold,
            _options.OverlapThreshold, _options.ReorderClassWeight, _options.ReorderMaskWeight);
        return Common.MaskClassificationOps.MergeKMaX(classProbabilities, maskProbabilities, ThingClassFlags(), settings);
    }

    private bool[] ThingClassFlags()
    {
        var flags = new bool[_numClasses];
        for (int c = NumStuffClasses; c < _numClasses; c++) flags[c] = true;
        return flags;
    }
    #endregion
}
