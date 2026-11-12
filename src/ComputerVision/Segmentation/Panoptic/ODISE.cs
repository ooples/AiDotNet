using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.VisionLanguage.Interfaces;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// ODISE: Open-vocabulary DIffusion-based panoptic SEgmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Open-vocabulary panoptic segmentation. Segmenting novel categories not seen during training.
///
/// Common use cases:
/// - Open-vocabulary panoptic segmentation
/// - Segmenting novel categories not seen during training
/// - Creative content understanding
/// - Cross-domain scene parsing
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Leverages Stable Diffusion internal representations for segmentation
/// - Text-image discriminative model (CLIP) for category assignment
/// - Mask generator trained on diffusion features
/// - Zero-shot panoptic segmentation capability
/// </para>
/// <para>
/// <b>Reference:</b> Xu et al., "Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models", CVPR 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an ODISE model for open-vocabulary panoptic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 133);
/// // textEncoder is any ITextEncoder (for example a CLIP model from AiDotNet.VisionLanguage.Encoders).
/// var model = new ODISE&lt;double&gt;(architecture, textEncoder);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new ODISE&lt;double&gt;(architecture, textEncoder,
///     options: new ODISEOptions { NumClasses = 133, DropRate = 0.1, ModelSize = ODISEModelSize.Base });
///
/// // Or load a pre-trained ONNX model for diffusion-based segmentation
/// var onnxModel = new ODISE&lt;double&gt;(architecture, "odise.onnx", textEncoder);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models", "https://arxiv.org/abs/2303.04803", Year = 2023, Authors = "Xu et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 0.0001, WeightDecay = 0.05,
                ReferenceBatchSize = 64,
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.1,
                MilestoneFractions = [0.9, 0.9556],
                Source = "Xu et al. 2023, Training Details: AdamW at a learning rate of 0.0001 with weight decay 0.05 and batch size 64, the rate reduced by a factor of 10 at 81k and 86k of a 90k-iteration run, which is the 0.9 and 0.9556 declared here.")]
public partial class ODISE<T> : Common.PanopticSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Does NOT downsample: measured [1,3,64,64] -> [1,C,64,64]. Its U-Net skip topology restores full
    /// resolution, so the inherited /32 law would be wrong by a factor of 32 on both spatial axes.
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 1);

    private readonly ODISEOptions _options;

    #region Fields
    // Only ODISE's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly ODISEModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    // True when the constructor caller supplied an explicit optimizer, so tape
    // training honors it instead of the built-in LR-warmup Adam default.
    private readonly bool _hasUserSuppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _baseTapeOptimizer;

    // Per-encoder-stage layer counts, used by Forward to tap each stage's output for
    // the U-Net skip connections. Null when layers come from a pre-supplied
    // Architecture (Forward then falls back to a plain sequential pass).
    private int[]? _encoderStageLayerCounts;

    // Each decoder upsampling stage is Deconv + GroupNorm + SiLU + ResidualBlock +
    // GroupNorm (see LayerHelper.CreateODISEDecoderLayers). Forward consumes exactly
    // this many layers per stage before inserting the skip-concat. Aliases the single
    // source of truth in LayerHelper so the builder and this consumer can't drift apart.
    private const int DecoderStageLayerCount = LayerHelper<T>.ODISEDecoderStageLayerCount;

    // Mask-classification head (Xu et al. 2023, Sec. 3.4-3.6): N queries refined by masked attention over a
    // D-wide pixel embedding; each query's embedding is classified against the text encoder's embeddings of
    // the class names.
    private readonly ITextEncoder<T> _textEncoder;
    private readonly int _numQueries;
    private readonly int _pixelDim;
    private readonly int _embeddingDim;
    private readonly int _decoderLayers;
    private readonly string[] _classNames;
    private Tensor<T>? _closedSetText;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, NumStuffClasses and
    // NumThingClasses are all inherited and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal ODISEModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes ODISE in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="textEncoder">CLIP-style text encoder whose embeddings classify the masks.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable ODISE model.
    /// </para>
    /// </remarks>
    public ODISE(NeuralNetworkArchitecture<T> architecture,
        ITextEncoder<T> textEncoder,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ODISEOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // stores `optimizer` verbatim (including null) - ODISE reads _optimizer directly in
        // GetOrCreateBaseOptimizer and deliberately does NOT want the AdamW default.
        // The stuff/thing split is the same one/two-thirds rule the explicit interface members used.
        : base(architecture, optimizer, lossFunction, (options ??= new ODISEOptions()).NumClasses,
            Math.Max(1, options.NumClasses / 3), options.NumClasses - Math.Max(1, options.NumClasses / 3))
    {
        _options = options; Options = _options;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        _hasUserSuppliedOptimizer = optimizer is not null;
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        (_numQueries, _pixelDim, _embeddingDim, _decoderLayers, _classNames) = ValidateHead(_options, textEncoder);
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes ODISE in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file. The graph must emit the head's raw
    /// channels: pixel embedding, query mask logits, text-space pixel embedding and void score.</param>
    /// <param name="textEncoder">CLIP-style text encoder whose embeddings classify the masks.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained ODISE from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public ODISE(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITextEncoder<T> textEncoder,
        ODISEOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, (options ??= new ODISEOptions()).NumClasses,
            Math.Max(1, options.NumClasses / 3), options.NumClasses - Math.Max(1, options.NumClasses / 3))
    {
        _options = options; Options = _options;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        (_numQueries, _pixelDim, _embeddingDim, _decoderLayers, _classNames) = ValidateHead(_options, textEncoder);
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Runs a forward pass to produce per-pixel class probabilities.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Probability tensor in [0, 1] with the same spatial layout as input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get a per-pixel class probability map.
    /// </para>
    /// <para>
    /// Inference applies softmax along the class dimension so the output is a
    /// probability distribution per pixel, matching the paper-canonical inference
    /// formulation (Xu et al. 2023 §3 outputs per-pixel class probabilities; CE
    /// loss applies log-softmax internally during training, so the training path
    /// continues to consume raw logits via <see cref="Forward"/>).
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Both modes must return the same semantics — per-pixel class
        // probabilities along the class dim (Xu et al. 2023 §3 paper
        // formulation). The ONNX path used to return raw output, making
        // Predict() behave differently between native and ONNX modes and
        // breaking SegmentPanoptic's confidence math (which assumes values
        // in [0, 1]). Apply the same softmax in both paths so the public
        // contract is consistent.
        var logits = _useNativeMode ? Forward(input) : SemanticScores(PredictOnnx(input));
        return Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
    }

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains the model. Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    /// <summary>
    /// ODISE trains the dense per-pixel segmentation head through a deep convolutional
    /// encoder/decoder. Its Stable-Diffusion U-Net encoder (SiLU + GroupNorm residual
    /// stack, per Xu et al. 2023 §3) is randomly initialised at test time, so the very
    /// first optimizer steps see large, poorly-conditioned gradients: at a fixed 1e-2 the
    /// per-pixel cross-entropy overshoots on step 1-2 and then DIVERGES (CE 2.35 → 5.5),
    /// while a fixed 1e-4 is too slow to memorise within the probe's 100 iterations.
    ///
    /// The standard remedy for this exact failure mode — and the regime diffusion U-Net
    /// training itself uses — is a linear learning-rate WARMUP: ramp the rate from ~0 up
    /// to the target over the first few steps so Adam's moment estimates stabilise on the
    /// ill-conditioned initial geometry before the full step size is applied, then hold it
    /// constant. This restores a monotonic decrease without weakening any test tolerance or
    /// touching the architecture.
    ///
    /// NOTE: the scheduler must be stepped PER BATCH (<see cref="SchedulerStepMode.StepPerBatch"/>)
    /// — the tape training path advances the scheduler via OnBatchEnd once per Train() call,
    /// and the default StepPerEpoch mode would leave the warmup pinned at its initial rate.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        // Honor a constructor-supplied optimizer (public contract). Only fall back to
        // the built-in LR-warmup Adam — which the from-scratch SD-U-Net needs for a
        // stable descent — when the caller did not provide one.
        if (_hasUserSuppliedOptimizer && _optimizer is not null)
            return _optimizer;

        // Adam at 1e-3 (the standard deep-net rate), NOT 1e-2. A loss-trajectory probe on the
        // from-scratch SD-U-Net showed 1e-2 OVERSHOOTS: the CE loss descends to ~1.49 through the
        // low-LR warmup, then climbs monotonically (->1.82 by step 30) once the LR reaches the full
        // 1e-2 — the deep encoder-decoder can't take 1e-2 Adam steps stably, so Training_ShouldReduceLoss
        // / MemorizationTask / MoreData all saw the loss rise. 1e-3 with the same warmup gives a stable
        // monotonic descent.
        return _baseTapeOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                UseAMSGrad = false,
                InitialLearningRate = 0.001,
                SchedulerStepMode = LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                LearningRateScheduler = new LearningRateSchedulers.LinearWarmupScheduler(
                    baseLearningRate: 0.001,
                    warmupSteps: 10,
                    totalSteps: 0,          // 0 => no decay phase; hold at baseLearningRate after warmup
                    warmupInitLr: 0.0001)   // start small but NONZERO so the very first step still trains
            });
    }

    #endregion

    #region Private Methods
    private static (int Queries, int PixelDim, int EmbeddingDim, int Layers, string[] ClassNames) ValidateHead(
        ODISEOptions options, ITextEncoder<T> textEncoder)
    {
        if (options.NumQueries < 1) throw new ArgumentException("NumQueries must be positive.", nameof(options));
        if (options.PixelEmbeddingDimension < 1)
            throw new ArgumentException("PixelEmbeddingDimension must be positive.", nameof(options));
        if (options.NumDecoderLayers < 1) throw new ArgumentException("NumDecoderLayers must be positive.", nameof(options));
        if (!(options.Temperature > 0)) throw new ArgumentException("Temperature must be positive.", nameof(options));
        if (!(options.InVocabularyClipWeight >= 0 && options.InVocabularyClipWeight <= 1)
            || !(options.OutOfVocabularyClipWeight >= 0 && options.OutOfVocabularyClipWeight <= 1))
            throw new ArgumentException("The CLIP fusion weights must lie in [0, 1].", nameof(options));
        if (textEncoder.TextEmbeddingDimension < 1)
            throw new ArgumentException("The text encoder must declare a positive TextEmbeddingDimension.", nameof(textEncoder));
        string[] names = options.ClassNames?.ToArray()
            ?? Enumerable.Range(0, options.NumClasses).Select(i => $"class {i}").ToArray();
        if (names.Length != options.NumClasses)
            throw new ArgumentException(
                $"ClassNames has {names.Length} names but NumClasses is {options.NumClasses}.", nameof(options));
        return (options.NumQueries, options.PixelEmbeddingDimension, textEncoder.TextEmbeddingDimension,
            options.NumDecoderLayers, names);
    }

    /// <summary>Channels the decoder emits: pixel embedding, query mask logits, text-space embedding, void.</summary>
    private int HeadChannels => _pixelDim + _numQueries + _embeddingDim + 1;

    /// <summary>The L2-normalized text embeddings <c>[K, E]</c> of <paramref name="names"/> (the encoder is frozen).</summary>
    private Tensor<T> EncodeNames(IReadOnlyList<string> names)
    {
        using var noGrad = new NoGradScope<T>();
        return Common.TextEmbeddingClassifier.EncodeNormalized(_textEncoder, names);
    }

    /// <summary>
    /// ODISE's mask-classification head on the raw decoder output <c>[B, HeadChannels, H, W]</c>: masked
    /// attention, then the open-vocabulary classification of every query against <paramref name="text"/>.
    /// </summary>
    /// <returns>Mask logits <c>[B, N, HW]</c> and class probabilities <c>[B, N, K + 1]</c> (void last).</returns>
    /// <remarks>
    /// <para>
    /// The diffusion-branch prediction is <c>softmax_k(cos(z_n, t_k) / tau)</c> of the query embedding
    /// <c>z_n</c>; the discriminative prediction classifies the mask-pooled text-space pixel embedding
    /// (MaskPooling over <c>mask &gt; 0</c>) the same way. They are fused by a geometric mean,
    /// <c>p_diffusion^(1 - w) * p_clip^w</c>, with <c>w = alpha</c> (0.35) for in-vocabulary classes and
    /// <c>w = beta</c> (0.65) for the rest, renormalized over the classes and scaled by <c>1 - p_void</c>, where
    /// <c>p_void</c> comes from the softmax over the diffusion logits and the query's void logit (ODISE's
    /// <c>CategoryODISE</c> inference).
    /// </para>
    /// </remarks>
    internal (Tensor<T> MaskLogits, Tensor<T> ClassProbabilities) ClassifyQueries(
        Tensor<T> raw, Tensor<T> text, Tensor<T> clipWeights)
    {
        int b = raw.Shape[0], hw = raw.Shape[2] * raw.Shape[3], k = text.Shape[0];
        var (maskLogits, queryEmbeddings, voidLogits) = Common.MaskClassificationOps.MaskedAttention(
            raw, _pixelDim, _numQueries, _embeddingDim, _decoderLayers);
        var scale = NumOps.FromDouble(1.0 / _options.Temperature);
        var textT = Engine.TensorTranspose(text);                                                    // [E, K]

        var diffusionLogits = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(Common.MaskClassificationOps.NormalizeAxis(queryEmbeddings, 2), textT), scale);

        // MaskPooling(V(x), m_n): mean text-space pixel embedding over the query's binary mask.
        var pixelEmbedding = Engine.TensorPermute(Engine.TensorSlice(Engine.Reshape(raw, new[] { b, HeadChannels, hw }),
            new[] { 0, _pixelDim + _numQueries, 0 }, new[] { b, _embeddingDim, hw }), new[] { 0, 2, 1 });   // [B, HW, E]
        var binaryMask = Engine.TensorGreaterThan(maskLogits, NumOps.Zero);
        var pooled = Engine.TensorDivide(
            Engine.TensorMatMul(binaryMask, pixelEmbedding),
            Engine.TensorBroadcastTo(
                Engine.TensorAddScalar(Engine.ReduceSum(binaryMask, new[] { 2 }, true), NumOps.FromDouble(1e-8)),
                new[] { b, _numQueries, _embeddingDim }));
        var clipLogits = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(Common.MaskClassificationOps.NormalizeAxis(pooled, 2), textT), scale);

        var diffusionProbability = Engine.TensorSoftmax(diffusionLogits, 2);
        var clipProbability = Engine.TensorSoftmax(clipLogits, 2);
        var withVoid = Engine.TensorSoftmax(Engine.TensorConcatenate(new[] { diffusionLogits, voidLogits }, 2), 2);
        var voidProbability = Engine.TensorSlice(withVoid, new[] { 0, 0, k }, new[] { b, _numQueries, 1 });

        var shape = new[] { b, _numQueries, k };
        var clipExponent = Engine.TensorBroadcastTo(Engine.Reshape(clipWeights, new[] { 1, 1, k }), shape);
        var fused = Engine.TensorMultiply(
            Engine.TensorPower(diffusionProbability, Engine.ScalarMinusTensor(NumOps.One, clipExponent)),
            Engine.TensorPower(clipProbability, clipExponent));
        var normalized = Engine.TensorDivide(fused, Engine.TensorBroadcastTo(Engine.ReduceSum(fused, new[] { 2 }, true), shape));
        var classProbabilities = Engine.TensorConcatenate(new[]
        {
            Engine.TensorMultiply(normalized, Engine.TensorBroadcastTo(Engine.ScalarMinusTensor(NumOps.One, voidProbability), shape)),
            voidProbability,
        }, 2);
        return (maskLogits, classProbabilities);
    }

    /// <summary>The CLIP fusion weight of each name: alpha for in-vocabulary names, beta otherwise.</summary>
    internal Tensor<T> ClipWeights(IReadOnlyList<string> names)
    {
        var seen = _options.InVocabularyClassNames is null
            ? null
            : new HashSet<string>(_options.InVocabularyClassNames, StringComparer.Ordinal);
        var flags = new bool[names.Count];
        for (int i = 0; i < flags.Length; i++) flags[i] = seen is null || seen.Contains(names[i]);
        var inVocabulary = Common.MaskClassificationOps.BoolTable<T>(flags);
        var alpha = NumOps.FromDouble(_options.InVocabularyClipWeight);
        var beta = NumOps.FromDouble(_options.OutOfVocabularyClipWeight);
        return Engine.TensorAdd(
            Engine.TensorMultiplyScalar(inVocabulary, alpha),
            Engine.TensorMultiplyScalar(Engine.ScalarMinusTensor(NumOps.One, inVocabulary), beta));
    }

    /// <summary>MaskFormer semantic inference over the closed-set classes, as log scores <c>[B, K, H, W]</c>.</summary>
    private Tensor<T> SemanticScores(Tensor<T> raw)
    {
        bool hasBatch = raw.Rank == 4; if (!hasBatch) raw = AddBatchDimension(raw);
        int b = raw.Shape[0], h = raw.Shape[2], w = raw.Shape[3];
        _closedSetText ??= EncodeNames(_classNames);
        var (maskLogits, classProbabilities) = ClassifyQueries(raw, _closedSetText, ClipWeights(_classNames));
        var withoutVoid = Engine.TensorSlice(classProbabilities, new[] { 0, 0, 0 }, new[] { b, _numQueries, _numClasses });
        var scores = Common.MaskClassificationOps.SemanticLogScores(withoutVoid, Engine.TensorSigmoid(maskLogits), h, w);
        if (!hasBatch) scores = RemoveBatchDimension(scores); return scores;
    }

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(ODISEModelSize modelSize) => modelSize switch
    {
        ODISEModelSize.Base => ([64, 128, 320, 512], [2, 2, 4, 2], 256),
        ODISEModelSize.Large => ([96, 192, 384, 768], [2, 4, 8, 2], 256),
        _ => ([64, 128, 320, 512], [2, 2, 4, 2], 256)
    };

    /// <summary>
    /// Forward pass with Stable Diffusion U-Net skip connections (Xu et al. 2023 §3).
    /// The encoder is run stage-by-stage and each stage's output is tapped; the decoder
    /// then upsamples the bottleneck and concatenates the matching-resolution encoder
    /// tap before each successive upsampling stage. The skips inject the encoder's
    /// high-resolution (spatially-detailed) features into the decoder so the dense
    /// per-pixel head is conditioned on per-pixel input — without them the 32x
    /// downsample collapses a small input to a 1x1 bottleneck and the output is
    /// spatially uniform.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var scores = SemanticScores(RawHeadOutput(input));
        if (!hasBatch) scores = RemoveBatchDimension(scores); return scores;
    }

    /// <summary>The decoder's raw head channels <c>[B, HeadChannels, H, W]</c> for a batched input.</summary>
    private Tensor<T> RawHeadOutput(Tensor<T> input)
    {
        // When the layers come from a pre-supplied Architecture (no generated
        // encoder/decoder split), fall back to a plain sequential pass.
        if (_encoderStageLayerCounts is null)
        {
            var seq = input;
            for (int i = 0; i < Layers.Count; i++) seq = Layers[i].Forward(seq);
            return seq;
        }

        var features = input;
        int li = 0;
        int n = _encoderStageLayerCounts.Length;

        // Encoder: run each stage and tap its output (taps[n-1] is the bottleneck).
        var taps = new Tensor<T>[n];
        for (int s = 0; s < n; s++)
        {
            int stageLayers = _encoderStageLayerCounts[s];
            for (int k = 0; k < stageLayers; k++) features = Layers[li++].Forward(features);
            taps[s] = features;
        }

        // Decoder: n-1 transposed-conv upsampling stages (3 layers each). After each
        // upsample, concat the encoder tap at the now-matching resolution (the stage
        // whose output shares that resolution): ds=0 -> taps[n-2], ds=1 -> taps[n-3], ...
        for (int ds = 0; ds < n - 1; ds++)
        {
            for (int k = 0; k < DecoderStageLayerCount; k++) features = Layers[li++].Forward(features);
            features = Engine.TensorConcatenate(new[] { features, taps[n - 2 - ds] }, axis: 1);
        }

        // Stem-reversal upsample stage + dense head — no skip (encoder has no feature
        // map above the stem resolution). Run all remaining layers sequentially.
        while (li < Layers.Count) features = Layers[li++].Forward(features);

        return features;
    }

    /// <summary>
    /// Training forward MUST use the same skip-connection path as inference so the lazy
    /// decoder convolutions resolve their input channel counts against the CONCATENATED
    /// feature maps. A plain sequential pass would resolve them against non-concatenated
    /// inputs and then mismatch at the first real (skip) forward.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        NumOps.ToFloatSpan(input.Data.Span.Slice(0, input.Length), inputData);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        NumOps.FromFloatSpan(outputTensor.ToDenseTensor().Buffer.Span, outputData);
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }
    #endregion

    #region Abstract Implementation
    /// <summary>
    /// Initializes the encoder and decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the neural network layers.
    /// In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Architecture.Layers.Count / 2; }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateODISEEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            // Layers per encoder stage (mirrors CreateODISEEncoderLayers): a strided
            // downsample Conv + GroupNorm + SiLU, then (depth-1) residual blocks each
            // wrapped as ResidualLayer + GroupNorm.
            _encoderStageLayerCounts = new int[_depths.Length];
            for (int s = 0; s < _depths.Length; s++) _encoderStageLayerCounts[s] = 3 + (_depths[s] - 1) * 2;
            // The final 1x1 convolution emits the mask-classification head's channels; its query rows are the
            // learnable initial queries.
            var decoderLayers = LayerHelper<T>.CreateODISEDecoderLayers(_channelDims, _decoderDim, HeadChannels);
            Layers.AddRange(decoderLayers);
        }
    }

    private bool _lazyShapesResolved;

    /// <summary>
    /// Resolves every lazy encoder/decoder layer's input/output shape by running
    /// ONE real forward through ODISE's actual skip-connection topology.
    /// </summary>
    /// <remarks>
    /// ODISE's decoder is the Stable-Diffusion U-Net decoder: <see cref="Forward"/>
    /// CONCATENATES each encoder tap onto the upsampled feature map before the next
    /// transposed-conv stage (Xu et al. 2023 §3), so a post-concat deconv's true
    /// input channel count is DOUBLE its sequential predecessor's (e.g. 320 + 320 =
    /// 640). The base class's sequential shape-walk cannot model the concats — it
    /// would resolve that deconv against its non-concatenated predecessor (320),
    /// pin the lazy kernel to the wrong inChannels, and crash the first real
    /// (skip-path) forward with "Input inChannels (640) must match kernel inChannels
    /// (320)". Resolving through the real forward fixes the channel counts AND keeps
    /// <see cref="NeuralNetworkBase{T}.ParameterCount"/> non-zero before the first
    /// user forward (the contract the base method upholds for ModelFamily's
    /// pre-forward Parameters_ShouldBeNonEmpty invariant).
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesResolved) return;
        // Called during base construction before InitializeLayers has run (Layers still
        // empty, _useNativeMode not yet assigned): nothing to resolve and nothing to
        // latch — the post-construction call does the real work. Check this BEFORE the
        // native-mode guard so a base-ctor call can't prematurely mark shapes resolved.
        if (Layers is null || Layers.Count == 0) return;
        // ONNX mode has no native layers to resolve.
        if (!_useNativeMode) { _lazyShapesResolved = true; return; }
        // A single inference-mode forward through the real skip topology
        // materializes every lazy layer with its correct (concatenated) channels.
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "ODISE" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };
    #endregion

    #region IPanopticSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, NumStuffClasses and NumThingClasses
    // all come from the base; only the model-specific override remains.

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Classifies the queries against the closed-set <see cref="ODISEOptions.ClassNames"/> (classes below
    /// <c>NumStuffClasses</c> are stuff) and merges them with Mask2Former's panoptic inference.
    /// </para>
    /// </remarks>
    public override PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        _closedSetText ??= EncodeNames(_classNames);
        var isThing = new bool[_numClasses];
        for (int c = NumStuffClasses; c < _numClasses; c++) isThing[c] = true;
        return SegmentPanopticCore(image, _closedSetText, ClipWeights(_classNames), isThing);
    }

    /// <summary>
    /// Open-vocabulary panoptic segmentation: segments <paramref name="image"/> into the given stuff and thing
    /// categories, which need not be the training classes.
    /// </summary>
    /// <param name="image">Image <c>[C, H, W]</c> or <c>[1, C, H, W]</c>.</param>
    /// <param name="stuffClassNames">Names of the stuff categories (class ids 0 .. S-1).</param>
    /// <param name="thingClassNames">Names of the thing categories (class ids S .. S+T-1).</param>
    /// <returns>The panoptic result; void pixels are <c>-1</c> in the semantic and panoptic maps.</returns>
    /// <remarks>
    /// <para>
    /// Each query's class distribution fuses the diffusion classifier with the mask-pooled classifier per
    /// category: in-vocabulary names (<see cref="ODISEOptions.InVocabularyClassNames"/>) use alpha = 0.35,
    /// the rest beta = 0.65 (Xu et al. 2023).
    /// </para>
    /// </remarks>
    public PanopticSegmentationResult<T> SegmentPanoptic(
        Tensor<T> image, IReadOnlyList<string> stuffClassNames, IReadOnlyList<string> thingClassNames)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (stuffClassNames is null) throw new ArgumentNullException(nameof(stuffClassNames));
        if (thingClassNames is null) throw new ArgumentNullException(nameof(thingClassNames));
        var names = stuffClassNames.Concat(thingClassNames).ToArray();
        if (names.Length == 0) throw new ArgumentException("At least one category name is required.", nameof(stuffClassNames));
        var isThing = new bool[names.Length];
        for (int c = stuffClassNames.Count; c < names.Length; c++) isThing[c] = true;
        return SegmentPanopticCore(image, EncodeNames(names), ClipWeights(names), isThing);
    }

    private PanopticSegmentationResult<T> SegmentPanopticCore(Tensor<T> image, Tensor<T> text, Tensor<T> clipWeights, bool[] isThing)
    {
        var batched = image.Rank == 4 ? image : AddBatchDimension(image);
        if (batched.Shape[0] != 1)
            throw new ArgumentException("SegmentPanoptic takes a single image.", nameof(image));
        using var noGrad = new NoGradScope<T>();
        var raw = _useNativeMode ? RawHeadOutput(batched) : PredictOnnx(batched);
        int h = raw.Shape[2], w = raw.Shape[3], k = text.Shape[0];
        var (maskLogits, classProbabilities) = ClassifyQueries(raw, text, clipWeights);
        var masks = Engine.Reshape(Engine.TensorSigmoid(maskLogits), new[] { _numQueries, h, w });
        var probabilities = Engine.Reshape(classProbabilities, new[] { _numQueries, k + 1 });
        return Common.MaskClassificationOps.MergeMask2Former(
            probabilities, masks, isThing, _options.ObjectMaskThreshold, _options.OverlapThreshold);
    }
    #endregion
}
