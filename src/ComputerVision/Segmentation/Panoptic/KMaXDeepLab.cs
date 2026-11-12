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
/// - k-means cross-attention replacing standard cross-attention
/// - Cluster centers as mask queries updated iteratively
/// - Pixel-path and cluster-path dual decoder
/// - State-of-the-art panoptic quality on COCO and Cityscapes
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
    /// <param name="lossFunction">Loss function. When null, training uses kMaX-DeepLab's set-prediction loss (Hungarian
    /// matching, PQ-style loss and mask-ID cross-entropy); a supplied loss is applied to the semantic scores.</param>
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
    /// Trains with kMaX-DeepLab's set-prediction loss (Hungarian matching, PQ-style loss and mask-ID cross-entropy)
    /// when no loss function was given to the constructor.
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

    /// <summary>kMaX-DeepLab's set-prediction loss of one training batch, recomputed from the input.</summary>
    internal Tensor<T> SetPredictionLoss(Tensor<T> input, Tensor<T> expected)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        bool hasBatch = input.Rank == 4;
        var batched = hasBatch ? input : AddBatchDimension(input);
        var target = hasBatch ? expected : Engine.Reshape(expected, new[] { 1 }.Concat(expected.Shape.ToArray()).ToArray());
        var (classLogits, maskLogits, h, w) = Heads(batched);
        var segments = Common.MaskSetPredictionLoss.SemanticSegments(target, _numClasses, batched.Shape[0], h, w);
        return Common.MaskSetPredictionLoss.KMaX(classLogits, maskLogits, segments);
    }

    #endregion

    #region Private Methods
    /// <summary>Backbone stages the pixel decoder reads: strides 4, 8, 16 and 32 (Yu et al. 2022, Fig. 2).</summary>
    private const int PyramidLevels = 4;

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
    /// The mask-classification heads of a batched image: class logits <c>[B, N, K + 1]</c> and cluster logits
    /// <c>[B, N, H*W]</c> at the input resolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The pixel decoder is a top-down pathway over the backbone's stride-4, 8, 16 and 32 stages. The kMaX decoder's
    /// k-means cross-attention layers read its stride-32, 16 and 8 features, coarse to fine (with the default six
    /// layers, two per scale), and the final centers are compared with the stride-4 features to give the mask
    /// logits, which are bilinearly upsampled to the input (Yu et al. 2022, Sec. 3.3). The class head classifies the
    /// final centers.
    /// </para>
    /// </remarks>
    private (Tensor<T> ClassLogits, Tensor<T> MaskLogits, int Height, int Width) Heads(Tensor<T> batched)
    {
        if (!_useNativeMode) return OnnxHeads(batched);
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
        var flat = levels.Select(level => Engine.Reshape(level, new[] { b, _pixelDim, level.Shape[2] * level.Shape[3] })).ToList();
        // The decoder stages read the coarser levels, coarsest first; the masks use the finest level.
        var decoderScales = levelCount == 1 ? flat : Enumerable.Reverse(flat.Skip(1)).ToList();
        var identity = Engine.TensorOneHot<T>(
            new Tensor<int>(Enumerable.Range(0, _numClusters).ToArray(), new[] { _numClusters }), _numClusters);
        var initialCenters = Layers[_encoderLayerEnd + 2 * PyramidLevels].Forward(identity);         // [N, D]
        var (centers, maskLogits) = Common.MaskClassificationOps.KMeansDecoder(
            decoderScales, _kMeansIterations, initialCenters, flat[0]);
        var classLogits = Engine.Reshape(
            Layers[_encoderLayerEnd + 2 * PyramidLevels + 1].Forward(Engine.Reshape(centers, new[] { b * _numClusters, _pixelDim })),
            new[] { b, _numClusters, _numClasses + 1 });
        var finest = levels[0];
        var upsampled = Engine.Interpolate(
            Engine.Reshape(maskLogits, new[] { b, _numClusters, finest.Shape[2], finest.Shape[3] }),
            new[] { height, width }, InterpolateMode.Bilinear, false);
        return (classLogits, Engine.Reshape(upsampled, new[] { b, _numClusters, height * width }), height, width);
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
    /// decoder's four lateral 1x1 convolutions and four output 3x3 convolutions (to the D-wide pixel features),
    /// the learnable initial cluster centers (a dense layer read on the identity) and the class head on the centers.
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
        Layers.Add(new DenseLayer<T>(_pixelDim));
        Layers.Add(new DenseLayer<T>(_numClasses + 1));
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
        var classProbabilities = Engine.Reshape(Engine.TensorSoftmax(classLogits, 2), new[] { _numClusters, _numClasses + 1 });
        var maskProbabilities = Engine.Reshape(Engine.TensorSoftmax(maskLogits, 1), new[] { _numClusters, h, w });
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
