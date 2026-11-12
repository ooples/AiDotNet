using System.IO;
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
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
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
    /// <c>[B, K, H, W]</c>); its argmax is the semantic map. In ONNX mode the graph must emit the head's raw
    /// channels (pixel embedding, cluster logits, class projections).
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
        => _useNativeMode ? Forward(input) : SemanticScores(PredictOnnx(input));

    #endregion

    #region Private Methods
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

    /// <summary>Channels the decoder emits for the head: pixel embedding, cluster logits, classes plus void.</summary>
    private int HeadChannels => _pixelDim + _numClusters + _numClasses + 1;

    /// <summary>Runs the k-means head on the raw decoder output <c>[B, HeadChannels, H, W]</c>.</summary>
    private (Tensor<T> MaskLogits, Tensor<T> ClassLogits) RunHead(Tensor<T> raw)
        => Common.MaskClassificationOps.KMeansCrossAttention(raw, _pixelDim, _numClusters, _kMeansIterations);

    /// <summary>
    /// MaskFormer semantic inference over the head: class probabilities without void times the cluster
    /// masks (softmax over the clusters, as kMaX-DeepLab assigns every pixel to one cluster).
    /// </summary>
    private Tensor<T> SemanticScores(Tensor<T> raw)
    {
        bool hasBatch = raw.Rank == 4; if (!hasBatch) raw = AddBatchDimension(raw);
        int b = raw.Shape[0], h = raw.Shape[2], w = raw.Shape[3];
        var (maskLogits, classLogits) = RunHead(raw);
        var classProbabilities = Engine.TensorSlice(
            Engine.TensorSoftmax(classLogits, 2), new[] { 0, 0, 0 }, new[] { b, _numClusters, _numClasses });
        var maskProbabilities = Engine.TensorSoftmax(maskLogits, 1);
        var scores = Common.MaskClassificationOps.SemanticLogScores(classProbabilities, maskProbabilities, h, w);
        if (!hasBatch) scores = RemoveBatchDimension(scores); return scores;
    }

    private Tensor<T> RunLayers(Tensor<T> input)
    {
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        return features;
    }

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(KMaXDeepLabModelSize modelSize) => modelSize switch
    {
        KMaXDeepLabModelSize.R50 => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256),
        KMaXDeepLabModelSize.ConvNeXtLarge => ([192, 384, 768, 1536], [3, 3, 27, 3], 256),
        _ => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256)
    };

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var scores = SemanticScores(RunLayers(input));
        if (!hasBatch) scores = RemoveBatchDimension(scores); return scores;
    }

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
            var encoderLayers = LayerHelper<T>.CreateKMaXDeepLabEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            // The decoder's last 1x1 convolution emits the head's channels: the pixel embedding, the initial
            // cluster logits (its weight rows are the learnable initial cluster centers) and the per-pixel
            // class projections.
            var decoderLayers = LayerHelper<T>.CreateKMaXDeepLabDecoderLayers(_channelDims[^1], _decoderDim, HeadChannels, fH, fW);
            Layers.AddRange(decoderLayers);
        }
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
    /// kMaX-DeepLab's panoptic post-processing: a cluster's mask is the pixels whose softmax-over-clusters
    /// probability exceeds <see cref="KMaXDeepLabOptions.PixelConfidenceThreshold"/> (0.4); clusters are
    /// visited by <c>classScore^a * maskScore^b</c> and kept when their class score clears the thing (0.7) or
    /// stuff (0.5) threshold and more than <see cref="KMaXDeepLabOptions.OverlapThreshold"/> (0.8) of their
    /// mask is still unclaimed. Stuff clusters of one class merge into one segment. Classes below
    /// <c>NumStuffClasses</c> are stuff. Void pixels are <c>-1</c> in the semantic and panoptic maps.
    /// </para>
    /// </remarks>
    public override PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var batched = image.Rank == 4 ? image : AddBatchDimension(image);
        using var noGrad = new NoGradScope<T>();
        var raw = _useNativeMode ? RunLayers(batched) : PredictOnnx(batched);
        int h = raw.Shape[2], w = raw.Shape[3];
        var (maskLogits, classLogits) = RunHead(raw);
        var classProbabilities = Engine.Reshape(
            Engine.TensorSlice(Engine.TensorSoftmax(classLogits, 2), new[] { 0, 0, 0 }, new[] { 1, _numClusters, _numClasses + 1 }),
            new[] { _numClusters, _numClasses + 1 });
        var maskProbabilities = Engine.Reshape(
            Engine.TensorSlice(Engine.TensorSoftmax(maskLogits, 1), new[] { 0, 0, 0 }, new[] { 1, _numClusters, h * w }),
            new[] { _numClusters, h, w });
        var settings = new Common.KMaXPanopticSettings(
            _options.ClassThresholdThing, _options.ClassThresholdStuff, _options.PixelConfidenceThreshold,
            _options.OverlapThreshold, _options.ReorderClassWeight, _options.ReorderMaskWeight);
        return Common.MaskClassificationOps.MergeKMaX(classProbabilities, maskProbabilities, ThingClassFlags(), settings);
    }

    /// <summary>Classes at or above <c>NumStuffClasses</c> are things.</summary>
    private bool[] ThingClassFlags()
    {
        var flags = new bool[_numClasses];
        for (int c = NumStuffClasses; c < _numClasses; c++) flags[c] = true;
        return flags;
    }
    #endregion
}
