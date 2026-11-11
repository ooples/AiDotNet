using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// SegGPT: Segmenting Everything In Context via in-context learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In-context segmentation from examples. Few-shot segmentation without fine-tuning.
///
/// Common use cases:
/// - In-context segmentation from examples
/// - Few-shot segmentation without fine-tuning
/// - Interactive image editing
/// - Versatile segmentation across domains
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - In-context learning for segmentation (no fine-tuning needed)
/// - ViT-Large backbone with random color mapping training
/// - Feature ensemble from multiple in-context examples
/// - Unified framework for semantic, instance, and part segmentation
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "SegGPT: Segmenting Everything In Context", ICCV 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SegGPT model for in-context few-shot segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 448, inputWidth: 448, inputDepth: 3, outputSize: 1);
/// var model = new SegGPT&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new SegGPT&lt;double&gt;(architecture,
///     options: new SegGPTOptions { NumClasses = 1, DropRate = 0.1, ModelSize = SegGPTModelSize.ViTLarge });
///
/// // Or load a pre-trained ONNX model for example-guided segmentation
/// var onnxModel = new SegGPT&lt;double&gt;(architecture, "seggpt.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SegGPT: Segmenting Everything In Context", "https://arxiv.org/abs/2304.03284", Year = 2023, Authors = "Wang et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 0.05,
                ReferenceBatchSize = 2048, WarmupFraction = 0.2, MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Wang et al. 2023, Sec. 4: AdamW with a cosine schedule, base learning rate 1e-4, weight decay 0.05, batch size 2048, and a 1.8K warm-up over a 9K iteration run, which is the 20% declared here rather than a step count fixed to that run.")]
public partial class SegGPT<T> : Common.PromptableSegmentationBase<T>
{
    private readonly SegGPTOptions _options;

    #region Fields
    // Only SegGPT's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PromptableSegmentationBase -> SegmentationModelBase, as do _imageEmbedding and
    // _imageSet.
    private readonly SegGPTModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal SegGPTModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes SegGPT in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SegGPT model.
    /// </para>
    /// </remarks>
    public SegGPT(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SegGPTOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults the loss to CrossEntropyWithLogitsLoss - exactly what the deleted lines did by
        // hand. `optimizer` is passed straight through INCLUDING null; the base's lazy
        // CreateDefaultOptimizer() produces the same `new AdamWOptimizer<...>(this)` default, which
        // could never be written as a base-constructor argument because `this` is unavailable there.
        : base(architecture, optimizer, lossFunction, (options ??= new SegGPTOptions()).NumClasses)
    {
        _options = options; Options = _options;
        ApplySegGPTInputFallback(architecture);
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options);
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies SegGPT's 448x448 fallback for architectures that carry no input geometry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SegmentationModelBase falls back to 512x512 when the architecture supplies no input height
    /// or width. SegGPT's documented fallback is 448x448 (the in-context painting resolution the
    /// paper uses), so it is restored here for that unset case only - when the architecture does
    /// specify dimensions, the base's value already matches and nothing changes.
    /// </para>
    /// </remarks>
    private void ApplySegGPTInputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 448;
        if (architecture.InputWidth <= 0) _width = 448;
    }

    /// <summary>
    /// Initializes SegGPT in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SegGPT from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SegGPT(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SegGPTOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same lines this used to repeat.
        : base(architecture, onnxModelPath, (options ??= new SegGPTOptions()).NumClasses)
    {
        _options = options; Options = _options;
        ApplySegGPTInputFallback(architecture);
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    // PredictCore is inherited from SegmentationModelBase and dispatches to Forward / PredictOnnx
    // exactly as the deleted override did.

    #endregion

    #region Private Methods
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SegGPTOptions options)
    {
        if (options.ChannelDimensions is null || options.ChannelDimensions.Length != 4)
            throw new ArgumentException("ChannelDimensions must contain exactly four positive values.", nameof(options));
        if (options.StageDepths is null || options.StageDepths.Length != 4)
            throw new ArgumentException("StageDepths must contain exactly four positive values.", nameof(options));
        if (options.ChannelDimensions.Any(value => value <= 0))
            throw new ArgumentException("ChannelDimensions must contain only positive values.", nameof(options));
        if (options.StageDepths.Any(value => value <= 0))
            throw new ArgumentException("StageDepths must contain only positive values.", nameof(options));
        if (options.DecoderDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DecoderDimension must be positive.");

        return (options.ChannelDimensions.ToArray(), options.StageDepths.ToArray(), options.DecoderDimension);
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var onnxInput = CvTensorOps<T>.ToOnnxFloat(input);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var result = CvTensorOps<T>.FromOnnxFloat(outputTensor);
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    // AddBatchDimension and RemoveBatchDimension are inherited from SegmentationModelBase.
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
            var encoderLayers = LayerHelper<T>.CreateSegGPTEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateSegGPTDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "SegGPT" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // SegGPT owns no further unmanaged resources.
    #endregion

    #region IPromptableSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase.
    // _imageEmbedding, _imageSet, SetImage and the Supports*Prompts flags (point/box/mask true,
    // text false) all come from PromptableSegmentationBase with these exact values; only SegGPT's
    // cached class probabilities are model-specific.
    [Scratch]
    private Tensor<T>? _imageProbabilities;

    /// <summary>
    /// Encodes an image by running a full forward pass, as SegGPT's explicit SetImage did.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The base's SetImage stores this return value in <c>_imageEmbedding</c> and marks the image
    /// set, so the observable behaviour is unchanged; SetImage below only adds the probability cache.
    /// </para>
    /// </remarks>
    protected override Tensor<T> EncodeImage(Tensor<T> image) => Predict(image);

    /// <inheritdoc/>
    public override void SetImage(Tensor<T> image)
    {
        base.SetImage(image);
        var embedding = _imageEmbedding;
        if (embedding is not null)
        {
            _imageProbabilities = Common.SegmentationTensorOps.SoftmaxAlongClassDim(embedding);
        }
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.PointPromptScoreMap(features, points, labels);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.BoxPromptScoreMap(features, box);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.MaskPromptScoreMap(features, mask);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override List<PromptedSegmentationResult<T>> SegmentEverything()
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var probs = _imageProbabilities ?? Common.SegmentationTensorOps.SoftmaxAlongClassDim(features);
        return Common.SegmentationTensorOps.ClassMaskCandidates(features, probs);
    }

    private PromptedSegmentationResult<T> BuildPromptMaskResult(Tensor<T> scoreMap, int h, int w)
        => Common.SegmentationTensorOps.BuildPromptMaskResult(scoreMap, h, w);
    #endregion
}
