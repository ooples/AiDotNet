using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// PixelLM: Pixel Reasoning with Large Multimodal Model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Multi-target reasoning segmentation. Complex text-to-mask segmentation.
///
/// Common use cases:
/// - Multi-target reasoning segmentation
/// - Complex text-to-mask segmentation
/// - Pixel-level reasoning from language
/// - Detailed visual grounding
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Lightweight pixel decoder with codebook tokens
/// - Multi-scale feature integration for segmentation
/// - Handles multiple target masks from a single query
/// - End-to-end LLM + pixel decoder training
/// </para>
/// <para>
/// <b>Reference:</b> Ren et al., "PixelLM: Pixel Reasoning with Large Multimodal Model", CVPR 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a PixelLM model for multi-target reasoning segmentation
/// // Any CLIP-style ITextEncoder supplies the text embeddings, e.g. a CLIP model from VisionLanguage.Encoders.
/// ITextEncoder&lt;double&gt; textEncoder = clipModel;
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new PixelLM&lt;double&gt;(architecture, textEncoder);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new PixelLM&lt;double&gt;(architecture, textEncoder,
///     options: new PixelLMOptions { NumClasses = 1, DropRate = 0 });
///
/// // Or load a pre-trained ONNX model for pixel-level reasoning
/// var onnxModel = new PixelLM&lt;double&gt;(architecture, "pixellm.onnx", textEncoder);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PixelLM: Pixel Reasoning with Large Multimodal Model", "https://arxiv.org/abs/2312.02228", Year = 2024, Authors = "Ren et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.95,
                WeightDecay = 0, WarmupSteps = 100,
                Source = "Ren et al. 2024, config table: AdamW with a base learning rate of 3.0e-4, a "
                        + "weight decay of 0, optimizer momentum beta1 0.9 and beta2 0.95, and 100 "
                        + "warmup iterations. The stated batch size of 16 is not declared as a reference "
                        + "batch size: the table gives a base learning rate, and pairing a base rate "
                        + "with a batch size would apply the linear scaling rule to a figure that has "
                        + "already been scaled. Routing is handled by "
                        + "SegmentationModelBase.CreateDefaultOptimizer, which consults "
                        + "PaperOptimizerFactory itself, so no call is inserted here.")]
public partial class PixelLM<T> : Common.ReferringSegmentationBase<T>
{
    private readonly PixelLMOptions _options;

    #region Fields
    // Only PixelLM's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from ReferringSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode and MaxTextLength (512)
    // are all inherited and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// PixelLM's default optimizer is driven entirely by <see cref="PixelLMOptions"/>, so it
    /// overrides the base's plain-AdamW default rather than accepting it.
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        // The base's own default routes through PaperOptimizerFactory; this override would
        // bypass it, so the paper recipe is consulted here before falling back to the
        // options-driven optimizer, which is then checked against the declaration.
        => PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
        ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                BatchSize = _options.OptimizerBatchSize,
                InitialLearningRate = _options.LearningRate,
                Beta1 = _options.OptimizerBeta1,
                Beta2 = _options.OptimizerBeta2,
                Epsilon = _options.OptimizerEpsilon,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                EnableGradientClipping = false,
                MaxGradientNorm = 0.0,
            }));
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes PixelLM in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="textEncoder">CLIP-style text encoder whose embeddings the pixel embeddings are scored against; the decoder emits embeddings of its TextEmbeddingDimension.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable PixelLM model.
    /// </para>
    /// </remarks>
    public PixelLM(NeuralNetworkArchitecture<T> architecture,
        ITextEncoder<T> textEncoder,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        PixelLMOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer() - which PixelLM overrides above
        // with its options-driven AdamW - so null is passed straight through.
        : base(architecture, optimizer, lossFunction, textEncoder, options ??= new PixelLMOptions())
    {
        _options = options; Options = _options;
        // PixelLM's own fallback input geometry is 1024x1024, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
        _dropRate = _options.DropRate;
        ValidateArchitectureOptions(_options);
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.StageDepths.Clone();
        _decoderDim = _options.DecoderDimension;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes PixelLM in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="textEncoder">CLIP-style text encoder whose embeddings the pixel embeddings are scored against; the decoder emits embeddings of its TextEmbeddingDimension.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained PixelLM from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public PixelLM(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITextEncoder<T> textEncoder,
        PixelLMOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, textEncoder, options ??= new PixelLMOptions())
    {
        _options = options; Options = _options;
        // PixelLM's own fallback input geometry is 1024x1024, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
        _dropRate = _options.DropRate;
        ValidateArchitectureOptions(_options);
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.StageDepths.Clone();
        _decoderDim = _options.DecoderDimension;
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
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : ClassifyPixels(PredictOnnx(input));

    #endregion

    #region Private Methods
    private static void ValidateArchitectureOptions(PixelLMOptions options)
    {
        if (options.ChannelDimensions is null)
            throw new ArgumentNullException(nameof(options.ChannelDimensions));
        if (options.StageDepths is null)
            throw new ArgumentNullException(nameof(options.StageDepths));
        if (options.ChannelDimensions.Length == 0 ||
            options.ChannelDimensions.Length != options.StageDepths.Length)
        {
            throw new ArgumentException(
                "ChannelDimensions and StageDepths must be non-empty arrays with matching lengths.",
                nameof(options));
        }
        if (options.ChannelDimensions.Any(value => value <= 0) ||
            options.StageDepths.Any(value => value <= 0) ||
            options.DecoderDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "All PixelLM channel dimensions, stage depths, and the decoder dimension must be positive.");
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        features = ClassifyPixels(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
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
            var encoderLayers = LayerHelper<T>.CreatePixelLMEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreatePixelLMDecoderLayers(_channelDims[^1], _decoderDim, TextEmbeddingDimension, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "PixelLM" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };
    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and latches
    // _disposed, and PixelLM owns no other unmanaged resource.
    #endregion

    #region IReferringSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, MaxTextLength (512),
    // SupportsConversation (false) and SupportsVideoInput (false) all come from the base;
    // only the model-specific members remain.

    // SegmentFromExpression comes from ReferringSegmentationBase: the expression's CLIP text embedding
    // is scored against the decoder's pixel embeddings and the sigmoid mask is thresholded.

    /// <inheritdoc/>
    /// <remarks>
    /// Overrides the public method rather than the base's SupportsConversation-gated
    /// SegmentFromConversationInternal hook: PixelLM reports SupportsConversation = false yet has
    /// always concatenated the history and answered, and re-parenting must not change that.
    /// </remarks>
    public override ReferringSegmentationResult<T> SegmentFromConversation(
        Tensor<T> image, IReadOnlyList<(string Role, string Message)> conversationHistory, string currentQuery)
    {
        var context = string.Join(" ", conversationHistory.Select(c => c.Message));
        var fullQuery = string.IsNullOrEmpty(context) ? currentQuery : $"{context} {currentQuery}";
        return SegmentFromExpression(image, fullQuery);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Overrides the public method rather than the base's SupportsVideoInput-gated
    /// SegmentVideoFromExpressionInternal hook, for the same reason as SegmentFromConversation.
    /// </remarks>
    public override List<ReferringSegmentationResult<T>> SegmentVideoFromExpression(Tensor<T> frames, string expression)
    {
        return SegmentEachFrame(frames, expression);
    }
    #endregion
}
