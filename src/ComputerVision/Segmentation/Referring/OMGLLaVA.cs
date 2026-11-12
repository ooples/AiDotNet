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
/// OMG-LLaVA: Bridging Image-Level, Object-Level, and Pixel-Level understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Unified visual perception and understanding. Object-level and pixel-level reasoning.
///
/// Common use cases:
/// - Unified visual perception and understanding
/// - Object-level and pixel-level reasoning
/// - Visual question answering with segmentation
/// - Multi-granularity visual understanding
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Universal segmentation model (Mask2Former) + LLM (LLaVA)
/// - Bridges perception (segmentation) and understanding (reasoning)
/// - Object-level queries from segmentation feed into LLM
/// - Pixel-level grounding from LLM responses
/// </para>
/// <para>
/// <b>Reference:</b> Zhang et al., "OMG-LLaVA: Bridging Image-Level, Object-Level, Pixel-Level Reasoning and Understanding", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an OMG-LLaVA model for multi-granularity visual understanding
/// // Any CLIP-style ITextEncoder supplies the text embeddings, e.g. a CLIP model from VisionLanguage.Encoders.
/// ITextEncoder&lt;double&gt; textEncoder = clipModel;
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new OMGLLaVA&lt;double&gt;(architecture, textEncoder);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new OMGLLaVA&lt;double&gt;(architecture, textEncoder,
///     options: new OMGLLaVAOptions { NumClasses = 1, DropRate = 0 });
///
/// // Or load a pre-trained ONNX model for visual question answering with segmentation
/// var onnxModel = new OMGLLaVA&lt;double&gt;(architecture, "omgllava.onnx", textEncoder);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("OMG-LLaVA: Bridging Image-Level, Object-Level, Pixel-Level Reasoning and Understanding", "https://arxiv.org/abs/2406.19389", Year = 2024, Authors = "Zhang et al.")]
public partial class OMGLLaVA<T> : Common.ReferringSegmentationBase<T>
{
    private readonly OMGLLaVAOptions _options;

    #region Fields
    // Only OMGLLaVA's OWN configuration lives here. _height, _width, _channels, _numClasses,
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
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes OMGLLaVA in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="textEncoder">CLIP-style text encoder whose embeddings the pixel embeddings are scored against; the decoder emits embeddings of its TextEmbeddingDimension.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable OMGLLaVA model.
    /// </para>
    /// </remarks>
    public OMGLLaVA(NeuralNetworkArchitecture<T> architecture,
        ITextEncoder<T> textEncoder,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        OMGLLaVAOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer(), so null is passed straight through.
        : base(architecture, optimizer, lossFunction, textEncoder, options ??= new OMGLLaVAOptions())
    {
        _options = options; Options = _options;
        // OMGLLaVA's own fallback input geometry is 1024x1024, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
        _dropRate = _options.DropRate;
        _channelDims = [96, 192, 384, 768];
        _depths = [2, 2, 6, 2];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes OMGLLaVA in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="textEncoder">CLIP-style text encoder whose embeddings the pixel embeddings are scored against; the decoder emits embeddings of its TextEmbeddingDimension.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained OMGLLaVA from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public OMGLLaVA(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITextEncoder<T> textEncoder,
        OMGLLaVAOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, textEncoder, options ??= new OMGLLaVAOptions())
    {
        _options = options; Options = _options;
        // OMGLLaVA's own fallback input geometry is 1024x1024, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
        _dropRate = _options.DropRate;
        _channelDims = [96, 192, 384, 768];
        _depths = [2, 2, 6, 2];
        _decoderDim = 256;
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
            var encoderLayers = LayerHelper<T>.CreateOMGLLaVAEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateOMGLLaVADecoderLayers(_channelDims[^1], _decoderDim, TextEmbeddingDimension, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "OMGLLaVA" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };
    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and latches
    // _disposed, and OMGLLaVA owns no other unmanaged resource.
    #endregion

    #region IReferringSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, MaxTextLength (512) and
    // SupportsVideoInput (false) all come from the base; only the model-specific members remain.

    /// <inheritdoc/>
    public override bool SupportsConversation => true;

    // SegmentFromExpression comes from ReferringSegmentationBase: the expression's CLIP text embedding
    // is scored against the decoder's pixel embeddings and the sigmoid mask is thresholded.

    /// <inheritdoc/>
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
    /// SegmentVideoFromExpressionInternal hook: OMGLLaVA reports SupportsVideoInput = false yet has
    /// always returned real per-frame results here, and re-parenting must not change that.
    /// </remarks>
    public override List<ReferringSegmentationResult<T>> SegmentVideoFromExpression(Tensor<T> frames, string expression)
    {
        return SegmentEachFrame(frames, expression);
    }
    #endregion
}
