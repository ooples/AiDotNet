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

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// GLaMM: Grounding Large Multimodal Model for pixel-level understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Grounded conversation about images. Pixel-level visual understanding with language.
///
/// Common use cases:
/// - Grounded conversation about images
/// - Pixel-level visual understanding with language
/// - Region-specific captioning and segmentation
/// - Multi-turn visual dialogue with grounding
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Grounding LMM with pixel-level output capability
/// - Generates text with embedded segmentation masks
/// - Region-level and pixel-level visual features
/// - Trained on Grounding-anything Dataset (GranD)
/// </para>
/// <para>
/// <b>Reference:</b> Rasheed et al., "GLaMM: Pixel Grounding Large Multimodal Model", CVPR 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a GLaMM model for grounded pixel-level visual understanding
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new GLaMM&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for grounded conversation
/// var onnxModel = new GLaMM&lt;double&gt;(architecture, "glamm.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("GLaMM: Pixel Grounding Large Multimodal Model", "https://arxiv.org/abs/2311.03356", Year = 2024, Authors = "Rasheed et al.")]
public partial class GLaMM<T> : Common.ReferringSegmentationBase<T>
{
    private readonly GLaMMOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only GLaMM's OWN configuration lives here. _height, _width, _channels, _numClasses,
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
    /// Paper-faithful LR: Rasheed et al. 2024 MBZUAI uses 5e-5 for GLaMM fine-tuning. The framework
    /// default LR=1e-3 diverges on this VLM stack, so GLaMM overrides the base's plain-AdamW default.
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 5e-5 });
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes GLaMM in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable GLaMM model.
    /// </para>
    /// </remarks>
    public GLaMM(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        GLaMMOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer() - which GLaMM overrides above with
        // the paper's 5e-5 AdamW - so null is passed straight through.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new GLaMMOptions(); Options = _options;
        // GLaMM's own fallback input geometry is 1024x1024, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
        _dropRate = dropRate;
        _channelDims = [64, 128, 320, 768];
        _depths = [2, 2, 4, 12];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes GLaMM in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained GLaMM from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public GLaMM(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        GLaMMOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new GLaMMOptions(); Options = _options;
        // GLaMM's own fallback input geometry is 1024x1024, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
        _dropRate = 0;
        _channelDims = [64, 128, 320, 768];
        _depths = [2, 2, 4, 12];
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
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

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
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    #endregion

    #region Private Methods
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
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
            var encoderLayers = LayerHelper<T>.CreateGLaMMEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateGLaMMDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "GLaMM" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };
    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and latches
    // _disposed, and GLaMM owns no other unmanaged resource.
    #endregion

    #region IReferringSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, MaxTextLength (512) and
    // SupportsVideoInput (false) all come from the base; only the model-specific members remain.

    /// <inheritdoc/>
    public override bool SupportsConversation => true;

    /// <inheritdoc/>
    public override ReferringSegmentationResult<T> SegmentFromExpression(Tensor<T> image, string expression)
    {
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        int numC = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
        var weights = Common.SegmentationTensorOps.TextToWeights(expression, numC);
        var scoreMap = Common.SegmentationTensorOps.WeightedChannelSum(logits, weights);
        var probs = Common.SegmentationTensorOps.Sigmoid(scoreMap);
        var binaryMask = Common.SegmentationTensorOps.ThresholdMask(probs, 0.5);
        int minY = h, maxY = 0, minX = w, maxX = 0;
        double area = 0, confSum = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (NumOps.GreaterThan(binaryMask[y, x], NumOps.FromDouble(0.5)))
                {
                    area++; confSum += NumOps.ToDouble(probs[y, x]);
                    if (y < minY) minY = y; if (y > maxY) maxY = y;
                    if (x < minX) minX = x; if (x > maxX) maxX = x;
                }
        double confidence = area > 0 ? confSum / area : 0;
        var masks = new Tensor<T>([1, h, w]);
        binaryMask.Data.Span.CopyTo(masks.Data.Span);
        var boxes = new Tensor<T>([1, 4]);
        if (area > 0) { boxes[0, 0] = NumOps.FromDouble(minX); boxes[0, 1] = NumOps.FromDouble(minY); boxes[0, 2] = NumOps.FromDouble(maxX); boxes[0, 3] = NumOps.FromDouble(maxY); }
        string response = area > 0
            ? $"Segmented region matching '{expression}' with {(int)area} pixels ({confidence:F2} confidence)"
            : $"No region found matching '{expression}'";
        return new ReferringSegmentationResult<T> { Masks = masks, TextResponse = response, Confidence = confidence, BoundingBoxes = boxes };
    }

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
    /// SegmentVideoFromExpressionInternal hook: GLaMM reports SupportsVideoInput = false yet has
    /// always returned real per-frame results here, and re-parenting must not change that.
    /// </remarks>
    public override List<ReferringSegmentationResult<T>> SegmentVideoFromExpression(Tensor<T> frames, string expression)
    {
        var results = new List<ReferringSegmentationResult<T>>();
        if (frames.Rank == 3) { var r = SegmentFromExpression(frames, expression); r.FrameIndex = 0; results.Add(r); return results; }
        int nf = frames.Shape[0], c = frames.Shape[1], fh = frames.Shape[2], fw = frames.Shape[3];
        for (int f = 0; f < nf; f++)
        {
            var frame = new Tensor<T>([c, fh, fw]);
            for (int ch = 0; ch < c; ch++) for (int y = 0; y < fh; y++) for (int x = 0; x < fw; x++) frame[ch, y, x] = frames[f, ch, y, x];
            var r = SegmentFromExpression(frame, expression);
            r.FrameIndex = f; results.Add(r);
        }
        return results;
    }
    #endregion
}
