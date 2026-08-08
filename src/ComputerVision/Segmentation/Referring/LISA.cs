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
/// LISA: Reasoning Segmentation via Large Language Model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Reasoning-based segmentation from complex text queries. Implicit referring segmentation.
///
/// Common use cases:
/// - Reasoning-based segmentation from complex text queries
/// - Implicit referring segmentation
/// - Conversational image segmentation
/// - World-knowledge-powered segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Large Language Model (LLaVA) + SAM mask decoder
/// - Embedding-as-mask paradigm: LLM output tokens control SAM
/// - Handles complex reasoning queries (not just simple references)
/// - End-to-end trainable with LoRA fine-tuning
/// </para>
/// <para>
/// <b>Reference:</b> Lai et al., "LISA: Reasoning Segmentation via Large Language Model", CVPR 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a LISA model for reasoning-based segmentation from text queries
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new LISA&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for conversational segmentation
/// var onnxModel = new LISA&lt;double&gt;(architecture, "lisa.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LISA: Reasoning Segmentation via Large Language Model", "https://arxiv.org/abs/2308.00692", Year = 2024, Authors = "Lai et al.")]
public class LISA<T> : Common.ReferringSegmentationBase<T>
{
    private readonly LISAOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only LISA's OWN configuration lives here. _height, _width, _channels, _numClasses,
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
    /// Initializes LISA in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable LISA model.
    /// </para>
    /// </remarks>
    public LISA(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        LISAOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer(), so null is passed straight through.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new LISAOptions(); Options = _options;
        // LISA's own fallback input geometry is 1024x1024, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
        _dropRate = dropRate;
        _channelDims = [64, 128, 320, 768];
        _depths = [2, 2, 4, 12];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes LISA in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained LISA from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public LISA(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        LISAOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new LISAOptions(); Options = _options;
        // LISA's own fallback input geometry is 1024x1024, not the base's 512.
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
            var encoderLayers = LayerHelper<T>.CreateLISAEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateLISADecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">Flat vector of all model parameters.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces all model weights with new values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    { int o = 0; foreach (var l in Layers) { var p = l.GetParameters(); int c = p.Length; if (o + c <= parameters.Length) { var n = new Vector<T>(c); for (int i = 0; i < c; i++) n[i] = parameters[o + i]; l.UpdateParameters(n); o += c; } } }

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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "LISA" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    /// <summary>
    /// Writes configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves model configuration for later reconstruction.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    { writer.Write(_height); writer.Write(_width); writer.Write(_channels); writer.Write(_numClasses); writer.Write(_decoderDim); writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty); writer.Write(_encoderLayerEnd); writer.Write(_channelDims.Length); foreach (int d in _channelDims) writer.Write(d); writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d); }

    /// <summary>
    /// Reads configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    { _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32(); int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32(); int dd = reader.ReadInt32(); for (int i = 0; i < dd; i++) _ = reader.ReadInt32(); }

    /// <summary>
    /// Creates a new instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new LISA<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new LISA<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);
    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and latches
    // _disposed, and LISA owns no other unmanaged resource.
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
    /// SegmentVideoFromExpressionInternal hook: LISA reports SupportsVideoInput = false yet has
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
