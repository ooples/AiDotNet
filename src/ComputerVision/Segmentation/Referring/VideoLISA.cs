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
/// Video-LISA: Language-instructed video segmentation with reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Language-instructed video segmentation. Temporal reasoning in video segmentation.
///
/// Common use cases:
/// - Language-instructed video segmentation
/// - Temporal reasoning in video segmentation
/// - Referring video object segmentation
/// - Conversational video understanding
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Extends LISA to video with sparse token memory
/// - Sparse Dense Sampling for temporal video understanding
/// - One-Token-Seg-All: single token triggers multi-frame masks
/// - Handles temporal reasoning queries in video
/// </para>
/// <para>
/// <b>Reference:</b> Bai et al., "One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos", NeurIPS 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Video-LISA model for language-instructed video segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new VideoLISA&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for temporal reasoning segmentation
/// var onnxModel = new VideoLISA&lt;double&gt;(architecture, "videolisa.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos", "https://arxiv.org/abs/2409.19603", Year = 2024, Authors = "Zechen Bai, Tong He, Haiyang Mei, Pichao Wang, Ziteng Gao, Joya Chen, Lei Liu, Zheng Zhang, Mike Zheng Shou")]
public partial class VideoLISA<T> : Common.ReferringSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>Does NOT downsample: measured [1,3,64,64] -> [1,C,64,64].</remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 1);

    private readonly VideoLISAOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only VideoLISA's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from ReferringSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses and MaxTextLength are inherited: SegmentationModelBase supplies
    // the first two and ReferringSegmentationBase defaults MaxTextLength to 512, which is exactly
    // what the explicit interface implementation used to return.
    internal bool UseNativeMode => _useNativeMode;

    /// <inheritdoc/>
    public override bool SupportsConversation => true;

    /// <inheritdoc/>
    public override bool SupportsVideoInput => true;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes VideoLISA in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: binary cross-entropy with logits for one mask class; otherwise cross-entropy with logits).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable VideoLISA model.
    /// </para>
    /// </remarks>
    public VideoLISA(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        VideoLISAOptions? options = null)
        // VideoLISA's own loss default is preserved verbatim - the base would otherwise substitute
        // plain CrossEntropyWithLogitsLoss, which is wrong for the single-mask (numClasses == 1) case.
        // `optimizer` is passed straight through INCLUDING null: the base resolves it lazily through
        // CreateDefaultOptimizer() below, which builds the same AdamW this constructor used to.
        : base(architecture, optimizer, lossFunction ?? (numClasses == 1
            ? (ILossFunction<T>)new BinaryCrossEntropyWithLogitsLoss<T>()
            : new CrossEntropyWithLogitsLoss<T>(classAxis: 1)), numClasses)
    {
        _options = options is null ? new VideoLISAOptions() : new VideoLISAOptions(options);
        ValidateOptions(_options);
        Options = _options;
        _dropRate = dropRate;
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.EncoderDepths.Clone();
        _decoderDim = _options.DecoderDimension;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes VideoLISA in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained VideoLISA from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public VideoLISA(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        VideoLISAOptions? options = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options is null ? new VideoLISAOptions() : new VideoLISAOptions(options);
        ValidateOptions(_options);
        Options = _options;
        _dropRate = 0;
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.EncoderDepths.Clone();
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
            // Pass the configured optimizer through; the two-argument overload ignored it.
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

    /// <summary>Adds a leading batch axis. Recorded, so it stays on the autodiff tape.</summary>
    /// <remarks>
    /// DELIBERATELY shadows the inherited SegmentationModelBase helper rather than using it. The base
    /// copies raw spans into a freshly allocated tensor - precisely the tape-detaching behaviour
    /// documented on RemoveBatchDimension below - so inheriting it would reintroduce the zero-gradient
    /// bug this pair was written to fix.
    /// </remarks>
    private new Tensor<T> AddBatchDimension(Tensor<T> tensor)
        => Engine.Reshape(tensor, new[] { 1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2] });

    /// <summary>
    /// Drops the leading batch axis. Recorded, for the same reason.
    /// </summary>
    /// <remarks>
    /// Both of these copied raw spans into a freshly allocated tensor, which produces a value with no
    /// history on the autodiff tape. Forward ends by calling this one whenever the caller passed an
    /// unbatched clip, so the network's OUTPUT was detached and every gradient came back zero —
    /// GradientFlow_ShouldBeNonZeroAndFinite reported "No parameters changed after training".
    /// </remarks>
    private new Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] s = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < s.Length; i++) s[i] = tensor.Shape[i + 1];
        return Engine.Reshape(tensor, s);
    }

    /// <summary>
    /// Creates VideoLISA's AdamW default when the caller supplies no optimizer. The base resolves this
    /// lazily after construction, so <c>_options</c> is already assigned by the time it runs - which is
    /// exactly what a base-constructor argument could never express.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer() =>
        new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveLearningRate = false,
            });

    private static void ValidateOptions(VideoLISAOptions options)
    {
        if (options.ChannelDimensions is null || options.ChannelDimensions.Length == 0)
            throw new ArgumentException("At least one encoder channel dimension is required.", nameof(options));
        if (options.EncoderDepths is null || options.EncoderDepths.Length != options.ChannelDimensions.Length)
            throw new ArgumentException("EncoderDepths must contain one value per ChannelDimensions stage.", nameof(options));
        if (options.ChannelDimensions.Any(d => d <= 0))
            throw new ArgumentOutOfRangeException(nameof(options), "All encoder channel dimensions must be positive.");
        if (options.EncoderDepths.Any(d => d <= 0))
            throw new ArgumentOutOfRangeException(nameof(options), "All encoder depths must be positive.");
        if (options.DecoderDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DecoderDimension must be positive.");
        if (options.LearningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LearningRate must be positive.");
        if (options.WeightDecay < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "WeightDecay cannot be negative.");
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
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
            if (_encoderLayerEnd < 0 || _encoderLayerEnd > Architecture.Layers.Count)
                throw new ArgumentOutOfRangeException(nameof(_options.EncoderLayerCount), "EncoderLayerCount must be within the custom layer list.");
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateVideoLISAEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateVideoLISADecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "VideoLISA" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    #endregion

    #region IReferringSegmentation Implementation
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
    protected override ReferringSegmentationResult<T> SegmentFromConversationInternal(
        Tensor<T> image, IReadOnlyList<(string Role, string Message)> conversationHistory, string currentQuery)
    {
        var context = string.Join(" ", conversationHistory.Select(c => c.Message));
        var fullQuery = string.IsNullOrEmpty(context) ? currentQuery : $"{context} {currentQuery}";
        return SegmentFromExpression(image, fullQuery);
    }

    /// <inheritdoc/>
    protected override List<ReferringSegmentationResult<T>> SegmentVideoFromExpressionInternal(Tensor<T> frames, string expression)
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
