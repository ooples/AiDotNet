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

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// SED: A Simple Encoder-Decoder for open-vocabulary semantic segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Simple and efficient open-vocabulary segmentation. Category-adaptive prompting.
///
/// Common use cases:
/// - Simple and efficient open-vocabulary segmentation
/// - Category-adaptive prompting
/// - Cross-dataset generalization
/// - Balanced accuracy-efficiency open-vocab segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Simple encoder-decoder with hierarchical CLIP features
/// - Category-adaptive prompt generation
/// - Lightweight decoder head for open-vocabulary prediction
/// - Competitive accuracy with minimal architectural overhead
/// </para>
/// <para>
/// <b>Reference:</b> Xie et al., "SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SED model for simple open-vocabulary segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new SED&lt;double&gt;(architecture, numClasses: 150);
///
/// // Or load a pre-trained ONNX model for category-adaptive segmentation
/// var onnxModel = new SED&lt;double&gt;(architecture, "sed.onnx", numClasses: 150);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation", "https://arxiv.org/abs/2311.15537", Year = 2024, Authors = "Xie et al.")]
public partial class SED<T> : Common.OpenVocabSegmentationBase<T>
{
    private readonly SEDOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only SED's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from OpenVocabSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, MaxCategories and
    // MaxPromptLength are all inherited and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes SED in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 150).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SED model.
    /// </para>
    /// </remarks>
    public SED(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 150,
        double dropRate = 0.1,
        SEDOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer(), so null is passed straight through.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new SEDOptions(); Options = _options;
        // SED's own fallback input geometry is 640x640, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 640;
        if (architecture.InputWidth <= 0) _width = 640;
        _dropRate = dropRate;
        _channelDims = [64, 128, 320, 512];
        _depths = [2, 2, 4, 2];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes SED in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 150).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SED from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SED(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 150,
        SEDOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new SEDOptions(); Options = _options;
        // SED's own fallback input geometry is 640x640, not the base's 512.
        if (architecture.InputHeight <= 0) _height = 640;
        if (architecture.InputWidth <= 0) _width = 640;
        _dropRate = 0.1;
        _channelDims = [64, 128, 320, 512];
        _depths = [2, 2, 4, 2];
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
            var encoderLayers = LayerHelper<T>.CreateSEDEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateSEDDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "SED" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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


    /// <summary>
    /// Reads configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>


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
        ? new SED<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new SED<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);
    #endregion

    #region IOpenVocabSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, MaxCategories (256) and
    // MaxPromptLength (77) all come from the base; only the model-specific overrides remain.

    /// <inheritdoc/>
    public override OpenVocabSegmentationResult<T> SegmentWithText(Tensor<T> image, IReadOnlyList<string> classNames)
    {
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        int numC = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
        int numText = classNames.Count;
        var masks = new Tensor<T>([numText, h, w]);
        var scores = new double[numText];
        var semanticMap = new Tensor<T>([h, w]);
        var textProbs = new double[numText][];
        for (int t = 0; t < numText; t++)
        {
            var weights = Common.SegmentationTensorOps.TextToWeights(classNames[t], numC);
            var scoreMap = Common.SegmentationTensorOps.WeightedChannelSum(logits, weights);
            var probMap = Common.SegmentationTensorOps.Sigmoid(scoreMap);
            double area = 0, confSum = 0;
            textProbs[t] = new double[h * w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double v = NumOps.ToDouble(probMap[y, x]);
                    textProbs[t][y * w + x] = v;
                    if (v >= 0.5) { masks[t, y, x] = NumOps.FromDouble(1.0); area++; confSum += v; }
                }
            scores[t] = area > 0 ? confSum / area : 0;
        }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int best = 0; double bestV = -1;
                for (int t = 0; t < numText; t++) { double v = textProbs[t][y * w + x]; if (v > bestV) { bestV = v; best = t; } }
                semanticMap[y, x] = NumOps.FromDouble(best);
            }
        return new OpenVocabSegmentationResult<T> { Masks = masks, ClassNames = classNames.ToArray(), Scores = scores, SemanticMap = semanticMap };
    }

    /// <inheritdoc/>
    public override OpenVocabSegmentationResult<T> SegmentWithPrompt(Tensor<T> image, string prompt)
        => SegmentWithText(image, new[] { prompt });
    #endregion
}
