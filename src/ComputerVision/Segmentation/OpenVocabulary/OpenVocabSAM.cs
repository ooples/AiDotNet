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
/// Open-Vocabulary SAM: SAM with text-based open-vocabulary recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Text-prompted SAM segmentation. Open-vocabulary interactive segmentation.
///
/// Common use cases:
/// - Text-prompted SAM segmentation
/// - Open-vocabulary interactive segmentation
/// - Large-vocabulary object segmentation
/// - Combining SAM masks with CLIP recognition
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - SAM encoder-decoder with CLIP text alignment
/// - Supports 20,000+ categories via open vocabulary
/// - Interactive segmentation with automatic class labels
/// - Region-level CLIP feature extraction for classification
/// </para>
/// <para>
/// <b>Reference:</b> Yuan et al., "Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively", ECCV 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an OpenVocabSAM model for text-prompted interactive segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new OpenVocabSAM&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for 20,000+ category segmentation
/// var onnxModel = new OpenVocabSAM&lt;double&gt;(architecture, "openvocabsam.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively", "https://arxiv.org/abs/2401.02955", Year = 2024, Authors = "Yuan et al.")]
public partial class OpenVocabSAM<T> : Common.OpenVocabSegmentationBase<T>
{
    private readonly OpenVocabSAMOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only OpenVocabSAM's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from OpenVocabSegmentationBase -> SegmentationModelBase.
    private const int MaxCategoriesSupported = 20000;
    private readonly int[] _channelDims;
    private readonly int _neckEmbeddingDim;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, MaxCategories
    // (20000, passed to the base) and MaxPromptLength (77) are all supplied by the base.
    internal bool UseNativeMode => _useNativeMode;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes OpenVocabSAM in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable OpenVocabSAM model.
    /// </para>
    /// </remarks>
    public OpenVocabSAM(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        OpenVocabSAMOptions? options = null)
        // `optimizer` is passed straight through - INCLUDING null. The base resolves the default
        // lazily via CreateDefaultOptimizer(), which is now an override of the base's hook rather
        // than a private helper the base could never reach.
        : base(architecture, optimizer, lossFunction, numClasses, MaxCategoriesSupported)
    {
        _options = options ?? new OpenVocabSAMOptions(); Options = _options;
        // Open-Vocabulary SAM defaults to 1024x1024, not the base's 512x512, so the fallback stays.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _dropRate = dropRate;
        ValidateOptions(_options);
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.StageDepths.Clone();
        _neckEmbeddingDim = _options.NeckEmbeddingDimension;
        _decoderDim = _options.DecoderDimension;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes OpenVocabSAM in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained OpenVocabSAM from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public OpenVocabSAM(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        OpenVocabSAMOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses, MaxCategoriesSupported)
    {
        _options = options ?? new OpenVocabSAMOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _dropRate = 0;
        ValidateOptions(_options);
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.StageDepths.Clone();
        _neckEmbeddingDim = _options.NeckEmbeddingDimension;
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
        if (input.Shape.Length == 3) input = AddBatchDimension(input);
        if (expectedOutput.Shape.Length == 3) expectedOutput = AddBatchDimension(expectedOutput);
        if (input.Shape.Length != 4) throw new ArgumentException($"Tape-based training requires rank 3 (CHW) or rank 4 (NCHW), got rank {input.Shape.Length}.", nameof(input));
        if (expectedOutput.Shape.Length != 4) throw new ArgumentException($"Tape-based training target requires rank 3 (CHW) or rank 4 (NCHW), got rank {expectedOutput.Shape.Length}.", nameof(expectedOutput));
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

    // AddBatchDimension / RemoveBatchDimension are inherited from SegmentationModelBase; the copies
    // that used to live here were line-for-line identical apart from the base's extra rank guards.
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
            int encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
            if (encoderLayerEnd < 0 || encoderLayerEnd > Architecture.Layers.Count)
                throw new ArgumentOutOfRangeException(
                    nameof(_options.EncoderLayerCount),
                    $"EncoderLayerCount must be between 0 and {Architecture.Layers.Count}.");
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = encoderLayerEnd;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateOpenVocabSAMEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateOpenVocabSAMDecoderLayers(
                _channelDims[^1], _neckEmbeddingDim, _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "OpenVocabSAM" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
        ? new OpenVocabSAM<T>(Architecture, lossFunction: LossFunction, numClasses: _numClasses,
            dropRate: _dropRate, options: _options)
        : new OpenVocabSAM<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);

    /// <summary>
    /// Open-Vocabulary SAM's default optimizer is AdamW configured from the model options.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer() =>
        new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveLearningRate = false,
            }
        );

    private static void ValidateOptions(OpenVocabSAMOptions options)
    {
        if (options.ChannelDimensions is null || options.ChannelDimensions.Length == 0)
            throw new ArgumentException("At least one encoder channel dimension is required.", nameof(options));
        if (options.StageDepths is null || options.StageDepths.Length != options.ChannelDimensions.Length)
            throw new ArgumentException("StageDepths must contain one value per ChannelDimensions entry.", nameof(options));
        if (options.ChannelDimensions.Any(d => d <= 0) || options.StageDepths.Any(d => d <= 0))
            throw new ArgumentOutOfRangeException(nameof(options), "Encoder channel dimensions and stage depths must be positive.");
        if (options.NeckEmbeddingDimension <= 0 || options.DecoderDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Neck and decoder dimensions must be positive.");
        if (options.LearningRate <= 0 || double.IsNaN(options.LearningRate) || double.IsInfinity(options.LearningRate))
            throw new ArgumentOutOfRangeException(nameof(options), "LearningRate must be finite and positive.");
        if (options.WeightDecay < 0 || double.IsNaN(options.WeightDecay) || double.IsInfinity(options.WeightDecay))
            throw new ArgumentOutOfRangeException(nameof(options), "WeightDecay must be finite and non-negative.");
    }

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and flips _disposed,
    // and OpenVocabSAM owns no other unmanaged resource.
    #endregion

    #region IOpenVocabSegmentation Implementation
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
