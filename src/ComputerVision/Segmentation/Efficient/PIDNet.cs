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

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Real-time semantic segmentation for autonomous driving. Cityscapes and urban scene parsing.
///
/// Common use cases:
/// - Real-time semantic segmentation for autonomous driving
/// - Cityscapes and urban scene parsing
/// - Edge-device semantic segmentation
/// - Balanced speed-accuracy segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - PID-controller-inspired three-branch architecture (P, I, D)
/// - P-branch: detail features, I-branch: context, D-branch: boundary
/// - Bag of Tricks for efficient feature fusion
/// - Achieves Cityscapes mIoU 80+ at 90+ FPS
/// </para>
/// <para>
/// <b>Reference:</b> Xu et al., "PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers", CVPR 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Use AiModelBuilder facade for real-time semantic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 1024, inputDepth: 3, outputSize: 19);
///
/// var builder = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
///     .ConfigureModel(new PIDNet&lt;double&gt;(architecture, numClasses: 19));
///
/// var result = builder.Build(trainingImages, trainingMasks);
/// var segmentation = result.Predict(inputImage);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers", "https://arxiv.org/abs/2206.02066", Year = 2023, Authors = "Jiacong Xu, Zixiang Xiong, Shankar P. Bhatt, Ravi Tandon")]
public partial class PIDNet<T> : Common.SemanticSegmentationBase<T>
{
    private readonly PIDNetOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only PIDNet's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly PIDNetModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;
    internal PIDNetModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes PIDNet in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 19).</param>
    /// <param name="modelSize">Model size variant (default: Small).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable PIDNet model.
    /// </para>
    /// </remarks>
    public PIDNet(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 19,
        PIDNetModelSize modelSize = PIDNetModelSize.Small, double dropRate = 0.1,
        PIDNetOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new PIDNetOptions(); Options = _options;
        ApplyCityscapesDefaultGeometry(architecture);
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes PIDNet in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 19).</param>
    /// <param name="modelSize">Model size for metadata (default: Small).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained PIDNet from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public PIDNet(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 19, PIDNetModelSize modelSize = PIDNetModelSize.Small,
        PIDNetOptions? options = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new PIDNetOptions(); Options = _options;
        ApplyCityscapesDefaultGeometry(architecture);
        _modelSize = modelSize; _dropRate = 0.1;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Restores PIDNet's own 1024x2048 fallback for unspecified input geometry.
    /// </summary>
    /// <remarks>
    /// SegmentationModelBase falls back to a square 512x512 when the architecture leaves the input
    /// size unset; PIDNet is a real-time driving model and has always fallen back to Cityscapes'
    /// 1024x2048 frame instead, so that stays the model's own rule.
    /// </remarks>
    private void ApplyCityscapesDefaultGeometry(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 2048;
    }
    #endregion

    #region Public Methods
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
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(PIDNetModelSize modelSize) => modelSize switch
    {
        PIDNetModelSize.Small => ([32, 64, 128, 256], [2, 3, 3, 2], 128),
        PIDNetModelSize.Medium => ([64, 128, 256, 512], [2, 3, 3, 2], 256),
        PIDNetModelSize.Large => ([64, 128, 256, 512], [3, 4, 6, 3], 256),
        _ => ([32, 64, 128, 256], [2, 3, 3, 2], 128)
    };

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

    // AddBatchDimension / RemoveBatchDimension are inherited from SegmentationModelBase.
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
            var encoderLayers = LayerHelper<T>.CreatePIDNetEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreatePIDNetDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "PIDNet" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
        ? new PIDNet<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new PIDNet<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

    // Dispose of the ONNX session and the _disposed latch are handled by SegmentationModelBase.
    // NumClasses / InputHeight / InputWidth / IsOnnxMode / Segment / GetClassMap / GetProbabilityMap
    // all arrive from SemanticSegmentationBase with identical bodies.
    #endregion
}
