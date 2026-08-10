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

namespace AiDotNet.ComputerVision.Segmentation.Mamba;

/// <summary>
/// ViM-UNet: Vision Mamba for medical image segmentation in U-Net.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Medical image segmentation with Mamba. Biomedical image analysis.
///
/// Common use cases:
/// - Medical image segmentation with Mamba
/// - Biomedical image analysis
/// - Efficient U-Net alternative
/// - Long-range dependency modeling in medical images
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Vision Mamba encoder blocks in U-Net architecture
/// - Bidirectional SSM for capturing global context
/// - Skip connections between encoder and decoder
/// - Linear complexity for efficient medical image processing
/// </para>
/// <para>
/// <b>Reference:</b> Archit and Pape, "ViM-UNet: Vision Mamba for Biomedical Segmentation", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ViM-UNet model for biomedical image segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 14);
/// var model = new ViMUNet&lt;double&gt;(architecture, numClasses: 14);
///
/// // Or load a pre-trained ONNX model for medical image analysis
/// var onnxModel = new ViMUNet&lt;double&gt;(architecture, "vimunet.onnx", numClasses: 14);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ViM-UNet: Vision Mamba for Biomedical Segmentation", "https://arxiv.org/abs/2404.07705", Year = 2024, Authors = "Archit and Pape")]
public class ViMUNet<T> : Common.SemanticSegmentationBase<T>
{
    private readonly ViMUNetOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only ViMUNet's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes ViMUNet in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable ViMUNet model.
    /// </para>
    /// </remarks>
    public ViMUNet(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 14,
        double dropRate = 0,
        ViMUNetOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults the loss to CrossEntropyWithLogitsLoss - exactly what the deleted lines did by
        // hand. `optimizer` is passed straight through INCLUDING null; the base's lazy
        // CreateDefaultOptimizer() produces the same `new AdamWOptimizer<...>(this)` default, which
        // could never be written as a base-constructor argument because `this` is unavailable there.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new ViMUNetOptions(); Options = _options;
        ApplyViMUNetInputFallback(architecture);
        _dropRate = dropRate;
        _channelDims = [32, 64, 128, 256];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies ViMUNet's 256x256 fallback for architectures that carry no input geometry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SegmentationModelBase falls back to 512x512 when the architecture supplies no input height
    /// or width. ViMUNet's documented fallback is 256x256 (the biomedical patch size the paper
    /// trains at), so it is restored here for that unset case only - when the architecture does
    /// specify dimensions, the base's value already matches and nothing changes.
    /// </para>
    /// </remarks>
    private void ApplyViMUNetInputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 256;
        if (architecture.InputWidth <= 0) _width = 256;
    }

    /// <summary>
    /// Initializes ViMUNet in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained ViMUNet from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public ViMUNet(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 14,
        ViMUNetOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new ViMUNetOptions(); Options = _options;
        ApplyViMUNetInputFallback(architecture);
        _dropRate = 0;
        _channelDims = [32, 64, 128, 256];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    // PredictCore is inherited from SegmentationModelBase and dispatches to Forward / PredictOnnx
    // exactly as the deleted override did.

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
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < Layers.Count; i++)
            features = Layers[i].Forward(features);

        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
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
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateViMUNetEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateViMUNetDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "ViMUNet" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
        ? new ViMUNet<T>(Architecture, Optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new ViMUNet<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // ViMUNet owns no further unmanaged resources.
    #endregion

    #region ISemanticSegmentation Implementation

    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase.
    // GetClassMap and GetProbabilityMap come from SemanticSegmentationBase, which computes
    // ArgmaxAlongClassDim / SoftmaxAlongClassDim over Segment(image) - i.e. over Predict(image),
    // the identical computation these explicit implementations performed.

    #endregion
}
