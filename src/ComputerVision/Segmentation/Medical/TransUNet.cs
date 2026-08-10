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

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// TransUNet: Transformers make strong encoders for medical segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Multi-organ segmentation from CT scans. Cardiac segmentation from MRI.
///
/// Common use cases:
/// - Multi-organ segmentation from CT scans
/// - Cardiac segmentation from MRI
/// - Medical image analysis requiring global context
/// - Hybrid CNN-Transformer medical segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Hybrid CNN-Transformer encoder (ResNet + ViT)
/// - CNN captures local features, Transformer captures global context
/// - Cascaded upsampler decoder with skip connections
/// - Tokenized image patches enable self-attention over full image
/// </para>
/// <para>
/// <b>Reference:</b> Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", arXiv 2021.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TransUNet model for multi-organ CT segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 1, outputSize: 9);
/// var model = new TransUNet&lt;double&gt;(architecture, numClasses: 9);
///
/// // Or load a pre-trained ONNX model for cardiac MRI segmentation
/// var onnxModel = new TransUNet&lt;double&gt;(architecture, "transunet.onnx", numClasses: 9);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", "https://arxiv.org/abs/2102.04306", Year = 2021, Authors = "Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, Yuyin Zhou")]
public class TransUNet<T> : Common.MedicalSegmentationBase<T>
{
    private readonly TransUNetOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only TransUNet's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase.
    private static readonly string[] ModalitiesSupported = ["CT", "MRI_T1"];
    private readonly TransUNetModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, SupportedModalities,
    // Supports2D and SupportsFewShot are all supplied identically by the base.
    internal bool UseNativeMode => _useNativeMode;
    internal TransUNetModelSize ModelSize => _modelSize;

    /// <summary>TransUNet is a 2D model; volumetric input is not supported.</summary>
    public override bool Supports3D => false;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes TransUNet in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 9).</param>
    /// <param name="modelSize">Model size variant (default: Base).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable TransUNet model.
    /// </para>
    /// </remarks>
    public TransUNet(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 9,
        TransUNetModelSize modelSize = TransUNetModelSize.Base, double dropRate = 0.1,
        TransUNetOptions? options = null)
        // `optimizer` is passed straight through - INCLUDING null. The base resolves the default
        // AdamW lazily via CreateDefaultOptimizer(), which a base-constructor argument cannot do.
        : base(architecture, optimizer, lossFunction, numClasses, ModalitiesSupported)
    {
        _options = options ?? new TransUNetOptions(); Options = _options;
        // TransUNet defaults to 224x224, not the base's 512x512, so the geometry fallback stays here.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes TransUNet in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 9).</param>
    /// <param name="modelSize">Model size for metadata (default: Base).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained TransUNet from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public TransUNet(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 9, TransUNetModelSize modelSize = TransUNetModelSize.Base,
        TransUNetOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses, ModalitiesSupported)
    {
        _options = options ?? new TransUNetOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _modelSize = modelSize; _dropRate = 0.1;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
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
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expectedOutput is null) throw new ArgumentNullException(nameof(expectedOutput));
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
        if (input.Shape.Length == 3) { input = input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] }); }
        else if (input.Shape.Length != 4) throw new ArgumentException($"Training requires rank 3 [C,H,W] or 4 [B,C,H,W], got rank {input.Shape.Length}.", nameof(input));
        if (expectedOutput.Shape.Length == 3) { expectedOutput = expectedOutput.Reshape(new[] { 1, expectedOutput.Shape[0], expectedOutput.Shape[1], expectedOutput.Shape[2] }); }
        else if (expectedOutput.Shape.Length != 4) throw new ArgumentException($"Expected output requires rank 3 [C,H,W] or 4 [B,C,H,W], got rank {expectedOutput.Shape.Length}.", nameof(expectedOutput));
        try
        {
            SetTrainingMode(true);
            TrainWithTape(input, expectedOutput, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    #endregion

    #region Private Methods
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(TransUNetModelSize modelSize) => modelSize switch
    {
        TransUNetModelSize.Base => ([64, 128, 256, 768], [3, 4, 9, 3], 256),
        TransUNetModelSize.Large => ([64, 128, 256, 1024], [3, 4, 9, 3], 256),
        _ => ([64, 128, 256, 768], [3, 4, 9, 3], 256)
    };

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
        { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Architecture.Layers.Count / 2; }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateTransUNetEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateTransUNetDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int totalRequired = 0;
        foreach (var l in Layers) totalRequired += l.GetParameters().Length;
        if (parameters.Length < totalRequired)
            throw new ArgumentException($"Parameter vector length {parameters.Length} is less than required {totalRequired}.", nameof(parameters));

        int o = 0;
        foreach (var l in Layers)
        {
            var p = l.GetParameters();
            int c = p.Length;
            var n = new Vector<T>(c);
            for (int i = 0; i < c; i++) n[i] = parameters[o + i];
            l.UpdateParameters(n);
            o += c;
        }
    }
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "TransUNet" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    { writer.Write(_height); writer.Write(_width); writer.Write(_channels); writer.Write(_numClasses); writer.Write((int)_modelSize); writer.Write(_decoderDim); writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty); writer.Write(_encoderLayerEnd); writer.Write(_channelDims.Length); foreach (int d in _channelDims) writer.Write(d); writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d); }

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
    { _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32(); int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32(); int dd = reader.ReadInt32(); for (int i = 0; i < dd; i++) _ = reader.ReadInt32(); }

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
        ? new TransUNet<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new TransUNet<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and flips _disposed,
    // and TransUNet owns no other unmanaged resource.
    #endregion

    #region IMedicalSegmentation Implementation
    /// <summary>Segments a single 2D medical slice.</summary>
    public override MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice)
    {
        var output = Predict(slice);
        var labels = Common.SegmentationTensorOps.ArgmaxAlongClassDim(output);
        var probs = Common.SegmentationTensorOps.SoftmaxAlongClassDim(output);
        int h = labels.Shape[0], w = labels.Shape[1];
        int numC = probs.Shape[0];
        var structures = new List<SegmentedStructure>();
        for (int c = 0; c < numC; c++)
        {
            int area = 0; double confSum = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if ((int)NumOps.ToDouble(labels[y, x]) == c) { area++; confSum += NumOps.ToDouble(probs[c, y, x]); }
            if (area > 0)
                structures.Add(new SegmentedStructure { ClassId = c, Name = $"Class_{c}", VolumeOrArea = area, MeanConfidence = confSum / area });
        }
        return new MedicalSegmentationResult<T> { Labels = labels, Probabilities = probs, Structures = structures };
    }
    /// <inheritdoc/>
    public override MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume)
        => throw new NotSupportedException("TransUNet does not support 3D volumetric segmentation. Use SegmentSlice for 2D slices.");

    /// <inheritdoc/>
    public override MedicalSegmentationResult<T> SegmentFewShot(
        Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => throw new NotSupportedException("TransUNet does not support few-shot segmentation. Use SegmentSlice for standard inference.");
    #endregion
}
