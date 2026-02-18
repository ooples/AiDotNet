using System.IO;
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
/// Vision Mamba (Vim): Bidirectional State Space Model for vision.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Efficient image segmentation with linear complexity. Dense prediction replacing ViT backbone.
///
/// Common use cases:
/// - Efficient image segmentation with linear complexity
/// - Dense prediction replacing ViT backbone
/// - Large-scale visual understanding
/// - Memory-efficient visual feature extraction
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Bidirectional Mamba (SSM) for image sequence modeling
/// - Position-aware scanning (bi-directional for images)
/// - Linear complexity O(n) vs quadratic O(n^2) attention
/// - Competitive with ViT at significantly lower FLOPs
/// </para>
/// <para>
/// <b>Reference:</b> Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", ICML 2024.
/// </para>
/// </remarks>
public class VisionMamba<T> : NeuralNetworkBase<T>
{
    private readonly VisionMambaOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    private readonly int _height, _width, _channels, _numClasses;
    private readonly VisionMambaModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    private readonly bool _useNativeMode;
    private readonly string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private int _encoderLayerEnd;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether this VisionMamba instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode, <c>false</c> in ONNX mode.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal VisionMambaModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes VisionMamba in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 150).</param>
    /// <param name="modelSize">Model size variant (default: Tiny).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable VisionMamba model.
    /// </para>
    /// </remarks>
    public VisionMamba(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 150,
        VisionMambaModelSize modelSize = VisionMambaModelSize.Tiny, double dropRate = 0.1,
        VisionMambaOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new VisionMambaOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _modelSize = modelSize; _dropRate = dropRate;
        _useNativeMode = true; _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes VisionMamba in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 150).</param>
    /// <param name="modelSize">Model size for metadata (default: Tiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained VisionMamba from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public VisionMamba(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 150, VisionMambaModelSize modelSize = VisionMambaModelSize.Tiny,
        VisionMambaOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new VisionMambaOptions(); Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"VisionMamba ONNX model not found: {onnxModelPath}");
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _modelSize = modelSize; _dropRate = 0.1;
        _useNativeMode = false; _onnxModelPath = onnxModelPath; _optimizer = null;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load VisionMamba ONNX model: {ex.Message}", ex); }
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
    public override Tensor<T> Predict(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

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
        var predicted = Forward(input);
        var lossGradient = predicted.Transform((v, idx) => NumOps.Subtract(v, expectedOutput.Data.Span[idx]));
        BackwardPass(lossGradient); _optimizer?.UpdateParameters(Layers);
    }
    #endregion

    #region Private Methods
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(VisionMambaModelSize modelSize) => modelSize switch
    {
        VisionMambaModelSize.Tiny => ([192, 192, 192, 192], [2, 2, 2, 18], 256),
        VisionMambaModelSize.Small => ([384, 384, 384, 384], [2, 2, 2, 22], 256),
        VisionMambaModelSize.Base => ([768, 768, 768, 768], [2, 2, 2, 22], 256),
        _ => ([192, 192, 192, 192], [2, 2, 2, 18], 256)
    };

    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    { if (!_useNativeMode || Layers.Count == 0) return; for (int i = Layers.Count - 1; i >= 0; i--) gradient = Layers[i].Backward(gradient); }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    { var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]); tensor.Data.Span.CopyTo(result.Data.Span); return result; }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    { int[] s = new int[tensor.Shape.Length - 1]; for (int i = 0; i < s.Length; i++) s[i] = tensor.Shape[i + 1]; var r = new Tensor<T>(s); tensor.Data.Span.CopyTo(r.Data.Span); return r; }
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
            var encoderLayers = LayerHelper<T>.CreateVisionMambaEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateVisionMambaDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        ModelType = ModelType.SemanticSegmentation,
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "VisionMamba" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = this.Serialize()
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
        ? new VisionMamba<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new VisionMamba<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);

    /// <summary>
    /// Releases managed resources including the ONNX inference session.
    /// </summary>
    /// <param name="disposing">True when called from Dispose().</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees memory used by the ONNX runtime.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    { if (!_disposed) { if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; } _disposed = true; } base.Dispose(disposing); }
    #endregion
}
