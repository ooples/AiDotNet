using System.IO;
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
/// BiomedParse: Biomedical image parsing with text prompts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Text-prompted biomedical image segmentation. Multi-modality biomedical parsing.
///
/// Common use cases:
/// - Text-prompted biomedical image segmentation
/// - Multi-modality biomedical parsing
/// - Detection and recognition in biomedical images
/// - Joint segmentation-detection-recognition
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Text-prompted segmentation for biomedical images
/// - Joint segmentation, detection, and recognition in one model
/// - Trained on 6M+ triples across 9 imaging modalities
/// - GPT-4 assisted harmonization of biomedical datasets
/// </para>
/// <para>
/// <b>Reference:</b> Zhao et al., "BiomedParse: a biomedical foundation model for image parsing of everything everywhere all at once", Nature Methods 2024.
/// </para>
/// </remarks>
public class BiomedParse<T> : NeuralNetworkBase<T>, IMedicalSegmentation<T>
{
    private readonly BiomedParseOptions _options;
    public override ModelOptions GetOptions() => _options;

    // BiomedParse paper defaults: Swin-B backbone (Zhao et al., Nature Methods 2024)
    private static readonly int[] DefaultChannelDims = [96, 192, 384, 768];
    private static readonly int[] DefaultDepths = [2, 2, 6, 2];
    private const int DefaultDecoderDim = 256;

    #region Fields
    private int _height, _width, _channels, _numClasses;
    private int[] _channelDims;
    private int _decoderDim;
    private int[] _depths;
    private double _dropRate;
    private bool _useNativeMode;
    private string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private int _encoderLayerEnd;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether this BiomedParse instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode, <c>false</c> in ONNX mode.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal int NumClasses => _numClasses;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes BiomedParse in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable BiomedParse model.
    /// </para>
    /// </remarks>
    public BiomedParse(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0.1,
        BiomedParseOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new BiomedParseOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _dropRate = dropRate;
        _useNativeMode = true; _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _channelDims = DefaultChannelDims;
        _depths = DefaultDepths;
        _decoderDim = DefaultDecoderDim;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes BiomedParse in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained BiomedParse from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public BiomedParse(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        BiomedParseOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new BiomedParseOptions(); Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"BiomedParse ONNX model not found: {onnxModelPath}");
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _dropRate = 0.1;
        _useNativeMode = false; _onnxModelPath = onnxModelPath; _optimizer = null;
        _channelDims = DefaultChannelDims;
        _depths = DefaultDepths;
        _decoderDim = DefaultDecoderDim;
        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load BiomedParse ONNX model: {ex.Message}", ex); }
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
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");

        var predicted = Forward(input);
        var lossGradient = LossFunction.ComputeGradient(predicted, expectedOutput);
        BackwardPass(lossGradient);
        _optimizer?.UpdateParameters(Layers);
    }
    #endregion

    #region Private Methods
    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < Layers.Count; i++)
            features = Layers[i].Forward(features);

        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0) return;
        if (gradient.Rank == 3) gradient = AddBatchDimension(gradient);
        for (int i = Layers.Count - 1; i >= 0; i--)
            gradient = Layers[i].Backward(gradient);
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] shape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < shape.Length; i++)
            shape[i] = tensor.Shape[i + 1];
        var result = new Tensor<T>(shape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
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
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateBiomedParseEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateBiomedParseDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
    {
        int totalRequired = 0;
        foreach (var l in Layers)
            totalRequired += l.GetParameters().Length;

        if (parameters.Length < totalRequired)
            throw new ArgumentException(
                $"Parameter vector length {parameters.Length} is less than required {totalRequired}.",
                nameof(parameters));

        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = layer.GetParameters().Length;
            var newParams = new Vector<T>(count);
            for (int i = 0; i < count; i++)
                newParams[i] = parameters[offset + i];
            layer.UpdateParameters(newParams);
            offset += count;
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
        ModelType = ModelType.SemanticSegmentation,
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "BiomedParse" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        _decoderDim = reader.ReadInt32();
        _dropRate = reader.ReadDouble();
        _useNativeMode = reader.ReadBoolean();
        _onnxModelPath = reader.ReadString();
        _encoderLayerEnd = reader.ReadInt32();
        int dc = reader.ReadInt32();
        _channelDims = new int[dc];
        for (int i = 0; i < dc; i++) _channelDims[i] = reader.ReadInt32();
        int dd = reader.ReadInt32();
        _depths = new int[dd];
        for (int i = 0; i < dd; i++) _depths[i] = reader.ReadInt32();
    }

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
        ? new BiomedParse<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new BiomedParse<T>(Architecture, _onnxModelPath!, _numClasses, _options);

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

    #region IMedicalSegmentation Implementation
    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    IReadOnlyList<string> IMedicalSegmentation<T>.SupportedModalities => ["CT", "MRI_T1", "MRI_T2", "Xray", "Ultrasound", "Pathology", "Dermoscopy", "Fundus", "Microscopy"];
    bool IMedicalSegmentation<T>.Supports3D => false;
    bool IMedicalSegmentation<T>.Supports2D => true;
    bool IMedicalSegmentation<T>.SupportsFewShot => false;
    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentSlice(Tensor<T> slice)
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
    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentVolume(Tensor<T> volume)
        => throw new NotSupportedException("BiomedParse does not support 3D volumetric segmentation. Use SegmentSlice for 2D slices.");
    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentFewShot(Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => throw new NotSupportedException("BiomedParse does not support few-shot segmentation. Use SegmentSlice for standard inference.");
    #endregion
}
