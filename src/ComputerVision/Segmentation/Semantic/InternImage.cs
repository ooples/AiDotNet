using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> InternImage is a semantic segmentation model that proves CNNs can compete
/// with Vision Transformers when using modern deformable convolutions. It uses DCNv3 (Deformable
/// Convolution v3) which can adaptively adjust where it "looks" in the image based on the content,
/// allowing it to focus on relevant regions for better segmentation.
///
/// Common use cases:
/// - Large-scale scene parsing (Cityscapes, ADE20K)
/// - Object detection and segmentation pipelines
/// - Foundation model applications requiring dense predictions
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - DCNv3 operator with multi-group deformable attention
/// - 4-stage hierarchical architecture (like ConvNeXt/Swin but with DCNv3)
/// - UPerNet decoder for multi-scale feature aggregation
/// - Scales from 30M (Tiny) to 1.08B (Huge) parameters
/// - Competitive with ViT-based models on ADE20K and COCO
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "InternImage: Exploring Large-Scale Vision Foundation Models
/// with Deformable Convolutions", CVPR 2023.
/// </para>
/// </remarks>
public class InternImage<T> : NeuralNetworkBase<T>, ISemanticSegmentation<T>
{
    private readonly InternImageOptions _options;

    /// <summary>
    /// Gets the configuration options for this InternImage model.
    /// </summary>
    /// <returns>The <see cref="InternImageOptions"/> used to configure this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numClasses;
    private readonly InternImageModelSize _modelSize;
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
    /// Gets whether this InternImage instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode (trainable) and <c>false</c>
    /// in ONNX mode (inference only). To fine-tune on your data, use the native constructor.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal InternImageModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of InternImage in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration defining
    /// input dimensions (height, width, channels).</param>
    /// <param name="optimizer">The gradient-based optimizer (default: AdamW, as used in the
    /// InternImage paper for all experiments on ADE20K and COCO).</param>
    /// <param name="lossFunction">The loss function (default: CrossEntropyLoss for multi-class segmentation).</param>
    /// <param name="numClasses">Number of semantic classes (default: 150 for ADE20K).</param>
    /// <param name="modelSize">Model size variant (default: Tiny, 30M params).</param>
    /// <param name="dropRate">Dropout rate for regularization (default: 0.1).</param>
    /// <param name="options">Optional model options including random seed.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable InternImage from scratch. The model uses
    /// deformable convolutions that adaptively adjust their sampling positions based on
    /// image content, giving CNN-level efficiency with transformer-level accuracy.
    /// Start with Tiny for experiments, then scale to Base or larger for production.
    /// </para>
    /// </remarks>
    public InternImage(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        InternImageModelSize modelSize = InternImageModelSize.Tiny,
        double dropRate = 0.1,
        InternImageOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new InternImageOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = dropRate;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of InternImage in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of semantic classes the ONNX model predicts (default: 150).</param>
    /// <param name="modelSize">Model size variant for metadata (default: Tiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained InternImage from an ONNX file for fast inference.
    /// ONNX mode does not support training. Use the native constructor for fine-tuning.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if ONNX path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX runtime fails to load the model.</exception>
    public InternImage(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        InternImageModelSize modelSize = InternImageModelSize.Tiny,
        InternImageOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new InternImageOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"InternImage ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load InternImage ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through InternImage to produce per-pixel segmentation logits.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in an image and get back a map where each pixel has scores for
    /// every class. The DCNv3 encoder adaptively focuses on relevant regions, while the UPerNet
    /// decoder aggregates multi-scale features for accurate predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step with forward pass, loss computation, backward pass, and update.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation map.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each training step processes an image through the DCNv3 encoder,
    /// compares the prediction to ground truth, and updates weights to improve future predictions.
    /// Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException(
                "Training is not supported in ONNX mode. Use the native mode constructor for training.");

        var predicted = Forward(input);
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data.Span[idx]));
        BackwardPass(lossGradient);
        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Private Methods

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(
        InternImageModelSize modelSize)
    {
        return modelSize switch
        {
            InternImageModelSize.Tiny => ([64, 128, 256, 512], [4, 4, 18, 4], 512),
            InternImageModelSize.Small => ([80, 160, 320, 640], [4, 4, 21, 4], 512),
            InternImageModelSize.Base => ([112, 224, 448, 896], [4, 4, 21, 4], 512),
            InternImageModelSize.XL => ([192, 384, 768, 1536], [4, 4, 21, 4], 1024),
            InternImageModelSize.Huge => ([320, 640, 1280, 2560], [6, 6, 32, 6], 1024),
            _ => ([64, 128, 256, 512], [4, 4, 18, 4], 512)
        };
    }

    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);

        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
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
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        var result = new Tensor<T>(outputShape, new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0) return;
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
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
            newShape[i] = tensor.Shape[i + 1];
        var result = new Tensor<T>(newShape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the DCNv3 encoder and UPerNet decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, creates the 4-stage DCNv3 encoder that uses deformable
    /// convolutions to adaptively sample relevant image regions, followed by a UPerNet decoder that
    /// aggregates multi-scale features for segmentation. In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateInternImageEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();

            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] patchKernels = [7, 3, 3, 3];
            int[] patchStrides = [4, 2, 2, 2];
            int[] patchPaddings = [3, 1, 1, 1];
            int featureH = _height, featureW = _width;
            for (int stage = 0; stage < 4; stage++)
            {
                featureH = (featureH + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
                featureW = (featureW + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
            }

            Layers.AddRange(LayerHelper<T>.CreateInternImageDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW));
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat vector.
    /// </summary>
    /// <param name="parameters">Flat parameter vector ordered by layer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces all model weights with new values, used during optimization
    /// and when loading saved models.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int count = layerParams.Length;
            if (offset + count <= parameters.Length)
            {
                var newParams = new Vector<T>(count);
                for (int i = 0; i < count; i++) newParams[i] = parameters[offset + i];
                layer.UpdateParameters(newParams);
                offset += count;
            }
        }
    }

    /// <summary>
    /// Collects metadata describing this InternImage model.
    /// </summary>
    /// <returns>Model metadata with type, configuration, and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary of the model including its type, dimensions,
    /// class count, and serialized weights for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SemanticSegmentation,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "InternImage" },
                { "Description", "InternImage Semantic Segmentation (DCNv3)" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() },
                { "DecoderDim", _decoderDim }, { "DropRate", _dropRate },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes InternImage-specific configuration to a binary stream.
    /// </summary>
    /// <param name="writer">Binary writer for persistence.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves the model's configuration so it can be restored later.
    /// The order must match <see cref="DeserializeNetworkSpecificData"/>.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height); writer.Write(_width); writer.Write(_channels);
        writer.Write(_numClasses); writer.Write((int)_modelSize);
        writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length);
        foreach (int dim in _channelDims) writer.Write(dim);
        writer.Write(_depths.Length);
        foreach (int depth in _depths) writer.Write(depth);
    }

    /// <summary>
    /// Deserializes InternImage-specific configuration from a binary stream.
    /// </summary>
    /// <param name="reader">Binary reader for loading.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reads back the saved configuration in the same order it was written.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString();
        _ = reader.ReadInt32();
        int channelCount = reader.ReadInt32();
        for (int i = 0; i < channelCount; i++) _ = reader.ReadInt32();
        int depthCount = reader.ReadInt32();
        for (int i = 0; i < depthCount; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new InternImage with the same config but fresh weights.
    /// </summary>
    /// <returns>A new model instance with reinitialized weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used for cross-validation or ensemble training where multiple
    /// independent copies of the same architecture are needed.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new InternImage<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
            : new InternImage<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);
    }

    /// <summary>
    /// Releases managed resources including the ONNX session.
    /// </summary>
    /// <param name="disposing">True when called from Dispose().</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees memory and file handles. Use a <c>using</c> statement:
    /// <c>using var model = new InternImage&lt;float&gt;(...);</c>
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; }
            _disposed = true;
        }
        base.Dispose(disposing);
    }

    #endregion

    #region ISemanticSegmentation Implementation

    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);

    Tensor<T> ISemanticSegmentation<T>.GetClassMap(Tensor<T> image)
        => Common.SegmentationTensorOps.ArgmaxAlongClassDim(Predict(image));

    Tensor<T> ISemanticSegmentation<T>.GetProbabilityMap(Tensor<T> image)
        => Common.SegmentationTensorOps.SoftmaxAlongClassDim(Predict(image));

    #endregion
}
