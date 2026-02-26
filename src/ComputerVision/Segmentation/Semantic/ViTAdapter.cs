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
/// ViT-Adapter: Vision Transformer Adapter for Dense Predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-Adapter enables plain Vision Transformers (ViTs) to handle dense
/// prediction tasks like semantic segmentation without changing the ViT architecture. It adds
/// lightweight "adapter" modules that inject multi-scale spatial information into the ViT,
/// letting it produce features at different resolutions needed for pixel-level predictions.
///
/// Common use cases:
/// - Semantic segmentation using pre-trained ViT backbones (DINOv2, MAE, BEiT)
/// - Object detection with ViT encoders
/// - Any dense prediction task where you want to leverage powerful ViT pretraining
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Spatial Prior Module: injects multi-scale spatial priors into plain ViT
/// - Spatial Interaction Module: enables feature interaction between adapter and ViT
/// - Works with any plain ViT (ViT-S, ViT-B, ViT-L) without modifying the backbone
/// - Adds only ~3% extra parameters on top of the base ViT
/// - SOTA on ADE20K with BEiT-L backbone (58.0 mIoU)
/// </para>
/// <para>
/// <b>Reference:</b> Chen et al., "Vision Transformer Adapter for Dense Predictions",
/// ICLR 2023 Spotlight.
/// </para>
/// </remarks>
public class ViTAdapter<T> : NeuralNetworkBase<T>, ISemanticSegmentation<T>
{
    private readonly ViTAdapterOptions _options;

    /// <summary>
    /// Gets the configuration options for this ViT-Adapter model.
    /// </summary>
    /// <returns>The <see cref="ViTAdapterOptions"/> used to configure this model instance.</returns>
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
    private readonly ViTAdapterModelSize _modelSize;
    private readonly int _embedDim;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly int[] _numHeads;
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
    /// Gets whether this ViT-Adapter instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode (trainable) and <c>false</c>
    /// in ONNX mode (inference only). To fine-tune on your data, use the native constructor.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal ViTAdapterModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of ViT-Adapter in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW, as used in all ViT-Adapter
    /// experiments with layer-wise learning rate decay).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of semantic classes (default: 150 for ADE20K).</param>
    /// <param name="modelSize">Model size variant (default: Base, 86M params).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable ViT-Adapter. The adapter modules add spatial
    /// information to a plain ViT so it can produce multi-scale features needed for segmentation.
    /// If you have a pre-trained ViT backbone (from DINOv2, MAE, etc.), ViT-Adapter lets you
    /// adapt it for segmentation with minimal extra parameters.
    /// </para>
    /// </remarks>
    public ViTAdapter(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        ViTAdapterModelSize modelSize = ViTAdapterModelSize.Base,
        double dropRate = 0.1,
        ViTAdapterOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new ViTAdapterOptions();
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

        (_embedDim, _depths, _numHeads, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of ViT-Adapter in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of semantic classes (default: 150).</param>
    /// <param name="modelSize">Model size for metadata (default: Base).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained ViT-Adapter from an ONNX file for fast inference.
    /// ONNX mode does not support training.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if ONNX path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX runtime fails.</exception>
    public ViTAdapter(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        ViTAdapterModelSize modelSize = ViTAdapterModelSize.Base,
        ViTAdapterOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new ViTAdapterOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ViT-Adapter ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        (_embedDim, _depths, _numHeads, _decoderDim) = GetModelConfig(modelSize);

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ViT-Adapter ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through the adapted ViT to produce per-pixel segmentation logits.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The image passes through the ViT backbone enhanced with adapter modules
    /// that provide multi-scale spatial features, then through the decoder for per-pixel predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation map.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training updates the adapter modules and decoder while optionally
    /// fine-tuning the ViT backbone. Uses AdamW with layer-wise learning rate decay as specified
    /// in the paper.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var predicted = Forward(input);
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data.Span[idx]));
        BackwardPass(lossGradient);
        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Private Methods

    private static (int EmbedDim, int[] Depths, int[] NumHeads, int DecoderDim) GetModelConfig(
        ViTAdapterModelSize modelSize)
    {
        return modelSize switch
        {
            ViTAdapterModelSize.Small => (384, [2, 2, 2, 2], [6, 6, 6, 6], 256),
            ViTAdapterModelSize.Base => (768, [2, 2, 2, 2], [12, 12, 12, 12], 512),
            ViTAdapterModelSize.Large => (1024, [2, 2, 2, 2], [16, 16, 16, 16], 768),
            _ => (768, [2, 2, 2, 2], [12, 12, 12, 12], 512)
        };
    }

    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        using var results = _onnxSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) });
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0) return;
        if (gradient.Rank == 3) gradient = AddBatchDimension(gradient);
        for (int i = Layers.Count - 1; i >= 0; i--) gradient = Layers[i].Backward(gradient);
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
        for (int i = 0; i < newShape.Length; i++) newShape[i] = tensor.Shape[i + 1];
        var result = new Tensor<T>(newShape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the ViT + adapter encoder and decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, creates the ViT backbone with spatial prior adapter
    /// modules and a decoder head. The adapter modules add multi-scale spatial information that
    /// plain ViTs lack. In ONNX mode, no layers are created.
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
            var encoderLayers = LayerHelper<T>.CreateViTAdapterEncoderLayers(
                _channels, _height, _width, _embedDim, _depths, _numHeads, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] patchKernels = [7, 3, 3, 3]; int[] patchStrides = [4, 2, 2, 2]; int[] patchPaddings = [3, 1, 1, 1];
            int featureH = _height, featureW = _width;
            for (int stage = 0; stage < 4; stage++)
            {
                featureH = (featureH + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
                featureW = (featureW + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
            }

            int encoderOutputChannels = _embedDim; // last stage uses full embed dim
            Layers.AddRange(LayerHelper<T>.CreateViTAdapterDecoderLayers(
                encoderOutputChannels, _decoderDim, _numClasses, featureH, featureW));
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat vector.
    /// </summary>
    /// <param name="parameters">Flat parameter vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces all model weights, used during optimization and loading.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var lp = layer.GetParameters();
            if (offset + lp.Length <= parameters.Length)
            {
                var np = new Vector<T>(lp.Length);
                for (int i = 0; i < lp.Length; i++) np[i] = parameters[offset + i];
                layer.UpdateParameters(np);
                offset += lp.Length;
            }
        }
    }

    /// <summary>
    /// Collects model metadata.
    /// </summary>
    /// <returns>Model metadata with configuration and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Summary of the model for saving, comparing, or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SemanticSegmentation,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "ViTAdapter" }, { "Description", "ViT-Adapter Semantic Segmentation" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() },
                { "EmbedDim", _embedDim }, { "DecoderDim", _decoderDim }, { "DropRate", _dropRate },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes ViT-Adapter configuration for persistence.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves configuration so the model can be restored later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height); writer.Write(_width); writer.Write(_channels);
        writer.Write(_numClasses); writer.Write((int)_modelSize);
        writer.Write(_embedDim); writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_depths.Length);
        foreach (int d in _depths) writer.Write(d);
        writer.Write(_numHeads.Length);
        foreach (int h in _numHeads) writer.Write(h);
    }

    /// <summary>
    /// Deserializes ViT-Adapter configuration.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reads saved configuration matching the write order.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString();
        _ = reader.ReadInt32();
        int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
        int hc = reader.ReadInt32(); for (int i = 0; i < hc; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new ViT-Adapter with same config but fresh weights.
    /// </summary>
    /// <returns>New model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new ViTAdapter<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
            : new ViTAdapter<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);
    }

    /// <summary>
    /// Releases managed resources.
    /// </summary>
    /// <param name="disposing">True from Dispose().</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees ONNX session resources. Use <c>using</c> statement.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed) { if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; } _disposed = true; }
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
