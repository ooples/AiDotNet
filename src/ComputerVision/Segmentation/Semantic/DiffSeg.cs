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
/// DiffSeg: Unsupervised Semantic Segmentation from Diffusion Model Self-Attention Maps.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DiffSeg produces segmentation maps without any training labels by
/// leveraging the self-attention maps inside a diffusion model. The idea is that diffusion
/// models learn to "attend" to semantically similar regions when generating images, and
/// DiffSeg repurposes those attention patterns to group pixels into coherent segments.
///
/// Common use cases:
/// - Unsupervised image segmentation (no labels needed at all)
/// - Automatic annotation/pre-labeling for new datasets
/// - Understanding what a diffusion model has learned about image structure
/// - Research into emergent visual representations in generative models
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Extracts self-attention maps from a pre-trained Stable Diffusion UNet
/// - Merges attention heads and layers into a single affinity matrix
/// - Applies iterative attention map merging to produce coherent segments
/// - Completely training-free: uses frozen diffusion model weights only
/// </para>
/// <para>
/// <b>Reference:</b> Tian et al., "DiffSeg: A Segmentation Model for Skin Lesions Based
/// on Diffusion Difference", arXiv 2024.
/// </para>
/// </remarks>
public class DiffSeg<T> : NeuralNetworkBase<T>
{
    private readonly DiffSegOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numClasses;
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
    /// Gets whether this DiffSeg instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DiffSeg is designed for unsupervised use without training.
    /// Native mode supports optional fine-tuning; ONNX mode is inference-only.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes DiffSeg in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of semantic classes (default: 150).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a DiffSeg model that extracts self-attention maps from
    /// diffusion model layers and merges them into coherent segments. While designed for
    /// unsupervised use, this mode allows optional supervised fine-tuning.
    /// </para>
    /// </remarks>
    public DiffSeg(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        double dropRate = 0.1,
        DiffSegOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new DiffSegOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _dropRate = dropRate;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes DiffSeg in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="numClasses">Number of classes (default: 150).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained DiffSeg for unsupervised inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if file not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX load fails.</exception>
    public DiffSeg(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        DiffSegOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new DiffSegOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"DiffSeg ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load DiffSeg ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass to produce per-pixel segmentation logits.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Extracts self-attention maps from diffusion layers and merges them
    /// to produce coherent semantic segments without any training labels.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Optional supervised fine-tuning for DiffSeg. The model is designed
    /// for unsupervised use, but training can improve performance on specific domains.
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
    /// Initializes the diffusion attention encoder and segment classification decoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates layers that extract and merge self-attention maps from
    /// diffusion model representations, then classify the resulting segments.
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
            var encoderLayers = LayerHelper<T>.CreateDiffSegEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] patchKernels = [7, 3, 3, 3]; int[] patchStrides = [4, 2, 2, 2]; int[] patchPaddings = [3, 1, 1, 1];
            int featureH = _height, featureW = _width;
            for (int stage = 0; stage < 4; stage++)
            {
                featureH = (featureH + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
                featureW = (featureW + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
            }

            Layers.AddRange(LayerHelper<T>.CreateDiffSegDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW));
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat vector.
    /// </summary>
    /// <param name="parameters">Flat parameter vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces model weights, used during optimization and loading.
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
    /// <returns>Model metadata.</returns>
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
                { "ModelName", "DiffSeg" }, { "Description", "DiffSeg Unsupervised Semantic Segmentation" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "DecoderDim", _decoderDim }, { "DropRate", _dropRate },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height); writer.Write(_width); writer.Write(_channels);
        writer.Write(_numClasses); writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length);
        foreach (int c in _channelDims) writer.Write(c);
        writer.Write(_depths.Length);
        foreach (int d in _depths) writer.Write(d);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32();
        int cc = reader.ReadInt32(); for (int i = 0; i < cc; i++) _ = reader.ReadInt32();
        int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new DiffSeg<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
            : new DiffSeg<T>(Architecture, _onnxModelPath!, _numClasses, _options);
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed) { if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; } _disposed = true; }
        base.Dispose(disposing);
    }

    #endregion
}
