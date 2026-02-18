using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// SAM-HQ: Segment Anything in High Quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM-HQ extends Meta's Segment Anything Model with a High-Quality output
/// token that produces significantly sharper and more accurate mask boundaries. While SAM sometimes
/// produces coarse masks (especially on thin structures like bicycle spokes or fences), SAM-HQ adds
/// learnable components that refine boundaries to pixel-level precision.
///
/// Common use cases:
/// - High-precision object segmentation where boundary quality matters
/// - Thin structure segmentation (wires, fences, poles)
/// - Medical imaging requiring precise boundaries
/// - Any SAM use case where mask quality needs improvement
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Adds an HQ output token alongside SAM's original output tokens
/// - Global-local feature fusion: combines early ViT features (local) with final features (global)
/// - Trained on only 44K fine-grained masks from HQSeg-44K dataset
/// - +17.6 mBIoU improvement on DIS-val5K (thin/complex structures)
/// - Backbone: ViT-B/L/H (same as SAM)
/// </para>
/// <para>
/// <b>Reference:</b> Ke et al., "Segment Anything in High Quality", NeurIPS 2023.
/// </para>
/// </remarks>
public class SAMHQ<T> : NeuralNetworkBase<T>
{
    private readonly SAMHQOptions _options;

    /// <summary>
    /// Gets the configuration options for this SAM-HQ model.
    /// </summary>
    /// <returns>The <see cref="SAMHQOptions"/> for this model instance.</returns>
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
    private readonly SAMHQModelSize _modelSize;
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
    /// Gets whether this SAM-HQ instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode (trainable) and <c>false</c>
    /// in ONNX mode (inference only). SAM-HQ can be fine-tuned on custom high-quality masks.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal SAMHQModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes SAM-HQ in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW, as used in the SAM-HQ paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss; the paper uses focal loss
    /// + dice loss + IoU loss).</param>
    /// <param name="numClasses">Number of output mask classes (default: 1 for binary segmentation,
    /// as SAM-HQ produces per-prompt binary masks).</param>
    /// <param name="modelSize">ViT backbone size (default: ViTBase, 91M params).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SAM-HQ model. The HQ output token learns to
    /// produce sharper boundaries than the original SAM. Training uses only 44K fine-grained masks
    /// from the HQSeg-44K dataset, making it efficient to fine-tune.
    /// </para>
    /// </remarks>
    public SAMHQ(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 1,
        SAMHQModelSize modelSize = SAMHQModelSize.ViTBase,
        double dropRate = 0.1,
        SAMHQOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new SAMHQOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
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
    /// Initializes SAM-HQ in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output mask classes (default: 1).</param>
    /// <param name="modelSize">ViT backbone size for metadata (default: ViTBase).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SAM-HQ model from an ONNX file for fast inference.
    /// ONNX mode does not support training.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SAMHQ(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 1,
        SAMHQModelSize modelSize = SAMHQModelSize.ViTBase,
        SAMHQOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new SAMHQOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SAM-HQ ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
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
            throw new InvalidOperationException($"Failed to load SAM-HQ ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through SAM-HQ to produce high-quality segmentation masks.
    /// </summary>
    /// <param name="input">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel mask logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image and get back high-quality segmentation masks.
    /// The HQ output token ensures boundary precision is significantly better than standard SAM.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : PredictOnnx(input);
    }

    /// <summary>
    /// Performs one training step: forward pass, loss computation, backward pass, and parameter update.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="expectedOutput">The ground-truth high-quality mask tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model to produce better HQ masks by comparing
    /// predictions to ground truth. Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SAMHQModelSize modelSize)
    {
        return modelSize switch
        {
            SAMHQModelSize.ViTBase => ([768, 768, 768, 768], [12, 0, 0, 0], 256),
            SAMHQModelSize.ViTLarge => ([1024, 1024, 1024, 1024], [24, 0, 0, 0], 256),
            SAMHQModelSize.ViTHuge => ([1280, 1280, 1280, 1280], [32, 0, 0, 0], 256),
            _ => ([768, 768, 768, 768], [12, 0, 0, 0], 256)
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
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

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
        for (int i = 0; i < newShape.Length; i++) newShape[i] = tensor.Shape[i + 1];
        var result = new Tensor<T>(newShape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the encoder and decoder layers for SAM-HQ.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the ViT encoder and HQ mask decoder layers.
    /// In ONNX mode, no layers are created since the runtime handles everything.
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
            var encoderLayers = LayerHelper<T>.CreateSAMHQEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int featureH = _height / 16;
            int featureW = _width / 16;
            int encoderOutputChannels = _channelDims[^1];
            var decoderLayers = LayerHelper<T>.CreateSAMHQDecoderLayers(
                encoderOutputChannels, _decoderDim, _numClasses, featureH, featureW);
            Layers.AddRange(decoderLayers);
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">Flat vector of all model parameters.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces all model weights with new values. Used internally during
    /// optimization and when loading saved weights.
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
    /// Collects metadata describing this SAM-HQ model's configuration.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> with model type, architecture, and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary of the model's configuration for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SemanticSegmentation,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "SAMHQ" }, { "InputHeight", _height }, { "InputWidth", _width },
                { "InputChannels", _channels }, { "NumClasses", _numClasses },
                { "ModelSize", _modelSize.ToString() }, { "DecoderDim", _decoderDim },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Writes SAM-HQ configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves model configuration for later reconstruction.
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
    /// Reads SAM-HQ configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString();
        _ = reader.ReadInt32();
        int dimCount = reader.ReadInt32();
        for (int i = 0; i < dimCount; i++) _ = reader.ReadInt32();
        int depthCount = reader.ReadInt32();
        for (int i = 0; i < depthCount; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new SAM-HQ instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new <see cref="SAMHQ{T}"/> model with reinitialized weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy of the model's configuration with fresh random weights.
    /// Used for cross-validation and ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new SAMHQ<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
            : new SAMHQ<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);
    }

    /// <summary>
    /// Releases managed resources including the ONNX inference session.
    /// </summary>
    /// <param name="disposing">True when called from Dispose(), false from finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees memory used by the ONNX runtime when done with the model.
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
}
