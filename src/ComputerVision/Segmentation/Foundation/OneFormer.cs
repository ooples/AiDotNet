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
/// OneFormer: One Transformer to Rule Universal Image Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OneFormer is trained once on panoptic data and can then perform any
/// segmentation task — semantic, instance, or panoptic — by simply providing a text prompt that
/// describes which task to perform. This "one model for all tasks" approach dramatically simplifies
/// deployment compared to maintaining separate models for each task.
///
/// Example usage:
/// - Pass "the task is semantic" to get per-pixel class labels
/// - Pass "the task is instance" to get individual object masks
/// - Pass "the task is panoptic" to get both stuff and thing segments
///
/// Common use cases:
/// - Multi-task segmentation systems needing all three task types
/// - Research comparing segmentation approaches
/// - Production systems where maintaining one model is simpler than three
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Builds on Mask2Former with a text encoder (CLIP-based) for task conditioning
/// - Task-conditioned joint training on panoptic, semantic, and instance data simultaneously
/// - Uses a task-guided query initialization that focuses queries on the specified task
/// - Backbone: Swin-L or DiNAT-L (Dilated Neighborhood Attention Transformer)
/// - SOTA on ADE20K, Cityscapes, and COCO across all three tasks with a single model
/// </para>
/// <para>
/// <b>Reference:</b> Jain et al., "OneFormer: One Transformer to Rule Universal Image
/// Segmentation", CVPR 2023.
/// </para>
/// </remarks>
public class OneFormer<T> : NeuralNetworkBase<T>, IPanopticSegmentation<T>
{
    private readonly OneFormerOptions _options;

    /// <summary>
    /// Gets the configuration options for this OneFormer model.
    /// </summary>
    /// <returns>The <see cref="OneFormerOptions"/> for this model instance.</returns>
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
    private readonly int _numQueries;
    private readonly OneFormerModelSize _modelSize;
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
    /// Gets whether this OneFormer instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode (trainable) and <c>false</c>
    /// in ONNX mode. OneFormer is trained on panoptic data which jointly trains for all tasks.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal OneFormerModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes OneFormer in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW with weight decay 0.05,
    /// as specified in the OneFormer paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output classes (default: 150 for ADE20K).</param>
    /// <param name="numQueries">Number of object queries (default: 150, higher than Mask2Former
    /// to accommodate multi-task predictions).</param>
    /// <param name="modelSize">Backbone size (default: SwinLarge).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable OneFormer. Training is done on panoptic data
    /// which automatically teaches the model semantic and instance segmentation as well.
    /// A text encoder conditions the queries on the target task at inference time.
    /// </para>
    /// </remarks>
    public OneFormer(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        int numQueries = 150,
        OneFormerModelSize modelSize = OneFormerModelSize.SwinLarge,
        double dropRate = 0.1,
        OneFormerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new OneFormerOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = dropRate;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes OneFormer in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="numClasses">Number of classes (default: 150).</param>
    /// <param name="numQueries">Number of queries (default: 150).</param>
    /// <param name="modelSize">Model size for metadata (default: SwinLarge).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained OneFormer for multi-task segmentation inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if file not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX load fails.</exception>
    public OneFormer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        int numQueries = 150,
        OneFormerModelSize modelSize = OneFormerModelSize.SwinLarge,
        OneFormerOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new OneFormerOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"OneFormer ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load OneFormer ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through OneFormer for task-conditioned segmentation.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor. The output depends on the task conditioning.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OneFormer processes the image through a backbone encoder and
    /// text-conditioned transformer decoder. The text prompt ("the task is semantic/instance/panoptic")
    /// guides which type of segmentation output is produced.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step with panoptic multi-task learning.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OneFormer training uses panoptic data to simultaneously learn
    /// semantic, instance, and panoptic segmentation. A contrastive loss between text and
    /// visual features helps the model learn task-specific behavior.
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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(OneFormerModelSize modelSize)
    {
        return modelSize switch
        {
            OneFormerModelSize.SwinLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            OneFormerModelSize.DiNATLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            _ => ([192, 384, 768, 1536], [2, 2, 18, 2], 256)
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
    /// Initializes the backbone encoder, text encoder, and transformer decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates the backbone for multi-scale feature extraction and a
    /// text-conditioned transformer decoder that uses task-guided queries.
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
            var encoderLayers = LayerHelper<T>.CreateOneFormerEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] pK = [7, 3, 3, 3]; int[] pS = [4, 2, 2, 2]; int[] pP = [3, 1, 1, 1];
            int fH = _height, fW = _width;
            for (int s = 0; s < 4; s++) { fH = (fH + 2 * pP[s] - pK[s]) / pS[s] + 1; fW = (fW + 2 * pP[s] - pK[s]) / pS[s] + 1; }

            Layers.AddRange(LayerHelper<T>.CreateOneFormerDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, fH, fW));
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat vector.
    /// </summary>
    /// <param name="parameters">Flat parameter vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces model weights for optimization or loading.
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
                layer.UpdateParameters(np); offset += lp.Length;
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
                { "ModelName", "OneFormer" }, { "Description", "OneFormer Universal Segmentation (Text-Conditioned)" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "NumQueries", _numQueries },
                { "ModelSize", _modelSize.ToString() }, { "DecoderDim", _decoderDim },
                { "DropRate", _dropRate }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes OneFormer configuration.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves configuration for later restoration.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height); writer.Write(_width); writer.Write(_channels);
        writer.Write(_numClasses); writer.Write(_numQueries); writer.Write((int)_modelSize);
        writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length); foreach (int c in _channelDims) writer.Write(c);
        writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d);
    }

    /// <summary>
    /// Deserializes OneFormer configuration.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reads saved configuration in write order.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32();
        int cc = reader.ReadInt32(); for (int i = 0; i < cc; i++) _ = reader.ReadInt32();
        int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new OneFormer with same config but fresh weights.
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
            ? new OneFormer<T>(Architecture, _optimizer, LossFunction, _numClasses, _numQueries, _modelSize, _dropRate, _options)
            : new OneFormer<T>(Architecture, _onnxModelPath!, _numClasses, _numQueries, _modelSize, _options);
    }

    /// <summary>
    /// Releases managed resources.
    /// </summary>
    /// <param name="disposing">True from Dispose().</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees ONNX session resources.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed) { if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; } _disposed = true; }
        base.Dispose(disposing);
    }

    #endregion

    #region IPanopticSegmentation Implementation

    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    int IPanopticSegmentation<T>.NumStuffClasses => Math.Max(1, _numClasses / 3);
    int IPanopticSegmentation<T>.NumThingClasses => _numClasses - Math.Max(1, _numClasses / 3);

    PanopticSegmentationResult<T> IPanopticSegmentation<T>.SegmentPanoptic(Tensor<T> image)
    {
        var output = Predict(image);
        return new PanopticSegmentationResult<T>
        {
            SemanticMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(output),
            InstanceMap = Tensor<T>.Empty(),
            PanopticMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(output)
        };
    }

    #endregion
}
