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
/// Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask DINO extends the powerful DINO object detector with a mask prediction
/// branch, creating a unified architecture that handles object detection, instance segmentation,
/// panoptic segmentation, and semantic segmentation all in one model. Instead of building separate
/// models for each task, Mask DINO uses a shared backbone and query-based transformer to do everything.
///
/// Common use cases:
/// - Joint object detection + instance segmentation
/// - Panoptic segmentation (detecting all things and stuff)
/// - Research requiring a unified detection-segmentation framework
/// - Production systems needing both detection boxes and segmentation masks
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Built on DINO detector with deformable attention transformer encoder-decoder
/// - Adds a mask branch using dot product between query embeddings and pixel embeddings
/// - Unified query matching for both box and mask predictions via Hungarian matching
/// - Backbone: ResNet-50 or Swin-L
/// - Achieves 54.5 AP on COCO instance, 59.4 PQ on COCO panoptic
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "Mask DINO: Towards A Unified Transformer-based Framework
/// for Object Detection and Segmentation", CVPR 2023.
/// </para>
/// </remarks>
public class MaskDINO<T> : NeuralNetworkBase<T>, IPanopticSegmentation<T>
{
    private readonly MaskDINOOptions _options;

    /// <summary>
    /// Gets the configuration options for this Mask DINO model.
    /// </summary>
    /// <returns>The <see cref="MaskDINOOptions"/> for this model instance.</returns>
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
    private readonly MaskDINOModelSize _modelSize;
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
    /// Gets whether this Mask DINO instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode (trainable) and <c>false</c>
    /// in ONNX mode (inference only).
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal MaskDINOModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes Mask DINO in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW as specified in the paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output classes (default: 80 for COCO).</param>
    /// <param name="numQueries">Number of object queries (default: 300, as in the paper).</param>
    /// <param name="modelSize">Backbone size (default: R50).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable Mask DINO model. 300 queries means the model
    /// can detect and segment up to 300 objects per image. The unified architecture jointly
    /// optimizes both detection boxes and segmentation masks.
    /// </para>
    /// </remarks>
    public MaskDINO(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 80,
        int numQueries = 300,
        MaskDINOModelSize modelSize = MaskDINOModelSize.R50,
        double dropRate = 0.1,
        MaskDINOOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new MaskDINOOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 800;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1333;
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
    /// Initializes Mask DINO in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output classes (default: 80).</param>
    /// <param name="numQueries">Number of object queries (default: 300).</param>
    /// <param name="modelSize">Backbone size for metadata (default: R50).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained Mask DINO from ONNX for fast inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public MaskDINO(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 80,
        int numQueries = 300,
        MaskDINOModelSize modelSize = MaskDINOModelSize.R50,
        MaskDINOOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new MaskDINOOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"Mask DINO ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 800;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1333;
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
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load Mask DINO ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass to produce detection boxes and segmentation masks.
    /// </summary>
    /// <param name="input">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get joint detection and segmentation predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : PredictOnnx(input);
    }

    /// <summary>
    /// Performs one training step with forward, loss, backward, and parameter update.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains the model by comparing predictions to correct answers.
    /// Only available in native mode.
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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(MaskDINOModelSize modelSize)
    {
        return modelSize switch
        {
            MaskDINOModelSize.R50 => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256),
            MaskDINOModelSize.SwinLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            _ => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256)
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
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++) newShape[i] = tensor.Shape[i + 1];
        var result = new Tensor<T>(newShape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the encoder and decoder layers for Mask DINO.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the backbone encoder (ResNet/Swin), deformable
    /// transformer encoder-decoder, and mask prediction head. In ONNX mode, layers are skipped.
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
            var encoderLayers = LayerHelper<T>.CreateMaskDINOEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int featureH = _height / 32;
            int featureW = _width / 32;
            int encoderOutputChannels = _channelDims[^1];
            var decoderLayers = LayerHelper<T>.CreateMaskDINODecoderLayers(
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
    /// <b>For Beginners:</b> Replaces all model weights with new values.
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
    /// Collects metadata describing this Mask DINO model's configuration.
    /// </summary>
    /// <returns>Model metadata including type, architecture, and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SemanticSegmentation,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "MaskDINO" }, { "InputHeight", _height }, { "InputWidth", _width },
                { "InputChannels", _channels }, { "NumClasses", _numClasses },
                { "NumQueries", _numQueries }, { "ModelSize", _modelSize.ToString() },
                { "DecoderDim", _decoderDim }, { "UseNativeMode", _useNativeMode },
                { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Writes Mask DINO configuration to a binary stream.
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
        writer.Write(_numClasses); writer.Write(_numQueries); writer.Write((int)_modelSize);
        writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length);
        foreach (int dim in _channelDims) writer.Write(dim);
        writer.Write(_depths.Length);
        foreach (int depth in _depths) writer.Write(depth);
    }

    /// <summary>
    /// Reads Mask DINO configuration from a binary stream.
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
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString();
        _ = reader.ReadInt32();
        int dimCount = reader.ReadInt32();
        for (int i = 0; i < dimCount; i++) _ = reader.ReadInt32();
        int depthCount = reader.ReadInt32();
        for (int i = 0; i < depthCount; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new Mask DINO instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new <see cref="MaskDINO{T}"/> model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new MaskDINO<T>(Architecture, _optimizer, LossFunction, _numClasses, _numQueries, _modelSize, _dropRate, _options)
            : new MaskDINO<T>(Architecture, _onnxModelPath!, _numClasses, _numQueries, _modelSize, _options);
    }

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
    {
        if (!_disposed)
        {
            if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; }
            _disposed = true;
        }
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
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        var probMap = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var semanticMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = semanticMap.Shape[0], w = semanticMap.Shape[1];
        int numStuff = Math.Max(1, _numClasses / 3);
        var instanceMap = new Tensor<T>([h, w]);
        var panopticMap = new Tensor<T>([h, w]);
        var segments = new List<PanopticSegment<T>>();
        int nextInstId = 1;
        for (int cls = 0; cls < numStuff; cls++)
        {
            int area = 0; double sumConf = 0;
            for (int row = 0; row < h; row++)
                for (int col = 0; col < w; col++)
                    if (Math.Abs(NumOps.ToDouble(semanticMap[row, col]) - cls) < 0.5)
                    { panopticMap[row, col] = NumOps.FromDouble(cls * 1000); area++; sumConf += NumOps.ToDouble(probMap[cls, row, col]); }
            if (area > 0) segments.Add(new PanopticSegment<T> { SegmentId = cls, ClassId = cls, IsThing = false, Confidence = sumConf / area, Area = area });
        }
        for (int cls = numStuff; cls < _numClasses; cls++)
        {
            var (labelMap, count) = Common.SegmentationTensorOps.LabelConnectedComponents(semanticMap, cls);
            for (int comp = 1; comp <= count; comp++)
            {
                int instId = nextInstId++;
                int area = 0; double sumConf = 0; var compMask = new Tensor<T>([h, w]);
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < w; col++)
                        if (Math.Abs(NumOps.ToDouble(labelMap[row, col]) - comp) < 0.5)
                        { instanceMap[row, col] = NumOps.FromDouble(instId); panopticMap[row, col] = NumOps.FromDouble(cls * 1000 + instId); compMask[row, col] = NumOps.FromDouble(1.0); area++; sumConf += NumOps.ToDouble(probMap[cls, row, col]); }
                if (area > 0) segments.Add(new PanopticSegment<T> { SegmentId = instId, ClassId = cls, IsThing = true, Confidence = sumConf / area, Area = area, Mask = compMask });
            }
        }
        return new PanopticSegmentationResult<T> { SemanticMap = semanticMap, InstanceMap = instanceMap, PanopticMap = panopticMap, Segments = segments };
    }

    #endregion
}
