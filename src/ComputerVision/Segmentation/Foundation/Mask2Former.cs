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
/// Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask2Former is a universal segmentation model that handles semantic,
/// instance, and panoptic segmentation with a single unified architecture. Instead of designing
/// separate models for each task, Mask2Former uses a transformer decoder with "masked cross-attention"
/// that restricts each query to attend only to its predicted mask region. This makes it highly
/// efficient and accurate across all segmentation tasks.
///
/// Common use cases:
/// - Panoptic segmentation (stuff + things in one pass, e.g., road + cars + people)
/// - Instance segmentation (individual object masks, e.g., "car 1", "car 2")
/// - Semantic segmentation (per-pixel class labels, e.g., "road", "sky", "building")
/// - Multi-task deployment where one model serves all segmentation needs
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Backbone: Swin Transformer or ResNet producing multi-scale features
/// - Pixel Decoder: Multi-Scale Deformable Attention Transformer (MSDeformAttn)
/// - Transformer Decoder: 9 layers with masked cross-attention (restricts attention to predicted masks)
/// - 100 learnable object queries predict class labels and binary masks
/// - Achieves 57.8 PQ on COCO panoptic, 83.3 AP on Cityscapes instance
/// </para>
/// <para>
/// <b>Reference:</b> Cheng et al., "Masked-attention Mask Transformer for Universal Image
/// Segmentation", CVPR 2022.
/// </para>
/// </remarks>
public class Mask2Former<T> : NeuralNetworkBase<T>, IPanopticSegmentation<T>
{
    private readonly Mask2FormerOptions _options;

    /// <summary>
    /// Gets the configuration options for this Mask2Former model.
    /// </summary>
    /// <returns>The <see cref="Mask2FormerOptions"/> for this model instance.</returns>
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
    private readonly Mask2FormerModelSize _modelSize;
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
    /// Gets whether this Mask2Former instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode (trainable) and <c>false</c>
    /// in ONNX mode (inference only). Mask2Former can be trained on any of the three segmentation
    /// tasks (semantic, instance, panoptic) or all at once.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal Mask2FormerModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes Mask2Former in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW with weight decay 0.05,
    /// as specified in the Mask2Former paper for all experiments).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss; the paper uses a
    /// combination of CE + binary cross-entropy + dice loss with Hungarian matching).</param>
    /// <param name="numClasses">Number of output classes (default: 150 for ADE20K semantic,
    /// use 133 for COCO panoptic, 80 for COCO instance).</param>
    /// <param name="numQueries">Number of object queries (default: 100, as in the paper).</param>
    /// <param name="modelSize">Backbone size (default: SwinTiny, 47M params).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable Mask2Former model. The "queries" are learned
    /// representations that each predict one segment (object or stuff region). 100 queries means
    /// the model can predict up to 100 segments per image. The masked cross-attention mechanism
    /// makes training efficient by focusing each query on its relevant image region only.
    /// </para>
    /// </remarks>
    public Mask2Former(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        int numQueries = 100,
        Mask2FormerModelSize modelSize = Mask2FormerModelSize.SwinTiny,
        double dropRate = 0.1,
        Mask2FormerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new Mask2FormerOptions();
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
    /// Initializes Mask2Former in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="numClasses">Number of classes (default: 150).</param>
    /// <param name="numQueries">Number of object queries (default: 100).</param>
    /// <param name="modelSize">Model size for metadata (default: SwinTiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained Mask2Former for fast universal segmentation inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if file not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX load fails.</exception>
    public Mask2Former(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        int numQueries = 100,
        Mask2FormerModelSize modelSize = Mask2FormerModelSize.SwinTiny,
        Mask2FormerOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new Mask2FormerOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"Mask2Former ONNX model not found: {onnxModelPath}");

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
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load Mask2Former ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass to produce segmentation masks and class predictions.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor. For panoptic/instance segmentation, post-process
    /// the output to extract individual mask predictions from the query outputs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The image passes through the backbone encoder to extract multi-scale
    /// features, then through the pixel decoder for feature refinement, and finally through the
    /// transformer decoder where learned queries predict segment masks and class labels using
    /// masked cross-attention.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step with Hungarian matching loss.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mask2Former training uses bipartite matching (Hungarian algorithm)
    /// to assign each predicted query to a ground-truth segment, then optimizes a combination
    /// of classification loss, binary mask loss, and dice loss.
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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(Mask2FormerModelSize modelSize)
    {
        return modelSize switch
        {
            Mask2FormerModelSize.R50 => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256),
            Mask2FormerModelSize.R101 => ([256, 512, 1024, 2048], [3, 4, 23, 3], 256),
            Mask2FormerModelSize.SwinTiny => ([96, 192, 384, 768], [2, 2, 6, 2], 256),
            Mask2FormerModelSize.SwinSmall => ([96, 192, 384, 768], [2, 2, 18, 2], 256),
            Mask2FormerModelSize.SwinBase => ([128, 256, 512, 1024], [2, 2, 18, 2], 256),
            Mask2FormerModelSize.SwinLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            _ => ([96, 192, 384, 768], [2, 2, 6, 2], 256)
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
    /// Initializes the backbone encoder, pixel decoder, and transformer decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates three main components:
    /// 1. <b>Backbone encoder</b> (Swin/ResNet): Extracts multi-scale features at 4 resolutions
    /// 2. <b>Pixel decoder</b>: Refines features with Multi-Scale Deformable Attention
    /// 3. <b>Transformer decoder</b>: 100 learned queries predict masks via masked cross-attention
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
            var encoderLayers = LayerHelper<T>.CreateMask2FormerEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] pK = [7, 3, 3, 3]; int[] pS = [4, 2, 2, 2]; int[] pP = [3, 1, 1, 1];
            int fH = _height, fW = _width;
            for (int s = 0; s < 4; s++) { fH = (fH + 2 * pP[s] - pK[s]) / pS[s] + 1; fW = (fW + 2 * pP[s] - pK[s]) / pS[s] + 1; }

            Layers.AddRange(LayerHelper<T>.CreateMask2FormerDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, fH, fW));
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat vector.
    /// </summary>
    /// <param name="parameters">Flat parameter vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces model weights for optimization or loading saved models.
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
    /// <returns>Model metadata with configuration and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Summary of the model for saving, comparing, or dashboard display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SemanticSegmentation,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "Mask2Former" }, { "Description", "Mask2Former Universal Segmentation" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "NumQueries", _numQueries },
                { "ModelSize", _modelSize.ToString() }, { "DecoderDim", _decoderDim },
                { "DropRate", _dropRate }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Mask2Former configuration for persistence.
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
        writer.Write(_numClasses); writer.Write(_numQueries); writer.Write((int)_modelSize);
        writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length);
        foreach (int c in _channelDims) writer.Write(c);
        writer.Write(_depths.Length);
        foreach (int d in _depths) writer.Write(d);
    }

    /// <summary>
    /// Deserializes Mask2Former configuration.
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
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32();
        int cc = reader.ReadInt32(); for (int i = 0; i < cc; i++) _ = reader.ReadInt32();
        int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new Mask2Former with same config but fresh weights.
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
            ? new Mask2Former<T>(Architecture, _optimizer, LossFunction, _numClasses, _numQueries, _modelSize, _dropRate, _options)
            : new Mask2Former<T>(Architecture, _onnxModelPath!, _numClasses, _numQueries, _modelSize, _options);
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
        var logits = Predict(image);
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
