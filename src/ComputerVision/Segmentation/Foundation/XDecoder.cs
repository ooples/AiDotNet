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
/// X-Decoder: Generalized Decoding for Pixel, Image, and Language.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> X-Decoder is a generalist model that simultaneously handles referring
/// segmentation (find and segment an object from a text description), open-vocabulary segmentation
/// (segment objects from any text class list), and image captioning — all with one shared decoder.
/// It bridges the gap between pixel-level understanding and language understanding.
///
/// Common use cases:
/// - Referring segmentation ("segment the red car on the left")
/// - Open-vocabulary semantic segmentation with arbitrary class names
/// - Image captioning and visual question answering
/// - Multi-modal vision-language systems
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Two-path decoder: pixel path (mask predictions) and token path (text predictions)
/// - Both paths share the same cross-attention mechanism
/// - Supports any combination of text, pixel, and image inputs/outputs
/// - Backbone: Focal-T/B/L transformer
/// - Single model handles 7+ vision-language tasks
/// </para>
/// <para>
/// <b>Reference:</b> Zou et al., "Generalized Decoding for Pixel, Image, and Language", CVPR 2023.
/// </para>
/// </remarks>
public class XDecoder<T> : NeuralNetworkBase<T>, IPanopticSegmentation<T>
{
    private readonly XDecoderOptions _options;

    /// <summary>
    /// Gets the configuration options for this X-Decoder model.
    /// </summary>
    /// <returns>The <see cref="XDecoderOptions"/> for this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numClasses;
    private int _numQueries;
    private XDecoderModelSize _modelSize;
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
    private int _numStuffClasses;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this X-Decoder instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode, <c>false</c> in ONNX mode.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal XDecoderModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes X-Decoder in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW as in the paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output classes (default: 150 for ADE20K).</param>
    /// <param name="numQueries">Number of pixel/token queries (default: 100).</param>
    /// <param name="modelSize">Backbone size (default: Tiny).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable X-Decoder. The queries are split between pixel-path
    /// (for masks) and token-path (for text). Both paths share attention layers for efficient
    /// multi-modal learning.
    /// </para>
    /// </remarks>
    public XDecoder(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        int numQueries = 100,
        XDecoderModelSize modelSize = XDecoderModelSize.Tiny,
        double dropRate = 0.1,
        XDecoderOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "numClasses must be > 0.");
        if (numQueries <= 0)
            throw new ArgumentOutOfRangeException(nameof(numQueries), "numQueries must be > 0.");
        _options = options ?? new XDecoderOptions();
        Options = _options;
        if (_options.NumStuffClasses is int stuff && (stuff <= 0 || stuff >= numClasses))
            throw new ArgumentOutOfRangeException(nameof(options), "NumStuffClasses must be between 1 and numClasses-1.");
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = dropRate;
        _numStuffClasses = _options.NumStuffClasses ?? Math.Max(1, _numClasses / 3);
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes X-Decoder in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output classes (default: 150).</param>
    /// <param name="numQueries">Number of queries (default: 100).</param>
    /// <param name="modelSize">Backbone size for metadata (default: Tiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained X-Decoder from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public XDecoder(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        int numQueries = 100,
        XDecoderModelSize modelSize = XDecoderModelSize.Tiny,
        XDecoderOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "numClasses must be > 0.");
        if (numQueries <= 0)
            throw new ArgumentOutOfRangeException(nameof(numQueries), "numQueries must be > 0.");
        _options = options ?? new XDecoderOptions();
        Options = _options;
        if (_options.NumStuffClasses is int stuff && (stuff <= 0 || stuff >= numClasses))
            throw new ArgumentOutOfRangeException(nameof(options), "NumStuffClasses must be between 1 and numClasses-1.");

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"X-Decoder ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _numStuffClasses = _options.NumStuffClasses ?? Math.Max(1, _numClasses / 3);
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load X-Decoder ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through X-Decoder for generalist vision-language segmentation.
    /// </summary>
    /// <param name="input">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get segmentation or captioning predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : PredictOnnx(input);
    }

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains X-Decoder on vision-language data. Only native mode.
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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(XDecoderModelSize modelSize)
    {
        return modelSize switch
        {
            XDecoderModelSize.Tiny => ([96, 192, 384, 768], [2, 2, 6, 2], 256),
            XDecoderModelSize.Base => ([128, 256, 512, 1024], [2, 2, 18, 2], 256),
            XDecoderModelSize.Large => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            _ => throw new ArgumentOutOfRangeException(nameof(modelSize), modelSize, "Unknown XDecoder model size.")
        };
    }

    private Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 3 && input.Rank != 4)
            throw new ArgumentException("Input must be rank 3 [C,H,W] or rank 4 [N,C,H,W].", nameof(input));

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
        if (input.Rank != 3 && input.Rank != 4)
            throw new ArgumentException("Input must be rank 3 [C,H,W] or rank 4 [N,C,H,W].", nameof(input));
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
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
    /// Initializes the encoder and dual-path decoder layers for X-Decoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the Focal backbone and dual-path decoder
    /// (pixel path + token path). In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            var enc = _options.EncoderLayerCount;
            if (enc is int encVal && (encVal <= 0 || encVal >= Architecture.Layers.Count))
                throw new ArgumentOutOfRangeException(nameof(_options.EncoderLayerCount),
                    $"EncoderLayerCount must be between 1 and {Architecture.Layers.Count - 1}.");
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = enc ?? Architecture.Layers.Count / 2;
        }
        else
        {
            const int BackboneStride = 32;
            if (_height % BackboneStride != 0 || _width % BackboneStride != 0)
                throw new InvalidOperationException(
                    $"Input dimensions ({_height}x{_width}) must be divisible by backbone stride ({BackboneStride}).");
            var encoderLayers = LayerHelper<T>.CreateXDecoderEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);
            int featureH = _height / BackboneStride;
            int featureW = _width / BackboneStride;
            var decoderLayers = LayerHelper<T>.CreateXDecoderDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW);
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
        int expected = 0;
        foreach (var layer in Layers) expected += layer.GetParameters().Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Parameter vector length {parameters.Length} does not match expected {expected}.", nameof(parameters));

        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = layer.GetParameters().Length;
            var newParams = new Vector<T>(count);
            for (int i = 0; i < count; i++) newParams[i] = parameters[offset + i];
            layer.UpdateParameters(newParams);
            offset += count;
        }
    }

    /// <summary>
    /// Collects metadata describing this X-Decoder model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
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
                { "ModelName", "XDecoder" }, { "InputHeight", _height }, { "InputWidth", _width },
                { "InputChannels", _channels }, { "NumClasses", _numClasses },
                { "NumQueries", _numQueries }, { "ModelSize", _modelSize.ToString() },
                { "DecoderDim", _decoderDim }, { "UseNativeMode", _useNativeMode },
                { "NumLayers", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Writes X-Decoder configuration to a binary stream.
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
        writer.Write(_encoderLayerEnd); writer.Write(_numStuffClasses);
        writer.Write(_channelDims.Length);
        foreach (int dim in _channelDims) writer.Write(dim);
        writer.Write(_depths.Length);
        foreach (int depth in _depths) writer.Write(depth);
    }

    /// <summary>
    /// Reads X-Decoder configuration from a binary stream.
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
        _numQueries = reader.ReadInt32();
        _modelSize = (XDecoderModelSize)reader.ReadInt32();
        _decoderDim = reader.ReadInt32();
        _dropRate = reader.ReadDouble();
        _useNativeMode = reader.ReadBoolean();
        _onnxModelPath = reader.ReadString();
        _encoderLayerEnd = reader.ReadInt32();
        _numStuffClasses = reader.ReadInt32();
        int dimCount = reader.ReadInt32();
        _channelDims = new int[dimCount];
        for (int i = 0; i < dimCount; i++) _channelDims[i] = reader.ReadInt32();
        int depthCount = reader.ReadInt32();
        _depths = new int[depthCount];
        for (int i = 0; i < depthCount; i++) _depths[i] = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new X-Decoder instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new <see cref="XDecoder{T}"/> model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new XDecoder<T>(Architecture, _optimizer, LossFunction, _numClasses, _numQueries, _modelSize, _dropRate, _options)
            : new XDecoder<T>(Architecture, _onnxModelPath!, _numClasses, _numQueries, _modelSize, _options);
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
    int IPanopticSegmentation<T>.NumStuffClasses => _numStuffClasses;
    int IPanopticSegmentation<T>.NumThingClasses => _numClasses - _numStuffClasses;

    PanopticSegmentationResult<T> IPanopticSegmentation<T>.SegmentPanoptic(Tensor<T> image)
    {
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        var probMap = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var semanticMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = semanticMap.Shape[0];
        int w = semanticMap.Shape[1];
        const int PanopticLabelDivisor = 1000;
        int numStuff = _numStuffClasses;
        var instanceMap = new Tensor<T>([h, w]);
        var panopticMap = new Tensor<T>([h, w]);
        var segments = new List<PanopticSegment<T>>();
        int nextInstId = 1;

        // Stuff classes: non-countable background regions (sky, road, etc.)
        for (int cls = 0; cls < numStuff; cls++)
        {
            int area = 0;
            double sumConf = 0;
            int segmentId = cls * PanopticLabelDivisor;
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    if (Math.Abs(NumOps.ToDouble(semanticMap[row, col]) - cls) < 0.5)
                    {
                        panopticMap[row, col] = NumOps.FromDouble(segmentId);
                        area++;
                        sumConf += NumOps.ToDouble(probMap[cls, row, col]);
                    }
                }
            }

            if (area > 0)
            {
                segments.Add(new PanopticSegment<T>
                {
                    SegmentId = segmentId,
                    ClassId = cls,
                    IsThing = false,
                    Confidence = sumConf / area,
                    Area = area
                });
            }
        }

        // Thing classes: countable object instances (person, car, etc.)
        for (int cls = numStuff; cls < _numClasses; cls++)
        {
            var (labelMap, count) = Common.SegmentationTensorOps.LabelConnectedComponents(semanticMap, cls);
            for (int comp = 1; comp <= count; comp++)
            {
                int instId = nextInstId++;
                int segmentId = cls * PanopticLabelDivisor + instId;
                int area = 0;
                double sumConf = 0;
                var compMask = new Tensor<T>([h, w]);

                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        if (Math.Abs(NumOps.ToDouble(labelMap[row, col]) - comp) < 0.5)
                        {
                            instanceMap[row, col] = NumOps.FromDouble(instId);
                            panopticMap[row, col] = NumOps.FromDouble(segmentId);
                            compMask[row, col] = NumOps.FromDouble(1.0);
                            area++;
                            sumConf += NumOps.ToDouble(probMap[cls, row, col]);
                        }
                    }
                }

                if (area > 0)
                {
                    segments.Add(new PanopticSegment<T>
                    {
                        SegmentId = segmentId,
                        ClassId = cls,
                        IsThing = true,
                        Confidence = sumConf / area,
                        Area = area,
                        Mask = compMask
                    });
                }
            }
        }

        return new PanopticSegmentationResult<T>
        {
            SemanticMap = semanticMap,
            InstanceMap = instanceMap,
            PanopticMap = panopticMap,
            Segments = segments
        };
    }

    #endregion
}
