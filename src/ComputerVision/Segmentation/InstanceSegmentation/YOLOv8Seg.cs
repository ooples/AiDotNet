using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// YOLOv8-Seg: Ultralytics real-time instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLOv8-Seg is a fast, accurate instance segmentation model that detects
/// individual objects and produces per-pixel masks for each. It builds on the YOLO family's
/// anchor-free detection with a YOLACT-style prototype mask generation branch.
///
/// Common use cases:
/// - Real-time instance segmentation in video streams
/// - Object counting and measurement
/// - Autonomous driving perception
/// - Industrial quality inspection
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - CSPDarknet backbone with C2f (Cross-Stage Partial with 2 convolutions) blocks
/// - Anchor-free decoupled detection head
/// - YOLACT-style prototype masks with 32 coefficients
/// - Available in 5 sizes: N (3.4M), S (11.8M), M (27.3M), L (46.0M), X (71.8M)
/// - Input resolution: 640x640 by default
/// </para>
/// <para>
/// <b>Reference:</b> Ultralytics, "YOLOv8", 2023.
/// </para>
/// </remarks>
public class YOLOv8Seg<T> : NeuralNetworkBase<T>, IInstanceSegmentation<T>
{
    private readonly YOLOv8SegOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    private readonly int _height, _width, _channels, _numClasses;
    private readonly YOLOv8SegModelSize _modelSize;
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
    /// Gets whether this YOLOv8Seg instance supports training.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal YOLOv8SegModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;
    #endregion

    #region Constructors

    /// <summary>
    /// Initializes YOLOv8Seg in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW with weight decay 0.0005 per paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss; paper uses CIoU + DFL + BCE).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 80 for COCO).</param>
    /// <param name="modelSize">Model size variant (default: N for fastest inference).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    public YOLOv8Seg(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 80,
        YOLOv8SegModelSize modelSize = YOLOv8SegModelSize.N,
        double dropRate = 0,
        YOLOv8SegOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new YOLOv8SegOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 640;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
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
    /// Initializes YOLOv8Seg in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 80).</param>
    /// <param name="modelSize">Model size for metadata (default: N).</param>
    /// <param name="options">Optional model options.</param>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public YOLOv8Seg(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 80,
        YOLOv8SegModelSize modelSize = YOLOv8SegModelSize.N,
        YOLOv8SegOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new YOLOv8SegOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"YOLOv8Seg ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 640;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = 0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load YOLOv8Seg ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass to produce segmentation logits.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
        => _useNativeMode ? Forward(input) : PredictOnnx(input);

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var predicted = Forward(input);
        var lossGradient = LossFunction.ComputeGradient(predicted, expectedOutput);
        BackwardPass(lossGradient);
        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Private Methods

    // YOLOv8 architecture configs: width/depth multipliers per size
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(YOLOv8SegModelSize modelSize)
    {
        return modelSize switch
        {
            YOLOv8SegModelSize.N => ([16, 32, 64, 128], [1, 2, 2, 1], 64),
            YOLOv8SegModelSize.S => ([32, 64, 128, 256], [1, 2, 2, 1], 128),
            YOLOv8SegModelSize.M => ([48, 96, 192, 384], [2, 4, 4, 2], 192),
            YOLOv8SegModelSize.L => ([64, 128, 256, 512], [2, 4, 4, 2], 256),
            YOLOv8SegModelSize.X => ([80, 160, 320, 640], [2, 4, 4, 2], 320),
            _ => ([16, 32, 64, 128], [1, 2, 2, 1], 64)
        };
    }

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
    /// Initializes the CSPDarknet encoder and detection + mask decoder layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            // Prefer explicit encoder layer count from options; fall back to midpoint heuristic for custom layers
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateYOLOv8SegEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);
            int fH = _height / 32;
            int fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateYOLOv8SegDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat parameter vector.
    /// </summary>
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
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.SemanticSegmentation,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "YOLOv8Seg" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumClasses", _numClasses },
            { "ModelSize", _modelSize.ToString() },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        },
        ModelData = this.Serialize()
    };

    /// <summary>
    /// Writes configuration to a binary stream.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numClasses);
        writer.Write((int)_modelSize);
        writer.Write(_decoderDim);
        writer.Write(_dropRate);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length);
        foreach (int d in _channelDims) writer.Write(d);
        writer.Write(_depths.Length);
        foreach (int d in _depths) writer.Write(d);
    }

    /// <summary>
    /// Reads configuration from a binary stream.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();
        _ = reader.ReadBoolean();
        _ = reader.ReadString();
        _ = reader.ReadInt32();
        int dc = reader.ReadInt32();
        for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
        int dd = reader.ReadInt32();
        for (int i = 0; i < dd; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new YOLOv8Seg<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new YOLOv8Seg<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);

    /// <summary>
    /// Releases managed resources.
    /// </summary>
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

    #region IInstanceSegmentation Implementation

    private double _confidenceThreshold = 0.5;
    private double _nmsThreshold = 0.5;

    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    int IInstanceSegmentation<T>.MaxInstances => 100;

    double IInstanceSegmentation<T>.ConfidenceThreshold
    {
        get => _confidenceThreshold;
        set => _confidenceThreshold = value;
    }

    double IInstanceSegmentation<T>.NmsThreshold
    {
        get => _nmsThreshold;
        set => _nmsThreshold = value;
    }

    InstanceSegmentationResult<T> IInstanceSegmentation<T>.DetectInstances(Tensor<T> image)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var logits = Predict(image);
        var probMap = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var classMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = classMap.Shape[0], w = classMap.Shape[1];
        var instances = new List<InstanceMask<T>>();

        for (int cls = 1; cls < _numClasses && instances.Count < 100; cls++)
        {
            var (labelMap, count) = Common.SegmentationTensorOps.LabelConnectedComponents(classMap, cls);
            for (int comp = 1; comp <= count && instances.Count < 100; comp++)
            {
                var mask = new Tensor<T>([h, w]);
                int area = 0;
                double sumConf = 0;
                int minX = w, minY = h, maxX = 0, maxY = 0;

                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        if (Math.Abs(NumOps.ToDouble(labelMap[row, col]) - comp) < 0.5)
                        {
                            mask[row, col] = NumOps.FromDouble(1.0);
                            area++;
                            sumConf += NumOps.ToDouble(probMap[cls, row, col]);
                            if (col < minX) minX = col;
                            if (col > maxX) maxX = col;
                            if (row < minY) minY = row;
                            if (row > maxY) maxY = row;
                        }
                    }
                }

                if (area < 4) continue;
                double confidence = sumConf / area;
                if (confidence < _confidenceThreshold) continue;

                var box = new BoundingBox<T>(
                    NumOps.FromDouble(minX), NumOps.FromDouble(minY),
                    NumOps.FromDouble(maxX + 1), NumOps.FromDouble(maxY + 1),
                    BoundingBoxFormat.XYXY, cls);
                instances.Add(new InstanceMask<T>(box, mask, cls, NumOps.FromDouble(confidence)));
            }
        }

        instances = instances.OrderByDescending(i => NumOps.ToDouble(i.Confidence)).ToList();
        var kept = new List<InstanceMask<T>>();
        while (instances.Count > 0)
        {
            var best = instances[0];
            kept.Add(best);
            instances.RemoveAt(0);
            instances = instances.Where(inst => best.ComputeMaskIoU(inst, NumOps) < _nmsThreshold).ToList();
        }

        sw.Stop();
        return new InstanceSegmentationResult<T>
        {
            Instances = kept,
            ImageHeight = h,
            ImageWidth = w,
            InferenceTime = sw.Elapsed
        };
    }

    #endregion
}
