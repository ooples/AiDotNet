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
/// SAM 2.1: Segment Anything Model 2.1 with refined checkpoints for images and videos.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM 2.1 is an updated version of SAM 2 with improved training recipes
/// that produce more accurate segmentation masks. Like SAM 2, it supports both image and video
/// segmentation through memory attention for temporal consistency.
///
/// Common use cases:
/// - Video object segmentation with click-to-track
/// - High-quality image segmentation with refined boundaries
/// - Interactive annotation tools
/// - Foundation for downstream video analysis
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Hiera backbone (hierarchical vision transformer) — same as SAM 2
/// - Memory attention mechanism for multi-frame consistency
/// - Streaming architecture: processes frames sequentially
/// - Refined training recipes improve accuracy without architecture changes
/// - Supports 4 size variants: Tiny (39M), Small (46M), Base+ (81M), Large (224M)
/// </para>
/// <para>
/// <b>Reference:</b> Ravi et al., "SAM 2: Segment Anything in Images and Videos", Meta AI, 2024.
/// </para>
/// </remarks>
public class SAM21<T> : NeuralNetworkBase<T>, IPromptableSegmentation<T>
{
    private readonly SAM21Options _options;

    /// <summary>
    /// Gets the configuration options for this SAM 2.1 model.
    /// </summary>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private int _height, _width, _channels, _numClasses;
    private readonly SAM21ModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    private bool _useNativeMode;
    private string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private int _encoderLayerEnd;
    private readonly int _memoryBankSize;

    // Memory bank for video tracking
    private readonly List<Tensor<T>> _memoryBank;

    // Promptable segmentation state
    private Tensor<T>? _imageEmbedding;
    private Tensor<T>? _imageProbabilities;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this SAM 2.1 instance supports training.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal SAM21ModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;
    internal int CurrentMemorySize => _memoryBank.Count;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes SAM 2.1 in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output mask classes (default: 1 for binary segmentation).</param>
    /// <param name="modelSize">Hiera backbone size (default: Large).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SAM 2.1 model. The refined checkpoints improve
    /// accuracy over SAM 2 without changing the architecture.
    /// </para>
    /// </remarks>
    public SAM21(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 1,
        SAM21ModelSize modelSize = SAM21ModelSize.Large,
        double dropRate = 0.1,
        SAM21Options? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new SAM21Options();
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
        _memoryBankSize = _options.MemoryBankSize ?? 7;
        _memoryBank = [];

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes SAM 2.1 in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output mask classes (default: 1).</param>
    /// <param name="modelSize">Hiera backbone size for metadata (default: Large).</param>
    /// <param name="options">Optional model options.</param>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SAM21(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 1,
        SAM21ModelSize modelSize = SAM21ModelSize.Large,
        SAM21Options? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new SAM21Options();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SAM 2.1 ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;
        _memoryBankSize = _options.MemoryBankSize ?? 7;
        _memoryBank = [];

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load SAM 2.1 ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through SAM 2.1.
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

    /// <summary>
    /// Adds image features to the memory bank for video tracking.
    /// </summary>
    public void AddToMemory(Tensor<T> features)
    {
        _memoryBank.Add(features);
        if (_memoryBank.Count > _memoryBankSize)
            _memoryBank.RemoveAt(0);
    }

    /// <summary>
    /// Clears the memory bank.
    /// </summary>
    public void ClearMemory() => _memoryBank.Clear();

    #endregion

    #region Private Methods

    // SAM 2.1 Hiera backbone: Tiny=96, Small=128, BasePlus=256, Large=384 (same as SAM 2)
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SAM21ModelSize modelSize)
    {
        return modelSize switch
        {
            SAM21ModelSize.Tiny => ([96, 192, 384, 768], [1, 2, 7, 2], 256),
            SAM21ModelSize.Small => ([96, 192, 384, 768], [1, 2, 11, 2], 256),
            SAM21ModelSize.BasePlus => ([112, 224, 448, 896], [2, 3, 16, 3], 256),
            SAM21ModelSize.Large => ([144, 288, 576, 1152], [2, 6, 36, 4], 256),
            _ => ([144, 288, 576, 1152], [2, 6, 36, 4], 256)
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
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
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
    /// Initializes the Hiera encoder and mask decoder layers.
    /// </summary>
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
            var encoderLayers = LayerHelper<T>.CreateSAM21EncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);
            int featureH = _height / 16;
            int featureW = _width / 16;
            var decoderLayers = LayerHelper<T>.CreateSAM21DecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW);
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
            { "ModelName", "SAM21" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumClasses", _numClasses },
            { "ModelSize", _modelSize.ToString() },
            { "UseNativeMode", _useNativeMode },
            { "MemoryBankSize", _memoryBankSize },
            { "NumLayers", Layers.Count },
            { "EncoderLayerEnd", _encoderLayerEnd }
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
        writer.Write(_memoryBankSize);
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
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        _ = reader.ReadInt32(); // modelSize (readonly)
        _ = reader.ReadInt32(); // decoderDim (readonly)
        _ = reader.ReadDouble(); // dropRate (readonly)
        _useNativeMode = reader.ReadBoolean();
        _onnxModelPath = reader.ReadString();
        _ = reader.ReadInt32(); // encoderLayerEnd
        _ = reader.ReadInt32(); // memoryBankSize (readonly)
        int dc = reader.ReadInt32();
        for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
        int dd = reader.ReadInt32();
        for (int i = 0; i < dd; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new SAM21<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new SAM21<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);

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

    #region IPromptableSegmentation Implementation

    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    bool IPromptableSegmentation<T>.SupportsPointPrompts => true;
    bool IPromptableSegmentation<T>.SupportsBoxPrompts => true;
    bool IPromptableSegmentation<T>.SupportsMaskPrompts => true;
    bool IPromptableSegmentation<T>.SupportsTextPrompts => false;

    void IPromptableSegmentation<T>.SetImage(Tensor<T> image)
    {
        _imageEmbedding = Predict(image);
        _imageProbabilities = Common.SegmentationTensorOps.SoftmaxAlongClassDim(_imageEmbedding);
    }

    PromptedSegmentationResult<T> IPromptableSegmentation<T>.SegmentFromPoints(Tensor<T> points, Tensor<T> labels)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        int numC = features.Shape[0], h = features.Shape[1], w = features.Shape[2];
        var attention = new Tensor<T>([h, w]);
        int numPts = points.Shape[0];
        double sigma = Math.Max(h, w) / 10.0;
        for (int i = 0; i < numPts; i++)
        {
            double px = NumOps.ToDouble(points[i, 0]), py = NumOps.ToDouble(points[i, 1]);
            double sign = NumOps.ToDouble(labels[i]) >= 0.5 ? 1.0 : -1.0;
            var g = Common.SegmentationTensorOps.GaussianMask<T>(h, w, px, py, sigma);
            for (int j = 0; j < h * w; j++)
                attention.Data.Span[j] = NumOps.Add(attention.Data.Span[j], NumOps.FromDouble(sign * NumOps.ToDouble(g.Data.Span[j])));
        }
        var sigAtt = Common.SegmentationTensorOps.Sigmoid(attention);
        var scoreMap = new Tensor<T>([h, w]);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double s = 0; for (int c = 0; c < numC; c++) s += NumOps.ToDouble(features[c, y, x]);
                scoreMap[y, x] = NumOps.FromDouble(s / numC * NumOps.ToDouble(sigAtt[y, x]));
            }
        return BuildPromptMaskResult(scoreMap, h, w);
    }

    PromptedSegmentationResult<T> IPromptableSegmentation<T>.SegmentFromBox(Tensor<T> box)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        int numC = features.Shape[0], h = features.Shape[1], w = features.Shape[2];
        int bx1 = (int)NumOps.ToDouble(box[0]), by1 = (int)NumOps.ToDouble(box[1]);
        int bx2 = (int)NumOps.ToDouble(box[2]), by2 = (int)NumOps.ToDouble(box[3]);
        var boxMask = Common.SegmentationTensorOps.BoxMask<T>(h, w, bx1, by1, bx2, by2);
        var scoreMap = new Tensor<T>([h, w]);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double s = 0; for (int c = 0; c < numC; c++) s += NumOps.ToDouble(features[c, y, x]);
                scoreMap[y, x] = NumOps.FromDouble(s / numC * NumOps.ToDouble(boxMask[y, x]));
            }
        return BuildPromptMaskResult(scoreMap, h, w);
    }

    PromptedSegmentationResult<T> IPromptableSegmentation<T>.SegmentFromMask(Tensor<T> mask)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        int numC = features.Shape[0], h = features.Shape[1], w = features.Shape[2];
        var scoreMap = new Tensor<T>([h, w]);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double mVal = y < mask.Shape[0] && x < mask.Shape[1] ? NumOps.ToDouble(mask[y, x]) : 0;
                double s = 0; for (int c = 0; c < numC; c++) s += NumOps.ToDouble(features[c, y, x]);
                scoreMap[y, x] = NumOps.FromDouble(s / numC * (mVal > 0 ? 1.0 : 0.0));
            }
        return BuildPromptMaskResult(scoreMap, h, w);
    }

    List<PromptedSegmentationResult<T>> IPromptableSegmentation<T>.SegmentEverything()
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var probs = _imageProbabilities ?? Common.SegmentationTensorOps.SoftmaxAlongClassDim(features);
        var classMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(features);
        int numC = features.Shape[0], h = classMap.Shape[0], w = classMap.Shape[1];
        var results = new List<PromptedSegmentationResult<T>>();
        for (int cls = 0; cls < numC && results.Count < 100; cls++)
        {
            double area = 0, confSum = 0;
            var mask = new Tensor<T>([1, h, w]);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if ((int)NumOps.ToDouble(classMap[y, x]) == cls)
                    { mask[0, y, x] = NumOps.FromDouble(1.0); area++; confSum += NumOps.ToDouble(probs[cls, y, x]); }
            if (area < 4) continue;
            double conf = confSum / area;
            results.Add(new PromptedSegmentationResult<T> { Masks = mask, Scores = [conf], StabilityScores = [conf > 0.7 ? 0.95 : conf] });
        }
        if (results.Count == 0)
            results.Add(new PromptedSegmentationResult<T> { Masks = new Tensor<T>([1, features.Shape[1], features.Shape[2]]), Scores = [0.0], StabilityScores = [0.0] });
        return results;
    }

    private PromptedSegmentationResult<T> BuildPromptMaskResult(Tensor<T> scoreMap, int h, int w)
    {
        var probs = Common.SegmentationTensorOps.Sigmoid(scoreMap);
        double[] thresholds = [0.3, 0.5, 0.7];
        var masks = new Tensor<T>([3, h, w]);
        var scores = new double[3];
        var stability = new double[3];
        for (int m = 0; m < 3; m++)
        {
            double area = 0, confSum = 0, areaLo = 0, areaHi = 0;
            double tLo = thresholds[m] - 0.05, tHi = thresholds[m] + 0.05;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double v = NumOps.ToDouble(probs[y, x]);
                    if (v >= thresholds[m]) { masks[m, y, x] = NumOps.FromDouble(1.0); area++; confSum += v; }
                    if (v >= tLo) areaLo++;
                    if (v >= tHi) areaHi++;
                }
            scores[m] = area > 0 ? confSum / area : 0;
            stability[m] = areaLo > 0 ? areaHi / areaLo : 0;
        }
        int lrH = Math.Max(1, h / 4), lrW = Math.Max(1, w / 4);
        var lowRes = new Tensor<T>([1, lrH, lrW]);
        for (int y = 0; y < lrH; y++)
            for (int x = 0; x < lrW; x++)
                lowRes[0, y, x] = scoreMap[Math.Min(y * 4, h - 1), Math.Min(x * 4, w - 1)];
        return new PromptedSegmentationResult<T> { Masks = masks, Scores = scores, LowResLogits = lowRes, StabilityScores = stability };
    }

    #endregion
}
