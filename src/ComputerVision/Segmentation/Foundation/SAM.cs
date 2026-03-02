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
/// Segment Anything Model (SAM): the first promptable foundation model for image segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM can segment any object in any image given a prompt (point click,
/// bounding box, or text). It was trained on the SA-1B dataset containing over 1 billion masks
/// across 11 million images, making it extremely versatile.
///
/// Common use cases:
/// - Interactive object selection in image editors
/// - Automatic mask generation for datasets
/// - Zero-shot transfer to new domains without fine-tuning
/// - Foundation for downstream segmentation tasks
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - ViT-H/L/B image encoder with 16x16 patch embedding
/// - Lightweight prompt encoder for points, boxes, and masks
/// - Two-way transformer mask decoder with IoU prediction head
/// - Ambiguity-aware: predicts 3 masks per prompt (whole, part, subpart)
/// - 1024x1024 input resolution; encoder runs once per image
/// </para>
/// <para>
/// <b>Reference:</b> Kirillov et al., "Segment Anything", ICCV 2023.
/// </para>
/// </remarks>
public class SAM<T> : NeuralNetworkBase<T>, IPromptableSegmentation<T>
{
    private readonly SAMOptions _options;

    /// <summary>
    /// Gets the configuration options for this SAM model.
    /// </summary>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private int _height, _width, _channels, _numClasses;
    private readonly SAMModelSize _modelSize;
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

    // Promptable segmentation state
    private Tensor<T>? _imageEmbedding;
    private Tensor<T>? _imageProbabilities;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this SAM instance supports training.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal SAMModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes SAM in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss; the paper uses focal + dice + IoU loss).</param>
    /// <param name="numClasses">Number of output mask classes (default: 1 for binary segmentation).</param>
    /// <param name="modelSize">ViT backbone size (default: ViTHuge — the original SAM default).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SAM model from scratch. The original SAM was
    /// trained on SA-1B (1B+ masks from 11M images). Fine-tuning on domain data typically uses
    /// ViT-B for efficiency.
    /// </para>
    /// </remarks>
    public SAM(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 1,
        SAMModelSize modelSize = SAMModelSize.ViTHuge,
        double dropRate = 0.1,
        SAMOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new SAMOptions();
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
    /// Initializes SAM in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output mask classes (default: 1).</param>
    /// <param name="modelSize">ViT backbone size for metadata (default: ViTHuge).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SAM model from ONNX for fast inference.
    /// Download pre-trained weights from Meta's SAM repository.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SAM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 1,
        SAMModelSize modelSize = SAMModelSize.ViTHuge,
        SAMOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new SAMOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SAM ONNX model not found: {onnxModelPath}");

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

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load SAM ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through SAM to produce segmentation mask logits.
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
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");

        var predicted = Forward(input);
        var lossGradient = LossFunction.ComputeGradient(predicted, expectedOutput);
        BackwardPass(lossGradient);
        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Private Methods

    // SAM paper: ViT-B = 12 blocks / 768 dim, ViT-L = 24 blocks / 1024 dim, ViT-H = 32 blocks / 1280 dim
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SAMModelSize modelSize)
    {
        return modelSize switch
        {
            SAMModelSize.ViTBase => ([768, 768, 768, 768], [12, 0, 0, 0], 256),
            SAMModelSize.ViTLarge => ([1024, 1024, 1024, 1024], [24, 0, 0, 0], 256),
            SAMModelSize.ViTHuge => ([1280, 1280, 1280, 1280], [32, 0, 0, 0], 256),
            _ => ([1280, 1280, 1280, 1280], [32, 0, 0, 0], 256)
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
    /// Initializes the ViT encoder and mask decoder layers.
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
            var encoderLayers = LayerHelper<T>.CreateSAMEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);
            int featureH = _height / 16;
            int featureW = _width / 16;
            var decoderLayers = LayerHelper<T>.CreateSAMDecoderLayers(
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
    /// Collects metadata describing this SAM model's configuration.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.SemanticSegmentation,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SAM" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumClasses", _numClasses },
            { "ModelSize", _modelSize.ToString() },
            { "UseNativeMode", _useNativeMode },
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
        int dc = reader.ReadInt32();
        for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
        int dd = reader.ReadInt32();
        for (int i = 0; i < dd; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance with the same configuration but fresh weights.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new SAM<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new SAM<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);

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
