using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// RepViT-SAM: Real-time SAM with RepViT backbone.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Real-time segment anything. Mobile and edge SAM inference.
///
/// Common use cases:
/// - Real-time segment anything
/// - Mobile and edge SAM inference
/// - Interactive segmentation on devices
/// - Low-latency promptable segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - RepViT (Re-parameterized Vision Transformer) backbone
/// - Structural re-parameterization for inference speed
/// - Multi-scale feature extraction for SAM decoder
/// - Achieves real-time SAM on mobile devices
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "RepViT-SAM: Towards Real-Time Segmenting Anything", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a RepViT-SAM model for real-time promptable segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new RepViTSAM&lt;double&gt;(architecture, numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RepViT-SAM: Towards Real-Time Segmenting Anything", "https://arxiv.org/abs/2312.05760", Year = 2024, Authors = "Ao Wang, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding")]
public partial class RepViTSAM<T> : Common.PromptableSegmentationBase<T>
{
    private readonly RepViTSAMOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only RepViT-SAM's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed, _encoderLayerEnd and
    // _imageEmbedding all come from PromptableSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes RepViTSAM in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable RepViTSAM model.
    /// </para>
    /// </remarks>
    public RepViTSAM(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        RepViTSAMOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new RepViTSAMOptions(); Options = _options;
        ApplySamDefaultGeometry(architecture);
        _dropRate = dropRate;
        _channelDims = [48, 96, 192, 384];
        _depths = [3, 4, 16, 3];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// RepViTSAM's default training recipe, taken from its options rather than the base's generic one.
    /// </summary>
    /// <remarks>
    /// Expressed by overriding the rate the base already exposes for this purpose -- "derived paper
    /// implementations override this instead of reimplementing optimizer plumbing" -- so a caller
    /// who passes their own optimizer still wins, and one who only wants a different rate can set
    /// it on the options.
    /// </remarks>
    protected override double DefaultLearningRate => _options.LearningRate;

    /// <inheritdoc cref="DefaultLearningRate"/>
    protected override double DefaultWeightDecay => _options.WeightDecay;

    /// <summary>
    /// Initializes RepViTSAM in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained RepViTSAM from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public RepViTSAM(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        RepViTSAMOptions? options = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new RepViTSAMOptions(); Options = _options;
        ApplySamDefaultGeometry(architecture);
        _dropRate = 0;
        _channelDims = [48, 96, 192, 384];
        _depths = [3, 4, 16, 3];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Restores RepViT-SAM's own 1024x1024 fallback for unspecified input geometry.
    /// </summary>
    /// <remarks>
    /// SegmentationModelBase falls back to 512x512 when the architecture leaves the input size
    /// unset; every SAM variant has always fallen back to SAM's native 1024x1024 instead, so that
    /// stays the model's own rule rather than becoming the shared default.
    /// </remarks>
    private void ApplySamDefaultGeometry(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains the model. Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    #endregion

    #region Private Methods
    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    // AddBatchDimension / RemoveBatchDimension are inherited from SegmentationModelBase.
    #endregion

    #region Abstract Implementation
    /// <summary>
    /// Initializes the encoder and decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the neural network layers.
    /// In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Architecture.Layers.Count / 2; }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateRepViTSAMEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateRepViTSAMDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "RepViTSAM" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    /// <summary>
    /// Releases managed resources including the ONNX inference session.
    /// </summary>
    /// <param name="disposing">True when called from Dispose().</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees memory used by the ONNX runtime.
    /// </para>
    /// </remarks>
    // Dispose of the ONNX session and the _disposed latch are handled by SegmentationModelBase.
    #endregion

    #region IPromptableSegmentation Implementation
    // NumClasses / InputHeight / InputWidth / IsOnnxMode / Segment and the four Supports*Prompts
    // flags all arrive from PromptableSegmentationBase with identical values.
    [Scratch]
    private Tensor<T>? _imageProbabilities;

    /// <inheritdoc />
    protected override Tensor<T> EncodeImage(Tensor<T> image) => Predict(image);

    /// <inheritdoc />
    public override void SetImage(Tensor<T> image)
    {
        base.SetImage(image);
        var embedding = _imageEmbedding;
        if (embedding is not null)
        {
            _imageProbabilities = Common.SegmentationTensorOps.SoftmaxAlongClassDim(embedding);
        }
    }

    /// <inheritdoc />
    public override PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        int numC = features.Shape[0], h = features.Shape[1], w = features.Shape[2];
        var attention = new Tensor<T>([h, w]);
        int numPts = points.Shape[0];
        double sigma = Math.Max(h, w) / 10.0;
        for (int i = 0; i < numPts; i++)
        {
            double px = NumOps.ToDouble(points[i, 0]), py = NumOps.ToDouble(points[i, 1]);
            double sign = NumOps.Compare(labels[i], NumOps.One) == 0 ? 1.0 : -1.0;
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

    /// <inheritdoc />
    public override PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box)
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

    /// <inheritdoc />
    public override PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask)
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

    /// <inheritdoc />
    public override List<PromptedSegmentationResult<T>> SegmentEverything()
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
