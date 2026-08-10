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

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// SegGPT: Segmenting Everything In Context via in-context learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In-context segmentation from examples. Few-shot segmentation without fine-tuning.
///
/// Common use cases:
/// - In-context segmentation from examples
/// - Few-shot segmentation without fine-tuning
/// - Interactive image editing
/// - Versatile segmentation across domains
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - In-context learning for segmentation (no fine-tuning needed)
/// - ViT-Large backbone with random color mapping training
/// - Feature ensemble from multiple in-context examples
/// - Unified framework for semantic, instance, and part segmentation
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "SegGPT: Segmenting Everything In Context", ICCV 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SegGPT model for in-context few-shot segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 448, inputWidth: 448, inputDepth: 3, outputSize: 1);
/// var model = new SegGPT&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for example-guided segmentation
/// var onnxModel = new SegGPT&lt;double&gt;(architecture, "seggpt.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SegGPT: Segmenting Everything In Context", "https://arxiv.org/abs/2304.03284", Year = 2023, Authors = "Wang et al.")]
public class SegGPT<T> : Common.PromptableSegmentationBase<T>
{
    private readonly SegGPTOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only SegGPT's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PromptableSegmentationBase -> SegmentationModelBase, as do _imageEmbedding and
    // _imageSet.
    private readonly SegGPTModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal SegGPTModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes SegGPT in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size variant (default: ViTLarge).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SegGPT model.
    /// </para>
    /// </remarks>
    public SegGPT(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        SegGPTModelSize modelSize = SegGPTModelSize.ViTLarge, double dropRate = 0.1,
        SegGPTOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults the loss to CrossEntropyWithLogitsLoss - exactly what the deleted lines did by
        // hand. `optimizer` is passed straight through INCLUDING null; the base's lazy
        // CreateDefaultOptimizer() produces the same `new AdamWOptimizer<...>(this)` default, which
        // could never be written as a base-constructor argument because `this` is unavailable there.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new SegGPTOptions(); Options = _options;
        ApplySegGPTInputFallback(architecture);
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options);
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies SegGPT's 448x448 fallback for architectures that carry no input geometry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SegmentationModelBase falls back to 512x512 when the architecture supplies no input height
    /// or width. SegGPT's documented fallback is 448x448 (the in-context painting resolution the
    /// paper uses), so it is restored here for that unset case only - when the architecture does
    /// specify dimensions, the base's value already matches and nothing changes.
    /// </para>
    /// </remarks>
    private void ApplySegGPTInputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 448;
        if (architecture.InputWidth <= 0) _width = 448;
    }

    /// <summary>
    /// Initializes SegGPT in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size for metadata (default: ViTLarge).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SegGPT from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SegGPT(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1, SegGPTModelSize modelSize = SegGPTModelSize.ViTLarge,
        SegGPTOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new SegGPTOptions(); Options = _options;
        ApplySegGPTInputFallback(architecture);
        _modelSize = modelSize; _dropRate = 0.1;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    // PredictCore is inherited from SegmentationModelBase and dispatches to Forward / PredictOnnx
    // exactly as the deleted override did.

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
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SegGPTOptions options)
    {
        if (options.ChannelDimensions is null || options.ChannelDimensions.Length != 4)
            throw new ArgumentException("ChannelDimensions must contain exactly four positive values.", nameof(options));
        if (options.StageDepths is null || options.StageDepths.Length != 4)
            throw new ArgumentException("StageDepths must contain exactly four positive values.", nameof(options));
        if (options.ChannelDimensions.Any(value => value <= 0))
            throw new ArgumentException("ChannelDimensions must contain only positive values.", nameof(options));
        if (options.StageDepths.Any(value => value <= 0))
            throw new ArgumentException("StageDepths must contain only positive values.", nameof(options));
        if (options.DecoderDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DecoderDimension must be positive.");

        return (options.ChannelDimensions.ToArray(), options.StageDepths.ToArray(), options.DecoderDimension);
    }

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

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

    // AddBatchDimension and RemoveBatchDimension are inherited from SegmentationModelBase.
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
            var encoderLayers = LayerHelper<T>.CreateSegGPTEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateSegGPTDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "SegGPT" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    /// <summary>
    /// Writes configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves model configuration for later reconstruction.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    { writer.Write(_height); writer.Write(_width); writer.Write(_channels); writer.Write(_numClasses); writer.Write((int)_modelSize); writer.Write(_decoderDim); writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty); writer.Write(_encoderLayerEnd); writer.Write(_channelDims.Length); foreach (int d in _channelDims) writer.Write(d); writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d); }

    /// <summary>
    /// Reads configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    { _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32(); int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32(); int dd = reader.ReadInt32(); for (int i = 0; i < dd; i++) _ = reader.ReadInt32(); }

    /// <summary>
    /// Creates a new instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new SegGPT<T>(Architecture, optimizer: null, lossFunction: LossFunction,
            numClasses: _numClasses, modelSize: _modelSize, dropRate: _dropRate,
            options: new SegGPTOptions(_options))
        : new SegGPT<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, new SegGPTOptions(_options));

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // SegGPT owns no further unmanaged resources.
    #endregion

    #region IPromptableSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase.
    // _imageEmbedding, _imageSet, SetImage and the Supports*Prompts flags (point/box/mask true,
    // text false) all come from PromptableSegmentationBase with these exact values; only SegGPT's
    // cached class probabilities are model-specific.
    private Tensor<T>? _imageProbabilities;

    /// <summary>
    /// Encodes an image by running a full forward pass, as SegGPT's explicit SetImage did.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The base's SetImage stores this return value in <c>_imageEmbedding</c> and marks the image
    /// set, so the observable behaviour is unchanged; SetImage below only adds the probability cache.
    /// </para>
    /// </remarks>
    protected override Tensor<T> EncodeImage(Tensor<T> image) => Predict(image);

    /// <inheritdoc/>
    public override void SetImage(Tensor<T> image)
    {
        base.SetImage(image);
        var embedding = _imageEmbedding;
        if (embedding is not null)
        {
            _imageProbabilities = Common.SegmentationTensorOps.SoftmaxAlongClassDim(embedding);
        }
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
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

    /// <inheritdoc/>
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

    /// <inheritdoc/>
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
