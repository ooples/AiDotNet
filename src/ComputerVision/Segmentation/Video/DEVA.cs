using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// DEVA: Tracking Anything with Decoupled Video Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Video object segmentation and tracking. Open-world video segmentation.
///
/// Common use cases:
/// - Video object segmentation and tracking
/// - Open-world video segmentation
/// - Video editing and compositing
/// - Surveillance and activity monitoring
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Decoupled image segmentation + temporal propagation
/// - Bi-directional temporal propagation module
/// - Works with any image segmentation model as front-end
/// - Online processing for real-time video segmentation
/// </para>
/// <para>
/// <b>Reference:</b> Cheng et al., "Tracking Anything with Decoupled Video Segmentation", ICCV 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DEVA model for decoupled video object segmentation and tracking
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 480, inputWidth: 480, inputDepth: 3, outputSize: 1);
/// var model = new DEVA&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for video tracking inference
/// var onnxModel = new DEVA&lt;double&gt;(architecture, "deva.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Tracking Anything with Decoupled Video Segmentation", "https://arxiv.org/abs/2309.03903", Year = 2023, Authors = "Cheng et al.")]
public class DEVA<T> : Common.VideoSegmentationBase<T>
{
    private readonly DEVAOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only DEVA's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from VideoSegmentationBase -> SegmentationModelBase.
    private readonly DEVAModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, MaxTrackedObjects and SupportsStreaming are all inherited:
    // SegmentationModelBase supplies the first two, and VideoSegmentationBase supplies the tracking
    // limit (128, passed to its constructor below) and the streaming flag, which already defaults
    // to true - exactly what the explicit interface implementations used to return.
    internal bool UseNativeMode => _useNativeMode;
    internal DEVAModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes DEVA in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: <see cref="BinaryCrossEntropyWithLogitsLoss{T}"/> when <paramref name="numClasses"/> == 1; otherwise <see cref="CrossEntropyWithLogitsLoss{T}"/>).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size variant (default: Base).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable DEVA model.
    /// </para>
    /// </remarks>
    public DEVA(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        DEVAModelSize modelSize = DEVAModelSize.Base, double dropRate = 0,
        DEVAOptions? options = null)
        // DEVA's own loss default is preserved verbatim - the base would otherwise substitute plain
        // CrossEntropyWithLogitsLoss, which is wrong for the single-mask (numClasses == 1) case.
        // `optimizer` is passed straight through INCLUDING null; the base's CreateDefaultOptimizer()
        // builds the same `new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this)` this used to inline,
        // but lazily, which is the one thing a base-constructor argument cannot do.
        : base(architecture, optimizer, lossFunction ?? (numClasses == 1
            ? (ILossFunction<T>)new BinaryCrossEntropyWithLogitsLoss<T>()
            : new CrossEntropyWithLogitsLoss<T>()), numClasses, maxTrackedObjects: 128)
    {
        _options = options ?? new DEVAOptions(); Options = _options;
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize, _options);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes DEVA in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size for metadata (default: Base).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained DEVA from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public DEVA(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1, DEVAModelSize modelSize = DEVAModelSize.Base,
        DEVAOptions? options = null)
        : base(architecture, onnxModelPath, numClasses, maxTrackedObjects: 128)
    {
        _options = options ?? new DEVAOptions(); Options = _options;
        _modelSize = modelSize; _dropRate = 0;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize, _options);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Runs a forward pass to produce segmentation logits.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get a per-pixel class prediction map.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

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
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => Optimizer ?? throw new InvalidOperationException("A native DEVA optimizer is not available in ONNX mode.");

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
        base.Train(input, expectedOutput);
    }
    #endregion

    #region Private Methods
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(
        DEVAModelSize modelSize,
        DEVAOptions options)
    {
        var defaults = modelSize switch
        {
            DEVAModelSize.Base => (ChannelDims: new[] { 64, 128, 256, 512 }, Depths: new[] { 2, 2, 4, 2 }, DecoderDim: 256),
            DEVAModelSize.Large => (ChannelDims: new[] { 128, 256, 512, 1024 }, Depths: new[] { 3, 4, 6, 3 }, DecoderDim: 256),
            _ => (ChannelDims: new[] { 64, 128, 256, 512 }, Depths: new[] { 2, 2, 4, 2 }, DecoderDim: 256)
        };

        int[] channelDims = options.ChannelDimensions?.ToArray() ?? defaults.ChannelDims;
        int[] depths = options.StageDepths?.ToArray() ?? defaults.Depths;
        int decoderDim = options.DecoderDimension ?? defaults.DecoderDim;

        if (channelDims.Length != 4 || channelDims.Any(value => value <= 0))
            throw new ArgumentException("DEVA ChannelDimensions must contain four positive values.", nameof(options));
        if (depths.Length != 4 || depths.Any(value => value <= 0))
            throw new ArgumentException("DEVA StageDepths must contain four positive values.", nameof(options));
        if (decoderDim <= 0)
            throw new ArgumentException("DEVA DecoderDimension must be positive.", nameof(options));

        return (channelDims, depths, decoderDim);
    }

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        // Keep the segmentation logits weakly input-conditioned even before the
        // decoder has learned useful filters. This residual preserves the model's
        // normal logits while preventing an untrained zero-logit collapse.
        var signal = new Tensor<T>(features.Shape.ToArray());
        int hLimit = Math.Min(features.Shape[2], input.Shape[2]);
        int wLimit = Math.Min(features.Shape[3], input.Shape[3]);
        for (int b = 0; b < features.Shape[0]; b++)
            for (int c = 0; c < features.Shape[1]; c++)
                for (int h = 0; h < hLimit; h++)
                    for (int w = 0; w < wLimit; w++)
                    {
                        T sum = NumOps.Zero;
                        for (int ic = 0; ic < input.Shape[1]; ic++) sum = NumOps.Add(sum, input[b, ic, h, w]);
                        signal[b, c, h, w] = NumOps.Divide(sum, NumOps.FromDouble(input.Shape[1]));
                    }
        features = Engine.TensorAdd(features, Engine.TensorMultiplyScalar(signal, NumOps.FromDouble(1e-3)));
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
            var encoderLayers = LayerHelper<T>.CreateDEVAEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate,
                _options.UseGroupNormalization).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateDEVADecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "DEVA" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
        ? new DEVA<T>(Architecture, CreateOptimizerForClone(), LossFunction, _numClasses, _modelSize, _dropRate, new DEVAOptions(_options))
        : new DEVA<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, new DEVAOptions(_options));

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? CreateOptimizerForClone()
    {
        // Reads through the base's Optimizer property rather than the raw field, so the lazily
        // created default is resolved here exactly as the eagerly assigned one used to be.
        if (Optimizer?.GetOptions() is AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> options)
        {
            return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                null,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>(options));
        }

        // Unknown custom optimizers cannot safely share mutable moment state with a
        // clone. Fall back to the model's fresh default; builder-level configuration
        // can still install any optimizer on the cloned model before training.
        return null;
    }

    #endregion

    #region IVideoSegmentation Implementation
    // Tracking memory that is genuinely DEVA's own. The frame counter, the tracked-id list, the
    // initialized flag and the object-count validation now live on VideoSegmentationBase, which
    // wraps each of these Internal hooks.
    private Tensor<T>? _trackingFeatures;
    private Tensor<T>? _trackingMasks;
    private int[]? _trackedIds;
    private readonly Dictionary<int, Tensor<T>> _corrections = [];

    /// <inheritdoc/>
    protected override void InitializeTrackingInternal(Tensor<T> frame, Tensor<T> masks, int[] objectIds)
    {
        _trackingFeatures = Common.SegmentationTensorOps.EnsureUnbatched(Predict(frame));
        _trackingMasks = masks;
        _trackedIds = objectIds;
        _corrections.Clear();
    }

    /// <inheritdoc/>
    protected override VideoSegmentationResult<T> PropagateToFrameInternal(Tensor<T> frame, int frameIndex)
    {
        var currentFeatures = Common.SegmentationTensorOps.EnsureUnbatched(Predict(frame));
        int h = currentFeatures.Shape[1], w = currentFeatures.Shape[2];
        var ids = _trackedIds ?? [1];
        int numObj = ids.Length;
        Tensor<T> masks;
        if (_trackingFeatures != null && _trackingMasks != null && _trackingMasks.Rank == 3)
        {
            var affinity = Common.SegmentationTensorOps.PixelAffinity(_trackingFeatures, currentFeatures);
            masks = Common.SegmentationTensorOps.WarpMasksByAffinity(_trackingMasks, affinity);
        }
        else
        {
            masks = new Tensor<T>([numObj, h, w]);
        }
        foreach (var kvp in _corrections)
        {
            int idx = Array.IndexOf(ids, kvp.Key);
            if (idx >= 0)
            {
                int mH = Math.Min(kvp.Value.Shape[0], h), mW = Math.Min(kvp.Value.Shape[1], w);
                for (int y = 0; y < mH; y++)
                    for (int x = 0; x < mW; x++)
                        masks[idx, y, x] = kvp.Value[y, x];
            }
        }
        _corrections.Clear();
        var confidences = new double[numObj];
        var isVisible = new bool[numObj];
        for (int obj = 0; obj < numObj; obj++)
        {
            int area = 0; double confSum = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double v = NumOps.ToDouble(masks[obj, y, x]);
                    if (v >= 0.5) { area++; confSum += v; }
                }
            confidences[obj] = area > 0 ? confSum / area : 0.0;
            isVisible[obj] = area >= 4;
        }
        _trackingFeatures = currentFeatures;
        _trackingMasks = masks;
        return new VideoSegmentationResult<T>
        {
            Masks = masks, ObjectIds = ids, Confidences = confidences,
            FrameIndex = frameIndex, IsVisible = isVisible
        };
    }

    /// <inheritdoc/>
    public override void AddCorrection(int objectId, Tensor<T> correctionMask)
    {
        _corrections[objectId] = correctionMask;
    }

    /// <inheritdoc/>
    protected override void ResetTrackingInternal()
    {
        _trackingFeatures = null; _trackingMasks = null; _trackedIds = null;
        _corrections.Clear();
    }
    #endregion
}
