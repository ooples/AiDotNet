using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
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
/// var model = new DEVA&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new DEVA&lt;double&gt;(architecture,
///     options: new DEVAOptions { NumClasses = 1, DropRate = 0, ModelSize = DEVAModelSize.Base });
///
/// // Or load a pre-trained ONNX model for video tracking inference
/// var onnxModel = new DEVA&lt;double&gt;(architecture, "deva.onnx");
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
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-5,
                Phase = TrainingPhase.PreTraining,
                Source = "Cheng et al. 2023, Sec. 4: the AdamW optimizer, with a learning rate of 2e-5 "
                        + "for 80,000 iterations during pre-training. The Adam appearing elsewhere in "
                        + "the paper is the author Hartwig Adam in a reference. Routing is handled by "
                        + "SegmentationModelBase.CreateDefaultOptimizer, which consults "
                        + "PaperOptimizerFactory itself, so no call is inserted here.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5,
                Phase = TrainingPhase.FineTuning,
                Source = "Cheng et al. 2023, Sec. 4: main training uses a learning rate of 1e-5 for "
                        + "150,000 iterations.")]
public partial class DEVA<T> : Common.VideoSegmentationBase<T>
{
    private readonly DEVAOptions _options;

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
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable DEVA model.
    /// </para>
    /// </remarks>
    public DEVA(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DEVAOptions? options = null)
        // DEVA's own loss default is preserved verbatim - the base would otherwise substitute plain
        // CrossEntropyWithLogitsLoss, which is wrong for the single-mask (numClasses == 1) case.
        // `optimizer` is passed straight through INCLUDING null; the base's CreateDefaultOptimizer()
        // builds the same `new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this)` this used to inline,
        // but lazily, which is the one thing a base-constructor argument cannot do.
        : base(architecture, optimizer, lossFunction ?? ((options ??= new DEVAOptions()).NumClasses == 1
            ? (ILossFunction<T>)new BinaryCrossEntropyWithLogitsLoss<T>()
            : new CrossEntropyWithLogitsLoss<T>()), (options ??= new DEVAOptions()).NumClasses, maxTrackedObjects: 128)
    {
        _options = options; Options = _options;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize, _options);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes DEVA in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained DEVA from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public DEVA(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DEVAOptions? options = null)
        : base(architecture, onnxModelPath, (options ??= new DEVAOptions()).NumClasses, maxTrackedObjects: 128)
    {
        _options = options; Options = _options;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize, _options);
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

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        // The image-level segmentation network. (An earlier version added 1e-3 times the input's channel
        // mean to the logits; that residual is not in Cheng et al. 2023 and is gone.)
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
        NumOps.ToFloatSpan(input.Data.Span.Slice(0, input.Length), inputData);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        NumOps.FromFloatSpan(outputTensor.ToDenseTensor().Buffer.Span, outputData);
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
        ModelDataProvider = () => SerializeForMetadata()
    };

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
    // DEVA (Cheng et al. 2023) decouples the task: the network above is the image-level segmentation model,
    // and a class-agnostic temporal propagation (XMem-style memory reading) carries the tracked segments from
    // frame to frame. Every MergeEvery frames the two are integrated: propagated and new segments are
    // associated by IoU, matched pairs merge, new segments start new objects, and segments unmatched for
    // MaxMissedMerges integrations are deleted. The frame counter, tracked-id list and initialized flag live
    // on VideoSegmentationBase.
    [Scratch]
    private readonly List<Tensor<T>> _memoryKeys = [];
    [Scratch]
    private readonly List<Tensor<T>> _memoryValues = [];
    private List<int> _objectIds = [];
    private List<int> _missedMerges = [];
    private int _nextObjectId = 1;
    private int _maskHeight;
    private int _maskWidth;
    [Scratch]
    private readonly Dictionary<int, Tensor<T>> _corrections = [];

    /// <summary>
    /// Runs the network once on <paramref name="frame"/>, returning the memory key (the encoder features,
    /// <c>[Ck, h*w]</c>, or the ONNX output) and the image-level logits <c>[K, h, w]</c>.
    /// </summary>
    private (Tensor<T> Key, Tensor<T> Logits, int Height, int Width) EncodeFrame(Tensor<T> frame)
    {
        if (frame is null) throw new ArgumentNullException(nameof(frame));
        var batched = frame.Rank == 4 ? frame : AddBatchDimension(frame);
        using var noGrad = new NoGradScope<T>();
        Tensor<T> key, logits;
        if (_useNativeMode)
        {
            var features = batched;
            for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
            key = features;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
            logits = features;
        }
        else
        {
            logits = PredictOnnx(batched);
            key = logits;
        }

        int ck = key.Shape[1], h = key.Shape[2], w = key.Shape[3];
        var unbatchedLogits = Engine.Reshape(logits, new[] { logits.Shape[1], logits.Shape[2], logits.Shape[3] });
        return (Engine.Reshape(key, new[] { ck, h * w }), unbatchedLogits, h, w);
    }

    /// <inheritdoc/>
    protected override void InitializeTrackingInternal(Tensor<T> frame, Tensor<T> masks, int[] objectIds)
    {
        var objectMasks = Common.VideoTrackingOps.AsObjectMasks(masks, nameof(masks));
        if (objectMasks.Shape[0] != objectIds.Length)
            throw new ArgumentException($"Give one mask per object ({objectIds.Length}).", nameof(masks));
        _maskHeight = objectMasks.Shape[1];
        _maskWidth = objectMasks.Shape[2];
        var (key, _, h, w) = EncodeFrame(frame);
        _memoryKeys.Clear();
        _memoryValues.Clear();
        _corrections.Clear();
        _objectIds = objectIds.ToList();
        _missedMerges = objectIds.Select(_ => 0).ToList();
        _nextObjectId = objectIds.Length == 0 ? 1 : objectIds.Max() + 1;
        AddToMemory(key, objectMasks, h, w);
    }

    /// <summary>Stores a frame's key and its object masks, downsampled to the key grid, in the working memory.</summary>
    private void AddToMemory(Tensor<T> key, Tensor<T> objectMasks, int h, int w)
    {
        int o = objectMasks.Shape[0];
        var values = Common.VideoTrackingOps.Resize(objectMasks, h, w, InterpolateMode.Area);
        _memoryKeys.Add(key);
        _memoryValues.Add(Engine.Reshape(values, new[] { o, h * w }));
        // T_max frames at most; the first frame (the user's annotation) is never dropped.
        if (_memoryKeys.Count > _options.MaxMemoryFrames && _memoryKeys.Count > 1)
        {
            _memoryKeys.RemoveAt(1);
            _memoryValues.RemoveAt(1);
        }
    }

    /// <summary>
    /// XMem memory reading: the affinity of memory pixel <c>i</c> to query pixel <c>j</c> is the negative squared
    /// L2 distance of their keys, <c>(2 k_i . q_j - |k_i|^2) / sqrt(Ck)</c> (the <c>|q_j|^2</c> term cancels in the
    /// softmax); each query pixel reads a softmax over its <see cref="DEVAOptions.MemoryTopK"/> highest affinities.
    /// </summary>
    /// <returns>Object probabilities <c>[O, h*w]</c> read from the memory values.</returns>
    internal static Tensor<T> ReadMemory(Tensor<T> memoryKeys, Tensor<T> memoryValues, Tensor<T> queryKey, int topK)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int ck = memoryKeys.Shape[0], m = memoryKeys.Shape[1], hw = queryKey.Shape[1];
        var dot = engine.TensorMatMul(engine.TensorTranspose(memoryKeys), queryKey);                       // [M, HW]
        var keyNorm = engine.Reshape(engine.ReduceSum(engine.TensorSquare(memoryKeys), new[] { 0 }, false), new[] { m, 1 });
        var affinity = engine.TensorMultiplyScalar(
            engine.TensorSubtract(engine.TensorMultiplyScalar(dot, ops.FromDouble(2.0)), engine.TensorBroadcastTo(keyNorm, new[] { m, hw })),
            ops.FromDouble(1.0 / Math.Sqrt(ck)));
        int k = Math.Min(topK, m);
        var top = engine.TensorTopK(affinity, k, 0, out _);                                                  // [k, HW]
        var kth = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(top), new[] { 0 }, true));       // [1, HW]
        var keep = engine.TensorGreaterEqual(affinity, engine.TensorBroadcastTo(kth, new[] { m, hw }));
        var blocked = Tensor<T>.CreateDefault(new[] { m, hw }, ops.FromDouble(-1e9));
        var weights = engine.TensorSoftmax(engine.TensorWhere(keep, affinity, blocked), 0);
        return engine.TensorMatMul(memoryValues, weights);                                                    // [O, HW]
    }

    /// <summary>
    /// XMem's soft aggregation: the background probability is <c>prod_o (1 - p_o)</c>, and background plus objects
    /// are renormalized with a softmax over their clamped logits <c>log(p / (1 - p))</c>.
    /// </summary>
    /// <returns>Logits <c>[O + 1, h, w]</c>, background first.</returns>
    internal static Tensor<T> AggregateLogits(Tensor<T> probabilities, int h, int w)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int o = probabilities.Shape[0];
        // XMem clamps to [1e-7, 1 - 1e-7]. Half rounds 1 - 1e-7 to 1 (and the logit to +inf), so the bounds are the
        // nearest values T holds strictly inside (0, 1).
        var low = Common.MaskClassificationOps.PositiveTolerance<T>(1e-7);
        var high = Common.MaskClassificationOps.BelowOne<T>(1e-7);
        var clamped = engine.TensorClamp(probabilities, low, high);
        var background = engine.TensorExp(engine.ReduceSum(engine.TensorLog(engine.ScalarMinusTensor(ops.One, clamped)), new[] { 0 }, true));
        var all = engine.TensorClamp(engine.TensorConcatenate(new[] { background, clamped }, 0), low, high);
        var logits = engine.TensorLog(engine.TensorDivide(all, engine.ScalarMinusTensor(ops.One, all)));
        return engine.Reshape(logits, new[] { o + 1, h, w });
    }

    /// <inheritdoc/>
    protected override VideoSegmentationResult<T> PropagateToFrameInternal(Tensor<T> frame, int frameIndex)
    {
        var (key, logits, h, w) = EncodeFrame(frame);
        int hm = _maskHeight, wm = _maskWidth;
        int o = _objectIds.Count;
        using var noGrad = new NoGradScope<T>();

        // Temporal propagation: memory reading, soft aggregation, upsampling to the mask grid.
        Tensor<T> probabilities;
        if (o > 0)
        {
            var read = ReadMemory(
                Engine.TensorConcatenate(_memoryKeys.ToArray(), 1), Engine.TensorConcatenate(_memoryValues.ToArray(), 1),
                key, _options.MemoryTopK);
            var aggregated = Common.VideoTrackingOps.Resize(AggregateLogits(read, h, w), hm, wm, InterpolateMode.Bilinear);
            var softmax = Engine.TensorSoftmax(aggregated, 0);
            probabilities = Engine.TensorSlice(softmax, new[] { 1, 0, 0 }, new[] { o, hm, wm });
        }
        else
        {
            probabilities = new Tensor<T>(new[] { 0, hm, wm });
        }

        bool merge = _options.MergeEvery > 0 && frameIndex % _options.MergeEvery == 0;
        bool corrected = ApplyCorrections(ref probabilities);
        if (merge)
        {
            probabilities = Integrate(probabilities, logits);
        }

        if (merge || corrected)
        {
            RemapMemory();
            AddToMemory(key, probabilities, h, w);
        }

        _trackedObjectIds = [.. _objectIds];
        var (confidences, visible) = Common.VideoTrackingOps.Statistics(probabilities);
        return new VideoSegmentationResult<T>
        {
            Masks = probabilities, ObjectIds = _objectIds.ToArray(), Confidences = confidences,
            FrameIndex = frameIndex, IsVisible = visible
        };
    }

    /// <summary>Replaces the masks of corrected objects with their correction masks; true when any was applied.</summary>
    private bool ApplyCorrections(ref Tensor<T> probabilities)
    {
        if (_corrections.Count == 0) return false;
        int o = _objectIds.Count, hm = _maskHeight, wm = _maskWidth;
        var rows = new Tensor<T>[o];
        bool applied = false;
        for (int i = 0; i < o; i++)
        {
            if (_corrections.TryGetValue(_objectIds[i], out var correction))
            {
                if (correction.Rank != 2 || correction.Shape[0] != hm || correction.Shape[1] != wm)
                    throw new ArgumentException($"A correction mask must be [{hm}, {wm}].");
                rows[i] = Engine.Reshape(correction, new[] { 1, hm, wm });
                applied = true;
            }
            else
            {
                rows[i] = Engine.TensorSlice(probabilities, new[] { i, 0, 0 }, new[] { 1, hm, wm });
            }
        }

        _corrections.Clear();
        if (applied) probabilities = Engine.TensorConcatenate(rows, 0);
        return applied;
    }

    // Row of each current object in the memory values: its previous row, or -1 for a new object.
    private int[] _memoryRows = Array.Empty<int>();

    /// <summary>Rewrites every memory frame's values for the current object list (new objects read zeros).</summary>
    private void RemapMemory()
    {
        if (_memoryRows.Length == 0 && _memoryValues.Count > 0 && _memoryValues[0].Shape[0] == _objectIds.Count) return;
        int o = _objectIds.Count;
        for (int f = 0; f < _memoryValues.Count; f++)
        {
            var values = _memoryValues[f];
            int m = values.Shape[1];
            var rows = new Tensor<T>[o];
            for (int i = 0; i < o; i++)
            {
                rows[i] = i < _memoryRows.Length && _memoryRows[i] >= 0
                    ? Engine.TensorSlice(values, new[] { _memoryRows[i], 0 }, new[] { 1, m })
                    : new Tensor<T>(new[] { 1, m });
            }

            _memoryValues[f] = o == 0 ? new Tensor<T>(new[] { 0, m }) : Engine.TensorConcatenate(rows, 0);
        }

        _memoryRows = Array.Empty<int>();
    }

    /// <summary>
    /// Integrates the propagated segments with the image-level segmentation of this frame (Cheng et al. 2023,
    /// Sec. 3.3). Propagated hard masks (argmax over background and objects) and new segments (the classes of
    /// the image-level argmax, background class 0 excluded; the sigmoid foreground when there is one class)
    /// are associated where their IoU exceeds <see cref="DEVAOptions.AssociationIoUThreshold"/>; matched pairs
    /// become their union, unmatched new segments start new objects, and an object unmatched for
    /// <see cref="DEVAOptions.MaxMissedMerges"/> integrations in a row is deleted. Overlaps are resolved in
    /// favour of smaller segments (rendered by descending area).
    /// </summary>
    /// <returns>The integrated hard masks <c>[O', H, W]</c>.</returns>
    private Tensor<T> Integrate(Tensor<T> probabilities, Tensor<T> logits)
    {
        int hm = _maskHeight, wm = _maskWidth, plane = hm * wm, o = _objectIds.Count, k = logits.Shape[0];
        var propagated = HardMasks(probabilities, o, plane);                                              // [O, HW] int
        var proposals = ProposalMasks(Common.VideoTrackingOps.Resize(logits, hm, wm, InterpolateMode.Bilinear), k, plane);
        int p = proposals.Shape[0];

        var matchOfObject = new int[o];
        var proposalMatched = new bool[p];
        for (int i = 0; i < o; i++) matchOfObject[i] = -1;
        if (o > 0 && p > 0)
        {
            int[] matches = Engine.TensorGreaterThan(
                Iou(propagated, proposals), NumOps.FromDouble(_options.AssociationIoUThreshold)).Cast<int>().ToArray();
            for (int i = 0; i < o; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    if (matches[i * p + j] == 0 || proposalMatched[j] || matchOfObject[i] >= 0) continue;
                    matchOfObject[i] = j;
                    proposalMatched[j] = true;
                }
            }
        }

        var segments = new List<Tensor<int>>();
        var ids = new List<int>();
        var missed = new List<int>();
        var memoryRows = new List<int>();
        for (int i = 0; i < o; i++)
        {
            var mask = Engine.Reshape(Engine.TensorSlice(propagated, new[] { i, 0 }, new[] { 1, plane }), new[] { plane });
            int misses = 0;
            if (matchOfObject[i] >= 0)
            {
                var proposal = Engine.Reshape(Engine.TensorSlice(proposals, new[] { matchOfObject[i], 0 }, new[] { 1, plane }), new[] { plane });
                mask = Engine.TensorSubtract(Engine.TensorAdd(mask, proposal), Engine.TensorMultiply(mask, proposal));
            }
            else
            {
                misses = _missedMerges[i] + 1;
                if (misses >= _options.MaxMissedMerges) continue;
            }

            segments.Add(mask);
            ids.Add(_objectIds[i]);
            missed.Add(misses);
            memoryRows.Add(i);
        }

        for (int j = 0; j < p; j++)
        {
            if (proposalMatched[j] || ids.Count >= _maxTrackedObjects) continue;
            segments.Add(Engine.Reshape(Engine.TensorSlice(proposals, new[] { j, 0 }, new[] { 1, plane }), new[] { plane }));
            ids.Add(_nextObjectId++);
            missed.Add(0);
            memoryRows.Add(-1);
        }

        _objectIds = ids;
        _missedMerges = missed;
        _memoryRows = memoryRows.ToArray();
        if (segments.Count == 0) return new Tensor<T>(new[] { 0, hm, wm });

        // Smaller segments win overlaps: paint in descending area order.
        var stacked = Engine.TensorStack(segments.ToArray(), 0);                                           // [S, HW]
        int[] order = Common.MaskClassificationOps.DescendingOrder(Engine.ReduceSum(stacked, new[] { 1 }, false));
        var owner = new Tensor<int>(new[] { plane });
        foreach (int s in order)
        {
            var mask = Engine.Reshape(Engine.TensorSlice(stacked, new[] { s, 0 }, new[] { 1, plane }), new[] { plane });
            owner = Engine.TensorAdd(Engine.TensorMultiply(owner, Engine.ScalarMinusTensor(1, mask)), Engine.TensorMultiplyScalar(mask, s + 1));
        }

        int count = segments.Count;
        var oneHot = Engine.TensorPermute(Engine.TensorOneHot<T>(owner, count + 1), new[] { 1, 0 });
        return Engine.Reshape(Engine.TensorSlice(oneHot, new[] { 1, 0 }, new[] { count, plane }), new[] { count, hm, wm });
    }

    /// <summary>Hard object masks <c>[O, HW]</c> (int): each pixel goes to the argmax of background and objects.</summary>
    private Tensor<int> HardMasks(Tensor<T> probabilities, int o, int plane)
    {
        if (o == 0) return new Tensor<int>(new[] { 0, plane });
        var flat = Engine.Reshape(probabilities, new[] { o, plane });
        var background = Engine.ScalarMinusTensor(NumOps.One, Engine.ReduceSum(flat, new[] { 0 }, true));
        var owner = Engine.TensorArgMax(Engine.TensorConcatenate(new[] { background, flat }, 0), 0);     // [HW]
        var oneHot = Engine.TensorPermute(Engine.TensorOneHot<int>(owner, o + 1), new[] { 1, 0 });
        return Engine.TensorSlice(oneHot, new[] { 1, 0 }, new[] { o, plane });
    }

    /// <summary>The image-level segments <c>[P, HW]</c> (int) of upsampled logits <c>[K, H, W]</c>.</summary>
    private Tensor<int> ProposalMasks(Tensor<T> logits, int k, int plane)
    {
        var flat = Engine.Reshape(logits, new[] { k, plane });
        Tensor<int> candidates;
        if (k == 1)
        {
            candidates = Engine.TensorGreaterThan(flat, NumOps.Zero).Cast<int>();
        }
        else
        {
            var oneHot = Engine.TensorPermute(Engine.TensorOneHot<int>(Engine.TensorArgMax(flat, 0), k), new[] { 1, 0 });
            candidates = Engine.TensorSlice(oneHot, new[] { 1, 0 }, new[] { k - 1, plane });
        }

        int[] areas = Engine.ReduceSum(candidates, new[] { 1 }, false).ToArray();
        var present = new List<int>();
        for (int c = 0; c < areas.Length; c++)
        {
            if (areas[c] > 0) present.Add(c);
        }

        return present.Count == 0
            ? new Tensor<int>(new[] { 0, plane })
            : Engine.TensorIndexSelect(candidates, new Tensor<int>(present.ToArray(), new[] { present.Count }), 0);
    }

    /// <summary>IoU matrix <c>[A, B]</c> of binary masks <c>[A, HW]</c> and <c>[B, HW]</c> (int).</summary>
    /// <remarks>
    /// Intersections and areas are counted in int and converted to T once for the division: as a matrix product or
    /// a sum in T, Half stops counting at 2048, so two 3000-pixel masks sharing 3000 pixels would score 2048 / 2048
    /// against the wrong union.
    /// </remarks>
    internal static Tensor<T> Iou(Tensor<int> a, Tensor<int> b)
    {
        var engine = AiDotNetEngine.Current;
        int na = a.Shape[0], nb = b.Shape[0];
        var intersection = engine.TensorMatMul(a, engine.TensorTranspose(b));
        var areaA = engine.TensorBroadcastTo(engine.ReduceSum(a, new[] { 1 }, true), new[] { na, nb });
        var areaB = engine.TensorBroadcastTo(engine.Reshape(engine.ReduceSum(b, new[] { 1 }, false), new[] { 1, nb }), new[] { na, nb });
        var union = engine.TensorClampMin(engine.TensorSubtract(engine.TensorAdd(areaA, areaB), intersection), 1);
        return engine.TensorDivide(intersection.Cast<T>(), union.Cast<T>());
    }

    /// <inheritdoc/>
    public override void AddCorrection(int objectId, Tensor<T> correctionMask)
    {
        if (correctionMask is null) throw new ArgumentNullException(nameof(correctionMask));
        _corrections[objectId] = correctionMask;
    }

    /// <inheritdoc/>
    protected override void ResetTrackingInternal()
    {
        _memoryKeys.Clear();
        _memoryValues.Clear();
        _objectIds = [];
        _missedMerges = [];
        _memoryRows = Array.Empty<int>();
        _nextObjectId = 1;
        _corrections.Clear();
    }
    #endregion
}
