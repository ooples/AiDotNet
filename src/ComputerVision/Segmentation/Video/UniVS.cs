using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// UniVS: Unified and Universal Video Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Unified video segmentation across tasks. Video instance and panoptic segmentation.
///
/// Common use cases:
/// - Unified video segmentation across tasks
/// - Video instance and panoptic segmentation
/// - Referring video object segmentation
/// - Multi-task video understanding
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Prompts-as-queries unifying VIS, VPS, and VOS
/// - Target-specific prompt generation
/// - Per-object prompt memory: predicted masks are fed back as visual prompts
/// - Handles both automatic and prompted segmentation
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "UniVS: Unified and Universal Video Segmentation with Prompts as Queries", CVPR 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a UniVS model for unified multi-task video segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 480, inputWidth: 480, inputDepth: 3, outputSize: 80);
/// var model = new UniVS&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new UniVS&lt;double&gt;(architecture,
///     options: new UniVSOptions { NumClasses = 80, DropRate = 0.1, ModelSize = UniVSModelSize.R50 });
///
/// // Or load a pre-trained ONNX model for video instance/panoptic segmentation
/// var onnxModel = new UniVS&lt;double&gt;(architecture, "univs.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("UniVS: Unified and Universal Video Segmentation with Prompts as Queries", "https://arxiv.org/abs/2402.18115", Year = 2024, Authors = "Li et al.")]
public partial class UniVS<T> : Common.VideoSegmentationBase<T>
{
    private readonly UniVSOptions _options;

    #region Fields
    // Only UniVS's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from VideoSegmentationBase -> SegmentationModelBase. _optimizer in particular still holds
    // EXACTLY what the caller passed (the base never defaults it eagerly), which is what
    // GetOrCreateBaseOptimizer below relies on.
    private readonly UniVSModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    private readonly bool _hasUserSuppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _baseTapeOptimizer;
    private bool _lazyShapesWarmed;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, MaxTrackedObjects and SupportsStreaming are all inherited:
    // SegmentationModelBase supplies the first two, and VideoSegmentationBase supplies the tracking
    // limit (256, passed to its constructor below) and the streaming flag, which already defaults
    // to true - exactly what the explicit interface implementations used to return.
    internal bool UseNativeMode => _useNativeMode;
    internal UniVSModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes UniVS in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable UniVS model.
    /// </para>
    /// </remarks>
    public UniVS(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        UniVSOptions? options = null)
        // `optimizer` is passed straight through INCLUDING null, exactly as before: UniVS must NOT
        // get a defaulted AdamW, because GetOrCreateBaseOptimizer below installs its own tuned
        // warmup Adam whenever the caller supplied nothing. The base stores the argument verbatim
        // and only ever defaults it lazily through the Optimizer property, which UniVS never reads.
        : base(architecture, optimizer, lossFunction, (options ??= new UniVSOptions()).NumClasses, maxTrackedObjects: 256)
    {
        _options = options; Options = _options;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        _hasUserSuppliedOptimizer = optimizer is not null;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        InitializeLayers();
        // Materialize every lazy conv/BN layer up front (one real eval-mode forward). DeepCopy /
        // Clone serialize the trained weights then rebuild a fresh instance and apply them via
        // SetParameters — which silently skips any layer still in its lazy, shape-unresolved state,
        // dropping those weights (Clone_AfterTraining, #1221 class). Resolving here guarantees the
        // freshly-constructed clone target already has concrete-shaped layers to receive them.
        ResolveLazyLayerShapes();
    }

    /// <summary>
    /// Initializes UniVS in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained UniVS from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public UniVS(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        UniVSOptions? options = null)
        : base(architecture, onnxModelPath, (options ??= new UniVSOptions()).NumClasses, maxTrackedObjects: 256)
    {
        _options = options; Options = _options;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!_useNativeMode) return PredictOnnx(input);
        // Ensure inference runs in eval mode so any per-layer inference-weight cache is
        // (re)synced to the latest trained weights. Without this a Predict issued right after a
        // Train step can serve stale cached packs — the trained model then predicts differently
        // from a freshly-deserialized clone that re-packs from current weights
        // (Clone_AfterTraining, #1221 class).
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Supplies the training optimizer. Honors a constructor-supplied optimizer; otherwise
    /// uses a linear-warmup Adam at 1e-3 rather than the framework-default rate.
    /// </summary>
    /// <remarks>
    /// The from-scratch ResNet-50 backbone + Mask2Former-style decoder is a deep encoder-decoder
    /// whose per-pixel cross-entropy does NOT descend under a plain high-rate step — a loss probe
    /// showed the CE loss holding / rising (10.95 -> 11.11) with the default optimizer, i.e. the
    /// deep stack overshoots. A small linear warmup into 1e-3 gives a stable monotonic descent
    /// (same fix as ODISE's from-scratch SD-U-Net). Without this override the constructor-built
    /// optimizer was never consulted at all (GetOrCreateBaseOptimizer fell through to the base
    /// default), so training used an untuned rate.
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        if (_hasUserSuppliedOptimizer && _optimizer is not null)
            return _optimizer;

        // Small warmup into 3e-4, then linear DECAY to ~0 over the training horizon. A loss probe
        // showed a held 1e-3 rate lets the deep from-scratch backbone DIVERGE over longer training
        // (MoreData: 50-iter loss 11.9 -> 200-iter loss 18.9). The lower peak rate plus a decay
        // phase keeps the descent monotonic across both the short (Training_ShouldReduceLoss) and
        // long (MoreData) invariants. totalSteps=300 comfortably covers MoreData's 200 iterations.
        return _baseTapeOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                UseAMSGrad = false,
                InitialLearningRate = 0.0001,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(
                    baseLearningRate: 0.0001,
                    warmupSteps: 5,
                    totalSteps: 300,
                    warmupInitLr: 0.00001)
            });
    }

    /// <summary>
    /// Resolves each layer's lazy shape by running the REAL forward once, so serialize/clone
    /// round-trips preserve trained weights.
    /// </summary>
    /// <remarks>
    /// UniVS's backbone mixes BottleneckBlock / MaxPooling / BatchNorm layers whose output shapes
    /// the base's sequential shape walk cannot always advance, leaving some conv layers lazy
    /// (input depth unresolved). SetParameters silently skips an unresolved layer, so a
    /// serialize -> new-instance -> deserialize clone dropped those layers' trained weights and the
    /// clone predicted differently (Clone_AfterTraining, #1221 class). One real eval-mode forward
    /// materializes every layer up front. Idempotent.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesWarmed || !_useNativeMode) return;
        _lazyShapesWarmed = true; // set first so any reentrancy is a no-op

        var dummy = new Tensor<T>([1, _channels, _height, _width]);
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try { Forward(dummy); }
        catch { /* best-effort; a genuine forward failure surfaces on the real Train/Predict */ }
        finally { if (wasTraining) SetTrainingMode(true); }
    }
    #endregion

    #region Private Methods
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(UniVSModelSize modelSize) => modelSize switch
    {
        UniVSModelSize.R50 => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256),
        UniVSModelSize.SwinLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
        _ => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256)
    };

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
            var encoderLayers = LayerHelper<T>.CreateUniVSEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateUniVSDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "UniVS" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };


    #endregion

    #region IVideoSegmentation Implementation
    // UniVS (Li et al. 2024) treats prompts as queries. Every tracked object owns a prompt memory pool: the
    // visual sampler draws NumPromptPoints points from each visual prompt (the initial mask, then every
    // predicted mask fed back) and stores their pixel embeddings as prompt tokens. A frame's query for the
    // object is the mean of its pool's tokens, refined by prompt cross-attention (ProCA) over the same
    // tokens, and the mask is the query's dot product with the pixel embedding. The frame counter,
    // tracked-id list and initialized flag live on VideoSegmentationBase.
    [Scratch]
    private readonly Dictionary<int, List<Tensor<T>>> _promptMemory = [];
    private int[] _objectIds = Array.Empty<int>();
    private int _maskHeight;
    private int _maskWidth;
    [Scratch]
    private readonly Dictionary<int, Tensor<T>> _corrections = [];

    /// <summary>The pixel embedding <c>[D, h*w]</c> of <paramref name="frame"/> (the network output).</summary>
    private (Tensor<T> Embedding, int Height, int Width) EmbedFrame(Tensor<T> frame)
    {
        if (frame is null) throw new ArgumentNullException(nameof(frame));
        var batched = frame.Rank == 4 ? frame : AddBatchDimension(frame);
        using var noGrad = new NoGradScope<T>();
        var output = Predict(batched);
        int d = output.Shape[1], h = output.Shape[2], w = output.Shape[3];
        return (Engine.Reshape(output, new[] { d, h * w }), h, w);
    }

    /// <summary>
    /// The visual sampler: up to <paramref name="count"/> points spread evenly, in raster order, over the pixels
    /// of <paramref name="mask"/> (<c>[h*w]</c>) at or above 0.5. Null when the mask is empty.
    /// </summary>
    internal static Tensor<int>? SamplePromptPoints(Tensor<T> mask, int count)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int[] foreground = engine.TensorGreaterEqual(
            mask, Tensor<T>.CreateDefault(mask.Shape.ToArray(), ops.FromDouble(0.5))).Cast<int>().ToArray();
        var positions = new List<int>();
        for (int i = 0; i < foreground.Length; i++)
        {
            if (foreground[i] != 0) positions.Add(i);
        }

        if (positions.Count == 0) return null;
        int l = Math.Min(count, positions.Count);
        var picked = new int[l];
        for (int j = 0; j < l; j++) picked[j] = positions[(int)((long)j * positions.Count / l)];
        return new Tensor<int>(picked, new[] { l });
    }

    /// <summary>
    /// The object query from its prompt tokens <c>[L, D]</c>: the token mean <c>q0</c> (P-&gt;Q), then prompt
    /// cross-attention <c>q = q0 + softmax(q0 P^T / sqrt(D)) P</c>.
    /// </summary>
    internal static Tensor<T> PromptQuery(Tensor<T> tokens)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int d = tokens.Shape[1];
        var initial = engine.ReduceMean(tokens, new[] { 0 }, true);                                         // [1, D]
        // The 1 / sqrt(D) scale is applied to q0 before the product, so the dot products are formed already scaled:
        // unscaled, a 64-wide product of entries near 32 is 65536 and overflows Half before the scale could apply.
        var attention = engine.TensorSoftmax(engine.TensorMatMul(
            engine.TensorMultiplyScalar(initial, ops.FromDouble(1.0 / Math.Sqrt(d))), engine.TensorTranspose(tokens)), 1);
        return engine.TensorAdd(initial, engine.TensorMatMul(attention, tokens));
    }

    /// <summary>Adds the prompt tokens of <paramref name="mask"/> (<c>[h*w]</c>) to an object's pool.</summary>
    private void Remember(int objectId, Tensor<T> embedding, Tensor<T> mask)
    {
        var points = SamplePromptPoints(mask, _options.NumPromptPoints);
        if (points is null) return;
        var tokens = Engine.TensorTranspose(Engine.TensorIndexSelect(embedding, points, 1));               // [l, D]
        if (!_promptMemory.TryGetValue(objectId, out var pool))
        {
            pool = new List<Tensor<T>>();
            _promptMemory[objectId] = pool;
        }

        AppendToMemory(pool, tokens, _options.PromptMemoryLength);
    }

    /// <summary>
    /// Appends a visual prompt to an object's memory pool and keeps only the <paramref name="length"/> most recent
    /// prompts, as the official UniVS code does (<c>prompt_memory[:, -num_prev_frames_memory:]</c> with
    /// <c>NUM_PREV_FRAMES_MEMORY = 5</c>); no earlier prompt is pinned.
    /// </summary>
    internal static void AppendToMemory(List<Tensor<T>> pool, Tensor<T> tokens, int length)
    {
        if (pool is null) throw new ArgumentNullException(nameof(pool));
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (length < 1) throw new ArgumentOutOfRangeException(nameof(length), "The prompt memory keeps at least one prompt.");
        pool.Add(tokens);
        if (pool.Count > length) pool.RemoveRange(0, pool.Count - length);
    }

    /// <summary>Feeds the masks <c>[O, H, W]</c> back to the prompt memory as visual prompts.</summary>
    private void RememberMasks(Tensor<T> embedding, Tensor<T> masks, int h, int w)
    {
        var small = Common.VideoTrackingOps.Resize(masks, h, w, InterpolateMode.Area);
        for (int i = 0; i < _objectIds.Length; i++)
        {
            var mask = Engine.Reshape(Engine.TensorSlice(small, new[] { i, 0, 0 }, new[] { 1, h, w }), new[] { h * w });
            Remember(_objectIds[i], embedding, mask);
        }
    }

    /// <inheritdoc/>
    protected override void InitializeTrackingInternal(Tensor<T> frame, Tensor<T> masks, int[] objectIds)
    {
        var objectMasks = Common.VideoTrackingOps.AsObjectMasks(masks, nameof(masks));
        if (objectMasks.Shape[0] != objectIds.Length)
            throw new ArgumentException($"Give one mask per object ({objectIds.Length}).", nameof(masks));
        if (_options.NumPromptPoints < 1) throw new InvalidOperationException("NumPromptPoints must be positive.");
        if (_options.PromptMemoryLength < 1) throw new InvalidOperationException("PromptMemoryLength must be positive.");
        _maskHeight = objectMasks.Shape[1];
        _maskWidth = objectMasks.Shape[2];
        _objectIds = objectIds.ToArray();
        _promptMemory.Clear();
        _corrections.Clear();
        var (embedding, h, w) = EmbedFrame(frame);
        using var noGrad = new NoGradScope<T>();
        RememberMasks(embedding, objectMasks, h, w);
    }

    /// <inheritdoc/>
    protected override VideoSegmentationResult<T> PropagateToFrameInternal(Tensor<T> frame, int frameIndex)
    {
        var (embedding, h, w) = EmbedFrame(frame);
        int o = _objectIds.Length, hm = _maskHeight, wm = _maskWidth;
        using var noGrad = new NoGradScope<T>();
        var rows = new Tensor<T>[o];
        for (int i = 0; i < o; i++)
        {
            int id = _objectIds[i];
            if (_corrections.TryGetValue(id, out var correction))
            {
                // A corrected object is prompted with its correction mask on this frame.
                if (correction.Rank != 2 || correction.Shape[0] != hm || correction.Shape[1] != wm)
                    throw new ArgumentException($"A correction mask must be [{hm}, {wm}].");
                rows[i] = Engine.Reshape(correction, new[] { 1, hm, wm });
                continue;
            }

            if (!_promptMemory.TryGetValue(id, out var pool) || pool.Count == 0)
            {
                rows[i] = new Tensor<T>(new[] { 1, hm, wm });
                continue;
            }

            var query = PromptQuery(Engine.TensorConcatenate(pool.ToArray(), 0));                          // [1, D]
            var logits = Engine.Reshape(Engine.TensorMatMul(query, embedding), new[] { 1, h, w });
            rows[i] = Engine.TensorSigmoid(Common.VideoTrackingOps.Resize(logits, hm, wm, InterpolateMode.Bilinear));
        }

        _corrections.Clear();
        var masks = o == 0 ? new Tensor<T>(new[] { 0, hm, wm }) : Engine.TensorConcatenate(rows, 0);
        RememberMasks(embedding, masks, h, w);
        var (confidences, visible) = Common.VideoTrackingOps.Statistics(masks);
        return new VideoSegmentationResult<T>
        {
            Masks = masks, ObjectIds = _objectIds.ToArray(), Confidences = confidences,
            FrameIndex = frameIndex, IsVisible = visible
        };
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
        _promptMemory.Clear();
        _objectIds = Array.Empty<int>();
        _corrections.Clear();
    }
    #endregion
}
