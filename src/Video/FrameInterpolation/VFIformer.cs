using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// VFIformer: video frame interpolation with a transformer (Lu, Cui, Tian, Jiang and Jia, CVPR 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (dvlab-research/VFIformer: VFIformer_arch.py, transformer_layers.py). IFNet estimates
/// the flows from the midpoint to both frames at half size in three coarse-to-fine blocks; FlowRefineNet_Multis refines
/// them from 1/8 size up with local cosine correlations and warps both frames' four-level feature pyramids along them.
/// The warped frames are fused and passed to a four-scale transformer encoder-decoder whose windows also attend to a
/// coarser copy of their surroundings (cross-scale window attention); its output is a residual and a blending mask for
/// the two warped frames. The frame is the midpoint. Frames are zero-padded around the centre to multiples of
/// 8·window, as the release's demo pads.
/// </para>
/// <para>
/// Training follows the release (models/trainer.py): AdamW at 1e-4 with a weight decay of 1e-4, the rate rising
/// linearly over 2,000 steps and then falling by a cosine to zero over 300 epochs, on the L1 loss plus the ternary census
/// loss. A training input is the pair [6, H, W], optionally followed by the precomputed flows from the midpoint to frame 0
/// and to frame 1 ([10, H, W], in pixels; the release computes them with LiteFlowNet), which add 0.01 times the mean
/// end-point error of each of IFNet's three flows. Training crops are multiples of 8·window, and the transformer uses
/// stochastic depth.
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIformer first estimates how pixels move between the two frames, warps both frames to the
/// middle, and then uses a transformer, which compares each small window of the image with its surroundings at a coarser
/// scale, to fix the places where the warping went wrong.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Video Frame Interpolation with Transformer",
    "https://arxiv.org/abs/2205.07230",
    Year = 2022,
    Authors = "Liying Lu, Ruizheng Wu, Huaijia Lin, Jiangbo Lu, Jiaya Jia")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 1e-4, MinLearningRate = 0,
                WarmupSteps = 2000, ReferenceBatchSize = 24, Schedule = LearningRateSchedulerType.LinearWarmup,
                Source = "The release (models/trainer.py, train.py and the README's command): AdamW at 1e-4 with a weight "
                        + "decay of 1e-4, the rate rising linearly over 2,000 steps then a cosine to 0 over 300 epochs, "
                        + "batch 24 on 8 GPUs, 192x192 crops.")]
public partial class VFIformer<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly VFIformerOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private VfiformerNetwork<T>? _network;
    private Random? _dropRandom;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates a VFIformer model in ONNX inference mode.</summary>
    public VFIformer(NeuralNetworkArchitecture<T> architecture, string modelPath, VFIformerOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new VFIformerOptions() : new VFIformerOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a VFIformer model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's VFIformer and training when null.</param>
    /// <param name="optimizer">The optimizer; the release's AdamW with warmup and cosine decay when null.</param>
    public VFIformer(NeuralNetworkArchitecture<T> architecture, VFIformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new VFIformerOptions() : new VFIformerOptions(options);
        Options = _options;
        var o = _options;
        if (o.FlowWidths is null || o.FlowWidths.Length != 3 || o.FlowWidths.Any(c => c < 2 || c % 2 != 0) || o.RefineChannels < 1 || o.EmbedDim < 2
            || o.EmbedDim % 2 != 0 || o.WindowSize < 4 || o.WindowSize % 4 != 0 || o.NumHeads < 1 || o.EmbedDim % o.NumHeads != 0 || o.MlpRatio <= 0
            || o.Depths is null || o.Depths.Length != 4 || o.Depths.Any(d => d < 1) || o.CrossScaleAttention is null || o.CrossScaleAttention.Length != 4
            || o.CrossScaleAttention.Any(g => g is null || g.Length != 2) || o.DropPathRate < 0 || o.DropPathRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "VFIformer needs three even flow widths, positive widths, an even embedding divisible by the heads, a window that is a multiple of 4, four positive depths and two cross-scale flags per scale.");
        if (o.LearningRate <= 0 || o.WeightDecay < 0 || o.WarmupSteps < 0 || o.TotalSteps <= o.WarmupSteps || o.L1Weight < 0 || o.TernaryWeight < 0 || o.FlowWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The training settings must be valid.");
        _dropRandom = o.Seed.HasValue ? RandomHelper.CreateSeededRandom(o.Seed.Value) : RandomHelper.CreateSecureRandom();
        _useNativeMode = true;
        SupportsArbitraryTimestep = false;
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        if (!_useNativeMode || _usingCustomLayers) return;
        RegisterParameterComponent("VFIformer.Network", _network);
    }

    #region Layers

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.AddRange(Architecture.Layers);
            return;
        }
        _network = new VfiformerNetwork<T>(_options.FlowWidths.ToArray(), _options.RefineChannels, _options.EmbedDim, _options.WindowSize, _options.NumHeads,
            _options.MlpRatio, _options.Depths.ToArray(), _options.CrossScaleAttention.Select(g => (IReadOnlyList<bool>)g.ToArray()).ToArray(), _options.DropPathRate);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private VfiformerNetwork<T> Network => _network ?? throw new InvalidOperationException("VFIformer's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    /// <summary>The demo's zero padding around the centre to multiples of 8·window: (top, bottom, left, right).</summary>
    internal (int Top, int Bottom, int Left, int Right) Padding(int height, int width)
    {
        int d = 8 * _options.WindowSize, padH = (d - height % d) % d, padW = (d - width % d) % d;
        return (padH / 2, padH - padH / 2, padW / 2, padW - padW / 2);
    }

    private Tensor<T> Pad(Tensor<T> x)
    {
        var (top, bottom, left, right) = Padding(x.Shape[2], x.Shape[3]);
        return top + bottom + left + right == 0 ? x : Engine.PadNd(x, new[] { left, right, top, bottom }, PadMode.Constant, NumOps.Zero);
    }

    // The midpoint frame from frames [N, 3, H, W] in [0, 1].
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3];
        var (top, _, left, _) = Padding(h, w);
        var output = Network.Forward(Pad(frame0), Pad(frame1), null).Prediction;
        return output.Shape[2] == h && output.Shape[3] == w ? output
            : Engine.TensorSlice(output, new[] { 0, 0, top, left }, new[] { output.Shape[0], output.Shape[1], h, w });
    }

    // [6 + k, H, W] or [B, 6 + k, H, W] inputs (k = 0, or 4 flow channels) to the two frames [N, 3, H, W] and the flows.
    private (Tensor<T> Frame0, Tensor<T> Frame1, Tensor<T>? Flows, bool Batched) SplitInput(Tensor<T> input)
    {
        bool batched = input.Rank == 4;
        var x = batched ? input : Engine.Reshape(input, new[] { 1 }.Concat(input._shape).ToArray());
        if (x.Rank != 4 || (x.Shape[1] != 6 && x.Shape[1] != 10))
            throw new ArgumentException("VFIformer takes a frame pair [6, H, W], optionally followed by the flows from the midpoint to both frames "
                + $"([10, H, W]), optionally batched; got [{string.Join(", ", input._shape)}].", nameof(input));
        var flows = x.Shape[1] == 10 ? Engine.TensorNarrow(x, 1, 6, 4).Contiguous() : null;
        return (Engine.TensorNarrow(x, 1, 0, 3).Contiguous(), Engine.TensorNarrow(x, 1, 3, 3).Contiguous(), flows, batched);
    }

    // The frames divided by 255; the flow channels are in pixels.
    private Tensor<T> PreprocessInput(Tensor<T> input)
    {
        int axis = input.Rank == 4 ? 1 : 0;
        if (input.Rank < 3 || input.Shape[axis] <= 6) return PreprocessFrames(input);
        var frames = PreprocessFrames(Engine.TensorNarrow(input, axis, 0, 6).Contiguous());
        return Engine.TensorConcatenate(new[] { frames, Engine.TensorNarrow(input, axis, 6, input.Shape[axis] - 6).Contiguous() }, axis);
    }

    private Tensor<T> AsBatch(Tensor<T> frames) => frames.Rank == 4 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());

    private Tensor<T> RunInput(Tensor<T> input)
    {
        var (frame0, frame1, _, batched) = SplitInput(input);
        var output = Synthesize(frame0, frame1);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    #endregion

    #region Interpolation

    /// <inheritdoc />
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (t != 0.5)
            throw new ArgumentOutOfRangeException(nameof(t), t, "VFIformer synthesizes the midpoint only (t = 0.5).");
        var pair = ConcatenateFeatures(PreprocessFrames(frame0), PreprocessFrames(frame1));
        Tensor<T> output;
        if (IsOnnxMode)
        {
            output = RunOnnxInference(pair);
        }
        else if (_usingCustomLayers)
        {
            output = Forward(pair);
        }
        else
        {
            using var noGrad = new NoGradScope<T>();
            output = RunInput(pair);
        }
        return PostprocessOutput(output);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return PostprocessOutput(RunOnnxInference(PreprocessFrames(input)));
        return PostprocessOutput(Forward(_usingCustomLayers ? PreprocessFrames(input) : PreprocessInput(input)));
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        using var noGrad = new NoGradScope<T>();
        return RunInput(input);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunInput(PreprocessInput(input));

    /// <inheritdoc />
    /// <remarks>IFNet's three flows, the refined flows and the midpoint frame, on the padded frames.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _, _) = SplitInput(PreprocessInput(input));
        var (prediction, ifFlows, flow) = Network.Forward(Pad(frame0), Pad(frame1), null);
        var activations = new Dictionary<string, Tensor<T>>();
        for (int i = 0; i < ifFlows.Length; i++) activations[$"IFNetFlow{i}"] = ifFlows[i];
        activations["RefinedFlow"] = flow;
        activations["Prediction"] = prediction;
        return activations;
    }

    #endregion

    #region Training

    // The attention masks and stochastic depth are built on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // AdamW(lr, weight_decay), its rate set per step: a linear warmup from zero, then a cosine to zero.
        var adamw = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new LinearWarmupScheduler(_options.LearningRate, _options.WarmupSteps, _options.TotalSteps, 0.0,
                LinearWarmupScheduler.DecayMode.Cosine, 0.0),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamw);
        return _optimizer;
    }

    /// <summary>
    /// The trainer's loss on frames [N, 3, H, W] in [0, 1]: the L1 loss plus the ternary loss of the midpoint frame, plus
    /// (given the flows [N, 4, H, W]) the weighted mean end-point error of IFNet's flows against the flows halved in size
    /// and magnitude, averaged over the two directions.
    /// </summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target, Tensor<T>? flows, bool stochastic)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3], d = 8 * _options.WindowSize;
        if (h % d != 0 || w % d != 0)
            throw new ArgumentException($"VFIformer trains on crops whose sides are multiples of {d}; got {h}x{w}.", nameof(frame0));
        var (prediction, ifFlows, _) = Network.Forward(frame0, frame1, stochastic ? _dropRandom : null);
        var l1 = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(prediction, target)), new[] { 0, 1, 2, 3 }, false);
        var loss = Engine.TensorAdd(Engine.TensorMultiplyScalar(l1, NumOps.FromDouble(_options.L1Weight)),
            Engine.TensorMultiplyScalar(M2mOps.Ternary(Engine, prediction, target), NumOps.FromDouble(_options.TernaryWeight)));
        if (flows is null || _options.FlowWeight == 0) return loss;
        Tensor<T>? epe = null;
        foreach (var flow in ifFlows)
        {
            double scale = flow.Shape[3] / (double)w;
            var reference = Engine.StopGradient(Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flows, flow.Shape[2], flow.Shape[3]), NumOps.FromDouble(scale)));
            for (int direction = 0; direction < 2; direction++)
            {
                var difference = Engine.TensorSubtract(Engine.TensorNarrow(flow, 1, 2 * direction, 2), Engine.TensorNarrow(reference, 1, 2 * direction, 2));
                var error = Engine.TensorSqrt(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorSquare(difference), new[] { 1 }, true), NumOps.FromDouble(1e-6)));
                var term = Engine.ReduceMean(error, new[] { 0, 1, 2, 3 }, false);
                epe = epe is null ? term : Engine.TensorAdd(epe, term);
            }
        }
        return epe is null ? loss : Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(epe, NumOps.FromDouble(_options.FlowWeight / 2)));
    }

    /// <inheritdoc />
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var target = AsBatch(NormalizeFrames(expectedOutput));
        return new[]
        {
            new TrainingObjective<T>(new[] { input, target }, data =>
            {
                var (frame0, frame1, flows, _) = SplitInput(PreprocessInput(data[0]));
                return ReleaseLoss(frame0, frame1, data[1], flows, true);
            })
            {
                Optimizer = ReleaseOptimizer(),
                Parameters = () => NetworkParameters,
            },
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The release's loss, the quantity a training step lowers.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var (frame0, frame1, flows, _) = SplitInput(PreprocessInput(input));
        return ReleaseLoss(frame0, frame1, AsBatch(NormalizeFrames(target)), flows, false)[0];
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? ReleaseOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    #region NeuralNetworkBase

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <summary>Frames are pixels on the 0-255 scale; the network divides them by 255.</summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.Value;

    /// <summary>The interpolated frame comes back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VFIformer-Native" : "VFIformer-ONNX",
            Description = "VFIformer transformer frame interpolation (Lu et al., CVPR 2022)",
            Complexity = _options.EmbedDim
        };
        m.AdditionalInfo["EmbedDim"] = _options.EmbedDim.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        m.AdditionalInfo["Depths"] = string.Join(",", _options.Depths);
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VFIformer<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
