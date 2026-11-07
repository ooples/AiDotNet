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
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// XVFI: extreme video frame interpolation (Sim, Oh and Kim, ICCV 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (JihyongOh/XVFI, XVFInet.py). Frames are mapped to [−1, 1]. A shared feature
/// extractor reduces both frames by the scale factor, and one shared stride-2 convolution builds the feature at every
/// scale level. The same VFInet runs from the lowest level up: BiOF-I estimates the bidirectional flows (from the
/// features at the lowest level, else refining the level below's 2x-upsampled flows with warped features), and below
/// level 0 only the flows are passed on. At level 0 (and at every level in training) complementary flow reversal turns
/// them into flows from time t by z-weighted Gaussian forward warping, a refinement stack corrects those, and BiOF-T
/// blends the warped frames by a refinement U-Net's occlusion map and adds its residual. Sides are zero-padded on the
/// right and bottom to multiples of 2^S·scale·4, as the release's test pads. Any time in [0, 1] can be synthesized, and
/// the test depth can exceed the training depth.
/// </para>
/// <para>
/// Training follows main.py: Adam (betas 0.9 and 0.999, no weight decay) at 1e-4, multiplied by 0.25 at epochs 100, 150
/// and 180, on the L1 reconstruction loss averaged over every level's frame against the bicubically downscaled ground
/// truth, plus 0.5 times the edge-aware smoothness of level 0's refined flows. Training crops are multiples of
/// 2^S·scale·4.
/// </para>
/// <para>
/// <b>For Beginners:</b> XVFI handles very large motion in 4K video by estimating motion on a pyramid of ever smaller
/// copies of the frames, starting at the coarsest, and refining it one level at a time. The same small network is reused
/// at every level, so you can add levels at test time for bigger motion.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("XVFI: eXtreme Video Frame Interpolation",
    "https://arxiv.org/abs/2103.16206",
    Year = 2021,
    Authors = "Hyeonjun Sim, Jihyong Oh, Munchurl Kim")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.0, DecayRate = 0.25,
                ReferenceBatchSize = 8, Schedule = LearningRateSchedulerType.MultiStep,
                Source = "Sim et al. 2021, Sec. 4 and the release (main.py): Adam at an initial learning rate of "
                        + "1e-4, multiplied by 0.25 at epochs 100, 150 and 180 of 200, batch size 8.")]
public partial class XVFI<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly XVFIOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private XvfiNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates an XVFI model in ONNX inference mode.</summary>
    public XVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, XVFIOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new XVFIOptions() : new XVFIOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an XVFI model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's X4K1000FPS model and training when null.</param>
    /// <param name="optimizer">The optimizer; the release's Adam with its milestones when null.</param>
    public XVFI(NeuralNetworkArchitecture<T> architecture, XVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new XVFIOptions() : new XVFIOptions(options);
        Options = _options;
        var o = _options;
        if (o.NumFeatures < 1 || o.ScaleFactor < 2 || (o.ScaleFactor & (o.ScaleFactor - 1)) != 0 || (4 * o.NumFeatures) % (o.ScaleFactor * o.ScaleFactor) != 0
            || o.TrainingDepth < 0 || o.TestDepth < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "XVFI needs positive features, a power-of-two scale factor of at least 2 whose square divides 4·features, and non-negative depths.");
        if (o.LearningRate <= 0 || o.LearningRateMilestones is null || o.LearningRateMilestones.Any(m => m < 1) || o.LearningRateDecay <= 0
            || o.LearningRateDecay > 1 || o.ReconstructionWeight < 0 || o.SmoothnessWeight < 0 || o.EdgeConstant < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The training settings must be valid.");
        _useNativeMode = true;
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        if (!_useNativeMode || _usingCustomLayers) return;
        RegisterParameterComponent("XVFI.Network", _network);
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
        _network = new XvfiNetwork<T>(_options.NumFeatures, _options.ScaleFactor);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private XvfiNetwork<T> Network => _network ?? throw new InvalidOperationException("XVFI's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    /// <summary>The side multiple a scale depth needs: 2^depth·scale·4, the release's divide.</summary>
    internal int Divisor(int depth) => (1 << depth) * _options.ScaleFactor * 4;

    /// <summary>The right and bottom zero padding that brings a side to a multiple of <see cref="Divisor"/>(depth).</summary>
    internal (int Right, int Bottom) Padding(int height, int width, int depth)
    {
        int d = Divisor(depth);
        return ((d - width % d) % d, (d - height % d) % d);
    }

    // [0, 1] frames to the release's [−1, 1] and back.
    private Tensor<T> ToNetwork(Tensor<T> x) => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));

    private Tensor<T> FromNetwork(Tensor<T> x) => Engine.TensorMultiplyScalar(Engine.TensorAddScalar(x, NumOps.One), NumOps.FromDouble(0.5));

    private Tensor<T> Pad(Tensor<T> x, int depth)
    {
        var (right, bottom) = Padding(x.Shape[2], x.Shape[3], depth);
        return right + bottom == 0 ? x : Engine.PadNd(x, new[] { 0, right, 0, bottom }, PadMode.Constant, NumOps.Zero);
    }

    private Tensor<T> Crop(Tensor<T> x, int h, int w)
        => x.Shape[2] == h && x.Shape[3] == w ? x : Engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], h, w });

    // The frame at time t from frames [N, 3, H, W] in [0, 1], at the test depth.
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1, double t)
    {
        int depth = _options.TestDepth;
        var output = Network.Forward(Pad(ToNetwork(frame0), depth), Pad(ToNetwork(frame1), depth), t, depth, false).Outputs[0];
        return FromNetwork(Crop(output, frame0.Shape[2], frame0.Shape[3]));
    }

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"XVFI takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
        return (Engine.TensorNarrow(x, 1, 0, 3).Contiguous(), Engine.TensorNarrow(x, 1, 3, 3).Contiguous(), batched);
    }

    private Tensor<T> AsBatch(Tensor<T> frames) => frames.Rank == 4 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());

    private Tensor<T> RunPairs(Tensor<T> pairs, double t)
    {
        var (frame0, frame1, batched) = SplitPairs(pairs);
        var output = Synthesize(frame0, frame1, t);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    #endregion

    #region Interpolation

    /// <inheritdoc />
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (!(t >= 0.0 && t <= 1.0))
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep must be in [0, 1].");
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
            output = RunPairs(pair, t);
        }
        return PostprocessOutput(output);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return PostprocessOutput(RunOnnxInference(PreprocessFrames(input)));
        return PostprocessOutput(Forward(PreprocessFrames(input)));
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        using var noGrad = new NoGradScope<T>();
        return RunPairs(input, 0.5);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunPairs(PreprocessFrames(input), 0.5);

    /// <inheritdoc />
    /// <remarks>Every level's frame (finest first) and level 0's refined flows from the midpoint.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        int depth = _options.TestDepth;
        var output = Network.Forward(Pad(ToNetwork(frame0), depth), Pad(ToNetwork(frame1), depth), 0.5, depth, true);
        var activations = new Dictionary<string, Tensor<T>>();
        for (int level = 0; level < output.Outputs.Length; level++) activations[$"Level{level}"] = output.Outputs[level];
        if (output.RefinedFlow is not null) activations["RefinedFlow"] = output.RefinedFlow;
        return activations;
    }

    #endregion

    #region Training

    // The forward warps and the warps' masks are built on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // optim.Adam(betas=(0.9, 0.999), weight_decay=0) and MultiStepLR stepped per epoch.
        var adam = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            UseAMSGrad = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new MultiStepLRScheduler(_options.LearningRate, _options.LearningRateMilestones.ToArray(), _options.LearningRateDecay),
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adam);
        return _optimizer;
    }

    /// <summary>
    /// The release's loss on frames [N, 3, H, W] in [0, 1], computed in [−1, 1]: the L1 reconstruction loss averaged over
    /// every level's frame against the bicubically downscaled ground truth, plus the smoothness of level 0's refined
    /// flows against the ground truth downscaled by the scale factor.
    /// </summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3], depth = _options.TrainingDepth, d = Divisor(depth);
        if (h % d != 0 || w % d != 0)
            throw new ArgumentException($"XVFI trains on crops whose sides are multiples of {d}; got {h}x{w}.", nameof(frame0));
        var gt = ToNetwork(target);
        var output = Network.Forward(ToNetwork(frame0), ToNetwork(frame1), 0.5, depth, true);
        Tensor<T>? reconstruction = null;
        for (int level = 0; level < output.Outputs.Length; level++)
        {
            var term = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(output.Outputs[level], XvfiOps.Downscale(Engine, gt, 1 << level))),
                new[] { 0, 1, 2, 3 }, false);
            reconstruction = reconstruction is null ? term : Engine.TensorAdd(reconstruction, term);
        }
        if (reconstruction is null || output.RefinedFlow is null) throw new InvalidOperationException("XVFI's training pass produced no frame.");
        var loss = Engine.TensorMultiplyScalar(reconstruction, NumOps.FromDouble(_options.ReconstructionWeight / output.Outputs.Length));
        var smooth = XvfiOps.Smoothness(Engine, output.RefinedFlow, XvfiOps.Downscale(Engine, gt, _options.ScaleFactor), _options.EdgeConstant);
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(smooth, NumOps.FromDouble(_options.SmoothnessWeight)));
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
                var (frame0, frame1, _) = SplitPairs(PreprocessFrames(data[0]));
                return ReleaseLoss(frame0, frame1, data[1]);
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
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        return ReleaseLoss(frame0, frame1, AsBatch(NormalizeFrames(target)))[0];
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
            Name = _useNativeMode ? "XVFI-Native" : "XVFI-ONNX",
            Description = "XVFI extreme video frame interpolation (Sim, Oh and Kim, ICCV 2021)",
            Complexity = _options.NumFeatures
        };
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        m.AdditionalInfo["TrainingDepth"] = _options.TrainingDepth.ToString();
        m.AdditionalInfo["TestDepth"] = _options.TestDepth.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(XVFI<T>));
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
