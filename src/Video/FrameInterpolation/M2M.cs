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
/// M2M: many-to-many splatting for efficient video frame interpolation (Hu, Niklaus, Sclaroff and Saenko, CVPR 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release's M2M-PWC (feinanshan/M2M_VFI, m2m.py). Both frames are standardized by their joint
/// mean and standard deviation. PWC-Net estimates the bidirectional flows at 1/ratio size; the Motion Refinement Network
/// upsamples them and, from both frames' feature pyramids warped toward each other and a rank-16 attention cube at 1/16
/// size, adds four flow residuals per pixel each way and predicts a photometric weight. Every pixel of each frame is
/// splatted along all four of its flows to time t, weighted by its distance in time and exp(α·(1 − weight·photometric
/// error)²), and the sum is divided by the summed weights; pixels nothing reaches take the time-weighted blend. Sides are
/// replicate-padded on the right and bottom to multiples of 16·ratio, as the release pads. Any time in [0, 1] can be
/// synthesized from one motion estimate.
/// </para>
/// <para>
/// Training follows the release (train.py, optimizer.py, configs/vimeo90k.yml): AdamW at 1e-4 with a weight decay of
/// 1e-4, a 2,000-step linear warmup from zero and cosine decay to zero at 900,000 steps, gradients clipped to norm 1, on
/// the Charbonnier plus ternary census loss at t = 0.5. The release loads a pretrained PWC-Net and trains it at a
/// learning rate of zero; <see cref="M2MOptions.FlowLearningRateMultiplier"/> sets that multiple.
/// </para>
/// <para>
/// <b>For Beginners:</b> M2M estimates rough motion once, then refines it into several motion guesses for every pixel.
/// Each pixel is pushed forward along all of its guesses to the in-between time, and where several pixels land on the
/// same spot they are blended by how reliable each one is. There is no heavy synthesis network, so it is fast.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Many-to-many Splatting for Efficient Video Frame Interpolation",
    "https://arxiv.org/abs/2204.03513",
    Year = 2022,
    Authors = "Ping Hu, Simon Niklaus, Stan Sclaroff, Kate Saenko")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 1e-4, MinLearningRate = 0,
                WarmupSteps = 2000, MaxGradientNorm = 1.0, ReferenceBatchSize = 12, Schedule = LearningRateSchedulerType.LinearWarmup,
                Source = "The release (Train/configs/vimeo90k.yml, vfi/optimizer.py, vfi/solver.py): AdamW at 1e-4 with a "
                        + "weight decay of 1e-4, linear warmup over 2,000 steps then cosine decay to 0 at 900,000, batch 12, "
                        + "gradient norm clipped to 1, PWC-Net at a learning rate of 0.")]
public partial class M2M<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly M2MOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private M2mNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates an M2M model in ONNX inference mode.</summary>
    public M2M(NeuralNetworkArchitecture<T> architecture, string modelPath, M2MOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new M2MOptions() : new M2MOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an M2M model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's M2M-PWC and training when null.</param>
    /// <param name="optimizer">The optimizer; the release's AdamW with warmup and cosine decay when null.</param>
    public M2M(NeuralNetworkArchitecture<T> architecture, M2MOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new M2MOptions() : new M2MOptions(options);
        Options = _options;
        var o = _options;
        if (o.BaseChannels < 1 || o.NumBranches < 1 || o.CubeRank < 1 || o.Ratio < 2 || (o.Ratio & (o.Ratio - 1)) != 0 || o.FlowChannels < 1
            || o.FlowDecoderChannels is null || o.FlowDecoderChannels.Length == 0 || o.FlowDecoderChannels.Any(c => c < 1))
            throw new ArgumentOutOfRangeException(nameof(options), "M2M needs positive widths, branches and rank, and a power-of-two ratio of at least 2.");
        if (o.LearningRate <= 0 || o.WeightDecay < 0 || o.WarmupSteps < 0 || o.TotalSteps <= o.WarmupSteps || !(o.MaxGradientNorm > 0)
            || o.FlowLearningRateMultiplier < 0 || o.CharbonnierWeight < 0 || o.TernaryWeight < 0)
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
        RegisterParameterComponent("M2M.Network", _network);
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
        _network = new M2mNetwork<T>(_options.BaseChannels, _options.NumBranches, _options.CubeRank, _options.FlowChannels, _options.FlowDecoderChannels.ToArray());
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private M2mNetwork<T> Network => _network ?? throw new InvalidOperationException("M2M's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    /// <summary>PWC-Net's weights, trained at <see cref="M2MOptions.FlowLearningRateMultiplier"/> times the rate.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters => _network is null ? Array.Empty<Tensor<T>>() : _network.FlowWeights.ToList();

    #endregion

    #region Pipeline

    /// <summary>The release's right and bottom replicate padding to multiples of 16·ratio.</summary>
    internal (int Right, int Bottom) Padding(int height, int width)
    {
        int d = 16 * _options.Ratio;
        return ((d - width % d) % d, (d - height % d) % d);
    }

    private Tensor<T> Pad(Tensor<T> x)
    {
        var (right, bottom) = Padding(x.Shape[2], x.Shape[3]);
        return right + bottom == 0 ? x : Engine.PadNd(x, new[] { 0, right, 0, bottom }, PadMode.Replicate, NumOps.Zero);
    }

    private Tensor<T> Crop(Tensor<T> x, int h, int w)
        => x.Shape[2] == h && x.Shape[3] == w ? x : Engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], h, w });

    // The frame at time t from frames [N, 3, H, W] in [0, 1].
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1, double t)
    {
        var output = Network.Forward(Pad(frame0), Pad(frame1), new[] { t }, _options.Ratio)[0];
        return Crop(output, frame0.Shape[2], frame0.Shape[3]);
    }

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"M2M takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
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
    /// <remarks>The refined flows each way, the photometric weights and the frame at the midpoint.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        var (forward, backward, weight0, weight1) = Network.Motion(Pad(frame0), Pad(frame1), _options.Ratio);
        return new Dictionary<string, Tensor<T>>
        {
            ["ForwardFlows"] = forward,
            ["BackwardFlows"] = backward,
            ["Weight0"] = weight0,
            ["Weight1"] = weight1,
            ["Output"] = Synthesize(frame0, frame1, 0.5),
        };
    }

    #endregion

    #region Training

    // The splatting builds its indices on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // The flow estimator's rate is a mask keyed by parameter tensor.
        ResolveLazyLayerShapes();
        // optim.AdamW(lr0, weight_decay=wd), the rate set per step to a linear warmup then a cosine to zero, the gradient
        // norm clipped to 1.
        var adamw = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = true,
            GradientClippingMethod = GradientClippingMethod.ByNorm,
            MaxGradientNorm = _options.MaxGradientNorm,
            LearningRateScheduler = new LinearWarmupScheduler(_options.LearningRate, _options.WarmupSteps, _options.TotalSteps, 0.0,
                LinearWarmupScheduler.DecayMode.Cosine, 0.0),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        var flow = new HashSet<Tensor<T>>(Network.FlowWeights);
        var parameters = NetworkParameters;
        var rates = new List<T>();
        foreach (var p in parameters)
        {
            var rate = NumOps.FromDouble(flow.Contains(p) ? _options.FlowLearningRateMultiplier : 1.0);
            for (int i = 0; i < p.Length; i++) rates.Add(rate);
        }
        var options = (AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)adamw.GetOptions();
        options.LearningRateMask = new Vector<T>(rates.ToArray());
        options.MaskParameters = parameters.ToList();
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamw);
        return _optimizer;
    }

    /// <summary>
    /// The release's loss on frames [N, 3, H, W] in [0, 1]: the Charbonnier loss plus the ternary census loss of the
    /// frame at t = 0.5.
    /// </summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        var prediction = Synthesize(frame0, frame1, 0.5);
        var charbonnier = Engine.TensorMultiplyScalar(AmtOps.Charbonnier(Engine, prediction, target), NumOps.FromDouble(_options.CharbonnierWeight));
        var ternary = Engine.TensorMultiplyScalar(M2mOps.Ternary(Engine, prediction, target), NumOps.FromDouble(_options.TernaryWeight));
        return Engine.TensorAdd(charbonnier, ternary);
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
            Name = _useNativeMode ? "M2M-Native" : "M2M-ONNX",
            Description = "M2M many-to-many splatting frame interpolation (Hu et al., CVPR 2022)",
            Complexity = _options.BaseChannels
        };
        m.AdditionalInfo["BaseChannels"] = _options.BaseChannels.ToString();
        m.AdditionalInfo["NumBranches"] = _options.NumBranches.ToString();
        m.AdditionalInfo["Ratio"] = _options.Ratio.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(M2M<T>));
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
