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
/// BiM-VFI: bidirectional motion field-guided frame interpolation for video with non-uniform motions (Seo, Oh and Kim,
/// CVPR 2025).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (KAIST-VICLab/BiM-VFI, modules/components/bim_vfi):
/// <list type="bullet">
/// <item><b>Bidirectional motion field (BiM).</b> The motion from the intermediate time to the two frames is described
/// by the ratio r of their magnitudes and the angle φ between them. At inference the field is uniform: r = t and
/// φ = π.</item>
/// <item><b>BiMFN.</b> Motion features of both frames, warped by the previous level's flows, with their 9x9 cost
/// volumes, the flows and the occlusion, form a motion feature that the embedded r and φ modulate; it predicts the
/// residual of both flows at 1/4 size.</item>
/// <item><b>CAUN.</b> Context features warped by those flows predict convex-combination weights upsampling the flows
/// to 1/2 and full size.</item>
/// <item><b>Synthesis.</b> A U-Net over the warped frames, occlusion and warped context predicts a residual and the
/// occlusion logits that blend the two warped frames.</item>
/// </list>
/// The levels run coarse-to-fine over an anti-aliased image pyramid of frames normalized by their joint mean and
/// standard deviation and zero-padded on the right and bottom to multiples of 2^(levels + 1).
/// </para>
/// <para>
/// Training follows cfgs/bim_vfi.yaml with knowledge distillation for VFI-centric flow (KDVCF): a teacher that sees the
/// ground-truth frame estimates the flows to it, whose ratio and angle feed the student's BiMFN. The loss sums the
/// student's multi-level Charbonnier and ternary terms, the teacher's photometric terms, the distillation of the
/// teacher's flow residuals, the zero-flow regularization of the teacher's frame-side flows and the edge-aware
/// smoothness of its flows. AdamW (weight decay 1e-4) follows a cosine one-cycle schedule peaking at 1.5e-4 after 1%
/// of training, with gradients clipped to norm 1. The paper's text instead reports a 1e-4 initial rate under cosine
/// annealing at batch 32; the release's configuration is the one that trained its checkpoints. The library's one-cycle
/// schedule warms up linearly where torch's cosine-annealed one-cycle warms up along a cosine; the two meet at both
/// ends of the 1% warm-up.
/// </para>
/// <para>
/// <b>For Beginners:</b> BiM-VFI describes how each pixel moves between the frames by a speed ratio and an angle,
/// estimates the motion from the in-between moment back to each frame from coarse to fine, and blends the two warped
/// frames into the in-between frame.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions",
    "https://arxiv.org/abs/2412.11365",
    Year = 2024,
    Authors = "Wonyong Seo, Jihyong Oh, Munchurl Kim")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1.5e-4, ReferenceBatchSize = 32, WeightDecay = 1e-4,
                MaxGradientNorm = 1.0, WarmupFraction = 0.01,
                Schedule = LearningRateSchedulerType.OneCycle,
                Source = "KAIST-VICLab/BiM-VFI cfgs/bim_vfi.yaml: AdamW (lr 1.5e-4, weight decay 1e-4), "
                        + "OneCycleLR (max_lr 1.5e-4, pct_start 0.01, cosine annealing) stepped per "
                        + "iteration, gradient clipping at norm 1, 400 epochs. Seo et al. 2025, Sec. 4 "
                        + "states AdamW, batch 32 and 400 epochs on Vimeo90K septuplets.")]
public partial class BiMVFI<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly BiMVFIOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private BimVfiNetwork<T>? _network;
    private readonly Random _teacherRandom;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates a BiM-VFI model in ONNX inference mode.</summary>
    public BiMVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, BiMVFIOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new BiMVFIOptions() : new BiMVFIOptions(options);
        Options = _options;
        _useNativeMode = false;
        _teacherRandom = RandomHelper.CreateSecureRandom();
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a BiM-VFI model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's when null.</param>
    /// <param name="optimizer">The optimizer; the release's AdamW with one-cycle schedule when null.</param>
    public BiMVFI(NeuralNetworkArchitecture<T> architecture, BiMVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new BiMVFIOptions() : new BiMVFIOptions(options);
        Options = _options;
        var o = _options;
        if (o.FeatureChannels < 1 || o.PyramidLevels < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "BiM-VFI needs a positive feature width and at least one pyramid level.");
        if (o.LearningRate <= 0 || o.WeightDecay < 0 || o.TotalIterations < 1 || !(o.WarmupFraction >= 0 && o.WarmupFraction < 1)
            || o.MaxGradientNorm <= 0 || o.LevelDecay < 0 || o.FlowLevelDecay < 0 || o.DistillationWeight < 0
            || o.RegularizationWeight < 0 || o.SmoothnessWeight < 0 || o.SmoothnessEdgeConstant < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The training settings must be valid.");
        _teacherRandom = o.Seed is int seed ? RandomHelper.CreateSeededRandom(seed) : RandomHelper.CreateSecureRandom();
        _useNativeMode = true;
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        if (!_useNativeMode || _usingCustomLayers) return;
        RegisterParameterComponent("BiMVFI.Network", _network);
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
        _network = new BimVfiNetwork<T>(_options.FeatureChannels);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private BimVfiNetwork<T> Network => _network ?? throw new InvalidOperationException("BiM-VFI's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    // The frame at time t from frames [N, 3, H, W] in [0, 1], cropped from the padded finest level.
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1, double t)
    {
        var finest = Network.Run(frame0, frame1, t, _options.PyramidLevels)[_options.PyramidLevels - 1].Image;
        int h = frame0.Shape[2], w = frame0.Shape[3];
        return finest.Shape[2] == h && finest.Shape[3] == w ? finest : Engine.TensorSlice(finest, new[] { 0, 0, 0, 0 }, new[] { finest.Shape[0], 3, h, w });
    }

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"BiM-VFI takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
        return (Engine.TensorNarrow(x, 1, 0, 3), Engine.TensorNarrow(x, 1, 3, 3), batched);
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
    /// <remarks>Each pyramid level's synthesized frame and flows at the midpoint.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        var levels = Network.Run(frame0, frame1, 0.5, _options.PyramidLevels);
        var activations = new Dictionary<string, Tensor<T>>();
        for (int i = 0; i < levels.Count; i++)
        {
            int level = levels.Count - 1 - i;
            activations[$"Level{level}.Flow"] = levels[i].Flow;
            activations[$"Level{level}.Synthesis"] = levels[i].Image;
        }
        return activations;
    }

    #endregion

    #region Training

    // The cost volumes and warps build their offsets and sampling grids on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // torch.optim.AdamW keeps its default betas (0.9, 0.999) and epsilon 1e-8; OneCycleLR its div factors 25 and 1e4.
        var adamW = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = true,
            MaxGradientNorm = _options.MaxGradientNorm,
            LearningRateScheduler = new OneCycleLRScheduler(_options.LearningRate, _options.TotalIterations, _options.WarmupFraction),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamW);
        return _optimizer;
    }

    private T Scalar(double value) => NumOps.FromDouble(value);

    private Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, Scalar(weight));

    /// <summary>
    /// The release's six-term loss for frames [N, 3, H, W] in [0, 1] with sides divisible by 2^(levels + 1), the
    /// sizes the release trains on.
    /// </summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        var o = _options;
        int levels = o.PyramidLevels, divisor = 1 << (levels + 1);
        if (target.Shape[2] % divisor != 0 || target.Shape[3] % divisor != 0)
            throw new ArgumentException($"BiM-VFI trains on frames whose sides are multiples of {divisor}; got {target.Shape[2]}x{target.Shape[3]}.", nameof(target));
        var outputs = Network.Run(frame0, frame1, 0.5, levels, target, _teacherRandom);
        var targetPyramid = new Tensor<T>[levels];
        for (int k = 0; k < levels; k++) targetPyramid[k] = BimOps.AntialiasDownsample(Engine, target, 1 << k);

        Tensor<T>? total = null;
        void Add(Tensor<T> term) => total = total is null ? term : Engine.TensorAdd(total, term);
        Tensor<T> Narrow(Tensor<T> x, int start) => Engine.TensorNarrow(x, 1, start, 2);
        var smoothShape = target.Shape;
        for (int k = 0; k < levels; k++)
        {
            // k = 0 is the finest level.
            var output = outputs[levels - 1 - k];
            var teacher = output.Teacher ?? throw new InvalidOperationException("The teacher pass did not run.");
            var down = targetPyramid[k];
            double photo = Math.Pow(o.LevelDecay, k), flow = Math.Pow(o.FlowLevelDecay, k);
            // multiple_charbonnier, multiple_ternary, photo_teacher.
            Add(Weighted(Engine.TensorAdd(BimOps.Charbonnier(Engine, output.Image, down), BimOps.Ternary(Engine, output.Image, down)), photo));
            Add(Weighted(Engine.TensorAdd(BimOps.Ternary(Engine, teacher.Image, down), BimOps.Charbonnier(Engine, teacher.Image, down)), photo));
            // flow_distill_res: residuals within 32 px of the teacher's.
            Tensor<T> Distill(Tensor<T> student, Tensor<T> tea)
            {
                var magnitude = Engine.ReduceSum(Engine.TensorSquare(tea), new[] { 1 }, true);
                var mask = new Tensor<T>(magnitude._shape);
                for (int i = 0; i < mask.Length; i++) mask[i] = NumOps.ToDouble(magnitude[i]) < 32 * 32 ? NumOps.One : NumOps.Zero;
                var cb = BimOps.FlowCharbonnier(Engine, Engine.TensorSubtract(student, tea));
                return BimOps.MeanAll(Engine, Engine.TensorMultiply(cb, Engine.TensorBroadcastTo(mask, cb.Shape.ToArray())));
            }
            Add(Weighted(Engine.TensorAdd(Distill(Narrow(output.FlowResidual, 0), teacher.ResidualT0),
                Distill(Narrow(output.FlowResidual, 2), teacher.ResidualT1)), o.DistillationWeight * flow));
            // flow_teacher_reg: the teacher's residuals on the frame side stay zero.
            Add(Weighted(Engine.TensorAdd(BimOps.MeanAll(Engine, Engine.TensorAbs(teacher.ZeroSideFlow0)),
                BimOps.MeanAll(Engine, Engine.TensorAbs(teacher.ZeroSideFlow1))), o.RegularizationWeight * flow));
            // flow_smooth_tea1 (gaussian): Charbonnier flow gradients weighted by exp(−(c·mean|∇I|)²).
            Add(Weighted(Smoothness(Narrow(teacher.Flow, 0), Narrow(teacher.Flow, 2), down), o.SmoothnessWeight * flow));
        }
        return total ?? throw new InvalidOperationException("BiM-VFI has at least one pyramid level.");
    }

    private Tensor<T> Smoothness(Tensor<T> flow0, Tensor<T> flow1, Tensor<T> image)
    {
        Tensor<T> Dx(Tensor<T> x) => Engine.TensorSubtract(Engine.TensorNarrow(x, 3, 1, x.Shape[3] - 1), Engine.TensorNarrow(x, 3, 0, x.Shape[3] - 1));
        Tensor<T> Dy(Tensor<T> x) => Engine.TensorSubtract(Engine.TensorNarrow(x, 2, 1, x.Shape[2] - 1), Engine.TensorNarrow(x, 2, 0, x.Shape[2] - 1));
        Tensor<T> EdgeWeight(Tensor<T> gradient)
            => Engine.TensorExp(Engine.TensorNegate(Engine.TensorSquare(Engine.TensorMultiplyScalar(
                Engine.ReduceMean(Engine.TensorAbs(gradient), new[] { 1 }, true), Scalar(_options.SmoothnessEdgeConstant)))));
        var wx = Engine.StopGradient(EdgeWeight(Dx(image)));
        var wy = Engine.StopGradient(EdgeWeight(Dy(image)));
        Tensor<T> Term(Tensor<T> gradient, Tensor<T> weight)
        {
            var cb = BimOps.FlowCharbonnier(Engine, gradient);
            return BimOps.MeanAll(Engine, Engine.TensorMultiply(cb, Engine.TensorBroadcastTo(weight, cb.Shape.ToArray())));
        }
        return Engine.TensorAdd(Engine.TensorAdd(Term(Dx(flow0), wx), Term(Dx(flow1), wx)), Engine.TensorAdd(Term(Dy(flow0), wy), Term(Dy(flow1), wy)));
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
    /// <remarks>
    /// The student's photometric loss (multi-level Charbonnier and ternary) at the uniform midpoint field, the
    /// inference path a training step improves. The full release loss samples the teacher's random angles, so it
    /// is not a repeatable measure.
    /// </remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        var prediction = Synthesize(frame0, frame1, 0.5);
        var expected = AsBatch(NormalizeFrames(target));
        return Engine.TensorAdd(BimOps.Charbonnier(Engine, prediction, expected), BimOps.Ternary(Engine, prediction, expected))[0];
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
            Name = _useNativeMode ? "BiMVFI-Native" : "BiMVFI-ONNX",
            Description = "BiM-VFI bidirectional motion field-guided frame interpolation (Seo et al., CVPR 2025)",
            Complexity = _options.PyramidLevels
        };
        m.AdditionalInfo["FeatureChannels"] = _options.FeatureChannels.ToString();
        m.AdditionalInfo["PyramidLevels"] = _options.PyramidLevels.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BiMVFI<T>));
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
