using AiDotNet.Attributes;
using AiDotNet.Diffusion.VAE;
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
/// ABME: asymmetric bilateral motion estimation for video frame interpolation (Park et al., ICCV 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The pipeline follows the release (JunHeum/ABME, run.py):
/// <list type="bullet">
/// <item><b>Symmetric bilateral motion</b> (SBMENet, BMBC's estimator with a strided first convolution). Both frames,
/// resized to multiples of 64 (halved from a height of 512), give the motion V from the intermediate time to the second
/// frame at a quarter of that size; −V points to the first.</item>
/// <item><b>Anchor and reliability.</b> Both frames warped along ∓V average into the anchor frame, each filling the
/// other's out-of-frame pixels; Z = exp(−20·|I₀→t − I₁→t|), the channel-averaged difference at a quarter size.</item>
/// <item><b>Asymmetric bilateral motion refinement</b> (ABMRNet). Run once from the anchor toward each frame, it frees
/// the two motions from the symmetric constraint.</item>
/// <item><b>Synthesis</b> (SynthesisNet). The frames and their feature pyramids, warped along the symmetric and the
/// asymmetric motions at three scales, are blended by per-pixel 3x3 dynamic filters over the four candidates and
/// refined by a residual GridNet, at a multiple of 8, then resized back bicubically.</item>
/// </list>
/// </para>
/// <para>
/// Training follows the paper's three stages (Sec. 4.1), selected by <see cref="ABMEOptions.Stage"/>: the symmetric
/// estimator, then the refiner with the estimator fixed, each on the photometric loss of the frames warped to the
/// intermediate time (Charbonnier plus census); then all three networks end to end on the synthesized frame. Each
/// stage has its own Adam with the paper's step halving.
/// </para>
/// <para>
/// The networks are trained for the midpoint only: <see cref="Interpolate"/> accepts t = 0.5.
/// </para>
/// <para>
/// <b>For Beginners:</b> ABME first guesses one motion for both frames, mirrored around the missing middle frame, then
/// lets the motion toward each frame differ, which handles speeding up, slowing down and curved paths, and finally
/// blends the two frames moved to the middle into the missing frame.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Asymmetric Bilateral Motion Estimation for Video Frame Interpolation",
    "https://arxiv.org/abs/2108.06815",
    Year = 2021,
    Authors = "Junheum Park, Chul Lee, Chang-Su Kim")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Component = "symmetric",
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.5, Milestones = [100000, 140000, 180000],
                Source = "Park et al. 2021, Sec. 4.1: the symmetric estimator trains with Adam at 1e-4 until 0.1M "
                        + "iterations, halved after every 0.04M, for 0.2M iterations at batch 24.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Component = "asymmetric",
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.5, Milestones = [120000, 180000, 240000],
                Source = "Park et al. 2021, Sec. 4.1: ABMR-Net, the estimator fixed, at 1e-4 until 0.12M iterations, "
                        + "halved after every 0.06M, for 0.3M iterations at batch 16.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Component = "synthesis",
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.5, Milestones = [350000, 500000, 650000],
                Source = "Park et al. 2021, Sec. 4.1: the synthesis network end to end with both motion networks at "
                        + "1e-4 until 0.35M iterations, halved after every 0.15M, for 0.8M iterations at batch 6.")]
public partial class ABME<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    // run.py: frames from this height on are estimated at half size, on a 128-pixel grid.
    private const int HalfSizeHeight = 512;

    // Motion is predicted in units of this many pixels.
    private const double MotionUnit = 20.0;

    private readonly ABMEOptions _options;
    private readonly Dictionary<AbmeTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _customOptimizer;
    private AbmeSbmeNet<T>? _sbme;
    private AbmeAbmrNet<T>? _abmr;
    private AbmeSynthesisNet<T>? _synthesis;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates an ABME model in ONNX inference mode.</summary>
    public ABME(NeuralNetworkArchitecture<T> architecture, string modelPath, ABMEOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new ABMEOptions() : new ABMEOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an ABME model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The widths and training stage; the release's and the paper's when null.</param>
    /// <param name="optimizer">An optimizer every stage steps with in place of the paper's Adam, or null.</param>
    public ABME(NeuralNetworkArchitecture<T> architecture, ABMEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _customOptimizer = optimizer;
        _options = options is null ? new ABMEOptions() : new ABMEOptions(options);
        Options = _options;
        var o = _options;
        static bool Widths(int[]? widths, int count) => widths is not null && widths.Length == count && widths.All(w => w > 0);
        if (!Widths(o.SymmetricEncoderWidths, 6) || !Widths(o.SymmetricDecoderWidths, 5) || !Widths(o.SymmetricDisplacements, 5)
            || !Widths(o.AsymmetricEncoderWidths, 2) || !Widths(o.AsymmetricDecoderWidths, 5) || o.AsymmetricDisplacement < 1
            || !Widths(o.SynthesisWidths, 3))
            throw new ArgumentOutOfRangeException(nameof(options),
                "ABME needs 6 symmetric encoder widths, 5 decoder widths and 5 displacements, 2 asymmetric encoder widths, "
                + "5 decoder widths and a displacement, and 3 synthesis widths, all positive.");
        if (o.LearningRate <= 0 || o.DecayRate <= 0 || o.CharbonnierEpsilon <= 0 || o.CensusWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The learning rate, decay, epsilon and census weight must be valid.");
        _useNativeMode = true;
        SupportsArbitraryTimestep = false;
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        if (!_useNativeMode || _usingCustomLayers) return;
        // The networks a stage does not train keep their weights: the estimator while the refiner trains, and the
        // networks a stage has not reached yet.
        var stage = _options.Stage;
        AiDotNet.Models.Parameters.ParameterSlotRole Role(bool trained)
            => trained ? AiDotNet.Models.Parameters.ParameterSlotRole.Trainable : AiDotNet.Models.Parameters.ParameterSlotRole.Frozen;
        RegisterParameterComponent("ABME.SymmetricMotion", _sbme, Role(stage is AbmeTrainingStage.SymmetricMotion or AbmeTrainingStage.Synthesis));
        RegisterParameterComponent("ABME.AsymmetricMotion", _abmr, Role(stage is AbmeTrainingStage.AsymmetricMotion or AbmeTrainingStage.Synthesis));
        RegisterParameterComponent("ABME.Synthesis", _synthesis, Role(stage == AbmeTrainingStage.Synthesis));
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
        var o = _options;
        _sbme = new AbmeSbmeNet<T>(o.SymmetricEncoderWidths, o.SymmetricDecoderWidths, o.SymmetricDisplacements);
        _abmr = new AbmeAbmrNet<T>(o.AsymmetricEncoderWidths, o.AsymmetricDecoderWidths, o.AsymmetricDisplacement);
        _synthesis = new AbmeSynthesisNet<T>(o.SynthesisWidths);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private static InvalidOperationException NotInitialized() => new("ABME's native networks are not initialized.");

    private AbmeSbmeNet<T> Sbme => _sbme ?? throw NotInitialized();

    private AbmeAbmrNet<T> Abmr => _abmr ?? throw NotInitialized();

    private AbmeSynthesisNet<T> Synthesis => _synthesis ?? throw NotInitialized();

    private static IReadOnlyList<Tensor<T>> Chunks(LazyWeightModule<T>? source)
        => source is null ? Array.Empty<Tensor<T>>() : source.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    /// <summary>The weights the stage trains.</summary>
    internal IReadOnlyList<Tensor<T>> StageParameters(AbmeTrainingStage stage) => stage switch
    {
        AbmeTrainingStage.SymmetricMotion => Chunks(_sbme),
        AbmeTrainingStage.AsymmetricMotion => Chunks(_abmr),
        _ => Chunks(_sbme).Concat(Chunks(_abmr)).Concat(Chunks(_synthesis)).ToList(),
    };

    #endregion

    #region Pipeline

    /// <summary>The motion estimation size of run.py: multiples of 64, or half of multiples of 128 from a height of 512.</summary>
    internal static (int Height, int Width) MotionSize(int height, int width)
    {
        bool half = height >= HalfSizeHeight;
        int divisor = half ? 128 : 64;
        int Round(int side) => (int)(Math.Ceiling(side / (double)divisor) * divisor * (half ? 0.5 : 1.0));
        return (Round(height), Round(width));
    }

    private Tensor<T> Units(Tensor<T> motion) => Engine.TensorMultiplyScalar(motion, NumOps.FromDouble(MotionUnit));

    /// <summary>Both motions at the estimation size, in pixels.</summary>
    internal readonly struct AbmeMotion
    {
        internal AbmeMotion(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> symmetric, Tensor<T> toFirst, Tensor<T> toSecond)
        {
            Frame0 = frame0;
            Frame1 = frame1;
            Symmetric = symmetric;
            ToFirst = toFirst;
            ToSecond = toSecond;
        }

        /// <summary>The first frame at the estimation size.</summary>
        internal Tensor<T> Frame0 { get; }

        /// <summary>The second frame at the estimation size.</summary>
        internal Tensor<T> Frame1 { get; }

        /// <summary>The symmetric motion toward the second frame; its negation points to the first.</summary>
        internal Tensor<T> Symmetric { get; }

        /// <summary>The asymmetric motion toward the first frame (the negated symmetric one without the refiner).</summary>
        internal Tensor<T> ToFirst { get; }

        /// <summary>The asymmetric motion toward the second frame.</summary>
        internal Tensor<T> ToSecond { get; }
    }

    // run.py up to the synthesis network, on frames [N, 3, H, W] in [0, 1]. With refine false only the estimator runs
    // (the asymmetric motions are then the symmetric ones); a fixed estimator's motion is detached from the tape.
    private AbmeMotion EstimateMotion(Tensor<T> frame0, Tensor<T> frame1, bool refine, bool fixedEstimator = false)
    {
        var (mh, mw) = MotionSize(frame0.Shape[2], frame0.Shape[3]);
        var f0 = AbmeOps.Resize(Engine, frame0, mh, mw, InterpolateMode.Bicubic);
        var f1 = AbmeOps.Resize(Engine, frame1, mh, mw, InterpolateMode.Bicubic);
        var sbm = Sbme.Forward(Engine.TensorConcatenate(new[] { f0, f1 }, 1));
        if (fixedEstimator) sbm = Engine.StopGradient(sbm);
        var symmetric = Units(AbmeOps.Resize(Engine, sbm, mh, mw));
        if (!refine) return new AbmeMotion(f0, f1, symmetric, Engine.TensorNegate(symmetric), symmetric);

        var (warped0, mask0) = AbmeOps.WarpWithMask(Engine, f0, Engine.TensorNegate(symmetric));
        var (warped1, mask1) = AbmeOps.WarpWithMask(Engine, f1, symmetric);
        Tensor<T> Expand(Tensor<T> mask) => Engine.TensorBroadcastTo(mask, warped0.Shape.ToArray());
        Tensor<T> OneMinus(Tensor<T> mask) => Engine.TensorAddScalar(Engine.TensorNegate(Expand(mask)), NumOps.One);
        var half = NumOps.FromDouble(0.5);
        var anchor = Engine.TensorAdd(Engine.TensorMultiplyScalar(Engine.TensorAdd(warped0, warped1), half),
            Engine.TensorMultiplyScalar(Engine.TensorAdd(Engine.TensorMultiply(warped1, OneMinus(mask0)), Engine.TensorMultiply(warped0, OneMinus(mask1))), half));
        var difference = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(warped1, warped0)), new[] { 1 }, true);
        var reliability = Engine.TensorExp(Engine.TensorMultiplyScalar(
            AbmeOps.Resize(Engine, difference, sbm.Shape[2], sbm.Shape[3]), NumOps.FromDouble(-MotionUnit)));

        // Both refinements in one batch: from the anchor toward the first frame along −V, toward the second along V.
        var pairs = Engine.TensorConcatenate(new[]
        {
            Engine.TensorConcatenate(new[] { anchor, f0 }, 1),
            Engine.TensorConcatenate(new[] { anchor, f1 }, 1),
        }, 0);
        var (refined, _) = Abmr.Forward(pairs, Engine.TensorConcatenate(new[] { Engine.TensorNegate(sbm), sbm }, 0),
            Engine.TensorConcatenate(new[] { reliability, reliability }, 0));
        int n = frame0.Shape[0];
        return new AbmeMotion(f0, f1, symmetric, Units(Engine.TensorNarrow(refined, 0, 0, n)), Units(Engine.TensorNarrow(refined, 0, n, n)));
    }

    // A motion at the estimation size carried to height x width, its components rescaled with the sides.
    private Tensor<T> Carry(Tensor<T> motion, int height, int width)
        => AbmeOps.ScaleFlow(Engine, AbmeOps.Resize(Engine, motion, height, width),
            width / (double)motion.Shape[3], height / (double)motion.Shape[2]);

    // The interpolated frame [N, 3, H, W] in [0, 1].
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3];
        var motion = EstimateMotion(frame0, frame1, refine: true);
        int sh = (h + 7) / 8 * 8, sw = (w + 7) / 8 * 8;
        var input = Engine.TensorConcatenate(new[]
        {
            AbmeOps.Resize(Engine, frame0, sh, sw),
            AbmeOps.Resize(Engine, frame1, sh, sw),
            Carry(motion.Symmetric, sh, sw),
            Carry(motion.ToSecond, sh, sw),
            Carry(motion.ToFirst, sh, sw),
        }, 1);
        return AbmeOps.Resize(Engine, Synthesis.Forward(input), h, w, InterpolateMode.Bicubic);
    }

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"ABME takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
        return (Engine.TensorNarrow(x, 1, 0, 3), Engine.TensorNarrow(x, 1, 3, 3), batched);
    }

    private Tensor<T> AsBatch(Tensor<T> frames) => frames.Rank == 4 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());

    private Tensor<T> RunPairs(Tensor<T> pairs)
    {
        var (frame0, frame1, batched) = SplitPairs(pairs);
        var output = Synthesize(frame0, frame1);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    #endregion

    #region Interpolation

    /// <inheritdoc />
    /// <exception cref="ArgumentOutOfRangeException">t is not 0.5: ABME's networks synthesize the midpoint only.</exception>
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (Math.Abs(t - 0.5) > 1e-9)
            throw new ArgumentOutOfRangeException(nameof(t), t, "ABME is trained for the midpoint only; t must be 0.5.");
        var pair = ConcatenateFeatures(PreprocessFrames(frame0), PreprocessFrames(frame1));
        var output = IsOnnxMode ? RunOnnxInference(pair) : Forward(pair);
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
        return RunPairs(input);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunPairs(PreprocessFrames(input));

    #endregion

    #region Training

    // The warps' validity masks are read back on the host, which a compiled plan would bake at trace time.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private static string Component(AbmeTrainingStage stage) => stage switch
    {
        AbmeTrainingStage.SymmetricMotion => "symmetric",
        AbmeTrainingStage.AsymmetricMotion => "asymmetric",
        _ => "synthesis",
    };

    private int[] Milestones(AbmeTrainingStage stage) => stage switch
    {
        AbmeTrainingStage.SymmetricMotion => _options.SymmetricMilestones,
        AbmeTrainingStage.AsymmetricMotion => _options.AsymmetricMilestones,
        _ => _options.SynthesisMilestones,
    };

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> StageOptimizer(AbmeTrainingStage stage)
    {
        if (_customOptimizer is not null) return _customOptimizer;
        if (_optimizers.TryGetValue(stage, out var existing)) return existing;
        // The paper states no betas, so Adam keeps PyTorch's (0.9, 0.999) and epsilon 1e-8.
        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            UseAMSGrad = false,
            LearningRateScheduler = new MultiStepLRScheduler(_options.LearningRate, Milestones(stage).ToArray(), _options.DecayRate),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        var verified = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer, Component(stage));
        _optimizers[stage] = verified;
        return verified;
    }

    // ρ plus the weighted census loss of one prediction against the target.
    private Tensor<T> Reconstruction(Tensor<T> prediction, Tensor<T> target)
    {
        var loss = AbmeOps.Charbonnier(Engine, prediction, target, _options.CharbonnierEpsilon);
        if (_options.CensusWeight > 0)
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(AbmeOps.CensusLoss(Engine, prediction, target), NumOps.FromDouble(_options.CensusWeight)));
        return loss;
    }

    /// <summary>
    /// The stage's loss on frame pairs [N, 3, H, W] and the intermediate frame, in [0, 1]: the photometric loss of both
    /// frames warped to the intermediate time at the estimation size (the motion stages), or that of the synthesized
    /// frame.
    /// </summary>
    internal Tensor<T> StageLoss(AbmeTrainingStage stage, Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        if (stage == AbmeTrainingStage.Synthesis) return Reconstruction(Synthesize(frame0, frame1), target);
        // In the refiner's stage the estimator is fixed: its motion enters the refiner as data.
        var motion = stage == AbmeTrainingStage.SymmetricMotion
            ? EstimateMotion(frame0, frame1, refine: false)
            : EstimateMotion(frame0, frame1, refine: true, fixedEstimator: true);
        var truth = AbmeOps.Resize(Engine, target, motion.Frame0.Shape[2], motion.Frame0.Shape[3], InterpolateMode.Bicubic);
        var warped0 = AbmeOps.Warp(Engine, motion.Frame0, motion.ToFirst);
        var warped1 = AbmeOps.Warp(Engine, motion.Frame1, motion.ToSecond);
        return Engine.TensorAdd(Reconstruction(warped0, truth), Reconstruction(warped1, truth));
    }

    /// <inheritdoc />
    /// <remarks>One step of the configured stage, on that stage's networks with that stage's optimizer.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var stage = _options.Stage;
        var target = AsBatch(NormalizeFrames(expectedOutput));
        return new[]
        {
            new TrainingObjective<T>(new[] { input, target }, data =>
            {
                var (frame0, frame1, _) = SplitPairs(PreprocessFrames(data[0]));
                return StageLoss(stage, frame0, frame1, data[1]);
            })
            {
                Optimizer = StageOptimizer(stage),
                Parameters = () => StageParameters(stage),
            },
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The configured stage's loss, the quantity a training step lowers.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        return StageLoss(_options.Stage, frame0, frame1, AsBatch(NormalizeFrames(target)))[0];
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? StageOptimizer(_options.Stage) : base.GetOrCreateBaseOptimizer();

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

    /// <summary>Frames are pixels on the 0-255 scale; the networks divide them by 255.</summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.Value;

    /// <summary>The interpolated frame comes back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ABME-Native" : "ABME-ONNX",
            Description = "ABME asymmetric bilateral motion estimation for frame interpolation (Park et al., ICCV 2021)",
            Complexity = _options.SymmetricEncoderWidths.Length
        };
        m.AdditionalInfo["Stage"] = _options.Stage.ToString();
        m.AdditionalInfo["SymmetricEncoderWidths"] = string.Join(",", _options.SymmetricEncoderWidths);
        m.AdditionalInfo["SynthesisWidths"] = string.Join(",", _options.SynthesisWidths);
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ABME<T>));
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
