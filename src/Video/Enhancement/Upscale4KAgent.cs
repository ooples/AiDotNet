using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// 4KAgent (Zuo et al. 2025): an agent that upscales any image to 4K by planning super-resolution steps, running every
/// expert in its toolbox on each step, keeping the result a quality score prefers, and rolling back when no result
/// passes; here over AiDotNet's own video super-resolution networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The loop follows the release (taco-group/4KAgent, pipeline/the4kagent_pipeline.py):
/// <list type="bullet">
/// <item><b>Profile.</b> <see cref="Upscale4KAgentOptions.Upscale4K"/>, <see cref="Upscale4KAgentOptions.ScaleFactor"/>
/// and <see cref="Upscale4KAgentOptions.RestorePreference"/> are the profile's Upscale4K, ScaleFactor and
/// RestorePerference; the defaults are Gen4K-P.</item>
/// <item><b>Plan</b> (propose, get_target_factor). The factor is the smallest of 2, 4, 8 and 16 that brings the longer
/// side to 4000 pixels in 4K mode (1 when it is already there), or the profile's scale factor otherwise; 2 plans one
/// 2x step, 4 one 4x step, 8 a 2x then a 4x step, 16 two 4x steps.</item>
/// <item><b>Quality-driven mixture of experts</b> (execute_subtask). Every expert admitted by the preference upscales
/// the clip; each result is scored and the best kept. A best score below
/// <see cref="Upscale4KAgentOptions.RollbackThreshold"/> marks the step a failure.</item>
/// <item><b>Rollback</b> (roll_back, reschedule). A failed step backtracks to try another order of the remaining
/// subtasks. A plan of super-resolution steps has no other order, so every node is fully expanded and the release
/// compromises by continuing from the best descendant, the result the step already kept.</item>
/// <item><b>Color fix.</b> A super-resolution result whose longer side reaches 1024 pixels takes its input's
/// per-channel mean and standard deviation (adain_color_fix).</item>
/// </list>
/// </para>
/// <para>
/// Native substitutions, each because the release's component is a hosted model AiDotNet does not carry: the experts
/// are BasicVSR and IconVSR (fidelity) and RealBasicVSR and RealViformer (perception) in place of the release's image
/// toolbox (SwinIR, HAT, DiffBIR, OSEDiff, ...); a 2x step is a 4x expert followed by MATLAB's antialiased bicubic
/// halving, since every native expert is 4x; the reflection score is the release's NIQE term, 1 − NIQE / 10, averaged
/// over the frames, in place of HPSv2 plus the mean of CLIPIQA+, MANIQA, MUSIQ and that term; the perception agent is
/// absent because super-resolution is the only subtask the native toolbox serves.
/// </para>
/// <para>
/// Training trains the toolbox: each expert takes a step on its own paper's first-stage loss with its own paper's
/// optimizer against the 4x target. A target at a larger multiple of 4 is area-averaged down to 4x.
/// </para>
/// <para>
/// <b>For Beginners:</b> instead of one upscaling network, the agent runs several, measures which output looks most
/// natural, keeps that one, and repeats until the clip is large enough.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("4KAgent: Agentic Any Image to 4K Super-Resolution",
    "https://arxiv.org/abs/2507.07105",
    Year = 2025,
    Authors = "Yushen Zuo, Qi Zheng, Mingyang Wu, Xinrui Jiang, Renjie Li, Jian Wang, Yide Zhang, Gengchen Mai, "
            + "Lihong V. Wang, James Zou, Xiaoyu Wang, Ming-Hsuan Yang, Zhengzhong Tu")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, Beta2 = 0.99, Component = "basicvsr",
                Schedule = LearningRateSchedulerType.CosineAnnealing, MinLearningRate = 1e-7,
                Source = "4KAgent trains no network; its BasicVSR expert keeps configs/basicvsr/basicvsr_2xb4_reds4.py: "
                        + "Adam 2e-4, betas (0.9, 0.99), SPyNet lr_mult 0.125, cosine to 1e-7 over 300k iterations.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, Beta2 = 0.99, Component = "iconvsr",
                Schedule = LearningRateSchedulerType.CosineAnnealing, MinLearningRate = 1e-7,
                Source = "The IconVSR expert keeps configs/iconvsr/iconvsr_2xb4_reds4.py: BasicVSR's recipe.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99, Component = "realbasicvsr",
                Source = "The RealBasicVSR expert keeps realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py: "
                        + "Adam 1e-4, betas (0.9, 0.99), no schedule.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99, Component = "realviformer",
                Source = "The RealViformer expert keeps its reconstruction stage: Adam 1e-4, betas (0.9, 0.99).")]
public partial class Upscale4KAgent<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    // The longer side 4K mode aims for, and the release's AdaIN epsilon.
    private const int FourKSide = 4000;
    private const double ColorFixEpsilon = 1e-5;

    private readonly Upscale4KAgentOptions _options;
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly List<FourKAgentExpert<T>> _experts = new();
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _customOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;

    /// <summary>Creates a 4KAgent in ONNX inference mode.</summary>
    public Upscale4KAgent(NeuralNetworkArchitecture<T> architecture, string modelPath, Upscale4KAgentOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new Upscale4KAgentOptions() : new Upscale4KAgentOptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.Upscale4K ? 0 : _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a 4KAgent over its native experts.</summary>
    /// <param name="architecture">The architecture.</param>
    /// <param name="options">The profile and the experts' options.</param>
    /// <param name="optimizer">An optimizer every expert steps with in place of its paper's, or null for the papers'.</param>
    public Upscale4KAgent(NeuralNetworkArchitecture<T> architecture, Upscale4KAgentOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _customOptimizer = optimizer;
        _options = options is null ? new Upscale4KAgentOptions() : new Upscale4KAgentOptions(options);
        Options = _options;
        if (!_options.Upscale4K && _options.ScaleFactor is not (2 or 4 or 8 or 16))
            throw new ArgumentOutOfRangeException(nameof(options), "ScaleFactor must be one of 2, 4, 8 or 16.");
        _useNativeMode = true;
        // In 4K mode the factor depends on the input, so the family's fixed-factor law does not apply.
        ScaleFactor = _options.Upscale4K ? 0 : _options.ScaleFactor;
        InitializeLayers();
    }

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
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
        // Only the experts the preference admits are built, so the agent carries no network it can never choose.
        var preference = _options.RestorePreference;
        bool fidelity = preference != FourKAgentRestorePreference.Perception;
        bool perception = preference != FourKAgentRestorePreference.Fidelity;
        if (fidelity)
        {
            _experts.Add(new FourKAgentBasicVsr<T>(Layers, _options.BasicVsr));
            _experts.Add(new FourKAgentIconVsr<T>(Layers, _options.IconVsr));
        }
        if (perception)
        {
            _experts.Add(new FourKAgentRealBasicVsr<T>(Layers, _options.RealBasicVsr));
            _experts.Add(new FourKAgentRealViformer<T>(Layers, _options.RealViformer));
        }
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        foreach (var expert in _experts) expert.Run(Engine, Layers, new Tensor<T>(new[] { 1, 2, 3, 32, 32 }));
        MarkLayerShapesResolved();
    }

    /// <summary>The toolbox's expert names, in the order they compete.</summary>
    internal IReadOnlyList<string> ExpertNames => _experts.Select(e => e.Name).ToList();

    /// <summary>Every parameter of the named expert.</summary>
    internal IReadOnlyList<Tensor<T>> ExpertParameters(string name)
        => _experts.Single(e => e.Name == name).Parameters(Layers);

    #endregion

    #region Agent

    /// <summary>One executed step: the subtask's factor, the expert kept, its score, and whether it passed.</summary>
    internal readonly struct AgentStep
    {
        internal AgentStep(int factor, string expert, double score, bool success)
        {
            Factor = factor;
            Expert = expert;
            Score = score;
            Success = success;
        }

        internal int Factor { get; }
        internal string Expert { get; }
        internal double Score { get; }
        internal bool Success { get; }
    }

    /// <summary>The steps of the last clip the agent processed.</summary>
    internal IReadOnlyList<AgentStep> LastSteps { get; private set; } = Array.Empty<AgentStep>();

    /// <summary>get_target_factor: the upscaling the profile asks of an image whose longer side is
    /// <paramref name="maxSide"/>.</summary>
    internal static int TargetFactor(int maxSide, bool upscale4K, int scaleFactor)
    {
        if (!upscale4K) return scaleFactor;
        if (maxSide >= FourKSide) return 1;
        foreach (int factor in new[] { 2, 4, 8 })
            if (maxSide * factor >= FourKSide) return factor;
        return 16;
    }

    /// <summary>propose's add_sr_tasks: the super-resolution steps that make up a factor, as per-step factors.</summary>
    internal static IReadOnlyList<int> PlanSteps(int factor) => factor switch
    {
        1 => Array.Empty<int>(),
        2 => new[] { 2 },
        4 => new[] { 4 },
        8 => new[] { 2, 4 },
        16 => new[] { 4, 4 },
        _ => throw new ArgumentOutOfRangeException(nameof(factor), "The factor must be 1, 2, 4, 8 or 16."),
    };

    /// <summary>The reflection score of a clip [F, 3, H, W] in [0, 1]: the frames' mean of 1 − NIQE / 10, NaN when no
    /// frame holds a 96x96 block.</summary>
    internal double QualityScore(Tensor<T> clip)
    {
        int frames = clip.Shape[0], h = clip.Shape[2], w = clip.Shape[3], size = 3 * h * w;
        double sum = 0;
        int counted = 0;
        for (int f = 0; f < frames; f++)
        {
            var rgb = new double[size];
            for (int i = 0; i < size; i++) rgb[i] = NumOps.ToDouble(clip[f * size + i]);
            double niqe = Niqe.Score(rgb, h, w);
            if (double.IsNaN(niqe)) continue;
            sum += 1 - niqe / 10;
            counted++;
        }
        return counted == 0 ? double.NaN : sum / counted;
    }

    // execute_subtask over a clip [F, 3, H, W] in [0, 1]: every expert, the best score kept, the color fix.
    private Tensor<T> RunClip(Tensor<T> clip, List<AgentStep> steps)
    {
        var current = clip;
        int maxSide = Math.Max(clip.Shape[2], clip.Shape[3]);
        foreach (int factor in PlanSteps(TargetFactor(maxSide, _options.Upscale4K, _options.ScaleFactor)))
        {
            Tensor<T>? best = null;
            string bestName = string.Empty;
            double bestScore = double.NegativeInfinity;
            var batched = Engine.Reshape(current, new[] { 1 }.Concat(current._shape).ToArray());
            foreach (var expert in _experts)
            {
                var output = expert.Run(Engine, Layers, batched);
                var result = Engine.TensorClamp(Engine.Reshape(output, output._shape.Skip(1).ToArray()), NumOps.Zero, NumOps.One);
                if (factor == 2) result = HalveFrames(result);
                double score = QualityScore(result);
                // An unscorable result ranks last, so the toolbox order decides among unscorable results.
                double rank = double.IsNaN(score) ? double.NegativeInfinity : score;
                if (best is null || rank > bestScore)
                {
                    best = result;
                    bestName = expert.Name;
                    bestScore = rank;
                }
            }
            if (best is null) throw new InvalidOperationException("4KAgent's toolbox is empty.");
            bool success = !double.IsNegativeInfinity(bestScore) && bestScore >= _options.RollbackThreshold;
            if (Math.Max(best.Shape[2], best.Shape[3]) >= _options.ColorFixMinSide) best = AdainColorFix(best, current);
            // A failed super-resolution step rolls back to a fully expanded node and compromises on its best descendant,
            // which is this step's best result, so the loop continues from it.
            steps.Add(new AgentStep(factor, bestName, double.IsNegativeInfinity(bestScore) ? double.NaN : bestScore, success));
            current = best;
        }
        return current;
    }

    // A 2x step from a 4x expert: MATLAB's imresize(x, 0.5) on every plane.
    private Tensor<T> HalveFrames(Tensor<T> frames)
    {
        int f = frames.Shape[0], c = frames.Shape[1], h = frames.Shape[2], w = frames.Shape[3];
        int oh = (h + 1) / 2, ow = (w + 1) / 2;
        var result = new Tensor<T>(new[] { f, c, oh, ow });
        for (int p = 0; p < f * c; p++)
        {
            var plane = new double[h, w];
            for (int r = 0; r < h; r++)
                for (int x = 0; x < w; x++) plane[r, x] = NumOps.ToDouble(frames[(p * h + r) * w + x]);
            var half = Niqe.Halve(plane);
            for (int r = 0; r < oh; r++)
                for (int x = 0; x < ow; x++) result[(p * oh + r) * ow + x] = NumOps.FromDouble(half[r, x]);
        }
        return result;
    }

    /// <summary>adain_color_fix: each frame and channel of <paramref name="target"/> takes the mean and standard deviation
    /// (unbiased variance plus 1e-5) of the same frame and channel of <paramref name="source"/>, clamped to [0, 1].</summary>
    internal Tensor<T> AdainColorFix(Tensor<T> target, Tensor<T> source)
    {
        int f = target.Shape[0], c = target.Shape[1];
        int targetPlane = target.Shape[2] * target.Shape[3], sourcePlane = source.Shape[2] * source.Shape[3];
        (double Mean, double Std) Stats(Tensor<T> x, int offset, int count)
        {
            double mean = 0;
            for (int i = 0; i < count; i++) mean += NumOps.ToDouble(x[offset + i]);
            mean /= count;
            double variance = 0;
            for (int i = 0; i < count; i++)
            {
                double d = NumOps.ToDouble(x[offset + i]) - mean;
                variance += d * d;
            }
            return (mean, Math.Sqrt(variance / Math.Max(1, count - 1) + ColorFixEpsilon));
        }
        var result = new Tensor<T>(target._shape);
        for (int p = 0; p < f * c; p++)
        {
            var (tm, ts) = Stats(target, p * targetPlane, targetPlane);
            var (sm, ss) = Stats(source, p * sourcePlane, sourcePlane);
            for (int i = 0; i < targetPlane; i++)
            {
                double v = (NumOps.ToDouble(target[p * targetPlane + i]) - tm) / ts * ss + sm;
                result[p * targetPlane + i] = NumOps.FromDouble(Math.Min(Math.Max(v, 0.0), 1.0));
            }
        }
        return result;
    }

    // [F, 3, h, w] or [B, F, 3, h, w] in [0, 1] to the same layout at the planned size.
    private Tensor<T> RunVideo(Tensor<T> frames)
    {
        using var noGrad = new NoGradScope<T>();
        var steps = new List<AgentStep>();
        if (frames.Rank == 4)
        {
            var single = RunClip(frames, steps);
            LastSteps = steps;
            return single;
        }
        var outputs = new List<Tensor<T>>();
        for (int b = 0; b < frames.Shape[0]; b++)
        {
            steps = new List<AgentStep>();
            var clip = RunClip(Engine.Reshape(Engine.TensorNarrow(frames, 0, b, 1), frames._shape.Skip(1).ToArray()), steps);
            outputs.Add(Engine.Reshape(clip, new[] { 1 }.Concat(clip._shape).ToArray()));
        }
        LastSteps = steps;
        return Engine.TensorConcatenate(outputs.ToArray(), 0);
    }

    #endregion

    #region Forward and training

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
        => _usingCustomLayers ? base.Forward(input) : RunVideo(input);

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunVideo(PreprocessFrames(input));

    // The agent's choice reads host-side quality scores, which a compiled plan would bake at trace time.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ExpertOptimizer(FourKAgentExpert<T> expert)
    {
        if (_customOptimizer is not null) return _customOptimizer;
        if (_optimizers.TryGetValue(expert.Name, out var existing)) return existing;
        // The rate mask is keyed by parameter tensor, so the lazily shaped convolutions must exist first.
        ResolveLazyLayerShapes();
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer;
        if (expert.Cosine is { } cosine)
        {
            // Adam is AdamW without decay; AdamW carries the per-parameter rate mask that gives SPyNet its multiplier.
            var adamW = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = expert.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.99,
                Epsilon = 1e-8,
                WeightDecay = 0.0,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                EnableGradientClipping = false,
                LearningRateScheduler = new CosineAnnealingLRScheduler(expert.LearningRate, Math.Max(1, cosine.Iterations), cosine.Minimum),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
            var flow = new HashSet<Tensor<T>>(expert.Flow(Layers));
            var parameters = expert.Parameters(Layers);
            var rates = new List<T>();
            foreach (var p in parameters)
            {
                var rate = NumOps.FromDouble(flow.Contains(p) ? expert.FlowRateMultiplier : 1.0);
                for (int i = 0; i < p.Length; i++) rates.Add(rate);
            }
            var options = (AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)adamW.GetOptions();
            options.LearningRateMask = new Vector<T>(rates.ToArray());
            options.MaskParameters = parameters;
            optimizer = adamW;
        }
        else
        {
            optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = expert.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.99,
                Epsilon = 1e-8,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
            });
        }
        var verified = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer, expert.Name);
        _optimizers[expert.Name] = verified;
        return verified;
    }

    // Clips [B, F, 3, h, w] in [0, 1], from either layout.
    private Tensor<T> AsClips(Tensor<T> frames)
        => frames.Rank == 5 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());

    /// <summary>The experts' 4x target: the target in [0, 1], area-averaged when it is a larger multiple of 4x.</summary>
    internal Tensor<T> ExpertTarget(Tensor<T> input, Tensor<T> normalizedTarget)
    {
        int rank = input.Rank, h = input.Shape[rank - 2], w = input.Shape[rank - 1];
        int th = normalizedTarget.Shape[normalizedTarget.Rank - 2], tw = normalizedTarget.Shape[normalizedTarget.Rank - 1];
        if (th % (4 * h) != 0 || tw % (4 * w) != 0 || th / (4 * h) != tw / (4 * w))
            throw new ArgumentException($"The target ({th}x{tw}) must be the same multiple of the experts' 4x output ({4 * h}x{4 * w}).");
        int k = th / (4 * h);
        if (k == 1) return normalizedTarget;
        int lead = normalizedTarget.Length / (th * tw);
        var pooled = Engine.AvgPool2D(Engine.Reshape(normalizedTarget, new[] { lead, 1, th, tw }), k, k, 0, false);
        var shape = normalizedTarget.Shape.ToArray();
        shape[shape.Length - 2] = 4 * h;
        shape[shape.Length - 1] = 4 * w;
        return Engine.Reshape(pooled, shape);
    }

    /// <inheritdoc />
    /// <remarks>Each expert steps on its own paper's first-stage loss with its own optimizer.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        int step = ++_trainingSteps;
        var target = AsClips(ExpertTarget(input, NormalizeFrames(expectedOutput)));
        return _experts.Select(expert => new TrainingObjective<T>(new[] { input, target },
                data => expert.Loss(Engine, Layers, AsClips(PreprocessFrames(data[0])), data[1]))
            {
                Optimizer = ExpertOptimizer(expert),
                Parameters = () => expert.Trainable(Layers, step),
            }).ToList();
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The sum of the experts' training losses, the quantity a training step lowers.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var clips = AsClips(PreprocessFrames(input));
        var truth = AsClips(ExpertTarget(input, NormalizeFrames(target)));
        T total = NumOps.Zero;
        foreach (var expert in _experts) total = NumOps.Add(total, expert.Loss(Engine, Layers, clips, truth)[0]);
        return total;
    }

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

    /// <summary>Frames are pixels on the 0-255 scale; the agent divides them by 255.</summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.Value;

    /// <summary>Upscaled frames come back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Upscale4KAgent-Native" : "Upscale4KAgent-ONNX",
            Description = "4KAgent agentic any-resolution to 4K super-resolution over native experts (Zuo et al. 2025)",
            Complexity = _experts.Count
        };
        m.AdditionalInfo["Upscale4K"] = _options.Upscale4K.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        m.AdditionalInfo["RestorePreference"] = _options.RestorePreference.ToString();
        m.AdditionalInfo["Experts"] = string.Join(",", _experts.Select(e => e.Name));
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Upscale4KAgent<T>));
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
