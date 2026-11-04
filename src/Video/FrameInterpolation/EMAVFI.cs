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
/// EMA-VFI: extracting motion and appearance via inter-frame attention for efficient video frame interpolation
/// (Zhang et al., CVPR 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (MCG-NJU/EMA-VFI, model/feature_extractor.py, flow_estimation.py, refine.py):
/// <list type="bullet">
/// <item><b>MotionFormer.</b> Both frames pass three convolution stages, then a cross-scale patch embedding and an
/// overlapping one lead into two stages of (shifted) window inter-frame attention at 1/8 and 1/16 size. Each window's
/// queries attend to the other frame's window: the attended values update its appearance, and the attention-weighted
/// coordinate embedding minus its own is its motion feature.</item>
/// <item><b>Multi-scale flow.</b> A head at 1/16 and one at 1/8 predict the flow pair from time t and a blending mask
/// from the motion features scaled by t and 1 − t and both frames' appearance features; the second refines the
/// first's.</item>
/// <item><b>Refinement.</b> A U-Net over the frames, warped frames, mask, flow and both appearance pyramids warped by
/// the flows predicts a residual added to the blend, clamped to [0, 1].</item>
/// </list>
/// Frames are padded by edge replication to multiples of 32 (InputPadder). Any time in [0, 1] can be synthesized.
/// </para>
/// <para>
/// Training follows train.py and Trainer.py: AdamW (weight decay 1e-4) with a 2000-step linear warm-up from 0 to 2e-4
/// and cosine decay to 2e-5 over 300 epochs, on the Laplacian-pyramid L1 loss of the refined frame plus half that of
/// each stage's warped blend.
/// </para>
/// <para>
/// <b>For Beginners:</b> EMA-VFI compares small windows of the two frames with attention: what each window attends to
/// in the other frame describes its appearance, and where it attends describes its motion. Both feed a coarse-to-fine
/// motion estimator and a refinement network that produce the in-between frame.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation",
    "https://arxiv.org/abs/2303.00440",
    Year = 2023,
    Authors = "Guozhen Zhang, Yuhan Zhu, Haonan Wang, Youxin Chen, Gangshan Wu, Limin Wang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, ReferenceBatchSize = 32,
                WeightDecay = 1e-4, WarmupSteps = 2000, MinLearningRate = 2e-5,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Zhang et al. 2023, Sec. 4.1: AdamW with beta1 0.9, beta2 0.999 and a weight "
                        + "decay of 1e-4, warming up over 2000 steps to a learning rate of 2e-4 and then "
                        + "cosine annealing over 300 epochs down to 2e-5. The release (train.py, README) "
                        + "warms up linearly from 0 and trains four processes at batch 8.")]
public partial class EMAVFI<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    // InputPadder's divisor: the backbone reaches 1/16 size and the U-Net halves it once more.
    private const int Divisor = 32;

    private readonly EMAVFIOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private EmaVfiNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates an EMA-VFI model in ONNX inference mode.</summary>
    public EMAVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, EMAVFIOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new EMAVFIOptions() : new EMAVFIOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an EMA-VFI model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's main model and training when null.</param>
    /// <param name="optimizer">The optimizer; the release's AdamW with warm-up and cosine decay when null.</param>
    public EMAVFI(NeuralNetworkArchitecture<T> architecture, EMAVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new EMAVFIOptions() : new EMAVFIOptions(options);
        Options = _options;
        var o = _options;
        if (o.FeatureChannels < 4 || o.FeatureChannels % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "F must be a positive multiple of 4, so the attention stages have F/4 and F/2 heads.");
        if (o.Depths is not { Length: 5 } || o.Depths.Any(d => d < 1) || 32 % o.Depths[3] != 0 || 32 % o.Depths[4] != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "EMA-VFI needs five positive depths, the attention stages' dividing 32 so each head gets whole motion channels.");
        if (o.WindowSize < 1 || o.MlpRatio <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The window and MLP ratio must be positive.");
        if (o.LearningRate <= 0 || o.MinLearningRate < 0 || o.WeightDecay < 0 || o.WarmupSteps < 0 || o.TotalIterations <= o.WarmupSteps
            || o.MergedWeight < 0 || o.LaplacianLevels < 1)
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
        RegisterParameterComponent("EMAVFI.Network", _network);
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
        _network = new EmaVfiNetwork<T>(_options.FeatureChannels, _options.WindowSize, _options.Depths.ToArray(), _options.MlpRatio);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private EmaVfiNetwork<T> Network => _network ?? throw new InvalidOperationException("EMA-VFI's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    /// <summary>InputPadder(divisor = 32): the left, right, top and bottom padding, centred.</summary>
    internal static (int Left, int Right, int Top, int Bottom) Padding(int height, int width)
    {
        int padH = ((height / Divisor + 1) * Divisor - height) % Divisor;
        int padW = ((width / Divisor + 1) * Divisor - width) % Divisor;
        return (padW / 2, padW - padW / 2, padH / 2, padH - padH / 2);
    }

    // The forward pass on frames [N, 3, H, W] in [0, 1], padded by edge replication; outputs cropped back.
    private EmaVfiOutput<T> Run(Tensor<T> frame0, Tensor<T> frame1, double t)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3];
        var (left, right, top, bottom) = Padding(h, w);
        if (left + right + top + bottom == 0) return Network.Forward(frame0, frame1, t);
        var pad = new[] { left, right, top, bottom };
        var output = Network.Forward(Engine.PadNd(frame0, pad, PadMode.Replicate, NumOps.Zero),
            Engine.PadNd(frame1, pad, PadMode.Replicate, NumOps.Zero), t);
        Tensor<T> Crop(Tensor<T> x) => Engine.TensorSlice(x, new[] { 0, 0, top, left }, new[] { x.Shape[0], x.Shape[1], h, w });
        var cropped = new EmaVfiOutput<T> { Prediction = Crop(output.Prediction) };
        cropped.Merged.AddRange(output.Merged.Select(Crop));
        cropped.Masks.AddRange(output.Masks.Select(Crop));
        cropped.Flows.AddRange(output.Flows.Select(Crop));
        return cropped;
    }

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"EMA-VFI takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
        return (Engine.TensorNarrow(x, 1, 0, 3), Engine.TensorNarrow(x, 1, 3, 3), batched);
    }

    private Tensor<T> AsBatch(Tensor<T> frames) => frames.Rank == 4 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());

    private Tensor<T> RunPairs(Tensor<T> pairs, double t)
    {
        var (frame0, frame1, batched) = SplitPairs(pairs);
        var output = Run(frame0, frame1, t).Prediction;
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
    /// <remarks>Each flow stage's flow, mask and blend, and the refined frame, at the midpoint.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        var output = Run(frame0, frame1, 0.5);
        var activations = new Dictionary<string, Tensor<T>>();
        for (int i = 0; i < output.Flows.Count; i++)
        {
            activations[$"Stage{i}.Flow"] = output.Flows[i];
            activations[$"Stage{i}.Mask"] = output.Masks[i];
            activations[$"Stage{i}.Merged"] = output.Merged[i];
        }
        activations["Refined"] = output.Prediction;
        return activations;
    }

    #endregion

    #region Training

    // The attention masks, coordinate grids and warps build their tensors on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // torch.optim.AdamW keeps its default betas (0.9, 0.999) and epsilon 1e-8; train.py sets the rate each step.
        var adamW = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new LinearWarmupScheduler(_options.LearningRate, _options.WarmupSteps, _options.TotalIterations, 0.0,
                LinearWarmupScheduler.DecayMode.Cosine, _options.MinLearningRate),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamW);
        return _optimizer;
    }

    /// <summary>Trainer.update's loss on frames [N, 3, H, W] in [0, 1]: LapLoss of the refined frame plus the weighted
    /// LapLoss of each stage's blend.</summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        var output = Run(frame0, frame1, 0.5);
        var loss = EmaOps.LaplacianLoss(Engine, output.Prediction, target, _options.LaplacianLevels);
        foreach (var merged in output.Merged)
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(EmaOps.LaplacianLoss(Engine, merged, target, _options.LaplacianLevels),
                NumOps.FromDouble(_options.MergedWeight)));
        return loss;
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
            Name = _useNativeMode ? "EMAVFI-Native" : "EMAVFI-ONNX",
            Description = "EMA-VFI inter-frame attention frame interpolation (Zhang et al., CVPR 2023)",
            Complexity = _options.Depths.Sum()
        };
        m.AdditionalInfo["FeatureChannels"] = _options.FeatureChannels.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        m.AdditionalInfo["Depths"] = string.Join(",", _options.Depths);
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(EMAVFI<T>));
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
