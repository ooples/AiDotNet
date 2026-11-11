using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// IART: a recurrent video super-resolution transformer whose frame alignment is learned implicit resampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is the authors' IART for x4 video super-resolution (Xu et al., "Enhancing Video Super-Resolution via
/// Implicit Resampling-based Alignment", CVPR 2024; kai422/IART, archs/iart_arch.py and archs/implicit_alignment.py). It
/// is PSRT-recurrent's network with the alignment replaced: SPyNet flow, a 3x3 embedding to 120 channels, and four
/// recurrent passes (backward, forward, backward, forward), each stacking the aligned previous and second-previous
/// outputs around the current features and running the stack through a SwinIRFM (three RSTBs of six window-attention
/// layers over 3x8x8 windows). Alignment is one shared implicit warp: every current pixel attends, with a single query,
/// over the 2x2 propagated pixels around the point its flow lands on, with sine codes of the window position and of the
/// sub-pixel offset, so the resampling kernel is learned instead of bilinear. The last pass is reconstructed with x4
/// pixel-shuffle upsampling and added to the bilinearly upsampled input.
/// </para>
/// <para>
/// Training follows options/IART_REDS_N16_600K.yml: Charbonnier loss (eps 1e-12 under the square root), Adam (0.9,
/// 0.99) at 2e-4 with SPyNet at 0.125 times the rate and frozen for the first 5,000 iterations, cosine annealing to 1e-7
/// over 600,000 iterations, 16-frame sequences, stochastic depth up to 0.1, and an exponential moving average of the
/// weights (decay 0.999) used for inference. SPyNet is loaded pretrained in the release and starts from PyTorch's
/// default initialization here. The release's GELU is the erf form; the library's is the tanh approximation.
/// </para>
/// <para>
/// <b>For Beginners:</b> To reuse detail from the previous frame, most video upscalers follow the motion and blend the
/// four nearest pixels with fixed weights, which blurs fine texture. IART lets each pixel look at those four neighbours
/// and decide for itself how much to take from each.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 8, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new IART&lt;float&gt;(arch, new IARTOptions());
/// model.Train(lowResClip, highResClip);         // [8, 3, 64, 64] -> [8, 3, 256, 256], pixel values 0-255
/// var upscaled = model.Upscale(lowResClip);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Enhancing Video Super-Resolution via Implicit Resampling-based Alignment",
    "https://arxiv.org/abs/2305.00163",
    Year = 2024,
    Authors = "Kai Xu, Ziwei Yu, Xin Wang, Michael Bi Mi, Angela Yao")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, Beta2 = 0.99, ReferenceBatchSize = 8,
                Schedule = LearningRateSchedulerType.CosineAnnealing, MinLearningRate = 1e-7,
                Source = "Xu et al. 2024, Sec. 4.1; kai422/IART options/IART_REDS_N16_600K.yml: "
                        + "Adam lr 2e-4, betas (0.9, 0.99), SPyNet flow_lr_mul 0.125 and frozen for fix_flow = 5000, "
                        + "CosineAnnealingRestartLR over 600k iterations to eta_min 1e-7, batch 1 on each of 8 GPUs.")]
public partial class IART<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    #region Fields

    private readonly IARTOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;

    private PsrtBackbone<T>? _backbone;
    private ParameterAverage<T>? _average;

    #endregion

    #region Constructors

    /// <summary>Creates an IART model in ONNX inference mode.</summary>
    public IART(NeuralNetworkArchitecture<T> architecture, string modelPath, IARTOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new IARTOptions();
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an IART model in native training mode.</summary>
    public IART(NeuralNetworkArchitecture<T> architecture, IARTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>((options ?? new IARTOptions()).CharbonnierEpsilon))
    {
        _options = options ?? new IARTOptions();
        Options = _options;
        if (_options.MidChannels < 1 || _options.EmbedDim < 1 || _options.NumHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MidChannels, EmbedDim and NumHeads must be positive.");
        if (_options.WindowSize < 1 || _options.Depths is null || _options.Depths.Length == 0 || _options.Depths.Any(d => d < 1))
            throw new ArgumentOutOfRangeException(nameof(options), "WindowSize and every RSTB depth must be positive.");
        if (_options.DropPathRate < 0 || _options.DropPathRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "DropPathRate must lie in [0, 1).");
        if (_options.PositionTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "PositionTemperature must be positive.");
        _useNativeMode = true;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        InitializeLayers();
        if (_options.EmaMomentum is double momentum) _average = new ParameterAverage<T>(momentum);
        _optimizer = optimizer;
    }

    #endregion

    #region Video Super-Resolution

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    #endregion

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

        if (_options.EmbedDim % _options.NumHeads != 0 || _options.EmbedDim % 2 != 0)
            throw new InvalidOperationException("EmbedDim must be even and divide into NumHeads.");
        var random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
        _backbone = new PsrtBackbone<T>(Layers, _options.MidChannels, _options.EmbedDim, _options.Depths, _options.NumHeads,
            new[] { PsrtAlignmentFrames, _options.WindowSize, _options.WindowSize }, _options.MlpRatio, _options.DropPathRate, random,
            layers => new IartImplicitWarp<T>(layers, _options.EmbedDim, _options.NumHeads, _options.PositionTemperature));
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        Network(new Tensor<T>(new[] { 1, PsrtAlignmentFrames, 3, 8, 8 }));
        MarkLayerShapesResolved();
    }

    #endregion

    #region Network

    /// <summary>Frames each alignment stacks (previous, current, second previous): the release's num_frames = 3.</summary>
    internal const int PsrtAlignmentFrames = 3;

    private PsrtBackbone<T> Backbone
        => _backbone ?? throw new InvalidOperationException("IART's native layers are not initialized.");

    /// <summary>IART.forward: clips [N, T, 3, H, W] in [0, 1] to [N, T, 3, 4H, 4W]; stochastic depth
    /// is active only in training mode.</summary>
    private Tensor<T> Network(Tensor<T> clips) => Backbone.Run(Engine, Layers, clips, IsTrainingMode);

    private Tensor<T> RunClips(Tensor<T> frames)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        var output = Network(clips);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    /// <summary>Test hook: SPyNet's flow from <paramref name="reference"/> to <paramref name="support"/>, [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> FlowForTest(Tensor<T> reference, Tensor<T> support)
        => Backbone.SpyNet.Flow(Engine, Layers, reference, support);

    /// <summary>Test hook: the implicit warp of <paramref name="propagated"/> to <paramref name="current"/> along
    /// <paramref name="flow"/>.</summary>
    internal Tensor<T> ImplicitWarpForTest(Tensor<T> propagated, Tensor<T> current, Tensor<T> flow)
        => Backbone.Aligner.Align(Engine, Layers, propagated, current, flow);

    /// <summary>Test hook: mmagic's flow_warp with zero padding.</summary>
    internal Tensor<T> WarpForTest(Tensor<T> image, Tensor<T> flow) => BasicVsrBlocks.FlowWarp(Engine, image, flow, GridSamplePadding.Zeros);

    /// <summary>Test hook: the network on frames already scaled to [0, 1].</summary>
    internal Tensor<T> UpscaleNormalizedForTest(Tensor<T> frames) => RunClips(frames);

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters
        => _backbone is null ? Array.Empty<Tensor<T>>() : _backbone.FlowParameters(Layers);

    #endregion

    #region Forward and training

    /// <inheritdoc />
    /// <remarks>The release evaluates its moving average of the weights (ema_decay 0.999).</remarks>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        return _average is { HasAverage: true }
            ? _average.WithAverage(TapeTrainingStep<T>.CollectParameters(Layers), () => RunClips(input))
            : RunClips(input);
    }

    /// <inheritdoc />
    protected override void OnTrainingStepEnded(bool succeeded)
    {
        base.OnTrainingStepEnded(succeeded);
        if (succeeded && !_usingCustomLayers) _average?.Update(TapeTrainingStep<T>.CollectParameters(Layers));
    }

    /// <inheritdoc />
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        state.Declare(nameof(IART<T>) + ".EmaWeights", () => _average?.ToVector(), restored => _average?.Restore(restored));
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input));

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildPaperOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // The rate mask is keyed by parameter tensor, so the lazily shaped convolutions must exist first.
        ResolveLazyLayerShapes();
        // Adam is AdamW without decay; AdamW carries the per-parameter rate mask that gives SPyNet its 0.125.
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.99,
            Epsilon = 1e-8,
            WeightDecay = 0.0,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new CosineAnnealingLRScheduler(
                _options.LearningRate, Math.Max(1, _options.CosineIterations), _options.MinLearningRate),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        var flow = new HashSet<Tensor<T>>(FlowParameters);
        var parameters = TapeTrainingStep<T>.CollectParameters(Layers);
        var rates = new List<T>();
        foreach (var p in parameters)
        {
            var rate = NumOps.FromDouble(flow.Contains(p) ? _options.SpyNetLearningRateMultiplier : 1.0);
            for (int i = 0; i < p.Length; i++) rates.Add(rate);
        }
        var options = (AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
        options.LearningRateMask = new Vector<T>(rates.ToArray());
        options.MaskParameters = parameters;
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
        return _optimizer;
    }

    private IReadOnlyList<Tensor<T>> TrainableParameters()
    {
        var all = TapeTrainingStep<T>.CollectParameters(Layers);
        if (_trainingSteps > _options.FlowFreezeIterations) return all;
        var flow = new HashSet<Tensor<T>>(FlowParameters);
        return all.Where(p => !flow.Contains(p)).ToList();
    }

    /// <inheritdoc />
    /// <remarks>Charbonnier loss over every output frame; SPyNet joins training after fix_flow iterations.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        // Train on one NumFrames-frame clip of a longer sequence, as the paper's data pipeline does.
        (input, expectedOutput) = TrainingClip(input, expectedOutput);
        _trainingSteps++;
        var optimizer = BuildPaperOptimizer();
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, data =>
                LossFunction.ComputeTapeLoss(RunClips(PreprocessFrames(data[0])), NormalizeFrames(data[1])))
            {
                Optimizer = optimizer,
                Parameters = TrainableParameters,
            }
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The Charbonnier loss training minimizes, between the network's output and the target both scaled to
    /// [0, 1]; a loss on the 0-255 outputs is a different objective whose minimizer need not coincide on small data.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        return LossFunction.ComputeTapeLoss(RunClips(PreprocessFrames(input)), NormalizeFrames(target))[0];
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? BuildPaperOptimizer() : base.GetOrCreateBaseOptimizer();

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

    /// <summary>Upscaled frames come back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "IART-Native" : "IART-ONNX",
            Description = "IART implicit-resampling alignment video SR transformer (Xu et al., CVPR 2024)",
            Complexity = _options.Depths.Sum()
        };
        m.AdditionalInfo["MidChannels"] = _options.MidChannels.ToString();
        m.AdditionalInfo["EmbedDim"] = _options.EmbedDim.ToString();
        m.AdditionalInfo["Depths"] = string.Join(",", _options.Depths);
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["WindowSize"] = $"{PsrtAlignmentFrames}x{_options.WindowSize}x{_options.WindowSize}";
        m.AdditionalInfo["ScaleFactor"] = "4";
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IART<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
