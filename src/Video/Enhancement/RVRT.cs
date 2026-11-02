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
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// RVRT: recurrent video restoration transformer with guided deformable attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is the authors' RVRT for x4 video super-resolution (Liang et al., NeurIPS 2022; JingyunLiang/RVRT,
/// models/network_rvrt.py): SPyNet flow, an RSTB feature extractor over 1x8x8 windows, and four recurrent passes
/// (backward, forward, backward, forward) over two-frame clips. Each pass aligns the propagated clip to the current one
/// with guided deformable attention (offsets are residues on top of composed optical flows, and each query attends over
/// a deformable 3x3 window in both frames of the clip), then refines it with RSTB groups over 2x8x8 windows fed by
/// every earlier pass. An RSTB reconstruction and x4 pixel-shuffle upsampling add to the bilinearly upsampled input.
/// </para>
/// <para>
/// Training follows KAIR options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json: Charbonnier loss, Adam (0.9, 0.99)
/// at 4e-4 with SPyNet at 0.25 times the rate and frozen for the first 30,000 iterations, cosine annealing to 1e-7 over
/// 600,000 iterations. SPyNet is loaded pretrained in the release and starts from PyTorch's default initialization
/// here. The release's GELU is the erf form; the library's is the tanh approximation (they differ by under 1e-3).
/// </para>
/// <para>
/// <b>For Beginners:</b> RVRT works through a video two frames at a time, carrying what it learned forward and then
/// backward. To match each pair with the last one, it follows the estimated motion and then searches a small area
/// around each moved point for the best detail to borrow.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 10, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RVRT&lt;float&gt;(arch, new RVRTOptions());
/// model.Train(lowResClip, highResClip);         // [10, 3, 64, 64] -> [10, 3, 256, 256], pixel values 0-255
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
[ResearchPaper("Recurrent Video Restoration Transformer with Guided Deformable Attention",
    "https://arxiv.org/abs/2206.02146",
    Year = 2022,
    Authors = "Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, Luc Van Gool")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 4e-4, Beta2 = 0.99, ReferenceBatchSize = 8,
                Schedule = LearningRateSchedulerType.CosineAnnealing, MinLearningRate = 1e-7,
                Source = "Liang et al. 2022, Sec. 4.1; KAIR options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json: Adam "
                        + "lr 4e-4, betas (0.9, 0.99), SPyNet fix_lr_mul 0.25 and frozen for fix_iter = 30000, "
                        + "CosineAnnealingWarmRestarts over 600k iterations to eta_min 1e-7, batch 8.")]
public partial class RVRT<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    #region Fields

    private readonly RVRTOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;

    private RvrtBackbone<T>? _backbone;

    #endregion

    #region Constructors

    /// <summary>Creates an RVRT model in ONNX inference mode.</summary>
    public RVRT(NeuralNetworkArchitecture<T> architecture, string modelPath, RVRTOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new RVRTOptions();
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an RVRT model in native training mode.</summary>
    public RVRT(NeuralNetworkArchitecture<T> architecture, RVRTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>((options ?? new RVRTOptions()).CharbonnierEpsilon))
    {
        _options = options ?? new RVRTOptions();
        Options = _options;
        if (_options.EmbedDim < 1 || _options.NumHeads < 1 || _options.AttentionHeads < 1 || _options.DeformableGroups < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "EmbedDim and the head and group counts must be positive.");
        if (_options.WindowSize < 1 || _options.Depth < 1) throw new ArgumentOutOfRangeException(nameof(options), "WindowSize and Depth must be positive.");
        _useNativeMode = true;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        InitializeLayers();
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

        if (_options.NumFrames % RvrtBackbone<T>.ClipSize != 0)
            throw new InvalidOperationException("NumFrames must be a multiple of RVRT's two-frame clips.");
        if (_options.EmbedDim % _options.NumHeads != 0 || (2 * _options.EmbedDim) % _options.AttentionHeads != 0
            || (4 * _options.EmbedDim) % _options.DeformableGroups != 0)
            throw new InvalidOperationException("EmbedDim must divide into NumHeads, AttentionHeads and DeformableGroups.");
        _backbone = new RvrtBackbone<T>(Layers, _options.EmbedDim, _options.NumHeads, _options.WindowSize,
            new[] { _options.ExtractionBlocks, _options.PropagationBlocks, _options.ReconstructionBlocks }, _options.Depth,
            _options.MlpRatio, _options.DeformableGroups, _options.AttentionHeads, _options.MaxResidueMagnitude);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        Network(new Tensor<T>(new[] { 1, 2 * RvrtBackbone<T>.ClipSize, 3, 8, 8 }));
        MarkLayerShapesResolved();
    }

    #endregion

    #region Network

    private RvrtBackbone<T> Backbone
        => _backbone ?? throw new InvalidOperationException("RVRT's native layers are not initialized.");

    /// <summary>RVRT.forward: clips [N, T, 3, H, W] in [0, 1] to [N, T, 3, 4H, 4W].</summary>
    private Tensor<T> Network(Tensor<T> clips) => Backbone.Run(Engine, Layers, clips);

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
    protected override Tensor<T> Forward(Tensor<T> input)
        => _usingCustomLayers ? base.Forward(input) : RunClips(input);

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
    /// <remarks>Charbonnier loss over every output frame; SPyNet joins training after the freeze iterations.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
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
            Name = _useNativeMode ? "RVRT-Native" : "RVRT-ONNX",
            Description = "RVRT recurrent video restoration transformer (Liang et al., NeurIPS 2022)",
            Complexity = _options.PropagationBlocks * _options.Depth
        };
        m.AdditionalInfo["EmbedDim"] = _options.EmbedDim.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        m.AdditionalInfo["ClipSize"] = RvrtBackbone<T>.ClipSize.ToString();
        m.AdditionalInfo["ScaleFactor"] = "4";
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RVRT<T>));
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
