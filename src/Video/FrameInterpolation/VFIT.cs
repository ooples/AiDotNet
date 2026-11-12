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
/// VFIT: the video frame interpolation transformer (Shi, Xu, Liu, Chen and Yang, CVPR 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (zhshi0816/Video-Frame-Interpolation-Transformer: model/VFIT_B.py,
/// Sep_STS_Encoder.py, Sep_STS_Layer.py, cupy_module/adacof.py). The four frames are stacked into a clip and each
/// sample's per-channel mean subtracted; a 3x3x3 stem and residual block at full size feed four stages that each halve
/// the size with a strided 3D convolution and run Sep-STS blocks: spatial attention in 8x8 windows (shifted by half in
/// every second block) and then temporal attention across each pixel's frames, followed by an MLP. A 3D U-Net decoder of
/// transposed convolutions, joined to the encoder by concatenation, gives features at 1/8, 1/4 and 1/2 size, from which
/// AdaCoF synthesis predicts, per input frame, 5x5 kernels of weights and vertical and horizontal offsets at 1/4, 1/2
/// and full size, plus occlusion maps softmaxed over the frames; each frame, resized, is warped by its kernels and the
/// gated sum is added to the coarser prediction upsampled. The frame is the midpoint; frames need no padding.
/// </para>
/// <para>
/// The input is the window of <see cref="VFITOptions.NumInputFrames"/> frames stacked along the channels, earliest
/// first ([3·frames, H, W] or batched). A frame pair alone is accepted as the window with each frame repeated over its
/// half, which is what <see cref="Interpolate"/> passes.
/// </para>
/// <para>
/// Training follows main.py: Adamax (betas 0.9 and 0.999) on the L1 loss of the full-size frame, the rate stepped through
/// 2e-4, 1e-4, 5e-5, 2.5e-5, 5e-6 and 1e-6 at epochs 40, 60, 75, 85 and 95 of 100, batch 4 on 256x256 crops.
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIT looks at two frames before and two after the missing one. A transformer compares small
/// windows within each frame and the same pixel across the four frames, and the result decides, for every output pixel,
/// which nearby input pixels to blend and by how much.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VFIT: Video Frame Interpolation Transformer",
    "https://arxiv.org/abs/2111.13817",
    Year = 2022,
    Authors = "Zhihao Shi, Xiangyu Xu, Xiaohong Liu, Jun Chen, Ming-Hsuan Yang")]
[PaperOptimizer(OptimizerKind.Adamax, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, ReferenceBatchSize = 4, MinLearningRate = 1e-6,
                Schedule = LearningRateSchedulerType.Sequential,
                Source = "Shi et al. 2022, Sec. 4: AdaMax with beta1 0.9 and beta2 0.999, batch 4, 100 epochs, the rate "
                        + "decayed from 2e-4 to 1e-6. The release (main.py, config.py) steps it through 2e-4, 1e-4, 5e-5, "
                        + "2.5e-5, 5e-6 and 1e-6 at epochs 40, 60, 75, 85 and 95.")]
public partial class VFIT<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly VFITOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private VfitNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates a VFIT model in ONNX inference mode.</summary>
    public VFIT(NeuralNetworkArchitecture<T> architecture, string modelPath, VFITOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new VFITOptions() : new VFITOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a VFIT model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's VFIT-B and training when null.</param>
    /// <param name="optimizer">The optimizer; the release's Adamax with its stepped rate when null.</param>
    public VFIT(NeuralNetworkArchitecture<T> architecture, VFITOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new VFITOptions() : new VFITOptions(options);
        Options = _options;
        var o = _options;
        if (o.Widths is not { Length: 4 } || o.Widths.Any(c => c < 1) || o.Widths[3] < 2 || o.Widths[3] % 2 != 0 || o.NumInputFrames < 2 || o.NumInputFrames % 2 != 0
            || o.StageDepths is not { Length: 4 } || o.StageDepths.Any(d => d < 1) || o.StageHeads is not { Length: 4 }
            || Enumerable.Range(0, 4).Any(i => o.StageHeads[i] < 1 || o.Widths[3 - i] % o.StageHeads[i] != 0)
            || o.WindowSize < 2 || o.WindowSize % 2 != 0 || o.MlpRatio <= 0 || o.SynthesisChannels < 1 || o.KernelSize < 1 || o.KernelSize % 2 != 1 || o.Dilation < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "VFIT needs four positive widths (the last even), an even number of frames, four positive depths and heads dividing their widths, an even window, and an odd kernel.");
        if (o.LearningRate <= 0 || o.Beta1 < 0 || o.Beta1 >= 1 || o.Beta2 < 0 || o.Beta2 >= 1 || o.LearningRateMilestones is null || o.LearningRateSteps is null
            || o.LearningRateMilestones.Length != o.LearningRateSteps.Length || o.LearningRateMilestones.Any(m => m < 1)
            || o.LearningRateMilestones.Zip(o.LearningRateMilestones.Skip(1), (a, b) => b <= a).Any(x => x) || o.LearningRateSteps.Any(r => r <= 0))
            throw new ArgumentOutOfRangeException(nameof(options), "The training settings must be valid: one positive rate per increasing milestone.");
        _useNativeMode = true;
        SupportsArbitraryTimestep = false;
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        if (!_useNativeMode || _usingCustomLayers) return;
        RegisterParameterComponent("VFIT.Network", _network);
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
        _network = new VfitNetwork<T>(_options.Widths.ToArray(), _options.NumInputFrames, _options.StageDepths.ToArray(), _options.StageHeads.ToArray(),
            _options.WindowSize, _options.MlpRatio, _options.SynthesisChannels, _options.KernelSize, _options.Dilation);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private VfitNetwork<T> Network => _network ?? throw new InvalidOperationException("VFIT's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    // A window [3·frames, H, W] or [B, 3·frames, H, W] to its frames [N, 3, H, W]; a pair fills each half with its frame.
    private (Tensor<T>[] Frames, bool Batched) SplitWindow(Tensor<T> window)
    {
        bool batched = window.Rank == 4;
        var x = batched ? window : Engine.Reshape(window, new[] { 1 }.Concat(window._shape).ToArray());
        int frames = _options.NumInputFrames;
        if (x.Rank != 4 || (x.Shape[1] != 3 * frames && x.Shape[1] != 6))
            throw new ArgumentException($"VFIT takes {frames} frames [{3 * frames}, H, W] or a pair [6, H, W], optionally batched; got [{string.Join(", ", window._shape)}].", nameof(window));
        if (x.Shape[1] == 3 * frames)
            return (Enumerable.Range(0, frames).Select(k => Engine.TensorNarrow(x, 1, 3 * k, 3)).ToArray(), batched);
        var first = Engine.TensorNarrow(x, 1, 0, 3);
        var last = Engine.TensorNarrow(x, 1, 3, 3);
        return (Enumerable.Range(0, frames).Select(k => k < frames / 2 ? first : last).ToArray(), batched);
    }

    private Tensor<T> AsBatch(Tensor<T> frames) => frames.Rank == 4 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());

    private Tensor<T> RunWindow(Tensor<T> window)
    {
        var (frames, batched) = SplitWindow(window);
        var output = Network.Forward(frames);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    #endregion

    #region Interpolation

    /// <inheritdoc />
    /// <remarks>The pair is the window with each frame repeated over its half. VFIT synthesizes the midpoint only.</remarks>
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (t != 0.5)
            throw new ArgumentOutOfRangeException(nameof(t), t, "VFIT synthesizes the midpoint only (t = 0.5).");
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
            output = RunWindow(pair);
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
        return RunWindow(input);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunWindow(PreprocessFrames(input));

    /// <inheritdoc />
    /// <remarks>The stem's and every encoder stage's clip features and the frame after each synthesis scale.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frames, _) = SplitWindow(PreprocessFrames(input));
        var activations = new Dictionary<string, Tensor<T>>();
        activations["Synthesis"] = Network.Forward(frames, activations);
        return activations;
    }

    #endregion

    #region Training

    // Training runs through the release's objective on the eager tape.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // torch.optim.Adamax with the release's betas and epsilon 1e-8; adjust_learning_rate's table as constant pieces.
        // A scheduler at step s has seen s batches, and switches once s exceeds a milestone, so each is moved one back.
        var pieces = new List<ILearningRateScheduler> { new ConstantLRScheduler(_options.LearningRate) };
        pieces.AddRange(_options.LearningRateSteps.Select(rate => (ILearningRateScheduler)new ConstantLRScheduler(rate)));
        var adamax = new AdaMaxOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdaMaxOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = _options.Beta1,
            Beta2 = _options.Beta2,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false,
            LearningRateScheduler = new SequentialLRScheduler(pieces, _options.LearningRateMilestones.Select(m => m - 1).ToArray()),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamax);
        return _optimizer;
    }

    /// <summary>The release's loss: the L1 distance of the full-size frame [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> window, Tensor<T> target)
    {
        var (frames, _) = SplitWindow(window);
        var prediction = Network.Forward(frames);
        return Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(prediction, target)), Enumerable.Range(0, prediction.Rank).ToArray(), false);
    }

    /// <inheritdoc />
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var target = AsBatch(NormalizeFrames(expectedOutput));
        return new[]
        {
            new TrainingObjective<T>(new[] { input, target }, data => ReleaseLoss(PreprocessFrames(data[0]), data[1]))
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
        return ReleaseLoss(PreprocessFrames(input), AsBatch(NormalizeFrames(target)))[0];
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

    /// <summary>The interpolated frames come back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VFIT-Native" : "VFIT-ONNX",
            Description = "VFIT video frame interpolation transformer (Shi et al., CVPR 2022)",
            Complexity = _options.StageDepths.Sum()
        };
        m.AdditionalInfo["Widths"] = string.Join(",", _options.Widths);
        m.AdditionalInfo["StageDepths"] = string.Join(",", _options.StageDepths);
        m.AdditionalInfo["NumInputFrames"] = _options.NumInputFrames.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VFIT<T>));
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
