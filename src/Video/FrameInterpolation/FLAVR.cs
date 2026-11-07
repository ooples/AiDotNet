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
/// FLAVR: flow-agnostic video representations for fast frame interpolation (Kalluri et al., WACV 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (tarun005/FLAVR, model/FLAVR_arch.py and model/resnet_3D.py): the context frames
/// are stacked into a clip and its per-channel mean subtracted; an R3D-18 encoder with feature gating and no batch norm
/// (a (3, 7, 7) stem at half size, then four stages, the middle two halving the size, time never strided) feeds a 3D
/// U-Net decoder of gated convolutions and (3, 4, 4) transposed convolutions joined to the encoder by concatenation.
/// The time steps are then stacked into channels, fused by a 1x1 convolution and mapped by a reflection-padded 7x7
/// convolution to every output frame, to which the mean is added back. Frames whose sides are not multiples of 8 are
/// zero-padded on the right and bottom after the mean is subtracted.
/// </para>
/// <para>
/// The input is the window of <see cref="FLAVROptions.NumInputFrames"/> frames stacked along the channels, earliest
/// first ([3·frames, H, W] or batched); the output is the <see cref="FLAVROptions.NumOutputFrames"/> evenly spaced
/// frames in the middle gap, stacked the same way. A frame pair alone is accepted as the window with each frame
/// repeated over its half, which is what <see cref="Interpolate"/> passes.
/// </para>
/// <para>
/// Training follows main.py: Adam (betas 0.9 and 0.99) at 2e-4 on the L1 loss over every output frame, the rate halved
/// by ReduceLROnPlateau (patience 5) on the validation loss once per epoch.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLAVR never estimates motion. It stacks four frames into a short clip, runs 3D (space and
/// time) convolutions over it in a U-Net, and directly paints the in-between frame or frames.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation",
    "https://arxiv.org/abs/2012.08512",
    Year = 2023,
    Authors = "Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, Du Tran")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.99, DecayRate = 0.5, StepSize = 5,
                Schedule = LearningRateSchedulerType.ReduceOnPlateau,
                Source = "Kalluri et al. 2023, Sec. 4: Adam with an initial learning rate of 2e-4, halved "
                        + "whenever training plateaus. The release (config.py, main.py) sets betas (0.9, 0.99) "
                        + "and ReduceLROnPlateau(factor 0.5, patience 5) stepped on the validation loss each epoch.")]
public partial class FLAVR<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly FLAVROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private FlavrNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates a FLAVR model in ONNX inference mode.</summary>
    public FLAVR(NeuralNetworkArchitecture<T> architecture, string modelPath, FLAVROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new FLAVROptions() : new FLAVROptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a FLAVR model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's unet_18 2x model and training when null.</param>
    /// <param name="optimizer">The optimizer; the release's Adam with the plateau schedule when null.</param>
    public FLAVR(NeuralNetworkArchitecture<T> architecture, FLAVROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new FLAVROptions() : new FLAVROptions(options);
        Options = _options;
        var o = _options;
        if (o.BaseChannels < 1 || o.LayerBlocks is not { Length: 4 } || o.LayerBlocks.Any(b => b < 1)
            || o.NumInputFrames < 2 || o.NumInputFrames % 2 != 0 || o.NumOutputFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "FLAVR needs a positive width, four positive stage depths, an even number of context frames and at least one output frame.");
        if (o.LearningRate <= 0 || o.Beta1 < 0 || o.Beta1 >= 1 || o.Beta2 < 0 || o.Beta2 >= 1
            || o.PlateauFactor <= 0 || o.PlateauFactor >= 1 || o.PlateauPatience < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The training settings must be valid.");
        _useNativeMode = true;
        SupportsArbitraryTimestep = false;
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        if (!_useNativeMode || _usingCustomLayers) return;
        RegisterParameterComponent("FLAVR.Network", _network);
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
        _network = new FlavrNetwork<T>(_options.BaseChannels, _options.LayerBlocks.ToArray(), _options.NumInputFrames, _options.NumOutputFrames);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private FlavrNetwork<T> Network => _network ?? throw new InvalidOperationException("FLAVR's native network is not initialized.");

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
            throw new ArgumentException($"FLAVR takes {frames} frames [{3 * frames}, H, W] or a pair [6, H, W], optionally batched; got [{string.Join(", ", window._shape)}].", nameof(window));
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
    /// <remarks>
    /// The pair is the window with each frame repeated over its half. FLAVR synthesizes only the times k/(outputs + 1),
    /// so <paramref name="t"/> must be one of them (0.5 for the 2x model).
    /// </remarks>
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        int outputs = _options.NumOutputFrames;
        int index = (int)Math.Round(t * (outputs + 1)) - 1;
        if (!(t > 0.0 && t < 1.0) || index < 0 || index >= outputs || Math.Abs((index + 1) / (double)(outputs + 1) - t) > 1e-9)
            throw new ArgumentOutOfRangeException(nameof(t), t, $"FLAVR synthesizes only the times k/{outputs + 1} for k in 1..{outputs}.");
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
        if (outputs > 1) output = Engine.TensorNarrow(output, output.Rank - 3, 3 * index, 3);
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
    /// <remarks>The stem's and every encoder stage's clip features, the decoder's full-size clip, the fused features
    /// and the synthesized frames.</remarks>
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
        // torch.optim.Adam with the release's betas, epsilon 1e-8 and no weight decay; ReduceLROnPlateau keeps torch's
        // defaults (relative threshold 1e-4, no cooldown, no floor).
        var adam = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = _options.Beta1,
            Beta2 = _options.Beta2,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            UseAMSGrad = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new ReduceOnPlateauScheduler(_options.LearningRate, _options.PlateauFactor, _options.PlateauPatience),
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adam);
        return _optimizer;
    }

    /// <summary>The release's loss: the L1 distance over every output frame [N, 3·outputs, H, W] in [0, 1].</summary>
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
            Name = _useNativeMode ? "FLAVR-Native" : "FLAVR-ONNX",
            Description = "FLAVR flow-agnostic 3D U-Net frame interpolation (Kalluri et al., WACV 2023)",
            Complexity = _options.LayerBlocks.Sum()
        };
        m.AdditionalInfo["BaseChannels"] = _options.BaseChannels.ToString();
        m.AdditionalInfo["LayerBlocks"] = string.Join(",", _options.LayerBlocks);
        m.AdditionalInfo["NumInputFrames"] = _options.NumInputFrames.ToString();
        m.AdditionalInfo["NumOutputFrames"] = _options.NumOutputFrames.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FLAVR<T>));
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
