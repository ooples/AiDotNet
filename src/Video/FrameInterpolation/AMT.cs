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
/// AMT: all-pairs multi-field transforms for efficient frame interpolation (Li et al., CVPR 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (MCG-NKU/AMT, networks/AMT-S.py and AMT-L.py):
/// <list type="bullet">
/// <item><b>Bidirectional all-pairs correlation.</b> A RAFT feature encoder at 1/8 size; the correlation of every pixel
/// pair and its transpose, each pooled into a four-level pyramid.</item>
/// <item><b>Coarse-to-fine flow decoding.</b> An IFRNet context encoder and decoders from 1/16 size predict both flows
/// from the intermediate time and a feature; at 1/8, 1/4 and 1/2 size the flows, extended linearly to 0→1 and 1→0, look
/// up both correlation pyramids and an update block refines flows and feature.</item>
/// <item><b>Multi-field synthesis.</b> The last decoder predicts several flow pairs, masks and residuals per pixel at
/// full size; each pair warps both frames, the blends are averaged and corrected by a small convolution.</item>
/// </list>
/// Frames are padded by edge replication to multiples of 16 (InputPadder) and the mean of both frames is subtracted
/// and restored. Any time in (0, 1) can be synthesized.
/// </para>
/// <para>
/// Training follows trainers/base_trainer.py and cfgs/AMT-S.yaml at the midpoint: AdamW at 2e-4 with cosine decay to
/// 2e-5, on the Charbonnier and ternary (census) losses. The release's third term, the multi-field flow loss, needs
/// pseudo ground-truth flows from a pretrained flow network that a frame-pair → frame sample does not carry, so it is
/// not applied.
/// </para>
/// <para>
/// <b>For Beginners:</b> AMT compares every pixel of one frame with every pixel of the other to find where things
/// moved, predicts several candidate motions per pixel, and blends the frames moved along each into the in-between
/// frame.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation",
    "https://arxiv.org/abs/2304.09790",
    Year = 2023,
    Authors = "Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, Ming-Ming Cheng")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, ReferenceBatchSize = 24, WeightDecay = 0.0,
                MinLearningRate = 2e-5,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Li et al. 2023, Training Details: AMT is trained on Vimeo90K for 300 epochs "
                        + "with AdamW at a total batch size of 24, the learning rate following a cosine "
                        + "attenuation schedule from 2e-4 to 2e-5. The release (cfgs/AMT-S.yaml, "
                        + "trainers/base_trainer.py) sets weight decay 0 and steps the cosine per iteration.")]
public partial class AMT<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    // InputPadder's divisor: the context encoder reaches 1/16 size.
    private const int Divisor = 16;

    private readonly AMTOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private AmtNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates an AMT model in ONNX inference mode.</summary>
    public AMT(NeuralNetworkArchitecture<T> architecture, string modelPath, AMTOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new AMTOptions() : new AMTOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an AMT model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; AMT-S and the release's training when null.</param>
    /// <param name="optimizer">The optimizer; the release's AdamW with cosine decay when null.</param>
    public AMT(NeuralNetworkArchitecture<T> architecture, AMTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new AMTOptions() : new AMTOptions(options);
        Options = _options;
        var o = _options;
        if (o.Channels is not { Length: 4 } || o.Channels.Any(c => c < 1) || o.SkipChannels < 1
            || o.Channels.Any(c => c * 2 <= o.SkipChannels) || o.NumFlowFields < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "AMT needs four positive context widths, side channels below twice each, and flow fields.");
        if (o.FeatureEncoderWidths is not { Length: 3 } || o.FeatureEncoderWidths.Any(c => c < 4 || (!o.LargeEncoders && c % 4 != 0)) || o.FeatureDim < 1
            || o.NumCorrelationLevels < 1 || o.CorrelationRadius < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The feature encoder needs three widths (multiples of 4 for bottlenecks) and a correlation pyramid.");
        if (o.UpdateHiddenDim < 1 || o.UpdateFlowDim < 1 || o.UpdateCorrelationDim < 1 || o.UpdateSecondCorrelationDim < 0
            || o.UpdateFusionDim < 1 || o.CombineKernelSize < 1 || o.CombineKernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The update block widths and the odd combination kernel must be valid.");
        if (o.LearningRate <= 0 || o.MinLearningRate < 0 || o.TotalIterations < 1 || o.WeightDecay < 0 || o.TernaryWeight < 0)
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
        RegisterParameterComponent("AMT.Network", _network);
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
        _network = new AmtNetwork<T>(new AmtArchitecture
        {
            Channels = o.Channels.ToArray(),
            SkipChannels = o.SkipChannels,
            NumFlows = o.NumFlowFields,
            CorrelationRadius = o.CorrelationRadius,
            CorrelationLevels = o.NumCorrelationLevels,
            FeatureWidths = o.FeatureEncoderWidths.ToArray(),
            FeatureDim = o.FeatureDim,
            Large = o.LargeEncoders,
            Hidden = o.UpdateHiddenDim,
            FlowDim = o.UpdateFlowDim,
            CorrDim = o.UpdateCorrelationDim,
            CorrDim2 = o.UpdateSecondCorrelationDim > 0 ? o.UpdateSecondCorrelationDim : null,
            FcDim = o.UpdateFusionDim,
            CombineKernel = o.CombineKernelSize,
        });
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private AmtNetwork<T> Network => _network ?? throw new InvalidOperationException("AMT's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    /// <summary>InputPadder: the left, right, top and bottom padding that brings a side to a multiple of 16.</summary>
    internal static (int Left, int Right, int Top, int Bottom) Padding(int height, int width)
    {
        int padH = ((height / Divisor + 1) * Divisor - height) % Divisor;
        int padW = ((width / Divisor + 1) * Divisor - width) % Divisor;
        return (padW / 2, padW - padW / 2, padH / 2, padH - padH / 2);
    }

    // The frame at time t from frames [N, 3, H, W] in [0, 1]: padded by edge replication, synthesized, cropped back.
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1, double t)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3];
        var (left, right, top, bottom) = Padding(h, w);
        if (left + right + top + bottom == 0) return Network.Forward(frame0, frame1, t);
        var pad = new[] { left, right, top, bottom };
        var output = Network.Forward(Engine.PadNd(frame0, pad, PadMode.Replicate, NumOps.Zero),
            Engine.PadNd(frame1, pad, PadMode.Replicate, NumOps.Zero), t);
        return Engine.TensorSlice(output, new[] { 0, 0, top, left }, new[] { output.Shape[0], 3, h, w });
    }

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"AMT takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
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
    /// <exception cref="ArgumentOutOfRangeException">t is not strictly between 0 and 1, where the release's linear
    /// extension of the flows divides by t and 1 − t.</exception>
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (!(t > 0.0 && t < 1.0))
            throw new ArgumentOutOfRangeException(nameof(t), t, "AMT synthesizes times strictly between the two frames.");
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
    /// <remarks>The flow fields, masks and output of the midpoint synthesis.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        return new Dictionary<string, Tensor<T>> { ["Synthesis"] = RunPairs(PreprocessFrames(input), 0.5) };
    }

    #endregion

    #region Training

    // The correlation lookups build their sampling grids on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // torch.optim.AdamW keeps its default betas (0.9, 0.999) and epsilon 1e-8.
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
            LearningRateScheduler = new CosineAnnealingLRScheduler(_options.LearningRate, _options.TotalIterations, _options.MinLearningRate),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamW);
        return _optimizer;
    }

    /// <summary>The release's loss on frames [N, 3, H, W] in [0, 1]: Charbonnier plus the weighted ternary loss.</summary>
    internal Tensor<T> ReconstructionLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        var prediction = Synthesize(frame0, frame1, 0.5);
        var loss = AmtOps.Charbonnier(Engine, prediction, target);
        if (_options.TernaryWeight > 0)
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(AmtOps.Ternary(Engine, prediction, target), NumOps.FromDouble(_options.TernaryWeight)));
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
                return ReconstructionLoss(frame0, frame1, data[1]);
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
        return ReconstructionLoss(frame0, frame1, AsBatch(NormalizeFrames(target)))[0];
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
            Name = _useNativeMode ? "AMT-Native" : "AMT-ONNX",
            Description = "AMT all-pairs multi-field transforms for frame interpolation (Li et al., CVPR 2023)",
            Complexity = _options.NumFlowFields
        };
        m.AdditionalInfo["Channels"] = string.Join(",", _options.Channels);
        m.AdditionalInfo["NumFlowFields"] = _options.NumFlowFields.ToString();
        m.AdditionalInfo["CorrelationRadius"] = _options.CorrelationRadius.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AMT<T>));
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
