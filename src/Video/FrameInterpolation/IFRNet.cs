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
/// IFRNet: intermediate feature refine network for efficient frame interpolation (Kong et al., CVPR 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (ltkong218/IFRNet, models/IFRNet.py, IFRNet_L.py and IFRNet_S.py): a shared
/// four-level encoder (1/2 to 1/16 size) and four decoders from 1/16 size. The first decoder reads both frames'
/// coarsest features and the time t; each later one reads the previous intermediate feature, both frames' features
/// warped by the current flows, and the flows, and refines the flows (added to the 2x-upsampled ones) and the
/// intermediate feature. The last also predicts a blending mask and a residual. The mean of both frames is subtracted
/// and restored, and frames are zero-padded on the right and bottom to multiples of 16, as the release's benchmarks
/// pad. Any time in [0, 1] can be synthesized.
/// </para>
/// <para>
/// Training follows train_vimeo90k.py at the midpoint: AdamW (no weight decay) with cosine decay from 1e-4 to 1e-5 per
/// iteration, on the Charbonnier and ternary reconstruction loss plus 0.01 times the geometry consistency between each
/// decoder's intermediate feature and the encoder's features of the ground-truth frame. The release's task-oriented
/// flow distillation needs pseudo ground-truth flows from a pretrained LiteFlowNet that a frame-pair to frame sample
/// does not carry, so it is not applied.
/// </para>
/// <para>
/// <b>For Beginners:</b> IFRNet encodes both frames into a four-level pyramid, then decodes from the coarsest level up,
/// each decoder refining the two motions from the in-between frame and a feature of the in-between frame itself; the
/// last one also predicts how to blend the two warped frames and a correction.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation",
    "https://arxiv.org/abs/2205.14620",
    Year = 2022,
    Authors = "Lingtong Kong, Boyuan Jiang, Donghao Luo, Wenqing Chu, Xiaoming Huang, Ying Tai, Chengjie Wang, Jie Yang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, ReferenceBatchSize = 24, WeightDecay = 0.0,
                MinLearningRate = 1e-5,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Kong et al. 2022, Sec. 4: optimized by AdamW for 300 epochs at a total batch "
                        + "size of 24, the learning rate initially 1e-4 and decaying to 1e-5 on a cosine "
                        + "attenuation schedule. The release (train_vimeo90k.py) sets weight decay 0 and "
                        + "steps the cosine per iteration.")]
public partial class IFRNet<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    // The encoder reaches 1/16 size.
    private const int Divisor = 16;

    private readonly IFRNetOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IfrNetNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates an IFRNet model in ONNX inference mode.</summary>
    public IFRNet(NeuralNetworkArchitecture<T> architecture, string modelPath, IFRNetOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new IFRNetOptions() : new IFRNetOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an IFRNet model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; IFRNet's and the release's training when null.</param>
    /// <param name="optimizer">The optimizer; the release's AdamW with cosine decay when null.</param>
    public IFRNet(NeuralNetworkArchitecture<T> architecture, IFRNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new IFRNetOptions() : new IFRNetOptions(options);
        Options = _options;
        var o = _options;
        if (o.EncoderChannels is not { Length: 4 } || o.EncoderChannels.Any(c => c < 1) || o.SideChannels < 1
            || o.SideChannels > 3 * o.EncoderChannels[0] || o.FirstKernelSize < 1 || o.FirstKernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(options), "IFRNet needs four positive encoder widths, side channels within every decoder and an odd first kernel.");
        if (o.LearningRate <= 0 || o.MinLearningRate < 0 || o.TotalIterations < 1 || o.WeightDecay < 0 || o.GeometryWeight < 0)
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
        RegisterParameterComponent("IFRNet.Network", _network);
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
        _network = new IfrNetNetwork<T>(_options.EncoderChannels.ToArray(), _options.SideChannels, _options.FirstKernelSize);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private IfrNetNetwork<T> Network => _network ?? throw new InvalidOperationException("IFRNet's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    /// <summary>The right and bottom zero padding that brings a side to a multiple of 16.</summary>
    internal static (int Right, int Bottom) Padding(int height, int width)
        => ((Divisor - width % Divisor) % Divisor, (Divisor - height % Divisor) % Divisor);

    private Tensor<T> Pad(Tensor<T> x)
    {
        var (right, bottom) = Padding(x.Shape[2], x.Shape[3]);
        return right + bottom == 0 ? x : Engine.PadNd(x, new[] { 0, right, 0, bottom }, PadMode.Constant, NumOps.Zero);
    }

    private Tensor<T> Crop(Tensor<T> x, int h, int w)
        => x.Shape[2] == h && x.Shape[3] == w ? x : Engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], h, w });

    // The frame at time t from frames [N, 3, H, W] in [0, 1].
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1, double t)
        => Crop(Network.Forward(Pad(frame0), Pad(frame1), t).Prediction, frame0.Shape[2], frame0.Shape[3]);

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"IFRNet takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
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
    /// <remarks>The decoders' intermediate features, the final flows and the frame at the midpoint.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        var output = Network.Forward(Pad(frame0), Pad(frame1), 0.5);
        var activations = new Dictionary<string, Tensor<T>>();
        for (int k = 0; k < output.IntermediateFeatures.Length; k++) activations[$"Intermediate{k + 1}"] = output.IntermediateFeatures[k];
        activations["Flow0"] = output.Flow0;
        activations["Flow1"] = output.Flow1;
        activations["Synthesis"] = output.Prediction;
        return activations;
    }

    #endregion

    #region Training

    // The warps build their sampling grids on the host.
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

    /// <summary>
    /// The release's loss on frames [N, 3, H, W] in [0, 1]: Charbonnier plus ternary on the frame, plus the weighted
    /// geometry consistency of the three intermediate features with the encoder's features of the mean-subtracted target.
    /// </summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3];
        var img0 = Pad(frame0);
        var img1 = Pad(frame1);
        var output = Network.Forward(img0, img1, 0.5);
        var prediction = Crop(output.Prediction, h, w);
        var loss = Engine.TensorAdd(AmtOps.Charbonnier(Engine, prediction, target), AmtOps.Ternary(Engine, prediction, target));
        if (_options.GeometryWeight > 0)
        {
            var paddedTarget = Pad(target);
            var targetFeatures = Network.Encode(Engine.TensorSubtract(paddedTarget, Network.FrameMean(img0, img1, paddedTarget.Shape.ToArray())));
            Tensor<T>? geometry = null;
            for (int k = 0; k < 3; k++)
            {
                var term = IfrOps.Geometry(Engine, output.IntermediateFeatures[k], targetFeatures[k]);
                geometry = geometry is null ? term : Engine.TensorAdd(geometry, term);
            }
            if (geometry is not null)
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(geometry, NumOps.FromDouble(_options.GeometryWeight)));
        }
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
            Name = _useNativeMode ? "IFRNet-Native" : "IFRNet-ONNX",
            Description = "IFRNet intermediate feature refine network (Kong et al., CVPR 2022)",
            Complexity = _options.EncoderChannels.Sum()
        };
        m.AdditionalInfo["EncoderChannels"] = string.Join(",", _options.EncoderChannels);
        m.AdditionalInfo["SideChannels"] = _options.SideChannels.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IFRNet<T>));
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
