using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
/// SoftSplat: softmax splatting for video frame interpolation (Niklaus and Liu, CVPR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release (sniklaus/softmax-splatting, run.py and softsplat.py). Both frames are standardized
/// by their joint mean and standard deviation. A PWC-Net (a six-level feature extractor and six decoders on 9x9
/// correlation volumes) estimates the flows in both directions. The synthesis extracts a three-level feature pyramid
/// from each frame, computes each frame's importance metric from its brightness-constancy error with a small U-Net, and
/// softmax-splats every level, with its metric, forward to time t, the metric (scaled by 2t or 2(1 − t)) as the
/// splatting weight. A three-stream GridNet fuses the splatted pyramids into the frame, and the statistics are
/// restored. Sides are replicate-padded to even sizes, as run.estimate pads. Any time in [0, 1] can be synthesized.
/// </para>
/// <para>
/// Training follows the paper, which adopts the training of context-aware synthesis (Niklaus and Liu, CVPR 2018):
/// AdaMax (β1 0.9, β2 0.999) at 1e-3 on the Laplacian loss Σ 2^(i−1)·|L_i(Î) − L_i(I)| over five pyramid levels at the
/// midpoint, the flow estimator fine-tuned end to end. The norm is averaged over pixels rather than summed, which
/// AdaMax's scale-invariant step does not distinguish, and training crops are multiples of 16 so every pyramid level
/// halves exactly. The release ships trained weights but no training script.
/// </para>
/// <para>
/// <b>For Beginners:</b> SoftSplat estimates motion both ways between the two frames, pushes each frame's feature
/// pyramid forward along its motion to the in-between time (blending pixels that land on the same spot by a learned
/// importance), and then lets a grid of convolutions paint the in-between frame from the pushed features.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Softmax Splatting for Video Frame Interpolation",
    "https://arxiv.org/abs/2003.05534",
    Year = 2020,
    Authors = "Simon Niklaus, Feng Liu")]
[PaperOptimizer(OptimizerKind.Adamax, LearningRate = 1e-3, Beta1 = 0.9, Beta2 = 0.999, ReferenceBatchSize = 8,
                Source = "Niklaus and Liu 2020, Sec. 3.3: training adopted from Niklaus and Liu 2018 (CtxSyn), "
                        + "Sec. 3.2: AdaMax with beta1 0.9, beta2 0.999, a learning rate of 0.001 and a "
                        + "mini-batch size of 8 on the Laplacian loss.")]
public partial class SoftSplat<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    // Every training pyramid level halves the size exactly.
    private const int TrainingDivisor = 16;

    private readonly SoftSplatOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private SoftSplatNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates a SoftSplat model in ONNX inference mode.</summary>
    public SoftSplat(NeuralNetworkArchitecture<T> architecture, string modelPath, SoftSplatOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new SoftSplatOptions() : new SoftSplatOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a SoftSplat model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's widths and the paper's training when null.</param>
    /// <param name="optimizer">The optimizer; the paper's AdaMax when null.</param>
    public SoftSplat(NeuralNetworkArchitecture<T> architecture, SoftSplatOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new SoftSplatOptions() : new SoftSplatOptions(options);
        Options = _options;
        var o = _options;
        if (o.FlowChannels is not { Length: 6 } || o.FlowDecoderChannels is not { Length: 5 } || o.MetricChannels is not { Length: 4 }
            || o.SynthesisChannels is not { Length: 3 } || o.FlowChannels.Concat(o.FlowDecoderChannels).Concat(o.SynthesisChannels).Any(c => c < 1)
            || o.MetricChannels.Any(c => c < 1) || o.MetricChannels[0] <= 4)
            throw new ArgumentOutOfRangeException(nameof(options), "SoftSplat needs six flow widths, five decoder widths, four metric widths (the first above 4) and three synthesis widths, all positive.");
        if (o.LearningRate <= 0 || o.Beta1 < 0 || o.Beta1 >= 1 || o.Beta2 < 0 || o.Beta2 >= 1 || o.LaplacianLevels < 1)
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
        RegisterParameterComponent("SoftSplat.Network", _network);
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
        _network = new SoftSplatNetwork<T>(_options.FlowChannels.ToArray(), _options.FlowDecoderChannels.ToArray(),
            _options.MetricChannels.ToArray(), _options.SynthesisChannels.ToArray());
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private SoftSplatNetwork<T> Network => _network ?? throw new InvalidOperationException("SoftSplat's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    // run.estimate: replicate the last row and column of an odd side.
    private Tensor<T> PadEven(Tensor<T> x)
    {
        int bottom = x.Shape[2] % 2, right = x.Shape[3] % 2;
        if (bottom + right == 0) return x;
        if (bottom == 1) x = Engine.TensorConcatenate(new[] { x, Engine.TensorNarrow(x, 2, x.Shape[2] - 1, 1).Contiguous() }, 2);
        if (right == 1) x = Engine.TensorConcatenate(new[] { x, Engine.TensorNarrow(x, 3, x.Shape[3] - 1, 1).Contiguous() }, 3);
        return x;
    }

    private Tensor<T> Crop(Tensor<T> x, int h, int w)
        => x.Shape[2] == h && x.Shape[3] == w ? x : Engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], h, w });

    // The frame at time t from frames [N, 3, H, W] in [0, 1].
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1, double t)
        => Crop(Network.Forward(PadEven(frame0), PadEven(frame1), new[] { t })[0], frame0.Shape[2], frame0.Shape[3]);

    // [6, H, W] or [B, 6, H, W] pairs to the two frames [N, 3, H, W].
    private (Tensor<T> Frame0, Tensor<T> Frame1, bool Batched) SplitPairs(Tensor<T> pairs)
    {
        bool batched = pairs.Rank == 4;
        var x = batched ? pairs : Engine.Reshape(pairs, new[] { 1 }.Concat(pairs._shape).ToArray());
        if (x.Rank != 4 || x.Shape[1] != 6)
            throw new ArgumentException($"SoftSplat takes a frame pair [6, H, W] or [B, 6, H, W]; got [{string.Join(", ", pairs._shape)}].", nameof(pairs));
        return (Engine.TensorNarrow(x, 1, 0, 3).Contiguous(), Engine.TensorNarrow(x, 1, 3, 3).Contiguous(), batched);
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
    /// <remarks>The forward and backward flows and the frame at the midpoint.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _) = SplitPairs(PreprocessFrames(input));
        var flows = Network.Flows(PadEven(frame0), PadEven(frame1));
        return new Dictionary<string, Tensor<T>>
        {
            ["ForwardFlow"] = flows.Forward,
            ["BackwardFlow"] = flows.Backward,
            ["Synthesis"] = Synthesize(frame0, frame1, 0.5),
        };
    }

    #endregion

    #region Training

    // The splatting and warps build their indices and grids on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // torch.optim.Adamax's epsilon 1e-8, constant rate.
        var adamax = new AdaMaxOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdaMaxOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = _options.Beta1,
            Beta2 = _options.Beta2,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamax);
        return _optimizer;
    }

    /// <summary>
    /// CtxSyn's Laplacian loss on frames [N, 3, H, W]: Σ_i 2^(i−1)·mean|L_i(prediction) − L_i(target)| over the detail
    /// levels and, as the last level, the low-pass residual.
    /// </summary>
    internal static Tensor<T> LaplacianLoss(IEngine engine, Tensor<T> prediction, Tensor<T> target, int levels)
    {
        int n = prediction.Shape[0], c = prediction.Shape[1], h = prediction.Shape[2], w = prediction.Shape[3];
        var a = EmaOps.LaplacianPyramid(engine, engine.Reshape(prediction, new[] { n * c, 1, h, w }), levels - 1, out var lowA);
        var b = EmaOps.LaplacianPyramid(engine, engine.Reshape(target, new[] { n * c, 1, h, w }), levels - 1, out var lowB);
        a.Add(lowA);
        b.Add(lowB);
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T>? total = null;
        for (int i = 0; i < levels; i++)
        {
            var mean = engine.ReduceMean(engine.TensorAbs(engine.TensorSubtract(a[i], b[i])), new[] { 0, 1, 2, 3 }, false);
            var term = engine.TensorMultiplyScalar(mean, ops.FromDouble(Math.Pow(2, i)));
            total = total is null ? term : engine.TensorAdd(total, term);
        }
        return total ?? throw new InvalidOperationException("The Laplacian loss needs a level.");
    }

    /// <summary>The paper's loss on frames [N, 3, H, W] in [0, 1]: the Laplacian loss of the midpoint frame.</summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> target)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3];
        if (h % TrainingDivisor != 0 || w % TrainingDivisor != 0)
            throw new ArgumentException($"SoftSplat trains on crops whose sides are multiples of {TrainingDivisor}; got {h}x{w}.", nameof(frame0));
        var prediction = Network.Forward(frame0, frame1, new[] { 0.5 })[0];
        return LaplacianLoss(Engine, prediction, target, _options.LaplacianLevels);
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
    /// <remarks>The paper's loss, the quantity a training step lowers.</remarks>
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
            Name = _useNativeMode ? "SoftSplat-Native" : "SoftSplat-ONNX",
            Description = "SoftSplat softmax splatting frame interpolation (Niklaus and Liu, CVPR 2020)",
            Complexity = _options.SynthesisChannels.Sum()
        };
        m.AdditionalInfo["FlowChannels"] = string.Join(",", _options.FlowChannels);
        m.AdditionalInfo["MetricChannels"] = string.Join(",", _options.MetricChannels);
        m.AdditionalInfo["SynthesisChannels"] = string.Join(",", _options.SynthesisChannels);
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SoftSplat<T>));
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
