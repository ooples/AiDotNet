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
/// InterpAny-Clearer: distance indexing and iterative reference-based estimation for clearer anytime video frame
/// interpolation (Zhong et al., ECCV 2024), on its release's DR-RIFE.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the release's DR-RIFE (zzh-tech/InterpAny-Clearer, models/DI-RIFE: IFNet_sdi_recur.py, refine.py,
/// RIFE_sdi_recur.py). Instead of a time, every pixel is given a distance index: how far along its path from frame 0 to
/// frame 1 it should be. RIFE's three IFBlocks estimate the flows to both frames and a blending mask coarse to fine, each
/// reading the index map with the reference's index and the reference frame; a Contextnet-U-Net adds a residual. At
/// inference the target index is reached in <see cref="InterpAnyClearerOptions.Iterations"/> steps from frame 0, each
/// step's frame the next step's reference. <see cref="Interpolate"/> uses a uniform index (equal to the time), and
/// <see cref="InterpolateWithDistanceMap"/> takes a per-pixel one. Sides are zero-padded on the right and bottom to
/// multiples of 32, as the release pads.
/// </para>
/// <para>
/// Training follows the release (RIFE_sdi_recur.py, train_sdi_m_mask_recur.py): AdamW (weight decay 1e-3) with a
/// 2,000-step linear warmup to 3e-4 and a cosine to 3e-6, on the Laplacian loss of the refined frame, plus that of the
/// privileged teacher (which also reads the ground truth) and 0.002 times the distillation of the students' flows toward
/// the teacher's where the teacher's frame is better. A training input is the pair [6, H, W], optionally followed by the
/// target's distance index (one channel) or by that, the reference's index and the reference frame (five channels). The
/// release computes the indices from RAFT flows offline; without them the index is 0.5 and the reference is frame 0
/// with index 0.
/// </para>
/// <para>
/// <b>For Beginners:</b> Ordinary interpolation is told only "halfway in time", which is ambiguous when objects speed up
/// or slow down, so the result blurs. This model is instead told, for every pixel, how far along its path the object
/// should be. It reaches that position in a few small steps, each starting from the frame the previous step produced,
/// which keeps the result sharp.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation",
    "https://arxiv.org/abs/2311.08007",
    Year = 2024,
    Authors = "Zhihang Zhong, Gurunandan Krishnan, Xiao Sun, Yu Qiao, Sizhuo Ma, Jian Wang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 1e-3, MinLearningRate = 3e-6,
                WarmupSteps = 2000, ReferenceBatchSize = 64, Schedule = LearningRateSchedulerType.LinearWarmup,
                Source = "The release (models/DI-RIFE: RIFE_sdi_recur.py, train_sdi_m_mask_recur.py): AdamW with a weight "
                        + "decay of 1e-3, the rate rising linearly from 0 to 3e-4 over 2,000 steps then a cosine to 3e-6 "
                        + "over 300 epochs, batch 16 on each of 4 GPUs.")]
public partial class InterpAnyClearer<T> : FrameInterpolationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly InterpAnyClearerOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IacNetwork<T>? _network;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>Creates an InterpAny-Clearer model in ONNX inference mode.</summary>
    public InterpAnyClearer(NeuralNetworkArchitecture<T> architecture, string modelPath, InterpAnyClearerOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new InterpAnyClearerOptions() : new InterpAnyClearerOptions(options);
        Options = _options;
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an InterpAny-Clearer model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's DR-RIFE and training when null.</param>
    /// <param name="optimizer">The optimizer; the release's AdamW with warmup and cosine decay when null.</param>
    public InterpAnyClearer(NeuralNetworkArchitecture<T> architecture, InterpAnyClearerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _optimizer = optimizer;
        _options = options is null ? new InterpAnyClearerOptions() : new InterpAnyClearerOptions(options);
        Options = _options;
        var o = _options;
        if (o.BlockWidths is null || o.BlockWidths.Length != 3 || o.BlockWidths.Any(c => c < 2 || c % 2 != 0) || o.TeacherWidth < 2 || o.TeacherWidth % 2 != 0
            || o.ContextChannels < 1 || o.Iterations < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "InterpAny-Clearer needs three even block widths, an even teacher width, a positive context width and at least one iteration.");
        if (o.LearningRate <= 0 || o.MinLearningRate < 0 || o.MinLearningRate > o.LearningRate || o.WeightDecay < 0 || o.WarmupSteps < 0
            || o.TotalSteps <= o.WarmupSteps || o.DistillationWeight < 0 || o.LaplacianLevels < 1)
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
        RegisterParameterComponent("InterpAnyClearer.Network", _network);
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
        _network = new IacNetwork<T>(_options.BlockWidths.ToArray(), _options.TeacherWidth, _options.ContextChannels, _options.UseDistillation);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        // Every weight's shape is fixed by the options, so there is nothing to resolve from an input.
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        MarkLayerShapesResolved();
    }

    private IacNetwork<T> Network => _network ?? throw new InvalidOperationException("InterpAny-Clearer's native network is not initialized.");

    /// <summary>The network's weights.</summary>
    internal IReadOnlyList<Tensor<T>> NetworkParameters
        => _network is null ? Array.Empty<Tensor<T>>() : _network.GetParameterStateChunks().Select(chunk => chunk.Tensor).ToList();

    #endregion

    #region Pipeline

    /// <summary>The release's right and bottom zero padding to multiples of 32.</summary>
    internal static (int Right, int Bottom) Padding(int height, int width) => ((32 - width % 32) % 32, (32 - height % 32) % 32);

    private Tensor<T> Pad(Tensor<T> x, PadMode mode)
    {
        var (right, bottom) = Padding(x.Shape[2], x.Shape[3]);
        return right + bottom == 0 ? x : Engine.PadNd(x, new[] { 0, right, 0, bottom }, mode, NumOps.Zero);
    }

    private Tensor<T> Crop(Tensor<T> x, int h, int w)
        => x.Shape[2] == h && x.Shape[3] == w ? x : Engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], h, w });

    private Tensor<T> Uniform(Tensor<T> like, double value)
    {
        var map = new Tensor<T>(new[] { like.Shape[0], 1, like.Shape[2], like.Shape[3] });
        map.Fill(NumOps.FromDouble(value));
        return map;
    }

    /// <summary>
    /// RIFE_sdi_recur.Model.inference: the frame at the distance index [N, 1, H, W] from frames [N, 3, H, W] in [0, 1],
    /// reached in <see cref="InterpAnyClearerOptions.Iterations"/> steps from frame 0 at index 0.
    /// </summary>
    private Tensor<T> Synthesize(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> distance)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3], iterations = _options.Iterations;
        var one = Pad(frame0, PadMode.Constant);
        var two = Pad(frame1, PadMode.Constant);
        // The release builds the uniform index on the padded frame; a per-pixel index is extended from its border.
        var target = Pad(distance, PadMode.Replicate);
        var current = one;
        for (int i = 1; i <= iterations; i++)
        {
            var step = Engine.TensorMultiplyScalar(target, NumOps.FromDouble(i / (double)iterations));
            var previous = Engine.TensorMultiplyScalar(target, NumOps.FromDouble((i - 1) / (double)iterations));
            current = Network.Forward(one, two, Engine.TensorConcatenate(new[] { step, previous, current }, 1), null).Refined;
        }
        return Crop(current, h, w);
    }

    // [6 + k, H, W] or [B, 6 + k, H, W] inputs (k = 0, 1 or 5) to the two frames [N, 3, H, W] and the extra channels.
    private (Tensor<T> Frame0, Tensor<T> Frame1, Tensor<T>? Extra, bool Batched) SplitInput(Tensor<T> input)
    {
        bool batched = input.Rank == 4;
        var x = batched ? input : Engine.Reshape(input, new[] { 1 }.Concat(input._shape).ToArray());
        if (x.Rank != 4 || (x.Shape[1] != 6 && x.Shape[1] != 7 && x.Shape[1] != 11))
            throw new ArgumentException("InterpAny-Clearer takes a frame pair [6, H, W], followed by the distance index (7) or by "
                + $"the index, the reference's index and the reference frame (11), optionally batched; got [{string.Join(", ", input._shape)}].", nameof(input));
        var extra = x.Shape[1] == 6 ? null : Engine.TensorNarrow(x, 1, 6, x.Shape[1] - 6).Contiguous();
        return (Engine.TensorNarrow(x, 1, 0, 3).Contiguous(), Engine.TensorNarrow(x, 1, 3, 3).Contiguous(), extra, batched);
    }

    private Tensor<T> AsBatch(Tensor<T> frames) => frames.Rank == 4 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());

    // The pair's frame at index 0.5, or at its index channel; an 11-channel input is one pass with its own index map.
    private Tensor<T> RunInput(Tensor<T> input)
    {
        var (frame0, frame1, extra, batched) = SplitInput(input);
        Tensor<T> output;
        if (extra is not null && extra.Shape[1] == 5)
        {
            int h = frame0.Shape[2], w = frame0.Shape[3];
            output = Crop(Network.Forward(Pad(frame0, PadMode.Constant), Pad(frame1, PadMode.Constant), Pad(extra, PadMode.Replicate), null).Refined, h, w);
        }
        else
        {
            output = Synthesize(frame0, frame1, extra ?? Uniform(frame0, 0.5));
        }
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    #endregion

    #region Interpolation

    /// <inheritdoc />
    /// <remarks>A uniform distance index equal to <paramref name="t"/>.</remarks>
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (!(t >= 0.0 && t <= 1.0))
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep must be in [0, 1].");
        if (IsOnnxMode || _usingCustomLayers)
        {
            var pair = ConcatenateFeatures(PreprocessFrames(frame0), PreprocessFrames(frame1));
            return PostprocessOutput(IsOnnxMode ? RunOnnxInference(pair) : Forward(pair));
        }
        using var noGrad = new NoGradScope<T>();
        var one = AsBatch(PreprocessFrames(frame0));
        var output = Synthesize(one, AsBatch(PreprocessFrames(frame1)), Uniform(one, t));
        return PostprocessOutput(frame0.Rank == 4 ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray()));
    }

    /// <summary>
    /// The frame at a per-pixel distance index: for each pixel, the fraction of its path from <paramref name="frame0"/> to
    /// <paramref name="frame1"/> it should have travelled (0 at frame 0, 1 at frame 1).
    /// </summary>
    /// <param name="frame0">The first frame [3, H, W] (or batched), on the 0-255 scale.</param>
    /// <param name="frame1">The second frame, the same shape.</param>
    /// <param name="distanceMap">The index [H, W], [1, H, W] or [N, 1, H, W], each value in [0, 1].</param>
    public Tensor<T> InterpolateWithDistanceMap(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> distanceMap)
    {
        ThrowIfDisposed();
        if (!_useNativeMode || _usingCustomLayers)
            throw new NotSupportedException("A per-pixel distance index needs the native network.");
        var one = AsBatch(PreprocessFrames(frame0));
        var map = distanceMap.Rank == 2 ? Engine.Reshape(distanceMap, new[] { 1, 1, distanceMap.Shape[0], distanceMap.Shape[1] })
            : distanceMap.Rank == 3 ? Engine.Reshape(distanceMap, new[] { 1 }.Concat(distanceMap._shape).ToArray()) : distanceMap;
        if (map.Rank != 4 || map.Shape[0] != one.Shape[0] || map.Shape[1] != 1 || map.Shape[2] != one.Shape[2] || map.Shape[3] != one.Shape[3])
            throw new ArgumentException($"The distance index must be one channel the frames' size; got [{string.Join(", ", distanceMap._shape)}].", nameof(distanceMap));
        if (map.ToArray().Any(v => !(NumOps.ToDouble(v) >= 0.0 && NumOps.ToDouble(v) <= 1.0)))
            throw new ArgumentOutOfRangeException(nameof(distanceMap), "Every distance index must be in [0, 1].");
        using var noGrad = new NoGradScope<T>();
        var output = Synthesize(one, AsBatch(PreprocessFrames(frame1)), map);
        return PostprocessOutput(frame0.Rank == 4 ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray()));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return PostprocessOutput(RunOnnxInference(PreprocessFrames(input)));
        return PostprocessOutput(Forward(_usingCustomLayers ? PreprocessFrames(input) : PreprocessInput(input)));
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        using var noGrad = new NoGradScope<T>();
        return RunInput(input);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunInput(PreprocessInput(input));

    /// <inheritdoc />
    /// <remarks>Each student's flows and the blending mask and refined frame of one pass at index 0.5 from frame 0.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || _usingCustomLayers) return base.GetNamedLayerActivations(input);
        using var noGrad = new NoGradScope<T>();
        var (frame0, frame1, _, _) = SplitInput(PreprocessInput(input));
        var one = Pad(frame0, PadMode.Constant);
        var index = Engine.TensorConcatenate(new[] { Uniform(one, 0.5), Uniform(one, 0.0), one }, 1);
        var output = Network.Forward(one, Pad(frame1, PadMode.Constant), index, null);
        var activations = new Dictionary<string, Tensor<T>>();
        for (int i = 0; i < output.Flows.Length; i++) activations[$"Flow{i}"] = output.Flows[i];
        activations["Mask"] = output.Mask;
        activations["Refined"] = output.Refined;
        return activations;
    }

    #endregion

    #region Training

    // The distillation mask is built on the host.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ReleaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // AdamW(weight_decay=1e-3), its rate set per step: a linear warmup from zero, then a cosine to the minimum.
        var adamw = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new LinearWarmupScheduler(_options.LearningRate, _options.WarmupSteps, _options.TotalSteps, 0.0,
                LinearWarmupScheduler.DecayMode.Cosine, _options.MinLearningRate),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, adamw);
        return _optimizer;
    }

    /// <summary>
    /// RIFE_sdi_recur.Model.update's loss on frames [N, 3, H, W] in [0, 1] and the index map [N, 5, H, W]: the Laplacian
    /// loss of the refined frame, plus (with distillation) the teacher's and the weighted distillation loss.
    /// </summary>
    internal Tensor<T> ReleaseLoss(Tensor<T> frame0, Tensor<T> frame1, Tensor<T> indexMap, Tensor<T> target)
    {
        int h = frame0.Shape[2], w = frame0.Shape[3];
        if (h % 32 != 0 || w % 32 != 0)
            throw new ArgumentException($"InterpAny-Clearer trains on crops whose sides are multiples of 32; got {h}x{w}.", nameof(frame0));
        var output = Network.Forward(frame0, frame1, indexMap, target);
        var loss = EmaOps.LaplacianLoss(Engine, output.Refined, target, _options.LaplacianLevels);
        if (output.Teacher is not null && output.Distill is not null)
        {
            loss = Engine.TensorAdd(loss, EmaOps.LaplacianLoss(Engine, output.Teacher, target, _options.LaplacianLevels));
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(output.Distill, NumOps.FromDouble(_options.DistillationWeight)));
        }
        return loss;
    }

    // The training sample's index map: the target's index (0.5 by default), the reference's (0) and the reference (frame 0).
    private Tensor<T> TrainingIndex(Tensor<T> frame0, Tensor<T>? extra)
    {
        if (extra is not null && extra.Shape[1] == 5) return extra;
        return Engine.TensorConcatenate(new[] { extra ?? Uniform(frame0, 0.5), Uniform(frame0, 0.0), frame0 }, 1);
    }

    // The frames divided by 255; the index channels are already in [0, 1].
    private Tensor<T> PreprocessInput(Tensor<T> input)
    {
        int axis = input.Rank == 4 ? 1 : 0;
        if (input.Rank < 3 || input.Shape[axis] <= 6) return PreprocessFrames(input);
        var frames = PreprocessFrames(Engine.TensorNarrow(input, axis, 0, 6).Contiguous());
        var extra = Engine.TensorNarrow(input, axis, 6, input.Shape[axis] - 6).Contiguous();
        if (extra.Shape[axis] != 5) return Engine.TensorConcatenate(new[] { frames, extra }, axis);
        // The reference frame is on the 0-255 scale like the pair.
        var indices = Engine.TensorNarrow(extra, axis, 0, 2).Contiguous();
        var reference = PreprocessFrames(Engine.TensorNarrow(extra, axis, 2, 3).Contiguous());
        return Engine.TensorConcatenate(new[] { frames, indices, reference }, axis);
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
                var (frame0, frame1, extra, _) = SplitInput(PreprocessInput(data[0]));
                return ReleaseLoss(frame0, frame1, TrainingIndex(frame0, extra), data[1]);
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
        var (frame0, frame1, extra, _) = SplitInput(PreprocessInput(input));
        return ReleaseLoss(frame0, frame1, TrainingIndex(frame0, extra), AsBatch(NormalizeFrames(target)))[0];
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
            Name = _useNativeMode ? "InterpAnyClearer-Native" : "InterpAnyClearer-ONNX",
            Description = "InterpAny-Clearer DR-RIFE distance-indexed frame interpolation (Zhong et al., ECCV 2024)",
            Complexity = _options.BlockWidths.Sum()
        };
        m.AdditionalInfo["BlockWidths"] = string.Join(",", _options.BlockWidths);
        m.AdditionalInfo["Iterations"] = _options.Iterations.ToString();
        m.AdditionalInfo["UseDistillation"] = _options.UseDistillation.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(InterpAnyClearer<T>));
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
