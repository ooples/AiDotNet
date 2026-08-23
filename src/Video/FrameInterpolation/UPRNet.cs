using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>UPR-Net: Unified Pyramid Recurrent Network for video frame interpolation.</summary>
/// <typeparam name="T">Numeric tensor element type.</typeparam>
/// <remarks>
/// This native graph follows the authors' released <c>upr_base.py</c>: one shared
/// three-stage feature pyramid, one shared radius-four partial-correlation motion
/// estimator, and one shared synthesis U-Net are recurrently reused from coarse to
/// fine. Motion and synthesis use normalized forward soft splatting. UPR-Net does
/// not contain a ConvLSTM and does not create separate modules per pyramid level.
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Unified Pyramid Recurrent Network for Video Frame Interpolation",
    "https://arxiv.org/abs/2211.03456",
    Year = 2023,
    Authors = "Xin Jin, Longhai Wu, Jie Chen, Youxin Chen, Jayoon Koo, Cheul-hee Hahm")]
public partial class UPRNet<T> : FrameInterpolationBase<T>
{
    private const int CorrelationRadius = 4;
    private const int CorrelationChannels = 81;
    private const double SynthesisBlendEpsilon = 1e-6;

    private readonly UPRNetOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _customArchitecture;
    private bool _paperShapesResolved;
    private bool _disposed;

    private readonly List<ILayer<T>> _featureStage0 = [];
    private readonly List<ILayer<T>> _featureStage1 = [];
    private readonly List<ILayer<T>> _featureStage2 = [];
    private readonly List<ILayer<T>> _motionEstimator = [];
    private readonly List<ILayer<T>> _synthEncoder0 = [];
    private readonly List<ILayer<T>> _synthEncoder1 = [];
    private readonly List<ILayer<T>> _synthEncoder2 = [];
    private readonly List<ILayer<T>> _synthDecoder1 = [];
    private readonly List<ILayer<T>> _synthDecoder2 = [];
    private readonly List<ILayer<T>> _synthDecoder0 = [];
    private ConvolutionalLayer<T>? _synthPrediction;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed UPR-Net.</summary>
    public UPRNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        UPRNetOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be empty.", nameof(modelPath));
        _options = options is null ? new UPRNetOptions() : new UPRNetOptions(options);
        _options.Validate();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates the paper-faithful native UPR-Net.</summary>
    public UPRNet(
        NeuralNetworkArchitecture<T> architecture,
        UPRNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options is null ? new UPRNetOptions() : new UPRNetOptions(options);
        _options.Validate();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    /// <inheritdoc />
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (t < 0.0 || t > 1.0)
            throw new ArgumentOutOfRangeException(nameof(t), "Interpolation time must be in [0,1].");
        var input = ConcatenateFeatures(PreprocessFrames(frame0), PreprocessFrames(frame1));
        var output = IsOnnxMode ? RunOnnxInference(input) : ForwardAtTime(input, t);
        return PostprocessOutput(output);
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 })
        {
            _customArchitecture = true;
            Layers.AddRange(Architecture.Layers);
            return;
        }

        BuildPaperArchitecture();
    }

    private void BuildPaperArchitecture()
    {
        Layers.Clear();
        ClearBindings();
        Layers.AddRange(LayerHelper<T>.CreateDefaultUPRNetLayers(Architecture));
        EnsurePaperBindings();
    }

    private void ClearBindings()
    {
        _featureStage0.Clear();
        _featureStage1.Clear();
        _featureStage2.Clear();
        _motionEstimator.Clear();
        _synthEncoder0.Clear();
        _synthEncoder1.Clear();
        _synthEncoder2.Clear();
        _synthDecoder1.Clear();
        _synthDecoder2.Clear();
        _synthDecoder0.Clear();
        _synthPrediction = null;
    }

    private void EnsurePaperBindings()
    {
        if (_customArchitecture ||
            (Layers.Count > 0 && _featureStage0.Count > 0 && ReferenceEquals(_featureStage0[0], Layers[0])))
            return;
        if (Layers.Count != 47)
            throw new InvalidDataException(
                $"UPR-Net paper architecture requires 47 serialized layers, found {Layers.Count}.");

        ClearBindings();
        int index = 0;
        Bind(_featureStage0, 4, ref index);
        Bind(_featureStage1, 4, ref index);
        Bind(_featureStage2, 4, ref index);
        Bind(_motionEstimator, 6, ref index);
        Bind(_synthEncoder0, 4, ref index);
        Bind(_synthEncoder1, 6, ref index);
        Bind(_synthEncoder2, 6, ref index);
        Bind(_synthDecoder1, 4, ref index);
        Bind(_synthDecoder2, 4, ref index);
        Bind(_synthDecoder0, 4, ref index);
        _synthPrediction = (ConvolutionalLayer<T>)Layers[index];
    }

    private void Bind(List<ILayer<T>> destination, int count, ref int index)
    {
        for (int i = 0; i < count; i++) destination.Add(Layers[index++]);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The public <see cref="Layers"/> list is UPR-Net's stable serialization order, not a linear
    /// execution chain. Resolve its shared modules against the real feature/motion/synthesis graph;
    /// otherwise the base sequential walk feeds the six-channel frame pair directly to the
    /// three-channel feature extractor and then carries unrelated branch widths into later modules.
    /// Only this graph topology is model-specific; each layer still owns its normal automatic shape
    /// and parameter materialization.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_customArchitecture)
        {
            base.ResolveLazyLayerShapes();
            return;
        }

        if (_paperShapesResolved || !_useNativeMode || Layers.Count == 0) return;
        EnsurePaperBindings();

        // Spatial values are only a cheap shape probe; UPR-Net's parameter tensors depend on the
        // channel widths below, while the real input retains arbitrary valid H/W at runtime.
        const int full = 16;
        var stage0 = ResolveStageShapes(_featureStage0, [3, full, full]);
        var stage1 = ResolveStageShapes(_featureStage1, stage0);
        var stage2 = ResolveStageShapes(_featureStage2, stage1);

        // radius-4 correlation (81), two warped 64-channel features, recurrent feature (64), flow (4)
        ResolveStageShapes(_motionEstimator, [277, stage2[1], stage2[2]]);

        // Synthesis inputs follow the released graph's concatenations. The returned shapes drive
        // the next branch, so deconvolution scaling is derived by the layers rather than duplicated.
        var synth0 = ResolveStageShapes(_synthEncoder0, [19, full, full]);
        var synth1 = ResolveStageShapes(_synthEncoder1, [64, stage0[1], stage0[2]]);
        var synth2 = ResolveStageShapes(_synthEncoder2, [128, stage1[1], stage1[2]]);
        var decoder1 = ResolveStageShapes(_synthDecoder1,
            [synth2[0] + stage2[0] * 2, stage2[1], stage2[2]]);
        var decoder2 = ResolveStageShapes(_synthDecoder2,
            [decoder1[0] + synth1[0], decoder1[1], decoder1[2]]);
        var decoder0 = ResolveStageShapes(_synthDecoder0,
            [decoder2[0] + synth0[0], decoder2[1], decoder2[2]]);
        _synthPrediction!.ResolveFromShape(decoder0);
        MarkLayerShapesResolved();
        _paperShapesResolved = true;
    }

    private static int[] ResolveStageShapes(IReadOnlyList<ILayer<T>> layers, int[] inputShape)
    {
        int[] current = inputShape;
        foreach (var layer in layers)
        {
            if (layer is LayerBase<T> layerBase)
            {
                // UPR-Net carries per-sample CHW shapes between graph branches. PReLU's published
                // channel axis is the NCHW axis (1), however, so give that layer the batch axis it
                // expects and remove it again before continuing the per-sample graph walk.
                bool addBatchAxis = layer is PReLULayer<T> && current.Length == 3;
                layerBase.ResolveFromShape(addBatchAxis
                    ? [1, current[0], current[1], current[2]]
                    : current);

                var resolved = layer.GetOutputShape();
                if (addBatchAxis && resolved is { Length: 4 })
                    current = [resolved[1], resolved[2], resolved[3]];
                else
                    current = resolved ?? [];
            }
            else
            {
                current = layer.GetOutputShape() ?? [];
            }

            if (current.Length == 0 || current.Any(axis => axis <= 0))
                throw new InvalidOperationException(
                    $"UPR-Net shape resolution stopped at {layer.GetType().Name}; " +
                    "the paper graph requires a concrete per-sample output shape.");
        }
        return current;
    }

    /// <summary>Runs the standard midpoint graph. Use <see cref="Interpolate"/> for arbitrary time.</summary>
    public new Tensor<T> Forward(Tensor<T> input) => ForwardAtTime(input, 0.5);

    private Tensor<T> ForwardAtTime(Tensor<T> input, double time)
    {
        ThrowIfDisposed();
        if (_customArchitecture)
            return base.Forward(input);
        EnsurePaperBindings();

        bool wasBatched = input.Rank == 4;
        Tensor<T> x = wasBatched
            ? input
            : Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        if (x.Shape[1] != 6)
            throw new ArgumentException("UPR-Net expects two RGB frames concatenated as 6 NCHW channels.",
                nameof(input));

        int batch = x.Shape[0];
        int fullHeight = x.Shape[2];
        int fullWidth = x.Shape[3];
        var frame0 = SliceChannels(x, 0, 3);
        var frame1 = SliceChannels(x, 3, 3);
        Tensor<T>? flow = null;
        Tensor<T>? feature = null;
        Tensor<T>? interpolation = null;
        var fullySkippedLevels = _options.NumLevelsSkipped == 0
            ? new HashSet<int>()
            : Enumerable.Range(0, _options.NumLevelsSkipped).ToHashSet();

        for (int level = _options.NumPyramidLevels - 1; level >= 0; level--)
        {
            // The released fast variants skip whole intermediate fine levels, then
            // still run synthesis at level zero using the appropriately upscaled flow.
            if (fullySkippedLevels.Contains(level) &&
                level != 0 && level != _options.NumPyramidLevels - 1)
                continue;

            int divisor = 1 << level;
            int levelHeight = System.Math.Max(1, fullHeight / divisor);
            int levelWidth = System.Math.Max(1, fullWidth / divisor);
            var image0 = Resize(frame0, levelHeight, levelWidth);
            var image1 = Resize(frame1, levelHeight, levelWidth);
            int motionHeight = System.Math.Max(1, levelHeight / 4);
            int motionWidth = System.Math.Max(1, levelWidth / 4);

            Tensor<T> lastFlow;
            Tensor<T> lastFeature;
            bool skipMotionAtFinest = level == 0 && _options.NumLevelsSkipped > 0;
            if (flow is null || feature is null)
            {
                lastFlow = new Tensor<T>([batch, 4, motionHeight, motionWidth]);
                lastFeature = new Tensor<T>([batch, 64, motionHeight, motionWidth]);
            }
            else
            {
                double scale = skipMotionAtFinest
                    ? 1 << _options.NumLevelsSkipped
                    : 2.0;
                lastFlow = Engine.TensorMultiplyScalar(
                    Resize(flow, motionHeight, motionWidth), NumOps.FromDouble(scale));
                lastFeature = Engine.TensorMultiplyScalar(
                    Resize(feature, motionHeight, motionWidth), NumOps.FromDouble(scale));
                interpolation = Resize(interpolation!, levelHeight, levelWidth);
            }

            var pyramid0 = ExtractFeaturePyramid(image0);
            var pyramid1 = ExtractFeaturePyramid(image1);
            if (skipMotionAtFinest)
            {
                flow = lastFlow;
                feature = lastFeature;
            }
            else
            {
                (flow, feature) = EstimateMotion(
                    pyramid0[2], pyramid1[2], lastFeature, lastFlow);
            }
            interpolation = Synthesize(
                interpolation, image0, image1, pyramid0, pyramid1, flow, time);
        }

        var result = interpolation ?? throw new InvalidOperationException("UPR-Net produced no pyramid output.");
        return wasBatched ? result : Engine.Reshape(result, [3, result.Shape[2], result.Shape[3]]);
    }

    private Tensor<T>[] ExtractFeaturePyramid(Tensor<T> image)
    {
        var stage0 = Apply(_featureStage0, image);
        var stage1 = Apply(_featureStage1, stage0);
        var stage2 = Apply(_featureStage2, stage1);
        return [stage0, stage1, stage2];
    }

    private (Tensor<T> Flow, Tensor<T> Feature) EstimateMotion(
        Tensor<T> feature0,
        Tensor<T> feature1,
        Tensor<T> lastFeature,
        Tensor<T> lastFlow)
    {
        var flow0 = Engine.TensorMultiplyScalar(
            SliceChannels(lastFlow, 0, 2), NumOps.FromDouble(0.125));
        var flow1 = Engine.TensorMultiplyScalar(
            SliceChannels(lastFlow, 2, 2), NumOps.FromDouble(0.125));
        var warped0 = Engine.ForwardSplat(feature0, flow0);
        var warped1 = Engine.ForwardSplat(feature1, flow1);
        var volume = Engine.LeakyReLU(
            Engine.PartialCorrelationVolume(warped0, warped1, CorrelationRadius),
            NumOps.FromDouble(0.1));
        if (volume.Shape[1] != CorrelationChannels)
            throw new InvalidOperationException(
                $"Radius-four partial correlation must produce {CorrelationChannels} channels.");

        var motionInput = ConcatChannels([volume, warped0, warped1, lastFeature, lastFlow]);
        Tensor<T> current = motionInput;
        for (int i = 0; i < _motionEstimator.Count - 1; i++)
            current = _motionEstimator[i].Forward(current);
        var feature = current;
        var flow = _motionEstimator[^1].Forward(feature);
        return (flow, feature);
    }

    private Tensor<T> Synthesize(
        Tensor<T>? lastInterpolation,
        Tensor<T> image0,
        Tensor<T> image1,
        Tensor<T>[] pyramid0,
        Tensor<T>[] pyramid1,
        Tensor<T> flow,
        double time)
    {
        int height = image0.Shape[2];
        int width = image0.Shape[3];
        var fullFlow = Resize(flow, height, width);
        var flow0t = Engine.TensorMultiplyScalar(
            SliceChannels(fullFlow, 0, 2), NumOps.FromDouble(time));
        var flow1t = Engine.TensorMultiplyScalar(
            SliceChannels(fullFlow, 2, 2), NumOps.FromDouble(1.0 - time));
        var warpedImage0 = Engine.ForwardSplat(image0, flow0t);
        var warpedImage1 = Engine.ForwardSplat(image1, flow1t);
        lastInterpolation ??= Engine.TensorAdd(
            Engine.TensorMultiplyScalar(warpedImage0, NumOps.FromDouble(1.0 - time)),
            Engine.TensorMultiplyScalar(warpedImage1, NumOps.FromDouble(time)));

        var flowPairs = Engine.TensorConcatenate([flow0t, flow1t], axis: 1);
        var s0 = Apply(_synthEncoder0,
            ConcatChannels([lastInterpolation, warpedImage0, warpedImage1, image0, image1, flowPairs]));

        var warpedC00 = Engine.ForwardSplat(pyramid0[0], flow0t);
        var warpedC10 = Engine.ForwardSplat(pyramid1[0], flow1t);
        var s1 = Apply(_synthEncoder1, ConcatChannels([s0, warpedC00, warpedC10]));

        var halfFlow = Engine.TensorMultiplyScalar(
            Resize(fullFlow, pyramid0[1].Shape[2], pyramid0[1].Shape[3]),
            NumOps.FromDouble(0.5));
        var warpedC01 = Engine.ForwardSplat(
            pyramid0[1], Engine.TensorMultiplyScalar(SliceChannels(halfFlow, 0, 2), NumOps.FromDouble(time)));
        var warpedC11 = Engine.ForwardSplat(
            pyramid1[1], Engine.TensorMultiplyScalar(SliceChannels(halfFlow, 2, 2), NumOps.FromDouble(1.0 - time)));
        var s2 = Apply(_synthEncoder2, ConcatChannels([s1, warpedC01, warpedC11]));

        var quarterFlow = Engine.TensorMultiplyScalar(
            Resize(halfFlow, pyramid0[2].Shape[2], pyramid0[2].Shape[3]),
            NumOps.FromDouble(0.5));
        var warpedC02 = Engine.ForwardSplat(
            pyramid0[2], Engine.TensorMultiplyScalar(SliceChannels(quarterFlow, 0, 2), NumOps.FromDouble(time)));
        var warpedC12 = Engine.ForwardSplat(
            pyramid1[2], Engine.TensorMultiplyScalar(SliceChannels(quarterFlow, 2, 2), NumOps.FromDouble(1.0 - time)));

        var d1 = Apply(_synthDecoder1, ConcatChannels([s2, warpedC02, warpedC12]));
        var d2 = Apply(_synthDecoder2, ConcatChannels([d1, s1]));
        var d0 = Apply(_synthDecoder0, ConcatChannels([d2, s0]));
        var prediction = _synthPrediction!.Forward(d0);

        var residual = Engine.TensorSubtract(
            Engine.TensorMultiplyScalar(
                Engine.TensorSigmoid(SliceChannels(prediction, 0, 3)), NumOps.FromDouble(2.0)),
            Engine.TensorAddScalar(
                Engine.TensorMultiplyScalar(SliceChannels(prediction, 0, 3), NumOps.Zero),
                NumOps.One));
        var mask0 = Engine.TensorSigmoid(SliceChannels(prediction, 3, 1));
        var mask1 = Engine.TensorSigmoid(SliceChannels(prediction, 4, 1));
        var weight0 = Engine.TensorMultiplyScalar(mask0, NumOps.FromDouble(1.0 - time));
        var weight1 = Engine.TensorMultiplyScalar(mask1, NumOps.FromDouble(time));
        var numerator = Engine.TensorAdd(
            Engine.TensorMultiply(warpedImage0, weight0),
            Engine.TensorMultiply(warpedImage1, weight1));
        var denominator = Engine.TensorAddScalar(
            Engine.TensorAdd(weight0, weight1), NumOps.FromDouble(SynthesisBlendEpsilon));
        var merged = Engine.TensorDivide(numerator, denominator);
        return Engine.TensorClamp(Engine.TensorAdd(merged, residual), NumOps.Zero, NumOps.One);
    }

    private static Tensor<T> Apply(IReadOnlyList<ILayer<T>> layers, Tensor<T> input)
    {
        var output = input;
        foreach (var layer in layers) output = layer.Forward(output);
        return output;
    }

    private Tensor<T> Resize(Tensor<T> input, int height, int width)
    {
        if (input.Shape[2] == height && input.Shape[3] == width) return input;
        return Engine.Interpolate(input, [height, width], InterpolateMode.Bilinear, alignCorners: false);
    }

    private Tensor<T> SliceChannels(Tensor<T> input, int start, int count) =>
        Engine.TensorSlice(input,
            [0, start, 0, 0], [input.Shape[0], count, input.Shape[2], input.Shape[3]]);

    private Tensor<T> ConcatChannels(Tensor<T>[] tensors) =>
        Engine.TensorConcatenate(tensors, axis: 1);

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ForwardAtTime(input, 0.5);

    /// <inheritdoc />
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => rawFrames;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // Forward via Forward(), MSE loss against expected, then a single
            // optimizer step on the collected layer parameters. We don't
            // delegate to TrainWithTape because the UPR-Net forward isn't a
            // simple Layers iteration — it has the pyramid recurrence and
            // bilinear warps that can't be expressed as a flat layer chain.
            // For the smoke-test invariants this performs one supervised step
            // by gradient descent on the per-level Conv weights via
            // numerical-style finite-difference handled inside the engine's
            // tape (Layers contains the convs, so the optimizer sees them).
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            ["ModelName"] = "UPRNet",
            ["FeaturePyramidChannels"] = "16,32,64",
            ["NumPyramidLevels"] = _options.NumPyramidLevels,
            ["CorrelationRadius"] = CorrelationRadius,
            ["CorrelationChannels"] = CorrelationChannels,
            ["LayerCount"] = 47,
            ["MotionInputChannels"] = 277,
            ["MotionOutputChannels"] = 4,
            ["SynthesisOutputChannels"] = 5,
            ["RecurrentSharing"] = "one shared motion estimator and synthesis network across levels",
            ["Warp"] = "normalized forward soft splat"
        },
        ModelData = SerializeForMetadata()
    };

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumPyramidLevels);
        writer.Write(_options.NumLevelsSkipped);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumPyramidLevels = reader.ReadInt32();
        _options.NumLevelsSkipped = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
        _options.Validate();
        EnsurePaperBindings();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        !_useNativeMode && !string.IsNullOrWhiteSpace(_options.ModelPath)
            ? new UPRNet<T>(Architecture, _options.ModelPath!, new UPRNetOptions(_options))
            : new UPRNet<T>(Architecture, new UPRNetOptions(_options));

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(UPRNet<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
