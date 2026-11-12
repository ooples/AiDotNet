using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// FloRNN: unidirectional video denoising with a look-ahead recurrent module that mimics a backward one
/// (Li et al., ECCV 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network follows the authors' release (github.com/nagejacob/FloRNN, models/flornn.py). Two recurrent modules,
/// each a 3x3 convolution followed by residual blocks (conv-ReLU-conv plus identity, no biases), read the noisy
/// frame, its noise-level map and the previous hidden state:
/// </para>
/// <list type="bullet">
/// <item>The forward module runs at the output frame <c>t</c>; its state is backward-warped from <c>t-1</c> by the
/// flow from <c>t</c> to <c>t-1</c>.</item>
/// <item>The look-ahead ("backward") module runs <see cref="FloRNNOptions.ForwardCount"/> frames ahead; its state
/// is carried forward by softmax splatting ("average") over a zero border, and its newest state is brought back to
/// <c>t</c> by backward warps along the same flows, the border restored from the splats at each step.</item>
/// <item>A decoder (conv-ReLU-conv) maps both aligned states to the denoised frame.</item>
/// </list>
/// <para>
/// The sequence is reflected past its end so the last frames have frames to look ahead to. The paper's flow is a
/// pretrained PWC-Net that is never trained; here a SPyNet estimator plays that role and is likewise excluded from
/// training (load pretrained weights into it for meaningful alignment). Weights are Kaiming-normal scaled by 0.2,
/// as in the release. Training feeds the noise level the noise was drawn with (the RMS of noisy minus clean) and
/// scores frames 1 to T-ForwardCount-1, as the authors' script does; their summed MSE over 2N is a constant multiple
/// of the mean used here.
/// </para>
/// <para><b>For Beginners:</b> FloRNN denoises video frame by frame, but it also looks a few frames into the
/// future and warps what it saw there back to the current frame, getting most of the benefit of a two-way model
/// while still running in a single pass.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Generative,
///     inputHeight: 96, inputWidth: 96, inputDepth: 3);
/// var flornn = new FloRNN&lt;double&gt;(architecture, new FloRNNOptions());
/// var clean = flornn.Denoise(noisyFrames, noiseLevel: 25.0 / 255.0); // [T, C, H, W] in [0, 1]
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unidirectional Video Denoising by Mimicking Backward Recurrent Modules with Look-ahead Forward Ones",
    "https://arxiv.org/abs/2204.05532",
    Year = 2022,
    Authors = "Junyi Li, Xiaohe Wu, Zhenxing Niu, Wangmeng Zuo")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4,
                Source = "Li et al. 2022, Sec. 4.1: Adam with an initial learning rate of 1e-4.")]
public partial class FloRNN<T> : VideoDenoisingBase<T>
{
    private readonly FloRNNOptions _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>
    /// The released FloRNN model consumes and emits normalized image tensors.
    /// </summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.NormalizedValue;

    /// <inheritdoc/>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.NormalizedValue;

    /// <summary>
    /// Creates a FloRNN model for ONNX inference.
    /// </summary>
    public FloRNN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FloRNNOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FloRNNOptions();
        Options = _options;
        _useNativeMode = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FloRNN model for native training and inference.
    /// </summary>
    public FloRNN(
        NeuralNetworkArchitecture<T> architecture,
        FloRNNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FloRNNOptions();
        Options = _options;
        if (_options.NumFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(options), "NumFeatures must be positive.");
        if (_options.NumResBlocks < 0) throw new ArgumentOutOfRangeException(nameof(options), "NumResBlocks cannot be negative.");
        if (_options.ForwardCount < 1) throw new ArgumentOutOfRangeException(nameof(options), "ForwardCount must be at least 1.");
        if (_options.BorderRatio < 0 || _options.BorderRatio >= 0.5)
            throw new ArgumentOutOfRangeException(nameof(options), "BorderRatio must be in [0, 0.5).");
        _useNativeMode = true;
        // The released FloRNN training script uses torch.optim.Adam at 1e-4
        // with Adam's standard fixed moments (train_models/sRGB_train.py).
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    Beta1 = 0.9,
                    Beta2 = 0.999,
                    Epsilon = 1e-8,
                    UseAdaptiveLearningRate = false,
                    UseAdaptiveBetas = false,
                    UseAMSGrad = false
                }));
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Denoise(Tensor<T> noisyFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(noisyFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <summary>
    /// Denoises frames <c>[T, C, H, W]</c> (or <c>[N, T, C, H, W]</c>) corrupted by noise of the given standard
    /// deviation, the value the noise-level map carries.
    /// </summary>
    public Tensor<T> Denoise(Tensor<T> noisyFrames, double noiseLevel)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return Denoise(noisyFrames);
        using var noGrad = new NoGradScope<T>();
        int batch = noisyFrames.Rank == 5 ? noisyFrames.Shape[0] : 1;
        return Run(noisyFrames, Enumerable.Repeat(noiseLevel, batch).ToArray());
    }

    #region Layers

    // Layers: [0] the frozen flow estimator, then the forward module (first conv, two convs per residual block),
    // the look-ahead module laid out the same way, and the decoder's two convs.
    private int ModuleLength => 1 + 2 * _options.NumResBlocks;
    private const int ForwardStart = 1;
    private int BackwardStart => ForwardStart + ModuleLength;
    private int DecoderStart => BackwardStart + ModuleLength;

    private SpyNetLayer<T> FlowNet => Layers.Count > 0 && Layers[0] is SpyNetLayer<T> flow
        ? flow
        : throw new InvalidOperationException("FloRNN's flow estimator is missing.");

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.AddRange(Architecture.Layers);
            return;
        }

        int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        int f = _options.NumFeatures;
        var rng = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        Layers.Add(new SpyNetLayer<T>(_options.FlowPyramidLevels));
        for (int module = 0; module < 2; module++)
        {
            Layers.Add(Conv(ch + ch + f, f, relu: false, rng));
            for (int b = 0; b < _options.NumResBlocks; b++)
            {
                Layers.Add(Conv(f, f, relu: true, rng));
                Layers.Add(Conv(f, f, relu: false, rng));
            }
        }
        Layers.Add(Conv(2 * f, 2 * f, relu: true, rng));
        Layers.Add(Conv(2 * f, ch, relu: false, rng));
    }

    // models/init.py: kaiming_normal_(fan_in, relu) scaled by gain 0.2, no bias (the release also clamps the draw to
    // [-1, 1] before scaling, which at these fan-ins touches well under one weight in ten thousand).
    private static ConvolutionalLayer<T> Conv(int inChannels, int outChannels, bool relu, Random rng) =>
        new(outChannels, kernelSize: 3, stride: 1, padding: 1,
            activationFunction: relu ? new ReLUActivation<T>() : null,
            initializationStrategy: new NormalInitializationStrategy<T>(rng, 0.0, 0.2 * Math.Sqrt(2.0 / (9.0 * inChannels))),
            nonlinearityForInit: null, groups: 1, biasMode: BiasMode.Never);

    private Tensor<T> RunModule(int start, Tensor<T> input)
    {
        var h = Layers[start].Forward(input);
        for (int b = 0; b < _options.NumResBlocks; b++)
        {
            var residual = Layers[start + 2 + 2 * b].Forward(Layers[start + 1 + 2 * b].Forward(h));
            h = Engine.TensorAdd(residual, h);
        }
        return h;
    }

    /// <inheritdoc/>
    protected override void ResolveLazyLayerShapes()
    {
        if (_usingCustomLayers || !_useNativeMode) return;
        int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 32;
        int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 32;
        using var noGrad = new NoGradScope<T>();
        Run(new Tensor<T>(new[] { 2, ch, h, w }), new[] { 0.0 });
    }

    #endregion

    #region Forward

    /// <inheritdoc/>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        return Run(input, InferenceNoiseLevels(input));
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.ForwardForTraining(input);
        return Run(PreprocessFrames(input), InferenceNoiseLevels(input));
    }

    private double[] InferenceNoiseLevels(Tensor<T> input)
    {
        bool batched = input.Rank == 5;
        int batch = batched ? input.Shape[0] : 1;
        var levels = new double[batch];
        for (int n = 0; n < batch; n++)
        {
            levels[n] = _options.NoiseLevel
                ?? EstimateNoiseLevel(batched ? input.Slice(n) : input);
        }
        return levels;
    }

    // Reflects an index past the end of a sequence of `length` frames (seqn[:, T+i] = seqn[:, T-2-i]).
    internal static int ReflectFrame(int index, int length)
    {
        if (index < length) return index;
        if (length == 1) return 0;
        int period = 2 * (length - 1);
        int k = index % period;
        return k < length ? k : period - k;
    }

    /// <summary>models/flornn.py, FloRNN.forward, over frames [T, C, H, W] or [N, T, C, H, W].</summary>
    private Tensor<T> Run(Tensor<T> input, IReadOnlyList<double> noiseLevels)
    {
        if (input.Rank is not (4 or 5))
            throw new ArgumentException($"FloRNN expects frames [T, C, H, W] or [N, T, C, H, W]; got rank {input.Rank}.", nameof(input));
        bool batched = input.Rank == 5;
        var seq = batched ? input : Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3] });
        int n = seq.Shape[0], frames = seq.Shape[1], c = seq.Shape[2], h = seq.Shape[3], w = seq.Shape[4];
        int fc = _options.ForwardCount, f = _options.NumFeatures;
        int bh = (int)(h * _options.BorderRatio), bw = (int)(w * _options.BorderRatio);

        var x = new Tensor<T>[frames + fc];
        for (int i = 0; i < x.Length; i++)
            x[i] = Engine.Reshape(Engine.TensorNarrow(seq, 1, ReflectFrame(i, frames), 1), new[] { n, c, h, w });
        var noiseMap = NoiseMap(n, c, h, w, noiseLevels);
        var zeros = new Tensor<T>(new[] { n, f, h, w });

        var backwardH = RunModule(BackwardStart, Engine.TensorConcatenate(new[] { x[0], noiseMap, zeros }, 1));
        var forwardH = RunModule(ForwardStart, Engine.TensorConcatenate(new[] { x[0], noiseMap, zeros }, 1));

        var flowQueue = new List<ExpandedFlow>();
        var borderQueue = new List<Tensor<T>>();
        var outputs = new Tensor<T>[frames];
        var ring = RingMask(h + 2 * bh, w + 2 * bw, bh, bw);

        for (int i = 1; i < frames + fc; i++)
        {
            var forwardFlow = EstimateFlow(x[i - 1], x[i], bh, bw);
            flowQueue.Add(forwardFlow);
            var splatted = SoftSplatAverage(Pad(backwardH, bh, bw), forwardFlow);
            borderQueue.Add(splatted);
            var alignedBackward = Crop(splatted, bh, bw, h, w);
            backwardH = RunModule(BackwardStart, Engine.TensorConcatenate(new[] { x[i], noiseMap, alignedBackward }, 1));

            if (i < fc) continue;

            // Bring the look-ahead state at frame i back to frame i - fc, one flow at a time, restoring the border
            // the splat pushed features into before each warp.
            var aligned = backwardH;
            for (int j = fc - 1; j >= 0; j--)
            {
                var merged = Engine.TensorAdd(Pad(aligned, bh, bw), Engine.TensorMultiply(borderQueue[j], ring));
                aligned = Crop(WarpMasked(merged, flowQueue[j]), bh, bw, h, w);
            }

            outputs[i - fc] = Decode(Engine.TensorConcatenate(new[] { forwardH, aligned }, 1));

            var backwardFlow = EstimateFlow(x[i - fc + 1], x[i - fc], 0, 0);
            var alignedForward = WarpMasked(forwardH, backwardFlow);
            forwardH = RunModule(ForwardStart, Engine.TensorConcatenate(new[] { x[i - fc + 1], noiseMap, alignedForward }, 1));
            flowQueue.RemoveAt(0);
            borderQueue.RemoveAt(0);
        }

        var stacked = Engine.TensorConcatenate(
            outputs.Select(o => Engine.Reshape(o, new[] { n, 1, c, h, w })).ToArray(), 1);
        return batched ? stacked : Engine.Reshape(stacked, new[] { frames, c, h, w });
    }

    private Tensor<T> Decode(Tensor<T> features)
        => Layers[DecoderStart + 1].Forward(Layers[DecoderStart].Forward(features));

    private Tensor<T> NoiseMap(int n, int c, int h, int w, IReadOnlyList<double> levels)
    {
        var map = new Tensor<T>(new[] { n, c, h, w });
        var span = map.Data.Span;
        int perSample = c * h * w;
        for (int b = 0; b < n; b++)
        {
            var value = NumOps.FromDouble(levels[Math.Min(b, levels.Count - 1)]);
            for (int k = 0; k < perSample; k++) span[b * perSample + k] = value;
        }
        return map;
    }

    private Tensor<T> Pad(Tensor<T> tensor, int bh, int bw)
        => bh == 0 && bw == 0 ? tensor : Engine.Pad(tensor, bh, bh, bw, bw, NumOps.Zero);

    private Tensor<T> Crop(Tensor<T> tensor, int bh, int bw, int h, int w)
        => bh == 0 && bw == 0 ? tensor : Engine.TensorNarrow(Engine.TensorNarrow(tensor, 2, bh, h), 3, bw, w);

    // 1 on the border ring of an expanded [He, We] frame, 0 inside (merge_border keeps the splat there).
    private Tensor<T> RingMask(int he, int we, int bh, int bw)
    {
        var mask = new Tensor<T>(new[] { 1, 1, he, we });
        var span = mask.Data.Span;
        for (int y = 0; y < he; y++)
            for (int xx = 0; xx < we; xx++)
            {
                bool inside = y >= bh && y < he - bh && xx >= bw && xx < we - bw;
                span[y * we + xx] = inside ? NumOps.Zero : NumOps.One;
            }
        return mask;
    }

    /// <summary>A flow from one frame to the next, zero-padded by the border, as a tensor and as values.</summary>
    private sealed class ExpandedFlow
    {
        public ExpandedFlow(Tensor<T> tensor, double[] values, int batch, int height, int width)
        {
            Tensor = tensor;
            Values = values;
            Batch = batch;
            Height = height;
            Width = width;
        }

        public Tensor<T> Tensor { get; }
        public double[] Values { get; }
        public int Batch { get; }
        public int Height { get; }
        public int Width { get; }
        public double Dx(int b, int y, int x) => Values[((b * 2) * Height + y) * Width + x];
        public double Dy(int b, int y, int x) => Values[((b * 2 + 1) * Height + y) * Width + x];
    }

    // extract_flow_torch(pwcnet, from, to): the frozen estimator's flow, no gradient (the paper never trains it).
    private ExpandedFlow EstimateFlow(Tensor<T> from, Tensor<T> to, int bh, int bw)
    {
        Tensor<T> flow;
        using (new NoGradScope<T>())
        {
            flow = FlowNet.EstimateFlow(from, to);
            if (flow.Rank == 3) flow = Engine.Reshape(flow, new[] { 1, flow.Shape[0], flow.Shape[1], flow.Shape[2] });
            flow = Pad(flow, bh, bw);
        }
        return WrapFlow(flow);
    }

    private ExpandedFlow WrapFlow(Tensor<T> flow)
    {
        var values = new double[flow.Length];
        var span = flow.Data.Span;
        for (int k = 0; k < values.Length; k++) values[k] = NumOps.ToDouble(span[k]);
        return new ExpandedFlow(flow, values, flow.Shape[0], flow.Shape[2], flow.Shape[3]);
    }

    /// <summary>
    /// Softmax splatting in its "average" mode (Niklaus and Liu 2020): every source pixel is pushed to
    /// <c>p + flow(p)</c> with bilinear weights, and each target is divided by the weight it received (1 where none).
    /// The flow is a constant, so this is linear in the features and differentiable through them.
    /// </summary>
    private Tensor<T> SoftSplatAverage(Tensor<T> features, ExpandedFlow flow)
    {
        int n = features.Shape[0], f = features.Shape[1], he = features.Shape[2], we = features.Shape[3];
        int pixels = n * he * we;
        var rows = Engine.Reshape(Engine.TensorPermute(features, new[] { 0, 2, 3, 1 }), new[] { pixels, f });
        var norm = new double[pixels];
        Tensor<T>? sum = null;
        for (int corner = 0; corner < 4; corner++)
        {
            int dxCorner = corner & 1, dyCorner = corner >> 1;
            var indices = new Tensor<int>(new[] { pixels });
            var weights = new Tensor<T>(new[] { pixels, 1 });
            var indexSpan = indices.Data.Span;
            var weightSpan = weights.Data.Span;
            for (int b = 0; b < n; b++)
                for (int y = 0; y < he; y++)
                    for (int xx = 0; xx < we; xx++)
                    {
                        int source = (b * he + y) * we + xx;
                        double fx = xx + flow.Dx(b, y, xx), fy = y + flow.Dy(b, y, xx);
                        double x0 = Math.Floor(fx), y0 = Math.Floor(fy);
                        int tx = (int)x0 + dxCorner, ty = (int)y0 + dyCorner;
                        double ax = fx - x0, ay = fy - y0;
                        double weight = (dxCorner == 0 ? 1 - ax : ax) * (dyCorner == 0 ? 1 - ay : ay);
                        if (tx < 0 || tx >= we || ty < 0 || ty >= he || double.IsNaN(weight))
                        {
                            indexSpan[source] = 0;
                            weightSpan[source] = NumOps.Zero;
                            continue;
                        }
                        int target = (b * he + ty) * we + tx;
                        indexSpan[source] = target;
                        weightSpan[source] = NumOps.FromDouble(weight);
                        norm[target] += weight;
                    }
            var part = Engine.ScatterAdd(Engine.TensorMultiply(rows, weights), indices, 0, pixels);
            sum = sum is null ? part : Engine.TensorAdd(sum, part);
        }

        var inverse = new Tensor<T>(new[] { pixels, 1 });
        var inverseSpan = inverse.Data.Span;
        for (int k = 0; k < pixels; k++) inverseSpan[k] = NumOps.FromDouble(norm[k] == 0 ? 1.0 : 1.0 / norm[k]);
        var averaged = Engine.TensorMultiply(sum ?? rows, inverse);
        return Engine.TensorPermute(Engine.Reshape(averaged, new[] { n, he, we, f }), new[] { 0, 3, 1, 2 });
    }

    /// <summary>
    /// utils/warp.py: bilinear backward warp, with pixels whose sample reaches outside the frame set to zero.
    /// </summary>
    private Tensor<T> WarpMasked(Tensor<T> features, ExpandedFlow flow)
    {
        int n = flow.Batch, he = flow.Height, we = flow.Width;
        var mask = new Tensor<T>(new[] { n, 1, he, we });
        var span = mask.Data.Span;
        for (int b = 0; b < n; b++)
            for (int y = 0; y < he; y++)
                for (int xx = 0; xx < we; xx++)
                {
                    double fx = xx + flow.Dx(b, y, xx), fy = y + flow.Dy(b, y, xx);
                    double x0 = Math.Floor(fx), y0 = Math.Floor(fy), ax = fx - x0, ay = fy - y0, inBounds = 0;
                    for (int corner = 0; corner < 4; corner++)
                    {
                        int dxCorner = corner & 1, dyCorner = corner >> 1;
                        double tx = x0 + dxCorner, ty = y0 + dyCorner;
                        if (tx >= 0 && tx < we && ty >= 0 && ty < he)
                            inBounds += (dxCorner == 0 ? 1 - ax : ax) * (dyCorner == 0 ? 1 - ay : ay);
                    }
                    span[(b * he + y) * we + xx] = inBounds >= 0.9999 ? NumOps.One : NumOps.Zero;
                }
        return Engine.TensorMultiply(FlowWarpHelper.Warp(Engine, features, flow.Tensor), mask);
    }

    /// <summary>Test hook: softmax splatting ("average") of features [N, F, H, W] by a flow [N, 2, H, W].</summary>
    internal Tensor<T> SoftSplatForTest(Tensor<T> features, Tensor<T> flow) => SoftSplatAverage(features, WrapFlow(flow));

    /// <summary>Test hook: the masked backward warp of features [N, F, H, W] by a flow [N, 2, H, W].</summary>
    internal Tensor<T> WarpMaskedForTest(Tensor<T> features, Tensor<T> flow) => WarpMasked(features, WrapFlow(flow));

    /// <summary>Test hook: the parameters of the frozen flow estimator.</summary>
    internal IReadOnlyList<Tensor<T>> FlowEstimatorParameters
        => TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { FlowNet });

    #endregion

    #region Training

    // The splat indices and warp masks are host values computed from each step's flow, which a compiled plan would
    // bake at trace time and replay for every later step.
    /// <inheritdoc/>
    protected override bool SupportsFusedCompiledTraining => false;

    private IReadOnlyList<Tensor<T>> TrainableParameters()
        => TapeTrainingStep<T>.CollectParameters(Layers.Skip(1).Cast<ILayer<T>>());

    /// <inheritdoc/>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_usingCustomLayers || Layers.Count == 0) return parameters;
        var frozen = TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { Layers[0] });
        return parameters.Where(p => !frozen.Any(q => ReferenceEquals(p, q))).ToList();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// train_models/sRGB_train.py: the noise-level map carries the σ the noise was drawn with (here the RMS of noisy
    /// minus clean per sequence), the loss is the MSE over frames 1 to T-ForwardCount-1 (all frames when the clip is
    /// too short to leave any), and the flow estimator is not trained.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, data =>
            {
                var noisy = data[0];
                var clean = data[1];
                var denoised = Run(noisy, TrainingNoiseLevels(noisy, clean));
                int frameAxis = noisy.Rank == 5 ? 1 : 0;
                int frames = noisy.Shape[frameAxis];
                int scored = frames - _options.ForwardCount - 1;
                if (scored >= 1)
                {
                    denoised = Engine.TensorNarrow(denoised, frameAxis, 1, scored);
                    clean = Engine.TensorNarrow(clean, frameAxis, 1, scored);
                }
                return LossFunction.ComputeTapeLoss(denoised, clean);
            })
            {
                Optimizer = _optimizer,
                Parameters = TrainableParameters,
            }
        };
    }

    private double[] TrainingNoiseLevels(Tensor<T> noisy, Tensor<T> clean)
    {
        int batch = noisy.Rank == 5 ? noisy.Shape[0] : 1;
        int perSample = noisy.Length / batch;
        var levels = new double[batch];
        var a = noisy.Data.Span;
        var b = clean.Data.Span;
        for (int s = 0; s < batch; s++)
        {
            double sum = 0;
            for (int k = s * perSample; k < (s + 1) * perSample; k++)
            {
                double d = NumOps.ToDouble(a[k]) - NumOps.ToDouble(b[k]);
                sum += d * d;
            }
            levels[s] = Math.Sqrt(sum / perSample);
        }
        return levels;
    }

    #endregion

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => rawFrames;

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessTarget(Tensor<T> expectedOutput) => expectedOutput;

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FloRNN" },
                { "NumFeatures", _options.NumFeatures },
                { "NumResBlocks", _options.NumResBlocks },
                { "ForwardCount", _options.ForwardCount },
                { "BorderRatio", _options.BorderRatio }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FloRNN<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
