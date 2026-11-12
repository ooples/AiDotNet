using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// Shift-Net: video denoising with grouped spatial-temporal shift (Li et al., CVPR 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is the authors' denoising release (github.com/dasongli1/Shift-Net, gshift_denoise1.py, GShiftNet).
/// The frames of a clip are processed as a batch:
/// </para>
/// <list type="bullet">
/// <item>A feature extractor (3x3 conv over the frame and its noise-level map, then a channel-attention block, CAB)
/// and five refinement U-Nets (three-level U-Nets of CABs, stride-2 conv down, bilinear up with a 1x1 conv).</item>
/// <item>The shift encoder at half resolution: CABs whose input is first <b>temporally shifted</b> (half of each
/// frame's channels come from the previous frame, or the next in the reverse block), a three-level encoder, and
/// seven decoder blocks of eight shift stages. Each shift stage shifts in time, then <b>shifts in space</b> half of
/// the channels in 24 groups by (±s | ±2s | 0)², feeds them through a depthwise conv into a gated block (layer norm,
/// SimpleGate, channel attention, re-parameterizable convs, a zero-initialized per-channel scale), and alternates the
/// time direction. A pixel-shuffle upsample returns to full resolution.</item>
/// <item>A second set of five refinement U-Nets over the fused features and a 3x3 conv to the residual that is added
/// to the noisy frame.</item>
/// </list>
/// <para>
/// Training follows options/davis_denoise.yml: BasicSR's PSNR loss (weight 0.5), Adam at 4e-4 with betas
/// (0.9, 0.99) cosine-annealed to 1e-7 per iteration. The noise map carries the RMS of noisy minus clean, the σ the
/// noise was drawn with. Frames must be a multiple of 8 in each spatial size or are zero-padded to one.
/// </para>
/// <para>
/// The paper's text describes a displacement set of {-9, -5, 0, 5, 9}; the released network, whose numbers the paper
/// reports, shifts by multiples of s = 4, and that is what is built here (see
/// <see cref="GroupedSpatialTemporalShift{T}"/> for the text's version).
/// </para>
/// <para><b>For Beginners:</b> Instead of estimating motion, Shift-Net slides copies of each frame's features to
/// the neighbouring frames and by a few pixels in many directions, and lets the following layers pick the slide that
/// lines up. Sliding costs nothing, so the model is fast.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Generative,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var shiftNet = new ShiftNet&lt;double&gt;(architecture, new ShiftNetOptions());
/// var clean = shiftNet.Denoise(noisyFrames); // [T, 3, H, W], pixel values 0-255
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Simple Baseline for Video Restoration with Grouped Spatial-temporal Shift",
    "https://arxiv.org/abs/2206.10810",
    Year = 2023,
    Authors = "Dasong Li, Xiaoyu Shi, Yi Zhang, Ka Chun Cheung, Simon See, Xiaogang Wang, Hongwei Qin, Hongsheng Li")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 4e-4, ReferenceBatchSize = 8,
                Beta2 = 0.99,
                MinLearningRate = 1e-7,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Li et al. 2022, Sec. 4.1: the Adam optimizer with the learning rate decreased "
                        + "from 4e-4 to 1e-7 by cosine annealing, at a batch size of 8 over 750 epochs; "
                        + "betas (0.9, 0.99) from options/davis_denoise.yml.")]
public partial class ShiftNet<T> : VideoDenoisingBase<T>
{
    private readonly ShiftNetOptions _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private Net? _net;

    /// <summary>
    /// Creates a ShiftNet model for ONNX inference.
    /// </summary>
    public ShiftNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ShiftNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ShiftNetOptions();
        Options = _options;
        _useNativeMode = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a ShiftNet model for native training and inference.
    /// </summary>
    public ShiftNet(
        NeuralNetworkArchitecture<T> architecture,
        ShiftNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new PSNRLoss<T>((options ?? new ShiftNetOptions()).PsnrLossWeight))
    {
        _options = options ?? new ShiftNetOptions();
        Options = _options;
        if (_options.NumFeatures < 16 || _options.NumFeatures % 16 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFeatures must be a positive multiple of 16.");
        if (_options.RefinementFeatures < 4 || _options.RefinementFeatures % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "RefinementFeatures must be a positive multiple of 4.");
        if (_options.UNetGrowth < 0 || _options.UNetGrowth % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "UNetGrowth must be a non-negative multiple of 4.");
        if (_options.NumUNets < 1) throw new ArgumentOutOfRangeException(nameof(options), "NumUNets must be at least 1.");
        if (_options.ShiftStride < 1) throw new ArgumentOutOfRangeException(nameof(options), "ShiftStride must be at least 1.");
        _useNativeMode = true;
        // options/davis_denoise.yml: Adam (0.9, 0.99) at 4e-4, TrueCosineAnnealingLR to 1e-7 over 450k iterations.
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    Beta1 = 0.9,
                    Beta2 = 0.99,
                    Epsilon = 1e-8,
                    UseAdaptiveLearningRate = false,
                    UseAdaptiveBetas = false,
                    UseAMSGrad = false,
                    LearningRateScheduler = new CosineAnnealingLRScheduler(
                        _options.LearningRate, Math.Max(1, _options.CosineIterations), _options.MinLearningRate),
                    SchedulerStepMode = SchedulerStepMode.StepPerBatch,
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

    #region Network structure

    // Every block stores the indices of its layers in Layers, so a deserialized network that replaces the layer
    // instances in place stays wired.
    private sealed class ChannelAttention
    {
        public int Down;
        public int Up;
    }

    private sealed class Cab
    {
        public int Conv1;
        public int Conv2;
        public int Act;
        public ChannelAttention Attention = new();
    }

    private sealed class GatedBlock
    {
        public int Depthwise = -1;
        public int Norm;
        public int In;
        public int Rep2;
        public ChannelAttention Attention1 = new();
        public int RepA;
        public int RepB;
        public int Mid;
        public ChannelAttention Attention2 = new();
        public int Out;
        public int Beta;
        public int Features;
    }

    private sealed class Down
    {
        public int Conv;
        public int Act;
    }

    private sealed class UNet
    {
        public Cab[] Enc1 = Array.Empty<Cab>();
        public Cab[] Enc2 = Array.Empty<Cab>();
        public Cab[] Enc3 = Array.Empty<Cab>();
        public Cab[] Dec1 = Array.Empty<Cab>();
        public Cab[] Dec2 = Array.Empty<Cab>();
        public Cab[] Dec3 = Array.Empty<Cab>();
        public Down Down12 = new();
        public Down Down23 = new();
        public int Up32;
        public int Up21;
        public Cab Skip1 = new();
        public Cab Skip2 = new();
    }

    private sealed class ShiftBlock
    {
        public GatedBlock[] Shifted = Array.Empty<GatedBlock>();
        public GatedBlock[] Plain = Array.Empty<GatedBlock>();
    }

    private sealed class Net
    {
        public int FeatConv;
        public Cab FeatCab = new();
        public UNet[] Stage0 = Array.Empty<UNet>();
        public int ConvTrans;
        // Encoder2
        public Cab Concat = new();
        public Cab Level0 = new();
        public Cab Level01 = new();
        public Down Down01 = new();
        public Cab Level1 = new();
        public Cab Level11 = new();
        public Down Down12 = new();
        public Cab Level2 = new();
        public Cab Level21 = new();
        public Down Down23 = new();
        public Cab Level3 = new();
        public Cab Level31 = new();
        public ShiftBlock Dec3 = new();
        public ShiftBlock Dec31 = new();
        public ShiftBlock Dec2 = new();
        public ShiftBlock Dec21 = new();
        public ShiftBlock Dec1 = new();
        public ShiftBlock Dec11 = new();
        public ShiftBlock Dec12 = new();
        public Cab SkipAttn1 = new();
        public Cab SkipAttn2 = new();
        public int Up32;
        public int Up21;
        public int Upsample0;
        public int Shuffle0;
        public Cab SkipConv = new();
        public int ConvHr0;
        public Cab OutConv = new();
        // stage 2
        public int RConcat;
        public int LRelu;
        public UNet[] Stage2 = Array.Empty<UNet>();
        public int ConvLast;
    }

    private int Add(ILayer<T> layer)
    {
        Layers.Add(layer);
        return Layers.Count - 1;
    }

    private int Conv(int outChannels, int kernel, bool bias = false, int stride = 1, int groups = 1,
        IActivationFunction<T>? activation = null, IInitializationStrategy<T>? init = null)
        => Add(new ConvolutionalLayer<T>(outChannels, kernel, stride, kernel % 2 == 1 ? kernel / 2 : 0,
            activation, init ?? new TorchDefaultInitializationStrategy<T>(), null, groups,
            bias ? BiasMode.Always : BiasMode.Never));

    private int PRelu() => Add(new PReLULayer<T>());

    private ChannelAttention NewAttention(int channels, int reduction = 4) => new()
    {
        Down = Conv(Math.Max(1, channels / reduction), 1, activation: new ReLUActivation<T>()),
        Up = Conv(channels, 1, activation: new SigmoidActivation<T>()),
    };

    private Cab NewCab(int channels, int act) => new()
    {
        Conv1 = Conv(channels, 3),
        Act = act,
        Conv2 = Conv(channels, 3),
        Attention = NewAttention(channels),
    };

    private GatedBlock NewGated(int n, int addChannels)
    {
        var block = new GatedBlock { Features = n };
        if (addChannels > 0) block.Depthwise = Conv(addChannels, 3, groups: addChannels);
        block.Norm = Add(new LayerNormalizationLayer<T>(n + addChannels, 1e-6));
        block.In = Conv(2 * n, 1);
        block.Rep2 = Conv(2 * n, 3, groups: 2 * n);
        block.Attention1 = NewAttention(n);
        block.RepA = Conv(n, 5, groups: Math.Max(1, n / 8));
        block.RepB = Conv(n, 3, groups: Math.Max(1, n / 8));
        block.Mid = Conv(2 * n, 1);
        block.Attention2 = NewAttention(n);
        block.Out = Conv(n, 1, bias: true);
        // beta = zeros((1, n, 1, 1)): a per-channel scale, here a bias-free depthwise 1x1 conv initialized to zero.
        block.Beta = Conv(n, 1, groups: n, init: new ZeroInitializationStrategy<T>());
        return block;
    }

    private Down NewDown(int outChannels) => new() { Conv = Conv(outChannels, 3, stride: 2), Act = PRelu() };

    private UNet NewUNet(int n0, int growth)
    {
        int act = PRelu();
        Cab[] Cabs(int count, int channels) => Enumerable.Range(0, count).Select(_ => NewCab(channels, act)).ToArray();
        var u = new UNet { Enc1 = Cabs(1, n0) };
        u.Down12 = NewDown(n0 + growth);
        u.Enc2 = Cabs(3, n0 + growth);
        u.Down23 = NewDown(n0 + 2 * growth);
        u.Enc3 = Cabs(3, n0 + 2 * growth);
        u.Dec3 = Cabs(3, n0 + 2 * growth);
        u.Dec2 = Cabs(3, n0 + growth);
        u.Dec1 = Cabs(1, n0);
        u.Skip1 = NewCab(n0, act);
        u.Skip2 = NewCab(n0 + growth, act);
        u.Up32 = Conv(n0 + growth, 1);
        u.Up21 = Conv(n0, 1);
        return u;
    }

    private ShiftBlock NewShiftBlock(int n)
    {
        int shifted = 8 * (n / 16);
        return new ShiftBlock
        {
            Shifted = Enumerable.Range(0, 8).Select(_ => NewGated(n, shifted)).ToArray(),
            Plain = Enumerable.Range(0, 8).Select(_ => NewGated(n, 0)).ToArray(),
        };
    }

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
        int n0 = _options.RefinementFeatures, n = _options.NumFeatures, growth = _options.UNetGrowth;
        var net = new Net { FeatConv = Conv(n0, 3, bias: true) };
        net.FeatCab = NewCab(n0, PRelu());
        net.Stage0 = Enumerable.Range(0, _options.NumUNets).Select(_ => NewUNet(n0, growth)).ToArray();
        net.ConvTrans = Conv(n0, 3, bias: true);

        // Encoder2: one PReLU shared by its CABs, as `act` is in the release.
        int act = PRelu();
        net.Concat = NewCab(n0, act);
        net.Level0 = NewCab(n0, act);
        net.Level01 = NewCab(n0, act);
        net.Down01 = new Down { Conv = Conv(n, 2, stride: 2), Act = PRelu() };
        net.Level1 = NewCab(n, act);
        net.Level11 = NewCab(n, act);
        net.Down12 = NewDown(n);
        net.Level2 = NewCab(n, act);
        net.Level21 = NewCab(n, act);
        net.Down23 = NewDown(n);
        net.Level3 = NewCab(n, act);
        net.Level31 = NewCab(n, act);
        net.Dec1 = NewShiftBlock(n);
        net.Dec11 = NewShiftBlock(n);
        net.Dec12 = NewShiftBlock(n);
        net.Dec2 = NewShiftBlock(n);
        net.Dec21 = NewShiftBlock(n);
        net.Dec3 = NewShiftBlock(n);
        net.Dec31 = NewShiftBlock(n);
        net.SkipAttn1 = NewCab(n, act);
        net.SkipAttn2 = NewCab(n, act);
        net.Upsample0 = Conv(n0 * 4, 3, bias: true);
        net.Shuffle0 = Add(new PixelShuffleLayer<T>(2));
        net.SkipConv = NewCab(n0, act);
        net.OutConv = NewCab(n0, act);
        net.ConvHr0 = Conv(n0, 3, bias: true);
        net.Up21 = Conv(n, 1);
        net.Up32 = Conv(n, 1);

        net.LRelu = PRelu();
        net.Stage2 = Enumerable.Range(0, _options.NumUNets).Select(_ => NewUNet(n0, growth)).ToArray();
        net.RConcat = Conv(n0, 3);
        net.ConvLast = Conv(ch, 3);
        _net = net;
    }

    #endregion

    #region Blocks

    private Tensor<T> L(int index, Tensor<T> x) => Layers[index].Forward(x);

    private Tensor<T> Attend(ChannelAttention a, Tensor<T> x)
    {
        var pooled = Engine.ReduceMean(x, new[] { 2, 3 }, keepDims: true);
        return Engine.TensorMultiply(x, L(a.Up, L(a.Down, pooled)));
    }

    private Tensor<T> RunCab(Cab c, Tensor<T> x)
        => Engine.TensorAdd(Attend(c.Attention, L(c.Conv2, L(c.Act, L(c.Conv1, x)))), x);

    private Tensor<T> Chunk(Tensor<T> x, int half) => Engine.TensorNarrow(x, 1, half * (x.Shape[1] / 2), x.Shape[1] / 2);

    // LayerNorm2d: normalization over the channels of every pixel.
    private Tensor<T> ChannelNorm(int index, Tensor<T> x)
        => Engine.TensorPermute(L(index, Engine.TensorPermute(x, new[] { 0, 2, 3, 1 })), new[] { 0, 3, 1, 2 });

    // CAB1 (no extra channels) and CAB2 (a depthwise conv over the shifted channels appended after the first n):
    // norm, 1x1 to 2n, RepConv2, SimpleGate, channel attention, RepConv, 1x1 to 2n, SimpleGate2, channel attention,
    // 1x1, and the shortcut plus beta times the result.
    private Tensor<T> RunGated(GatedBlock g, Tensor<T> input)
    {
        int n = g.Features;
        var shortcut = input;
        var x = input;
        if (g.Depthwise >= 0)
        {
            shortcut = Engine.TensorNarrow(input, 1, 0, n);
            var hw = L(g.Depthwise, Engine.TensorNarrow(input, 1, n, input.Shape[1] - n));
            x = Engine.TensorConcatenate(new[] { shortcut, hw }, 1);
        }
        var y = L(g.In, ChannelNorm(g.Norm, x));
        y = Engine.TensorAdd(L(g.Rep2, y), y);
        y = Engine.TensorMultiply(Chunk(y, 0), Chunk(y, 1));
        y = Attend(g.Attention1, y);
        y = Engine.TensorAdd(Engine.TensorAdd(L(g.RepA, y), L(g.RepB, y)), y);
        y = L(g.Mid, y);
        y = Engine.TensorMultiply(Chunk(y, 0), Engine.Sigmoid(Chunk(y, 1)));
        y = Attend(g.Attention2, y);
        y = L(g.Out, y);
        return Engine.TensorAdd(shortcut, L(g.Beta, y));
    }

    /// <summary>
    /// The temporal shift over the frames of a clip (torch.roll of the flattened frame-channel axis by C/2): each
    /// frame takes the last C/2 channels of the previous frame in front of its own first C/2 (or, reversed, its own
    /// last C/2 followed by the next frame's first C/2); the first (last) frame is left as it is.
    /// </summary>
    internal Tensor<T> TemporalShift(Tensor<T> x, bool reverse)
    {
        int frames = x.Shape[0], c = x.Shape[1], s = c / 2;
        if (frames == 1 || s == 0) return x;
        if (!reverse)
        {
            var previousTail = Engine.TensorNarrow(Engine.TensorNarrow(x, 1, c - s, s), 0, 0, frames - 1);
            var ownHead = Engine.TensorNarrow(Engine.TensorNarrow(x, 1, 0, c - s), 0, 1, frames - 1);
            var shifted = Engine.TensorConcatenate(new[] { previousTail, ownHead }, 1);
            return Engine.TensorConcatenate(new[] { Engine.TensorNarrow(x, 0, 0, 1), shifted }, 0);
        }
        var ownTail = Engine.TensorNarrow(Engine.TensorNarrow(x, 1, s, c - s), 0, 0, frames - 1);
        var nextHead = Engine.TensorNarrow(Engine.TensorNarrow(x, 1, 0, s), 0, 1, frames - 1);
        var back = Engine.TensorConcatenate(new[] { ownTail, nextHead }, 1);
        return Engine.TensorConcatenate(new[] { back, Engine.TensorNarrow(x, 0, frames - 1, 1) }, 0);
    }

    // spatial_shift2: 16 groups of n2 channels on the ring at distance 2s and 8 groups of n1 at distance s, as
    // (dy, dx) with out[y, x] = in[y - dy, x - dx] and zeros where the source leaves the frame.
    private static readonly (int Dy, int Dx)[] OuterRing =
    {
        (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
        (1, 2), (1, -2), (0, 2), (0, -2), (-1, 2), (-1, -2),
    };

    private static readonly (int Dy, int Dx)[] InnerRing =
    {
        (1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1),
    };

    /// <summary>
    /// The release's grouped spatial shift of <c>8 * number</c> channels: the ring of displacements in units of s,
    /// outer ring at 2s (n2 = (number - 1) / 2 channels each), inner ring at s (n1 = number - 2 n2 channels each).
    /// </summary>
    internal Tensor<T> SpatialShift(Tensor<T> hw)
    {
        int number = hw.Shape[1] / 8, n2 = (number - 1) / 2, n1 = number - 2 * n2, s = _options.ShiftStride;
        var parts = new List<Tensor<T>>();
        int channel = 0;
        foreach (var (dy, dx) in OuterRing)
        {
            if (n2 > 0) parts.Add(Shift(Engine.TensorNarrow(hw, 1, channel, n2), dy * s, dx * s));
            channel += n2;
        }
        foreach (var (dy, dx) in InnerRing)
        {
            parts.Add(Shift(Engine.TensorNarrow(hw, 1, channel, n1), dy * s, dx * s));
            channel += n1;
        }
        return Engine.TensorConcatenate(parts.ToArray(), 1);
    }

    private Tensor<T> Shift(Tensor<T> x, int dy, int dx)
    {
        int h = x.Shape[2], w = x.Shape[3];
        var padded = Engine.Pad(x, Math.Max(dy, 0), Math.Max(-dy, 0), Math.Max(dx, 0), Math.Max(-dx, 0), NumOps.Zero);
        return Engine.TensorNarrow(Engine.TensorNarrow(padded, 2, Math.Max(-dy, 0), h), 3, Math.Max(-dx, 0), w);
    }

    private Tensor<T> ShiftStage(Tensor<T> x, bool reverse)
    {
        var y = TemporalShift(x, reverse);
        int k = 8 * (_options.NumFeatures / 16);
        var hw = reverse ? Engine.TensorNarrow(y, 1, y.Shape[1] - k, k) : Engine.TensorNarrow(y, 1, 0, k);
        return Engine.TensorConcatenate(new[] { y, SpatialShift(hw) }, 1);
    }

    // Encoder_shift_block: eight (CAB2, CAB1) stages, each after a spatial-temporal shift whose direction alternates.
    private Tensor<T> RunShiftBlock(ShiftBlock block, Tensor<T> x)
    {
        for (int stage = 0; stage < block.Shifted.Length; stage++)
        {
            x = ShiftStage(x, reverse: stage % 2 == 1);
            x = RunGated(block.Plain[stage], RunGated(block.Shifted[stage], x));
        }
        return x;
    }

    private Tensor<T> RunCabs(Cab[] cabs, Tensor<T> x)
    {
        foreach (var cab in cabs) x = RunCab(cab, x);
        return x;
    }

    private Tensor<T> RunDown(Down d, Tensor<T> x) => L(d.Act, L(d.Conv, x));

    // SkipUpSample: bilinear x2 (align_corners=False), 1x1 conv, plus the skip.
    private Tensor<T> SkipUp(int conv, Tensor<T> x, Tensor<T> skip)
    {
        var up = Engine.Interpolate(x, new[] { skip.Shape[2], skip.Shape[3] }, InterpolateMode.Bilinear, false);
        return Engine.TensorAdd(L(conv, up), skip);
    }

    private Tensor<T> RunUNet(UNet u, Tensor<T> x)
    {
        var enc1 = RunCabs(u.Enc1, x);
        var enc2 = RunCabs(u.Enc2, RunDown(u.Down12, enc1));
        var enc3 = RunCabs(u.Enc3, RunDown(u.Down23, enc2));
        var dec3 = RunCabs(u.Dec3, enc3);
        var dec2 = RunCabs(u.Dec2, SkipUp(u.Up32, dec3, RunCab(u.Skip2, enc2)));
        return RunCabs(u.Dec1, SkipUp(u.Up21, dec2, RunCab(u.Skip1, enc1)));
    }

    // Shift_CAB: the temporal shift, then a CAB whose residual is the shifted input.
    private Tensor<T> RunShiftCab(Cab c, Tensor<T> x, bool reverse) => RunCab(c, TemporalShift(x, reverse));

    private Tensor<T> RunEncoder(Net net, Tensor<T> x)
    {
        x = RunCab(net.Concat, x);
        var shortcut = x;
        x = RunShiftCab(net.Level01, RunShiftCab(net.Level0, x, false), true);
        x = RunDown(net.Down01, x);
        var enc11 = RunShiftCab(net.Level11, RunShiftCab(net.Level1, x, false), true);
        var enc22 = RunCab(net.Level21, RunCab(net.Level2, RunDown(net.Down12, enc11)));
        var enc33 = RunCab(net.Level31, RunCab(net.Level3, RunDown(net.Down23, enc22)));
        var dec33 = RunShiftBlock(net.Dec31, RunShiftBlock(net.Dec3, enc33));
        x = SkipUp(net.Up32, dec33, RunCab(net.SkipAttn2, enc22));
        var dec22 = RunShiftBlock(net.Dec21, RunShiftBlock(net.Dec2, x));
        x = SkipUp(net.Up21, dec22, RunCab(net.SkipAttn1, enc11));
        var dec11 = RunShiftBlock(net.Dec12, RunShiftBlock(net.Dec11, RunShiftBlock(net.Dec1, x)));
        var up = L(net.Shuffle0, L(net.Upsample0, dec11));
        var fused = L(net.ConvHr0, Engine.TensorConcatenate(new[] { up, RunCab(net.SkipConv, shortcut) }, 1));
        return RunCab(net.OutConv, fused);
    }

    /// <summary>GShiftNet.forward over one clip [F, C, H, W] (H and W multiples of 8) and its noise map [F, 1, H, W].</summary>
    private Tensor<T> RunClip(Net net, Tensor<T> frames, Tensor<T> noiseMap)
    {
        var x0 = RunCab(net.FeatCab, L(net.FeatConv, Engine.TensorConcatenate(new[] { frames, noiseMap }, 1)));
        var stage0 = x0;
        foreach (var u in net.Stage0) stage0 = RunUNet(u, stage0);
        var sam = L(net.ConvTrans, stage0);
        var decoded = RunEncoder(net, sam);
        var y = L(net.LRelu, L(net.RConcat, Engine.TensorConcatenate(new[] { x0, sam, decoded }, 1)));
        foreach (var u in net.Stage2) y = RunUNet(u, y);
        return Engine.TensorAdd(L(net.ConvLast, y), frames);
    }

    /// <summary>
    /// Denoises normalized frames [F, C, H, W] or [N, F, C, H, W]; <paramref name="sigma"/> holds one noise level per
    /// clip, shape [N], and is broadcast into each clip's one-channel noise map.
    /// </summary>
    private Tensor<T> Run(Tensor<T> input, Tensor<T> sigma)
    {
        var net = _net ?? throw new InvalidOperationException("ShiftNet's native network is not built.");
        if (input.Rank is not (4 or 5))
            throw new ArgumentException($"ShiftNet expects frames [F, C, H, W] or [N, F, C, H, W]; got rank {input.Rank}.", nameof(input));
        bool batched = input.Rank == 5;
        int clips = batched ? input.Shape[0] : 1;
        var outputs = new Tensor<T>[clips];
        for (int n = 0; n < clips; n++)
        {
            var clip = batched ? Engine.Reshape(Engine.TensorNarrow(input, 0, n, 1), input._shape.Skip(1).ToArray()) : input;
            int f = clip.Shape[0], h = clip.Shape[2], w = clip.Shape[3];
            int ph = (8 - h % 8) % 8, pw = (8 - w % 8) % 8;
            var padded = ph + pw > 0 ? Engine.Pad(clip, 0, ph, 0, pw, NumOps.Zero) : clip;
            var ones = new Tensor<T>(new[] { f, 1, h + ph, w + pw });
            ones.Fill(NumOps.One);
            var level = Engine.Reshape(Engine.TensorNarrow(sigma, 0, n, 1), new[] { 1, 1, 1, 1 });
            var denoised = RunClip(net, padded, Engine.TensorMultiply(ones, level));
            if (ph + pw > 0) denoised = Engine.TensorNarrow(Engine.TensorNarrow(denoised, 2, 0, h), 3, 0, w);
            outputs[n] = denoised;
        }
        if (!batched) return outputs[0];
        return Engine.TensorConcatenate(outputs.Select(o => Engine.Reshape(o, new[] { 1 }.Concat(o._shape).ToArray())).ToArray(), 0);
    }

    private Tensor<T> InferenceSigma(Tensor<T> normalized)
    {
        bool batched = normalized.Rank == 5;
        int clips = batched ? normalized.Shape[0] : 1;
        var sigma = new Tensor<T>(new[] { clips });
        for (int n = 0; n < clips; n++)
            sigma[n] = NumOps.FromDouble(_options.NoiseLevel ?? EstimateNoiseLevel(batched ? normalized.Slice(n) : normalized));
        return sigma;
    }

    // The σ the noise was drawn with, as tape operations on the step's data: sqrt(mean((noisy - clean)²)) per clip.
    private Tensor<T> TrainingSigma(Tensor<T> noisy, Tensor<T> clean)
    {
        var diff = Engine.TensorSubtract(noisy, clean);
        var squared = Engine.TensorMultiply(diff, diff);
        bool batched = squared.Rank == 5;
        var axes = Enumerable.Range(batched ? 1 : 0, batched ? squared.Rank - 1 : squared.Rank).ToArray();
        var mean = Engine.ReduceMean(squared, axes, keepDims: false);
        return Engine.Reshape(Engine.TensorSqrt(mean), new[] { batched ? squared.Shape[0] : 1 });
    }

    /// <summary>Test hook: one of the network's temporal shifts (the release's channel_shift).</summary>
    internal Tensor<T> TemporalShiftForTest(Tensor<T> x, bool reverse) => TemporalShift(x, reverse);

    /// <summary>Test hook: the release's grouped spatial shift of 8 * number channels.</summary>
    internal Tensor<T> SpatialShiftForTest(Tensor<T> hw) => SpatialShift(hw);

    /// <summary>Test hook: the denoised normalized frames for a given noise level.</summary>
    internal Tensor<T> DenoiseNormalizedForTest(Tensor<T> normalized, double noiseLevel)
    {
        int clips = normalized.Rank == 5 ? normalized.Shape[0] : 1;
        var sigma = new Tensor<T>(new[] { clips });
        sigma.Fill(NumOps.FromDouble(noiseLevel));
        return Run(normalized, sigma);
    }

    #endregion

    #region Forward and training

    /// <inheritdoc/>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        return Run(input, InferenceSigma(input));
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.ForwardForTraining(input);
        var normalized = PreprocessFrames(input);
        return Run(normalized, InferenceSigma(normalized));
    }

    /// <inheritdoc/>
    protected override void ResolveLazyLayerShapes()
    {
        if (_usingCustomLayers || !_useNativeMode) return;
        int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        Run(new Tensor<T>(new[] { 2, ch, 8, 8 }), new Tensor<T>(new[] { 1 }));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The noise map carries the RMS of noisy minus clean for each clip, computed on the tape from the step's data,
    /// and the objective is the model's loss (BasicSR's PSNR loss by default) on the normalized frames.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, data =>
            {
                var noisy = PreprocessFrames(data[0]);
                var clean = PreprocessTarget(data[1]);
                return LossFunction.ComputeTapeLoss(Run(noisy, TrainingSigma(noisy, clean)), clean);
            })
            {
                Optimizer = _optimizer,
            }
        };
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "ShiftNet" },
                { "NumFeatures", _options.NumFeatures },
                { "RefinementFeatures", _options.RefinementFeatures },
                { "NumUNets", _options.NumUNets },
                { "ShiftStride", _options.ShiftStride }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }
    #endregion

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ShiftNet<T>));
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
