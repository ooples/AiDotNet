using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// The sizes of an EfficientNet trunk: the B0 baseline scaled by a width and a depth multiplier.
/// </summary>
/// <param name="WidthMultiplier">Scales every channel count (rounded to a multiple of 8, never below 90% of the
/// unrounded width).</param>
/// <param name="DepthMultiplier">Scales every stage's block count (rounded up).</param>
/// <param name="InputChannels">The image channels.</param>
/// <param name="DropConnectRate">The stochastic-depth rate of the last block; block i of n drops its residual branch
/// with probability rate·i/n during training (the paper's survival probability 0.8).</param>
/// <param name="IncludeHead">Builds the final 1×1 convolution to <see cref="HeadChannels"/> (the classifier uses it, a
/// detection backbone does not).</param>
internal sealed record EfficientNetTrunkConfiguration(double WidthMultiplier, double DepthMultiplier, int InputChannels = 3,
    double DropConnectRate = 0.2, bool IncludeHead = true)
{
    /// <summary>One stage of the B0 baseline (Tan &amp; Le 2019, Table 1): expansion, kernel, stride, input and output
    /// channels, and blocks.</summary>
    internal readonly record struct Stage(int Expand, int Kernel, int Stride, int Inputs, int Outputs, int Blocks);

    internal static readonly Stage[] Baseline =
    {
        new(1, 3, 1, 32, 16, 1),
        new(6, 3, 2, 16, 24, 2),
        new(6, 5, 2, 24, 40, 2),
        new(6, 3, 2, 40, 80, 3),
        new(6, 5, 1, 80, 112, 3),
        new(6, 5, 2, 112, 192, 4),
        new(6, 3, 1, 192, 320, 1),
    };

    /// <summary>The official code's <c>round_filters</c>.</summary>
    public int RoundChannels(int channels)
    {
        double scaled = channels * WidthMultiplier;
        int rounded = Math.Max(8, (int)(scaled + 4) / 8 * 8);
        return rounded < 0.9 * scaled ? rounded + 8 : rounded;
    }

    /// <summary>The official code's <c>round_repeats</c>.</summary>
    public int RoundBlocks(int blocks) => (int)Math.Ceiling(DepthMultiplier * blocks);

    public int StemChannels => RoundChannels(32);
    public int HeadChannels => RoundChannels(1280);

    /// <summary>The scaled stages.</summary>
    public Stage[] Stages => Baseline
        .Select(s => s with { Inputs = RoundChannels(s.Inputs), Outputs = RoundChannels(s.Outputs), Blocks = RoundBlocks(s.Blocks) })
        .ToArray();

    /// <summary>The stages whose outputs are the C2–C5 pyramid (strides 4, 8, 16 and 32).</summary>
    public static readonly int[] PyramidStages = { 1, 2, 4, 6 };

    public int[] PyramidChannels => PyramidStages.Select(i => Stages[i].Outputs).ToArray();

    public static (double Width, double Depth) MultipliersOf(EfficientNetVariant variant) => variant switch
    {
        EfficientNetVariant.B0 => (1.0, 1.0),
        EfficientNetVariant.B1 => (1.0, 1.1),
        EfficientNetVariant.B2 => (1.1, 1.2),
        EfficientNetVariant.B3 => (1.2, 1.4),
        EfficientNetVariant.B4 => (1.4, 1.8),
        EfficientNetVariant.B5 => (1.6, 2.2),
        EfficientNetVariant.B6 => (1.8, 2.6),
        EfficientNetVariant.B7 => (2.0, 3.1),
        _ => throw new NotSupportedException($"EfficientNet variant {variant} has no published multipliers.")
    };
}

/// <summary>
/// The convolutional trunk of EfficientNet (Tan &amp; Le 2019) as the official TensorFlow code builds it: a 3×3 stride-2
/// stem, seven stages of MBConv blocks (an optional 1×1 expansion, a depthwise convolution, squeeze-and-excitation over a
/// quarter of the block's input channels, a 1×1 projection), and a 1×1 head convolution.
/// </summary>
/// <remarks>
/// <para>Every convolution is bias-free and followed by BatchNorm with the paper's momentum 0.99 and the official
/// ε = 1e-3; the activation is Swish (SiLU). Convolutions pad as TensorFlow's "SAME" does, which for a stride-2
/// convolution of an even input puts the extra row and column after the image, not before it. A block whose input and
/// output shapes agree adds its input back, its branch dropped with the stochastic-depth probability during training.
/// The image classifier and the detection backbone both run this trunk.</para>
/// </remarks>
internal sealed class EfficientNetTrunk<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _owned = new();

    public EfficientNetTrunk(IEngine engine, List<LayerBase<T>> layers, EfficientNetTrunkConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            _owned.Add(layer);
            return layer;
        }
        ConvolutionalLayer<T> Conv(int inputs, int outputs, int kernel, int stride, int groups = 1) => Own(
            ConvolutionalLayer<T>.WithInputDepth(inputs, outputs, kernel, stride, padding: stride == 1 ? kernel / 2 : 0,
                // InitializeLikeReference writes every weight; drawing a default initialization first is wasted work.
                initializationStrategy: new Initialization.ZeroInitializationStrategy<T>(), groups: groups,
                biasMode: Enums.BiasMode.Never));
        BatchNormalizationLayer<T> Norm(int channels) => Own(new BatchNormalizationLayer<T>(channels, 1e-3, 0.99));
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));

        StemConv = Conv(c.InputChannels, c.StemChannels, 3, 2);
        StemNorm = Norm(c.StemChannels);
        var stages = c.Stages;
        int total = stages.Sum(s => s.Blocks), index = 0;
        foreach (var s in stages)
        {
            var blocks = new List<Block>();
            for (int b = 0; b < s.Blocks; b++)
            {
                int inputs = b == 0 ? s.Inputs : s.Outputs, stride = b == 0 ? s.Stride : 1, hidden = inputs * s.Expand;
                var expand = s.Expand != 1 ? (Conv(inputs, hidden, 1, 1), Norm(hidden)) : default;
                var depthwise = Conv(hidden, hidden, s.Kernel, stride, groups: hidden);
                var depthwiseNorm = Norm(hidden);
                int squeezed = Math.Max(1, inputs / 4);
                var reduce = Linear(squeezed);
                var excite = Linear(hidden);
                var project = Conv(hidden, s.Outputs, 1, 1);
                var projectNorm = Norm(s.Outputs);
                blocks.Add(new Block(expand.Item1, expand.Item2, depthwise, depthwiseNorm, reduce, excite, project, projectNorm,
                    inputs, hidden, squeezed, s.Outputs, s.Kernel, stride,
                    Residual: stride == 1 && inputs == s.Outputs,
                    DropRate: c.DropConnectRate * index / total));
                index++;
            }
            Stages.Add(blocks);
        }
        if (c.IncludeHead)
        {
            HeadConv = Conv(stages[^1].Outputs, c.HeadChannels, 1, 1);
            HeadNorm = Norm(c.HeadChannels);
        }
    }

    /// <summary>An MBConv block (<see cref="Expand"/> is absent when the expansion ratio is 1).</summary>
    internal sealed record Block(ConvolutionalLayer<T>? Expand, BatchNormalizationLayer<T>? ExpandNorm,
        ConvolutionalLayer<T> Depthwise, BatchNormalizationLayer<T> DepthwiseNorm,
        DenseLayer<T> Reduce, DenseLayer<T> Excite,
        ConvolutionalLayer<T> Project, BatchNormalizationLayer<T> ProjectNorm,
        int Inputs, int Hidden, int Squeezed, int Outputs, int Kernel, int Stride, bool Residual, double DropRate);

    public EfficientNetTrunkConfiguration Configuration { get; }
    public ConvolutionalLayer<T> StemConv { get; }
    public BatchNormalizationLayer<T> StemNorm { get; }
    public List<List<Block>> Stages { get; } = new();
    public ConvolutionalLayer<T>? HeadConv { get; }
    public BatchNormalizationLayer<T>? HeadNorm { get; }

    /// <summary>The layers this trunk added to its owner's list, in order.</summary>
    public IReadOnlyList<LayerBase<T>> OwnedLayers => _owned;

    /// <summary>
    /// Every stage's output, then (with a head) the head's: <c>[B, C, H, W]</c> maps for an image batch
    /// <c>[B, C, H, W]</c>. Training draws the stochastic-depth decisions from <paramref name="random"/>.
    /// </summary>
    public List<Tensor<T>> Forward(Tensor<T> image, bool training, Random random)
    {
        var x = _engine.TensorSiLU(StemNorm.Forward(StemConv.Forward(SamePad(image, 3, 2))));
        var outputs = new List<Tensor<T>>(Stages.Count + 1);
        foreach (var stage in Stages)
        {
            foreach (var block in stage) x = Run(block, x, training, random);
            outputs.Add(x);
        }
        if (HeadConv is not null && HeadNorm is not null)
            outputs.Add(_engine.TensorSiLU(HeadNorm.Forward(HeadConv.Forward(x))));
        return outputs;
    }

    private Tensor<T> Run(Block block, Tensor<T> input, bool training, Random random)
    {
        var x = input;
        if (block.Expand is not null && block.ExpandNorm is not null)
            x = _engine.TensorSiLU(block.ExpandNorm.Forward(block.Expand.Forward(x)));
        x = _engine.TensorSiLU(block.DepthwiseNorm.Forward(block.Depthwise.Forward(SamePad(x, block.Kernel, block.Stride))));
        x = Excite(block, x);
        x = block.ProjectNorm.Forward(block.Project.Forward(x));
        if (!block.Residual) return x;
        if (training && block.DropRate > 0) x = DropPath(x, block.DropRate, random);
        return _engine.TensorAdd(x, input);
    }

    /// <summary>Squeeze-and-excitation: the channel means, a Swish bottleneck, and a sigmoid gate on every channel.</summary>
    private Tensor<T> Excite(Block block, Tensor<T> x)
    {
        var pooled = _engine.ReduceMean(x, new[] { 2, 3 }, false);                                       // [B, C]
        var gate = _engine.TensorSigmoid(block.Excite.Forward(_engine.TensorSiLU(block.Reduce.Forward(pooled))));
        var shape = x.Shape.ToArray();
        gate = _engine.Reshape(gate, new[] { shape[0], shape[1], 1, 1 });
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(gate, shape));
    }

    /// <summary>Drops the whole branch of each sample with probability <paramref name="rate"/>, scaling the kept ones
    /// by 1/(1 − rate).</summary>
    private Tensor<T> DropPath(Tensor<T> x, double rate, Random random)
    {
        var shape = x.Shape.ToArray();
        var mask = new Tensor<T>(new[] { shape[0], 1, 1, 1 });
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int b = 0; b < shape[0]; b++) mask[b] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(mask, shape));
    }

    /// <summary>TensorFlow's "SAME" padding for a stride-2 convolution (stride 1 pads symmetrically inside the
    /// convolution): <c>max((⌈n/s⌉ − 1)·s + k − n, 0)</c> zeros per axis, the smaller half before.</summary>
    private Tensor<T> SamePad(Tensor<T> x, int kernel, int stride)
    {
        if (stride == 1) return x;
        for (int axis = 2; axis <= 3; axis++)
        {
            int n = x.Shape[axis];
            int total = Math.Max(((n + stride - 1) / stride - 1) * stride + kernel - n, 0);
            int before = total / 2, after = total - before;
            if (total == 0) continue;
            var parts = new List<Tensor<T>>(3);
            var zerosShape = x.Shape.ToArray();
            if (before > 0)
            {
                zerosShape[axis] = before;
                parts.Add(new Tensor<T>(zerosShape.ToArray()));
            }
            parts.Add(x);
            if (after > 0)
            {
                zerosShape[axis] = after;
                parts.Add(new Tensor<T>(zerosShape.ToArray()));
            }
            x = _engine.TensorConcatenate(parts.ToArray(), axis);
        }
        return x;
    }

    private IEnumerable<(string Name, ConvolutionalLayer<T> Conv, string NormName, BatchNormalizationLayer<T> Norm)> NamedConvolutions()
    {
        yield return ("conv_stem", StemConv, "bn1", StemNorm);
        for (int s = 0; s < Stages.Count; s++)
            for (int b = 0; b < Stages[s].Count; b++)
            {
                var block = Stages[s][b];
                string p = $"blocks.{s}.{b}";
                if (block.Expand is not null && block.ExpandNorm is not null)
                {
                    yield return ($"{p}.conv_pw", block.Expand, $"{p}.bn1", block.ExpandNorm);
                    yield return ($"{p}.conv_dw", block.Depthwise, $"{p}.bn2", block.DepthwiseNorm);
                    yield return ($"{p}.conv_pwl", block.Project, $"{p}.bn3", block.ProjectNorm);
                }
                else
                {
                    yield return ($"{p}.conv_dw", block.Depthwise, $"{p}.bn1", block.DepthwiseNorm);
                    yield return ($"{p}.conv_pw", block.Project, $"{p}.bn2", block.ProjectNorm);
                }
            }
        if (HeadConv is not null && HeadNorm is not null) yield return ("conv_head", HeadConv, "bn2", HeadNorm);
    }

    private static int KernelInputs(ConvolutionalLayer<T> conv) => conv.Groups == 1 ? conv.InputDepth : conv.InputDepth / conv.Groups;

    /// <summary>The official initialization (timm's <c>_init_weight_goog</c>): convolutions normal with variance
    /// 2/fan-out (fan-out = k²·out/groups), the excitation layers likewise with zero biases, BatchNorm scale 1 and
    /// shift 0.</summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double[] Gaussian(int count, double std)
        {
            var a = new double[count];
            for (int i = 0; i < count; i++) a[i] = std * Normal();
            return a;
        }
        foreach (var (_, conv, _, norm) in NamedConvolutions())
        {
            int k = conv.KernelSize, outputs = conv.OutputDepth;
            conv.SetParameters(TextToSpeech.CodecBased.TorchParameters.Gaussian<T>(random, outputs * KernelInputs(conv) * k * k,
                Math.Sqrt(2.0 / (k * k * outputs / conv.Groups))));
            LoadNorm(norm, Enumerable.Repeat(1.0, outputs).ToArray(), new double[outputs], new double[outputs],
                Enumerable.Repeat(1.0, outputs).ToArray());
        }
        foreach (var block in Stages.SelectMany(s => s))
        {
            TextToSpeech.CodecBased.TorchParameters.Linear(_engine, block.Reduce, block.Hidden, block.Squeezed,
                Gaussian(block.Squeezed * block.Hidden, Math.Sqrt(2.0 / block.Squeezed)), new double[block.Squeezed]);
            TextToSpeech.CodecBased.TorchParameters.Linear(_engine, block.Excite, block.Squeezed, block.Hidden,
                Gaussian(block.Hidden * block.Squeezed, Math.Sqrt(2.0 / block.Hidden)), new double[block.Hidden]);
        }
    }

    /// <summary>Loads a timm <c>tf_efficientnet</c> state dict, given a reader of a named tensor of a given shape.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        foreach (var (name, conv, normName, norm) in NamedConvolutions())
        {
            int k = conv.KernelSize, outputs = conv.OutputDepth;
            LoadConv(conv, read($"{name}.weight", new[] { outputs, KernelInputs(conv), k, k }));
            var shape = new[] { outputs };
            LoadNorm(norm, read($"{normName}.weight", shape), read($"{normName}.bias", shape),
                read($"{normName}.running_mean", shape), read($"{normName}.running_var", shape));
        }
        for (int s = 0; s < Stages.Count; s++)
            for (int b = 0; b < Stages[s].Count; b++)
            {
                var block = Stages[s][b];
                string p = $"blocks.{s}.{b}.se";
                TextToSpeech.CodecBased.TorchParameters.Linear(_engine, block.Reduce, block.Hidden, block.Squeezed,
                    read($"{p}.conv_reduce.weight", new[] { block.Squeezed, block.Hidden, 1, 1 }),
                    read($"{p}.conv_reduce.bias", new[] { block.Squeezed }));
                TextToSpeech.CodecBased.TorchParameters.Linear(_engine, block.Excite, block.Squeezed, block.Hidden,
                    read($"{p}.conv_expand.weight", new[] { block.Hidden, block.Squeezed, 1, 1 }),
                    read($"{p}.conv_expand.bias", new[] { block.Hidden }));
            }
    }

    private static void LoadConv(ConvolutionalLayer<T> conv, double[] weight)
    {
        var v = new Vector<T>(weight.Length);
        for (int i = 0; i < weight.Length; i++) v[i] = NumOps.FromDouble(weight[i]);
        conv.SetParameters(v);
    }

    private static void LoadNorm(BatchNormalizationLayer<T> norm, double[] gamma, double[] beta, double[] mean, double[] variance)
    {
        // The layer's vector is gamma, beta, then the running mean and variance buffers.
        int c = gamma.Length;
        var v = new Vector<T>(4 * c);
        for (int j = 0; j < c; j++)
        {
            v[j] = NumOps.FromDouble(gamma[j]);
            v[c + j] = NumOps.FromDouble(beta[j]);
            v[2 * c + j] = NumOps.FromDouble(mean[j]);
            v[3 * c + j] = NumOps.FromDouble(variance[j]);
        }
        norm.SetParameters(v);
    }
}
