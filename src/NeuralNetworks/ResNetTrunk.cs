using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// The sizes of a ResNet trunk: the stem, then four stages of basic or bottleneck blocks.
/// </summary>
/// <param name="BlockCounts">The number of blocks in each of the four stages.</param>
/// <param name="Bottleneck">Bottleneck blocks (1×1, 3×3, 1×1 with expansion 4) instead of basic blocks (two 3×3).</param>
/// <param name="InputChannels">The image channels.</param>
/// <param name="Widths">The stem width and each stage's block width (the paper's 64, then 64, 128, 256 and 512).</param>
/// <param name="ZeroInitResidual">Zero-initializes each block's last BatchNorm scale (Goyal et al. 2017; off in the
/// paper and in torchvision's default).</param>
internal sealed record ResNetTrunkConfiguration(int[] BlockCounts, bool Bottleneck, int InputChannels = 3,
    int[]? Widths = null, bool ZeroInitResidual = false)
{
    public int[] StageWidths { get; } = Widths ?? new[] { 64, 64, 128, 256, 512 };

    public int Expansion => Bottleneck ? 4 : 1;

    /// <summary>The channels of the four stage outputs, C2 to C5.</summary>
    public int[] OutputChannels => StageWidths.Skip(1).Select(w => w * Expansion).ToArray();

    public static int[] BlockCountsOf(ResNetVariant variant) => variant switch
    {
        ResNetVariant.ResNet18 => new[] { 2, 2, 2, 2 },
        ResNetVariant.ResNet34 => new[] { 3, 4, 6, 3 },
        ResNetVariant.ResNet50 => new[] { 3, 4, 6, 3 },
        ResNetVariant.ResNet101 => new[] { 3, 4, 23, 3 },
        ResNetVariant.ResNet152 => new[] { 3, 8, 36, 3 },
        _ => throw new NotSupportedException($"ResNet variant {variant} is not supported.")
    };

    public static ResNetTrunkConfiguration For(ResNetVariant variant, int inputChannels = 3, bool zeroInitResidual = false)
        => new(BlockCountsOf(variant), variant >= ResNetVariant.ResNet50, inputChannels, null, zeroInitResidual);
}

/// <summary>
/// The convolutional trunk of ResNet (He et al. 2016) as torchvision defines it: a 7×7 stride-2 convolution, BatchNorm,
/// ReLU and a 3×3 stride-2 max-pool (padding 1), then four stages whose first block strides by 2 (except the first
/// stage) with a 1×1 convolution + BatchNorm projection shortcut where the shape changes.
/// </summary>
/// <remarks>
/// <para>Every convolution is bias-free and followed by BatchNorm (ε = 1e-5, momentum 0.1). A bottleneck block strides
/// in its 3×3 convolution (torchvision's "ResNet v1.5"; the paper's Caffe models stride in the first 1×1). The image
/// classifier and the detection backbone both run this trunk: one reads the C5 map, the other the C2–C5 pyramid.</para>
/// </remarks>
internal sealed class ResNetTrunk<T>
{
    private readonly IEngine _engine;

    public ResNetTrunk(IEngine engine, List<LayerBase<T>> layers, ResNetTrunkConfiguration configuration)
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
        ConvolutionalLayer<T> Conv(int inputs, int outputs, int kernel, int stride) => Own(ConvolutionalLayer<T>.WithInputDepth(
            inputs, outputs, kernel, stride, padding: kernel / 2,
            // InitializeLikeReference writes every weight; drawing a default initialization first is wasted work.
            initializationStrategy: new Initialization.ZeroInitializationStrategy<T>(), biasMode: Enums.BiasMode.Never));
        BatchNormalizationLayer<T> Norm(int channels) => Own(new BatchNormalizationLayer<T>(channels, 1e-5, 0.9));

        StemConv = Conv(c.InputChannels, c.StageWidths[0], 7, 2);
        StemNorm = Norm(c.StageWidths[0]);
        int inputs = c.StageWidths[0];
        for (int s = 0; s < 4; s++)
        {
            var stage = new List<Block>();
            int width = c.StageWidths[s + 1], outputs = width * c.Expansion;
            for (int b = 0; b < c.BlockCounts[s]; b++)
            {
                int stride = b == 0 && s > 0 ? 2 : 1;
                Block block = c.Bottleneck
                    ? new Block(Conv(inputs, width, 1, 1), Norm(width), Conv(width, width, 3, stride), Norm(width),
                        Conv(width, outputs, 1, 1), Norm(outputs), null, null)
                    : new Block(Conv(inputs, width, 3, stride), Norm(width), Conv(width, outputs, 3, 1), Norm(outputs),
                        null, null, null, null);
                if (stride != 1 || inputs != outputs)
                    block = block with { DownsampleConv = Conv(inputs, outputs, 1, stride), DownsampleNorm = Norm(outputs) };
                stage.Add(block);
                inputs = outputs;
            }
            Stages.Add(stage);
        }
    }

    /// <summary>A basic block (two 3×3 convolutions) or a bottleneck block (with <see cref="Conv3"/>), and its
    /// projection shortcut when it changes the shape.</summary>
    internal sealed record Block(ConvolutionalLayer<T> Conv1, BatchNormalizationLayer<T> Norm1,
        ConvolutionalLayer<T> Conv2, BatchNormalizationLayer<T> Norm2,
        ConvolutionalLayer<T>? Conv3, BatchNormalizationLayer<T>? Norm3,
        ConvolutionalLayer<T>? DownsampleConv, BatchNormalizationLayer<T>? DownsampleNorm)
    {
        public BatchNormalizationLayer<T> LastNorm => Norm3 ?? Norm2;
    }

    public ResNetTrunkConfiguration Configuration { get; }

    private readonly List<LayerBase<T>> _owned = new();

    /// <summary>The layers this trunk added to its owner's list, in order.</summary>
    public IReadOnlyList<LayerBase<T>> OwnedLayers => _owned;
    public ConvolutionalLayer<T> StemConv { get; }
    public BatchNormalizationLayer<T> StemNorm { get; }
    public List<List<Block>> Stages { get; } = new();

    /// <summary>The stem: convolution, BatchNorm, ReLU and the padded max-pool, on <c>[B, C, H, W]</c>.</summary>
    public Tensor<T> Stem(Tensor<T> image)
    {
        var x = _engine.ReLU(StemNorm.Forward(StemConv.Forward(image)));
        return AiDotNet.ComputerVision.CvTensorOps<T>.MaxPoolPadded(x, kernelSize: 3, stride: 2, padding: 1);
    }

    /// <summary>The four stage outputs C2–C5 (strides 4, 8, 16 and 32) of an image batch <c>[B, C, H, W]</c>.</summary>
    public List<Tensor<T>> Forward(Tensor<T> image)
    {
        var x = Stem(image);
        var outputs = new List<Tensor<T>>(4);
        foreach (var stage in Stages)
        {
            foreach (var block in stage) x = Run(block, x);
            outputs.Add(x);
        }
        return outputs;
    }

    private Tensor<T> Run(Block block, Tensor<T> input)
    {
        var y = _engine.ReLU(block.Norm1.Forward(block.Conv1.Forward(input)));
        y = block.Norm2.Forward(block.Conv2.Forward(y));
        if (block.Conv3 is not null && block.Norm3 is not null)
            y = block.Norm3.Forward(block.Conv3.Forward(_engine.ReLU(y)));
        var identity = block.DownsampleConv is not null && block.DownsampleNorm is not null
            ? block.DownsampleNorm.Forward(block.DownsampleConv.Forward(input))
            : input;
        return _engine.ReLU(_engine.TensorAdd(y, identity));
    }

    private IEnumerable<(string Name, ConvolutionalLayer<T> Conv, BatchNormalizationLayer<T> Norm)> Named()
    {
        yield return ("conv1", StemConv, StemNorm);
        for (int s = 0; s < Stages.Count; s++)
            for (int b = 0; b < Stages[s].Count; b++)
            {
                var block = Stages[s][b];
                string p = $"layer{s + 1}.{b}";
                yield return ($"{p}.conv1", block.Conv1, block.Norm1);
                yield return ($"{p}.conv2", block.Conv2, block.Norm2);
                if (block.Conv3 is not null && block.Norm3 is not null) yield return ($"{p}.conv3", block.Conv3, block.Norm3);
                if (block.DownsampleConv is not null && block.DownsampleNorm is not null)
                    yield return ($"{p}.downsample.0", block.DownsampleConv, block.DownsampleNorm);
            }
    }

    private static string NormName(string convName) => convName switch
    {
        "conv1" => "bn1",
        _ when convName.EndsWith(".downsample.0", StringComparison.Ordinal) => convName.Substring(0, convName.Length - 1) + "1",
        _ => convName.Substring(0, convName.Length - 5) + "bn" + convName.Substring(convName.Length - 1)
    };

    /// <summary>torchvision's initialization: Kaiming-normal convolutions (fan-out, ReLU gain), BatchNorm scale 1 and
    /// shift 0, and with <see cref="ResNetTrunkConfiguration.ZeroInitResidual"/> each block's last scale 0.</summary>
    public void InitializeLikeReference(Random random)
    {
        foreach (var (_, conv, norm) in Named())
        {
            int outputs = conv.OutputDepth, inputs = conv.InputDepth, k = conv.KernelSize;
            conv.SetParameters(TextToSpeech.CodecBased.TorchParameters.Gaussian<T>(random, outputs * inputs * k * k,
                Math.Sqrt(2.0 / ((double)outputs * k * k))));
            TextToSpeech.CodecBased.TorchParameters.BatchNorm(norm, Fill(norm, 1.0), new double[outputs], new double[outputs], Fill(norm, 1.0));
        }
        if (Configuration.ZeroInitResidual)
            foreach (var block in Stages.SelectMany(s => s))
            {
                int channels = block.LastNorm.GetGamma().Length;
                TextToSpeech.CodecBased.TorchParameters.BatchNorm(block.LastNorm, new double[channels], new double[channels], new double[channels], Fill(block.LastNorm, 1.0));
            }
        double[] Fill(BatchNormalizationLayer<T> norm, double value)
            => Enumerable.Repeat(value, norm.GetGamma().Length).ToArray();
    }

    /// <summary>Loads a torchvision ResNet state dict (<c>conv1</c>, <c>bn1</c>, <c>layerN.i.convK</c>,
    /// <c>layerN.i.bnK</c>, <c>layerN.i.downsample.{0,1}</c>), given a reader of a named tensor of a given shape.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        foreach (var (name, conv, norm) in Named())
        {
            int outputs = conv.OutputDepth, inputs = conv.InputDepth, k = conv.KernelSize;
            LoadConv(conv, read($"{name}.weight", new[] { outputs, inputs, k, k }));
            string n = NormName(name);
            var shape = new[] { outputs };
            TextToSpeech.CodecBased.TorchParameters.BatchNorm(norm, read($"{n}.weight", shape), read($"{n}.bias", shape), read($"{n}.running_mean", shape),
                read($"{n}.running_var", shape));
        }
    }

    private static void LoadConv(ConvolutionalLayer<T> conv, double[] weight)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var v = new Vector<T>(weight.Length);
        for (int i = 0; i < weight.Length; i++) v[i] = ops.FromDouble(weight[i]);
        conv.SetParameters(v);
    }
}
