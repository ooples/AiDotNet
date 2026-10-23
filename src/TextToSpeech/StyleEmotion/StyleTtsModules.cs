using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Reads a named PyTorch tensor's row-major values after checking its shape.</summary>
internal delegate double[] TorchReader(string name, int[] shape);

/// <summary>Shared operations of the StyleTTS family (yl4579/StyleTTS <c>models.py</c>) on <c>[1, channels, time]</c> and
/// <c>[1, channels, height, width]</c> tensors.</summary>
internal static class StyleTtsOps
{
    public static readonly double InverseSqrt2 = 1.0 / Math.Sqrt(2.0);

    public static TLayer Own<T, TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    /// <summary><c>LeakyReLU(slope)</c> as <c>slope · x + (1 − slope) · relu(x)</c>.</summary>
    public static Tensor<T> LeakyRelu<T>(IEngine engine, Tensor<T> x, double slope)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorAdd(engine.TensorMultiplyScalar(x, ops.FromDouble(slope)),
            engine.TensorMultiplyScalar(engine.ReLU(x), ops.FromDouble(1 - slope)));
    }

    public static Tensor<T> Dropout<T>(IEngine engine, Tensor<T> x, double rate, bool training, Random random) =>
        training && rate > 0 ? T5Seq2Seq<T>.Dropout(engine, x, rate, random) : x;

    public static Tensor<T> Scale<T>(IEngine engine, Tensor<T> x, double factor) =>
        engine.TensorMultiplyScalar(x, MathHelper.GetNumericOperations<T>().FromDouble(factor));

    /// <summary><c>F.interpolate(scale_factor=2, mode='nearest')</c> along time: every frame twice.</summary>
    public static Tensor<T> RepeatFrames<T>(IEngine engine, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        return engine.Reshape(engine.TensorTile(engine.Reshape(x, new[] { 1, c, t, 1 }), new[] { 1, 1, 1, 2 }), new[] { 1, c, 2 * t });
    }

    /// <summary>Repeats the last column once when the last axis is odd (the reference's padding before
    /// <c>avg_pool(2)</c>).</summary>
    public static Tensor<T> PadOddLast<T>(IEngine engine, Tensor<T> x)
    {
        int rank = x.Rank, last = x.Shape[rank - 1];
        if (last % 2 == 0) return x;
        var start = new int[rank];
        start[rank - 1] = last - 1;
        var length = x.Shape.ToArray();
        length[rank - 1] = 1;
        return engine.TensorConcatenate(new[] { x, engine.TensorSlice(x, start, length) }, rank - 1);
    }

    /// <summary><c>F.avg_pool1d(x, 2)</c> on <c>[1, C, T]</c> after padding an odd T.</summary>
    public static Tensor<T> HalveTime<T>(IEngine engine, Tensor<T> x)
    {
        x = PadOddLast(engine, x);
        int c = x.Shape[1], t = x.Shape[2] / 2;
        var pairs = engine.Reshape(x, new[] { 1, c, t, 2 });
        return Scale(engine, engine.ReduceSum(pairs, new[] { 3 }, keepDims: false), 0.5);
    }

    /// <summary><c>F.avg_pool2d(x, 2)</c> on <c>[1, C, H, W]</c> after padding an odd W (H floors).</summary>
    public static Tensor<T> HalveBoth<T>(IEngine engine, Tensor<T> x)
    {
        x = PadOddLast(engine, x);
        int c = x.Shape[1], h = x.Shape[2] / 2, w = x.Shape[3] / 2;
        if (x.Shape[2] != 2 * h) x = engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { 1, c, 2 * h, x.Shape[3] });
        var blocks = engine.Reshape(x, new[] { 1, c, h, 2, w, 2 });
        return Scale(engine, engine.ReduceSum(blocks, new[] { 3, 5 }, keepDims: false), 0.25);
    }

    /// <summary>Applies a linear layer to each time step of <c>[1, C, T]</c>.</summary>
    public static Tensor<T> PerFrame<T>(IEngine engine, LayerBase<T> linear, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var rows = engine.TensorTranspose(engine.Reshape(x, new[] { c, t }));                               // [T, C]
        var y = linear.Forward(rows);
        return engine.Reshape(engine.TensorTranspose(y), new[] { 1, y.Shape[1], t });
    }

    /// <summary>Loads a PyTorch <c>weight_norm</c> convolution (<c>weight_g</c>, <c>weight_v</c>, <c>bias</c>).</summary>
    public static void LoadWeightNorm<T>(NormedConv1DLayer<T> conv, TorchReader read, string name, bool bias = true)
    {
        var shape = conv.TorchWeightShape;
        var v = read(name + ".weight_v", shape);
        var g = read(name + ".weight_g", new[] { shape[0], 1, 1 });
        conv.LoadTorchWeights(v, g, bias ? read(name + ".bias", new[] { conv.OutputChannels }) : new double[conv.OutputChannels]);
    }

    /// <summary>Loads a plain PyTorch convolution (<c>weight</c>, <c>bias</c>).</summary>
    public static void LoadPlain<T>(NormedConv1DLayer<T> conv, TorchReader read, string name, bool bias = true)
    {
        var shape = conv.TorchWeightShape;
        conv.LoadTorchWeights(read(name + ".weight", shape), null,
            bias ? read(name + ".bias", new[] { conv.OutputChannels }) : new double[conv.OutputChannels]);
    }

    /// <summary>Loads a <c>spectral_norm</c> 2-D convolution (<c>weight_orig</c>, <c>weight_u</c>, <c>weight_v</c>,
    /// <c>bias</c>).</summary>
    public static void LoadSpectral<T>(NormedConv2DLayer<T> conv, TorchReader read, string name, bool bias = true)
    {
        var shape = conv.TorchWeightShape;
        int per = shape[1] * shape[2] * shape[3];
        conv.LoadTorchWeights(read(name + ".weight_orig", shape), bias ? read(name + ".bias", new[] { shape[0] }) : null,
            read(name + ".weight_u", new[] { shape[0] }), read(name + ".weight_v", new[] { per }));
    }

    /// <summary>Loads an <c>nn.Linear</c>.</summary>
    public static void LoadLinear<T>(IEngine engine, DenseLayer<T> layer, TorchReader read, string name, int inputs, int outputs) =>
        TorchParameters.Linear(engine, layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }), read(name + ".bias", new[] { outputs }));

    /// <summary>Loads an affine normalization's scale and shift.</summary>
    public static void LoadAffine<T>(IEngine engine, Tensor<T> gamma, Tensor<T> beta, double[] weight, double[] bias)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < weight.Length; i++)
        {
            gamma[i] = ops.FromDouble(weight[i]);
            beta[i] = ops.FromDouble(bias[i]);
        }
        engine.InvalidatePersistentTensor(gamma);
        engine.InvalidatePersistentTensor(beta);
    }

    public static void LoadInstanceNorm<T>(IEngine engine, InstanceNormalizationLayer<T> norm, TorchReader read, string name, int channels) =>
        LoadAffine(engine, norm.GetGammaTensor(), norm.GetBetaTensor(), read(name + ".weight", new[] { channels }), read(name + ".bias", new[] { channels }));

    public static DenseLayer<T> Linear<T>(int outputs) => new(outputs, (IActivationFunction<T>)new IdentityActivation<T>());

    public static NormedConv1DLayer<T> WeightNormConv<T>(int input, int output, int kernel, int stride = 1, int padding = -1,
        int groups = 1, bool bias = true) =>
        new(input, output, kernel, stride, 1, groups, padding < 0 ? (kernel - 1) / 2 : padding, false, ConvolutionNormalization.Weight, bias);
}

/// <summary>
/// A PyTorch single-layer <c>nn.LSTM(bidirectional=True, batch_first=True)</c> on <c>[T, input]</c>, returning
/// <c>[T, 2 · hidden]</c> (forward then backward), from two <see cref="LSTMCellLayer{T}"/> stepped over time.
/// </summary>
internal sealed class StyleTtsBiLstm<T>
{
    private readonly IEngine _engine;

    public StyleTtsBiLstm(IEngine engine, List<LayerBase<T>> layers, int input, int hidden)
    {
        _engine = engine;
        ForwardCell = StyleTtsOps.Own(layers, new LSTMCellLayer<T>(input, hidden));
        BackwardCell = StyleTtsOps.Own(layers, new LSTMCellLayer<T>(input, hidden));
        Hidden = hidden;
        Input = input;
    }

    public LSTMCellLayer<T> ForwardCell { get; }
    public LSTMCellLayer<T> BackwardCell { get; }
    public int Hidden { get; }
    public int Input { get; }

    private Tensor<T> Run(LSTMCellLayer<T> cell, Tensor<T> rows, bool reverse)
    {
        int t = rows.Shape[0];
        var state = new Tensor<T>(new[] { 1, 2 * Hidden });
        var outputs = new Tensor<T>[t];
        for (int s = 0; s < t; s++)
        {
            int i = reverse ? t - 1 - s : s;
            state = cell.Forward(_engine.TensorSlice(rows, new[] { i, 0 }, new[] { 1, rows.Shape[1] }), state);
            outputs[i] = cell.SplitState(state).Hidden;
        }
        return _engine.TensorConcatenate(outputs, 0);
    }

    public Tensor<T> Apply(Tensor<T> rows) =>
        _engine.TensorConcatenate(new[] { Run(ForwardCell, rows, false), Run(BackwardCell, rows, true) }, 1);

    public void Load(TorchReader read, string name)
    {
        int g = 4 * Hidden;
        foreach (var (cell, suffix) in new[] { (ForwardCell, "_l0"), (BackwardCell, "_l0_reverse") })
            cell.LoadTorchWeights(read($"{name}.weight_ih{suffix}", new[] { g, Input }), read($"{name}.weight_hh{suffix}", new[] { g, Hidden }),
                read($"{name}.bias_ih{suffix}", new[] { g }), read($"{name}.bias_hh{suffix}", new[] { g }));
    }
}

/// <summary>
/// The reference's 2-D <c>ResBlk</c> (StarGAN v2's) without normalization: <c>(shortcut(x) + residual(x)) / √2</c>, the
/// residual LeakyReLU(0.2) → spectrally normalized 3×3 conv → (learned depthwise 3×3 stride-2 downsampling) → LeakyReLU →
/// 3×3 conv to the output width; the shortcut a bias-free 1×1 conv when the width changes, then 2×2 average pooling.
/// </summary>
internal sealed class StyleTtsResBlock2d<T>
{
    private readonly IEngine _engine;
    private readonly bool _downsample;

    public StyleTtsResBlock2d(IEngine engine, List<LayerBase<T>> layers, int dimIn, int dimOut, bool downsample)
    {
        _engine = engine;
        _downsample = downsample;
        Conv1 = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(dimIn, dimIn, 3, 3, 1, 1, 1, 1, 1, ConvolutionNormalization.Spectral));
        Conv2 = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(dimIn, dimOut, 3, 3, 1, 1, 1, 1, 1, ConvolutionNormalization.Spectral));
        if (downsample) DownsampleResidual = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(dimIn, dimIn, 3, 3, 2, 2, 1, 1, dimIn, ConvolutionNormalization.Spectral));
        if (dimIn != dimOut) Shortcut = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(dimIn, dimOut, 1, 1, 1, 1, 0, 0, 1, ConvolutionNormalization.Spectral, useBias: false));
    }

    public NormedConv2DLayer<T> Conv1 { get; }
    public NormedConv2DLayer<T> Conv2 { get; }
    public NormedConv2DLayer<T>? DownsampleResidual { get; }
    public NormedConv2DLayer<T>? Shortcut { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var shortcut = Shortcut is null ? x : Shortcut.Forward(x);
        if (_downsample) shortcut = StyleTtsOps.HalveBoth(_engine, shortcut);
        var r = Conv1.Forward(StyleTtsOps.LeakyRelu(_engine, x, 0.2));
        if (DownsampleResidual is not null) r = DownsampleResidual.Forward(r);
        r = Conv2.Forward(StyleTtsOps.LeakyRelu(_engine, r, 0.2));
        return StyleTtsOps.Scale(_engine, _engine.TensorAdd(shortcut, r), StyleTtsOps.InverseSqrt2);
    }

    public void Load(TorchReader read, string name)
    {
        StyleTtsOps.LoadSpectral(Conv1, read, name + ".conv1");
        StyleTtsOps.LoadSpectral(Conv2, read, name + ".conv2");
        if (DownsampleResidual is not null) StyleTtsOps.LoadSpectral(DownsampleResidual, read, name + ".downsample_res.conv");
        if (Shortcut is not null) StyleTtsOps.LoadSpectral(Shortcut, read, name + ".conv1x1", bias: false);
    }
}

/// <summary>
/// The 2-D trunk the reference's style encoder and discriminator share: a spectrally normalized 3×3 conv from one
/// channel, four downsampling residual blocks doubling the width up to a cap, LeakyReLU, a 5×5 valid conv.
/// </summary>
internal sealed class StyleTtsTrunk2d<T>
{
    private readonly IEngine _engine;

    public StyleTtsTrunk2d(IEngine engine, List<LayerBase<T>> layers, int dimIn, int maxConvDim, int repeat = 4)
    {
        _engine = engine;
        Input = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(1, dimIn, 3, 3, 1, 1, 1, 1, 1, ConvolutionNormalization.Spectral));
        int dim = dimIn;
        for (int i = 0; i < repeat; i++)
        {
            int next = Math.Min(dim * 2, maxConvDim);
            Blocks.Add(new StyleTtsResBlock2d<T>(engine, layers, dim, next, downsample: true));
            dim = next;
        }
        OutputDim = dim;
        Output = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(dim, dim, 5, 5, 1, 1, 0, 0, 1, ConvolutionNormalization.Spectral));
    }

    public NormedConv2DLayer<T> Input { get; }
    public List<StyleTtsResBlock2d<T>> Blocks { get; } = new();
    public NormedConv2DLayer<T> Output { get; }
    public int OutputDim { get; }

    /// <summary>Runs the trunk on a mel spectrogram <c>[1, 1, mels, frames]</c>, collecting every layer's output (the
    /// reference's <c>nn.Sequential</c> entries) in <paramref name="features"/>.</summary>
    public Tensor<T> Forward(Tensor<T> x, List<Tensor<T>>? features)
    {
        x = Input.Forward(x);
        features?.Add(x);
        foreach (var block in Blocks)
        {
            x = block.Forward(x);
            features?.Add(x);
        }
        x = StyleTtsOps.LeakyRelu(_engine, x, 0.2);
        features?.Add(x);
        x = Output.Forward(x);
        features?.Add(x);
        return x;
    }

    /// <summary>Global average pooling of <c>[1, C, H, W]</c> to <c>[1, C, 1, 1]</c>.</summary>
    public Tensor<T> Pool(Tensor<T> x)
    {
        int c = x.Shape[1];
        return _engine.Reshape(_engine.ReduceMean(x, new[] { 2, 3 }, keepDims: false), new[] { 1, c, 1, 1 });
    }

    public void Load(TorchReader read, string name, int firstIndex = 0)
    {
        StyleTtsOps.LoadSpectral(Input, read, $"{name}.{firstIndex}");
        for (int i = 0; i < Blocks.Count; i++) Blocks[i].Load(read, $"{name}.{firstIndex + 1 + i}");
        StyleTtsOps.LoadSpectral(Output, read, $"{name}.{firstIndex + 2 + Blocks.Count}");
    }
}

/// <summary>The reference's <c>StyleEncoder</c>: the 2-D trunk, global average pooling, LeakyReLU and a linear map to the
/// style vector.</summary>
internal sealed class StyleTtsStyleEncoder<T>
{
    private readonly IEngine _engine;

    public StyleTtsStyleEncoder(IEngine engine, List<LayerBase<T>> layers, int dimIn, int styleDim, int maxConvDim)
    {
        _engine = engine;
        Trunk = new StyleTtsTrunk2d<T>(engine, layers, dimIn, maxConvDim);
        Projection = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(styleDim));
        StyleDim = styleDim;
    }

    public StyleTtsTrunk2d<T> Trunk { get; }
    public DenseLayer<T> Projection { get; }
    public int StyleDim { get; }

    /// <summary>The style vector <c>[1, style]</c> of a mel spectrogram <c>[mels, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> mel)
    {
        var x = Trunk.Forward(_engine.Reshape(mel, new[] { 1, 1, mel.Shape[0], mel.Shape[1] }), null);
        x = StyleTtsOps.LeakyRelu(_engine, Trunk.Pool(x), 0.2);
        return Projection.Forward(_engine.Reshape(x, new[] { 1, Trunk.OutputDim }));
    }

    public void Load(IEngine engine, TorchReader read, string name)
    {
        Trunk.Load(read, name + ".shared");
        StyleTtsOps.LoadLinear(engine, Projection, read, name + ".unshared", Trunk.OutputDim, StyleDim);
    }
}

/// <summary>The reference's <c>Discriminator2d</c>: the 2-D trunk, LeakyReLU, global average pooling and a spectrally
/// normalized 1×1 conv to one logit, with every layer's output kept for feature matching.</summary>
internal sealed class StyleTtsDiscriminator<T>
{
    private readonly IEngine _engine;

    public StyleTtsDiscriminator(IEngine engine, List<LayerBase<T>> layers, int dimIn, int maxConvDim, int repeat = 4)
    {
        _engine = engine;
        Trunk = new StyleTtsTrunk2d<T>(engine, layers, dimIn, maxConvDim, repeat);
        Logit = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(Trunk.OutputDim, 1, 1, 1, 1, 1, 0, 0, 1, ConvolutionNormalization.Spectral));
    }

    public StyleTtsTrunk2d<T> Trunk { get; }
    public NormedConv2DLayer<T> Logit { get; }

    /// <summary>The logit <c>[1]</c> and the feature maps of a mel spectrogram <c>[1, 1, mels, frames]</c>.</summary>
    public (Tensor<T> Logit, List<Tensor<T>> Features) Forward(Tensor<T> x)
    {
        var features = new List<Tensor<T>>();
        x = Trunk.Forward(x, features);
        x = StyleTtsOps.LeakyRelu(_engine, x, 0.2);
        features.Add(x);
        x = Trunk.Pool(x);
        features.Add(x);
        x = Logit.Forward(x);
        features.Add(x);
        return (_engine.Reshape(x, new[] { 1 }), features);
    }

    public void Load(TorchReader read, string name)
    {
        Trunk.Load(read, name + ".main");
        StyleTtsOps.LoadSpectral(Logit, read, $"{name}.main.{Trunk.Blocks.Count + 5}");
    }
}
