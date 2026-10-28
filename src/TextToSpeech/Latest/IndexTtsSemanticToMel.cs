using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>The sizes of IndexTTS2's semantic-to-mel module (the release's <c>s2mel</c> config).</summary>
internal sealed record IndexTtsSemanticToMelConfiguration(int MelChannels, int Hidden, int Heads, int Depth, int StyleDim, int SemanticDim,
    int RegulatorChannels, int RegulatorStages, int WavenetHidden, int WavenetLayers, int WavenetKernel, double WavenetDropout,
    double ClassDropout, int GptDim, int GptBottleneck1 = 256, int GptBottleneck2 = 128);

/// <summary>
/// IndexTTS2's length regulator (the release's <c>InterpolateRegulator</c> on continuous content): a linear map of the
/// semantic codec's quantized features, nearest interpolation to the mel frame count, stages of a 3-wide convolution,
/// a one-group GroupNorm and Mish, and a 1×1 convolution.
/// </summary>
internal sealed class IndexTtsLengthRegulator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _channels, _input;

    internal sealed record Stage(NormedConv1DLayer<T> Conv, BiasFreeLinearLayer<T> Gamma, BiasFreeLinearLayer<T> Beta);

    public IndexTtsLengthRegulator(IEngine engine, List<LayerBase<T>> layers, Random random, int input, int channels, int stages)
    {
        _engine = engine;
        _input = input;
        _channels = channels;
        ContentIn = IndexTtsOps.Linear(engine, layers, random, input, channels);
        for (int i = 0; i < stages; i++)
            Stages.Add(new Stage(IndexTtsS2MelOps.Conv(layers, random, channels, channels, 3, ConvolutionNormalization.None),
                IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, channels).ToArray()), IndexTtsOps.Values(layers, new double[channels])));
        Output = IndexTtsS2MelOps.Conv(layers, random, channels, channels, 1, ConvolutionNormalization.None);
    }

    public DenseLayer<T> ContentIn { get; }
    public List<Stage> Stages { get; } = new();
    public NormedConv1DLayer<T> Output { get; }

    /// <summary>The source frame of each of <paramref name="frames"/> outputs under <c>F.interpolate(mode='nearest')</c>
    /// (single-precision scale, as ATen computes it).</summary>
    internal static int[] NearestSource(int length, int frames)
    {
        var index = new int[frames];
        float scale = (float)length / frames;
        for (int i = 0; i < frames; i++)
            index[i] = frames == length ? i : Math.Min((int)Math.Floor((float)(i * scale)), length - 1);
        return index;
    }

    /// <summary>The conditioning <c>[frames, channels]</c> of quantized semantic features <c>[n, input]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> content, int frames)
    {
        int n = content.Shape[0];
        var x = ContentIn.Forward(content);                                                         // [n, C]
        var source = new Tensor<int>(new[] { frames });
        var index = NearestSource(n, frames);
        for (int i = 0; i < frames; i++) source[i] = index[i];
        x = _engine.TensorIndexSelect(x, source, 0);                                                // [frames, C]
        var h = _engine.Reshape(_engine.TensorTranspose(x), new[] { 1, _channels, frames });
        foreach (var stage in Stages)
        {
            h = stage.Conv.Forward(h);
            h = _engine.Mish(IndexTtsS2MelOps.GroupNorm(_engine, h, IndexTtsOps.Value(stage.Gamma), IndexTtsOps.Value(stage.Beta), 1e-5));
        }
        h = Output.Forward(h);
        return _engine.TensorTranspose(_engine.Reshape(h, new[] { _channels, frames }));
    }

    public void Load(IEngine engine, Func<string, int[], double[]> read, string prefix)
    {
        IndexTtsOps.LoadLinear(engine, read, ContentIn, prefix + "content_in_proj", _input, _channels);
        for (int i = 0; i < Stages.Count; i++)
        {
            IndexTtsOps.LoadConv(read, Stages[i].Conv, $"{prefix}model.{3 * i}", _channels);
            TorchParameters.Linear(Stages[i].Gamma, 1, _channels, read($"{prefix}model.{3 * i + 1}.weight", new[] { _channels }));
            TorchParameters.Linear(Stages[i].Beta, 1, _channels, read($"{prefix}model.{3 * i + 1}.bias", new[] { _channels }));
        }
        IndexTtsOps.LoadConv(read, Output, $"{prefix}model.{3 * Stages.Count}", _channels);
    }
}

/// <summary>Helpers of the semantic-to-mel module.</summary>
internal static class IndexTtsS2MelOps
{
    public static NormedConv1DLayer<T> Conv<T>(List<LayerBase<T>> layers, Random random, int inputs, int outputs, int kernel,
        ConvolutionNormalization normalization, int padding = -1, bool bias = true)
    {
        var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, 1, 1, 1, padding < 0 ? (kernel - 1) / 2 : padding, false, normalization, bias);
        layers.Add(conv);
        int fanIn = inputs * kernel;
        conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * kernel, fanIn), null,
            bias ? IndexTtsOps.Uniform(random, outputs, fanIn) : new double[outputs]);
        return conv;
    }

    /// <summary>Loads a weight-normalized module (<c>weight_g</c>/<c>weight_v</c>, or <c>weight</c> without the
    /// parametrization).</summary>
    public static void LoadWeightNorm<T>(Func<string, int[], double[]> read, Func<string, bool> has, NormedConv1DLayer<T> conv, string name, int outputs)
    {
        var bias = conv.HasBias ? read(name + ".bias", new[] { outputs }) : new double[outputs];
        if (has(name + ".weight_v"))
            conv.LoadTorchWeights(read(name + ".weight_v", conv.TorchWeightShape), read(name + ".weight_g", new[] { outputs }), bias);
        else
            conv.LoadTorchWeights(read(name + ".weight", conv.TorchWeightShape), null, bias);
    }

    /// <summary><c>GroupNorm(1, C)</c> of <c>[1, C, T]</c>: one mean and variance over every value, then per-channel
    /// affine.</summary>
    public static Tensor<T> GroupNorm<T>(IEngine engine, Tensor<T> x, Tensor<T> gamma, Tensor<T> beta, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int c = x.Shape[1], t = x.Shape[2];
        var shape = x.Shape.ToArray();
        var mean = engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 1, 2 }, true), shape);
        var centered = engine.TensorSubtract(x, mean);
        var variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 1, 2 }, true);
        var inverse = engine.TensorPower(engine.TensorAddScalar(variance, ops.FromDouble(epsilon)), ops.FromDouble(-0.5));
        var normed = engine.TensorMultiply(centered, engine.TensorBroadcastTo(inverse, shape));
        return engine.TensorAdd(engine.TensorMultiply(normed, engine.TensorBroadcastTo(engine.Reshape(gamma, new[] { 1, c, 1 }), shape)),
            engine.TensorBroadcastTo(engine.Reshape(beta, new[] { 1, c, 1 }), shape));
    }

    /// <summary>RMS norm over the last axis: <c>x · rsqrt(mean(x²) + ε) · w</c> on <c>[n, d]</c>.</summary>
    public static Tensor<T> RmsNorm<T>(IEngine engine, Tensor<T> x, Tensor<T> weight, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        int d = shape[^1];
        var meanSquare = engine.ReduceMean(engine.TensorMultiply(x, x), new[] { shape.Length - 1 }, true);
        var inverse = engine.TensorPower(engine.TensorAddScalar(meanSquare, ops.FromDouble(epsilon)), ops.FromDouble(-0.5));
        var normed = engine.TensorMultiply(x, engine.TensorBroadcastTo(inverse, shape));
        return engine.TensorMultiply(normed, engine.TensorBroadcastTo(engine.Reshape(weight, new[] { 1, d }), shape));
    }

    /// <summary>A 1×1 channel map applied to rows <c>[T, in]</c> → <c>[T, out]</c>.</summary>
    public static Tensor<T> PerRow<T>(IEngine engine, NormedConv1DLayer<T> conv, Tensor<T> x)
    {
        int t = x.Shape[0];
        var y = conv.Forward(engine.Reshape(engine.TensorTranspose(x), new[] { 1, x.Shape[1], t }));
        return engine.TensorTranspose(engine.Reshape(y, new[] { y.Shape[1], t }));
    }
}

/// <summary>DiT's sinusoidal timestep embedder: <c>[cos(1000·t·f), sin(1000·t·f)]</c> over 256 frequencies
/// <c>f_k = exp(−ln(10⁴)·k/128)</c>, then Linear → SiLU → Linear.</summary>
internal sealed class IndexTtsTimestepEmbedder<T>
{
    private const int Frequencies = 256;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _hidden;

    public IndexTtsTimestepEmbedder(IEngine engine, List<LayerBase<T>> layers, Random random, int hidden)
    {
        _engine = engine;
        _hidden = hidden;
        Mlp1 = IndexTtsOps.Linear(engine, layers, random, Frequencies, hidden);
        Mlp2 = IndexTtsOps.Linear(engine, layers, random, hidden, hidden);
    }

    public DenseLayer<T> Mlp1 { get; }
    public DenseLayer<T> Mlp2 { get; }

    /// <summary>The embedding <c>[1, hidden]</c> of time <paramref name="t"/>.</summary>
    public Tensor<T> Forward(double t)
    {
        int half = Frequencies / 2;
        var e = new Tensor<T>(new[] { 1, Frequencies });
        for (int k = 0; k < half; k++)
        {
            double angle = 1000 * t * Math.Exp(-Math.Log(10000) * k / half);
            e[0, k] = NumOps.FromDouble(Math.Cos(angle));
            e[0, half + k] = NumOps.FromDouble(Math.Sin(angle));
        }
        return Mlp2.Forward(_engine.Swish(Mlp1.Forward(e)));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        IndexTtsOps.LoadLinear(_engine, read, Mlp1, prefix + "mlp.0", Frequencies, _hidden);
        IndexTtsOps.LoadLinear(_engine, read, Mlp2, prefix + "mlp.2", _hidden, _hidden);
    }
}

/// <summary>
/// IndexTTS2's flow-matching estimator (the release's seed-vc <c>DiT</c> with a WaveNet head): the noisy mel, the prompt
/// mel, the projected content and the CAM++ style concatenated per frame and merged to the hidden width; a transformer
/// of AdaLN-conditioned (on the time embedding) RMS-norm blocks with interleaved rotary attention, a SwiGLU feed-forward
/// and U-ViT skips (block i &lt; L/2 feeds block L−1−i); a long skip with the noisy mel; a WaveNet conditioned on a second
/// time embedding plus a residual projection; an adaLN-modulated final layer and a 1×1 convolution to the mel bands.
/// </summary>
internal sealed class IndexTtsDiT<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(DenseLayer<T> AttentionModulation, BiasFreeLinearLayer<T> AttentionNorm, BiasFreeLinearLayer<T> QueryKeyValue,
        BiasFreeLinearLayer<T> Output, DenseLayer<T> FeedForwardModulation, BiasFreeLinearLayer<T> FeedForwardNorm, BiasFreeLinearLayer<T> W1,
        BiasFreeLinearLayer<T> W3, BiasFreeLinearLayer<T> W2, DenseLayer<T>? SkipIn);

    internal sealed record WavenetLayer(NormedConv1DLayer<T> In, NormedConv1DLayer<T> ResidualSkip);

    public IndexTtsDiT(IEngine engine, List<LayerBase<T>> layers, IndexTtsSemanticToMelConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int d = c.Hidden;
        if (d % c.Heads != 0 || (d / c.Heads) % 2 != 0) throw new ArgumentException("The width must divide into heads of even width.", nameof(configuration));
        Intermediate = FindMultiple((int)(2 * (4 * d) / 3.0), 256);
        Time = new IndexTtsTimestepEmbedder<T>(engine, layers, random, d);
        ContentProjection = IndexTtsOps.Linear(engine, layers, random, c.RegulatorChannels, d);
        Merge = IndexTtsOps.Linear(engine, layers, random, d + 2 * c.MelChannels + c.StyleDim, d);
        for (int i = 0; i < c.Depth; i++)
            Blocks.Add(new Block(IndexTtsOps.Linear(engine, layers, random, d, 2 * d), IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, d).ToArray()),
                IndexTtsOps.BiasFree(layers, random, d, 3 * d), IndexTtsOps.BiasFree(layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, 2 * d), IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, d).ToArray()),
                IndexTtsOps.BiasFree(layers, random, d, Intermediate), IndexTtsOps.BiasFree(layers, random, d, Intermediate),
                IndexTtsOps.BiasFree(layers, random, Intermediate, d),
                i > c.Depth / 2 ? IndexTtsOps.Linear(engine, layers, random, 2 * d, d) : null));
        NormModulation = IndexTtsOps.Linear(engine, layers, random, d, 2 * d);
        NormWeight = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, d).ToArray());
        SkipLinear = IndexTtsOps.Linear(engine, layers, random, d + c.MelChannels, d);
        int w = c.WavenetHidden;
        Conv1 = IndexTtsOps.Linear(engine, layers, random, d, w);
        Time2 = new IndexTtsTimestepEmbedder<T>(engine, layers, random, w);
        WavenetCondition = IndexTtsS2MelOps.Conv(layers, random, w, 2 * w * c.WavenetLayers, 1, ConvolutionNormalization.Weight);
        for (int i = 0; i < c.WavenetLayers; i++)
            Wavenet.Add(new WavenetLayer(IndexTtsS2MelOps.Conv(layers, random, w, 2 * w, c.WavenetKernel, ConvolutionNormalization.Weight, padding: 0),
                IndexTtsS2MelOps.Conv(layers, random, w, i < c.WavenetLayers - 1 ? 2 * w : w, 1, ConvolutionNormalization.Weight)));
        ResidualProjection = IndexTtsOps.Linear(engine, layers, random, d, w);
        FinalModulation = IndexTtsOps.Linear(engine, layers, random, w, 2 * w);
        FinalLinear = IndexTtsS2MelOps.Conv(layers, random, w, w, 1, ConvolutionNormalization.Weight);
        Conv2 = IndexTtsS2MelOps.Conv(layers, random, w, c.MelChannels, 1, ConvolutionNormalization.None);
    }

    private static int FindMultiple(int n, int k) => n % k == 0 ? n : n + k - n % k;

    public IndexTtsSemanticToMelConfiguration Configuration { get; }
    public int Intermediate { get; }
    public IndexTtsTimestepEmbedder<T> Time { get; }
    public DenseLayer<T> ContentProjection { get; }
    public DenseLayer<T> Merge { get; }
    public List<Block> Blocks { get; } = new();
    public DenseLayer<T> NormModulation { get; }
    public BiasFreeLinearLayer<T> NormWeight { get; }
    public DenseLayer<T> SkipLinear { get; }
    public DenseLayer<T> Conv1 { get; }
    public IndexTtsTimestepEmbedder<T> Time2 { get; }
    public NormedConv1DLayer<T> WavenetCondition { get; }
    public List<WavenetLayer> Wavenet { get; } = new();
    public DenseLayer<T> ResidualProjection { get; }
    public DenseLayer<T> FinalModulation { get; }
    public NormedConv1DLayer<T> FinalLinear { get; }
    public NormedConv1DLayer<T> Conv2 { get; }

    private Tensor<T> AdaptiveRmsNorm(Tensor<T> x, DenseLayer<T> modulation, BiasFreeLinearLayer<T> weight, Tensor<T> time)
    {
        int d = Configuration.Hidden, frames = x.Shape[0];
        var wb = modulation.Forward(time);                                                         // [1, 2d]
        var scale = _engine.TensorBroadcastTo(_engine.TensorNarrow(wb, 1, 0, d), new[] { frames, d });
        var shift = _engine.TensorBroadcastTo(_engine.TensorNarrow(wb, 1, d, d), new[] { frames, d });
        return _engine.TensorAdd(_engine.TensorMultiply(scale, IndexTtsS2MelOps.RmsNorm(_engine, x, IndexTtsOps.Value(weight), 1e-5)), shift);
    }

    /// <summary>The rotary tables <c>[T, dk/2]</c> (cos, sin) of positions 0 … T−1 with base 10⁴.</summary>
    private (Tensor<T> Cos, Tensor<T> Sin) Rotary(int frames, int dk)
    {
        var cos = new Tensor<T>(new[] { frames, dk / 2 });
        var sin = new Tensor<T>(new[] { frames, dk / 2 });
        for (int p = 0; p < frames; p++)
            for (int k = 0; k < dk / 2; k++)
            {
                double angle = p / Math.Pow(10000, 2.0 * k / dk);
                cos[p, k] = NumOps.FromDouble(Math.Cos(angle));
                sin[p, k] = NumOps.FromDouble(Math.Sin(angle));
            }
        return (cos, sin);
    }

    /// <summary>gpt-fast's <c>apply_rotary_emb</c> on <c>[heads, T, dk]</c>: consecutive pairs as complex numbers.</summary>
    private Tensor<T> ApplyRotary(Tensor<T> x, Tensor<T> cos, Tensor<T> sin)
    {
        int heads = x.Shape[0], frames = x.Shape[1], dk = x.Shape[2];
        var pairs = _engine.Reshape(x, new[] { heads, frames, dk / 2, 2 });
        var a = _engine.Reshape(_engine.TensorNarrow(pairs, 3, 0, 1), new[] { heads, frames, dk / 2 });
        var b = _engine.Reshape(_engine.TensorNarrow(pairs, 3, 1, 1), new[] { heads, frames, dk / 2 });
        var c = _engine.TensorBroadcastTo(_engine.Reshape(cos, new[] { 1, frames, dk / 2 }), new[] { heads, frames, dk / 2 });
        var s = _engine.TensorBroadcastTo(_engine.Reshape(sin, new[] { 1, frames, dk / 2 }), new[] { heads, frames, dk / 2 });
        var real = _engine.TensorSubtract(_engine.TensorMultiply(a, c), _engine.TensorMultiply(b, s));
        var imaginary = _engine.TensorAdd(_engine.TensorMultiply(b, c), _engine.TensorMultiply(a, s));
        var stacked = _engine.TensorConcatenate(new[] { _engine.Reshape(real, new[] { heads, frames, dk / 2, 1 }),
            _engine.Reshape(imaginary, new[] { heads, frames, dk / 2, 1 }) }, 3);
        return _engine.Reshape(stacked, new[] { heads, frames, dk });
    }

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random? random) =>
        training && rate > 0 && random is not null ? T5Seq2Seq<T>.Dropout(_engine, x, rate, random) : x;

    /// <summary>
    /// The predicted velocity <c>[T, mel]</c> of the noisy mel <c>x [T, mel]</c> at time <paramref name="t"/>, with the
    /// prompt mel <c>[T, mel]</c> (zero past the prompt), the style <c>[style]</c> and the content <c>[T, channels]</c>.
    /// <paramref name="dropCondition"/> zeroes everything but the noisy mel after the content projection (the class
    /// dropout in training; classifier-free guidance instead passes zero inputs).
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> prompt, double t, Tensor<T> style, Tensor<T> content, bool dropCondition,
        bool training = false, Random? random = null)
    {
        var c = Configuration;
        int frames = x.Shape[0], d = c.Hidden, heads = c.Heads, dk = d / heads, w = c.WavenetHidden;
        var t1 = Time.Forward(t);
        var styleRows = _engine.TensorBroadcastTo(_engine.Reshape(style, new[] { 1, c.StyleDim }), new[] { frames, c.StyleDim });
        var condition = _engine.TensorConcatenate(new[] { prompt, ContentProjection.Forward(content), styleRows }, 1);
        if (dropCondition) condition = new Tensor<T>(condition.Shape.ToArray());
        var h = Merge.Forward(_engine.TensorConcatenate(new[] { x, condition }, 1));                    // [T, d]

        var (cos, sin) = Rotary(frames, dk);
        var skips = new Stack<Tensor<T>>();
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            if (b.SkipIn is not null) h = b.SkipIn.Forward(_engine.TensorConcatenate(new[] { h, skips.Pop() }, 1));
            var normed = AdaptiveRmsNorm(h, b.AttentionModulation, b.AttentionNorm, t1);
            var qkv = b.QueryKeyValue.Forward(normed);
            var q = ApplyRotary(IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(qkv, 1, 0, d), heads), cos, sin);
            var k = ApplyRotary(IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(qkv, 1, d, d), heads), cos, sin);
            var v = IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(qkv, 1, 2 * d, d), heads);
            h = _engine.TensorAdd(h, b.Output.Forward(IndexTtsOps.MergeHeads(_engine, IndexTtsOps.Attend(_engine, q, k, v, 1.0 / Math.Sqrt(dk), null))));
            var f = AdaptiveRmsNorm(h, b.FeedForwardModulation, b.FeedForwardNorm, t1);
            h = _engine.TensorAdd(h, b.W2.Forward(_engine.TensorMultiply(_engine.Swish(b.W1.Forward(f)), b.W3.Forward(f))));
            if (i < Blocks.Count / 2) skips.Push(h);
        }
        var residual = AdaptiveRmsNorm(h, NormModulation, NormWeight, t1);
        residual = SkipLinear.Forward(_engine.TensorConcatenate(new[] { residual, x }, 1));            // long skip

        var y = _engine.Reshape(_engine.TensorTranspose(Conv1.Forward(residual)), new[] { 1, w, frames });
        var g = WavenetCondition.Forward(_engine.Reshape(Time2.Forward(t), new[] { 1, w, 1 }));        // [1, 2wL, 1]
        var output = new Tensor<T>(new[] { 1, w, frames });
        for (int i = 0; i < Wavenet.Count; i++)
        {
            var layer = Wavenet[i];
            int pad = c.WavenetKernel - 1;
            var input = layer.In.Forward(VocoderOps.ReflectPad(_engine, y, pad - pad / 2, pad / 2));
            var gl = _engine.TensorBroadcastTo(_engine.TensorNarrow(g, 1, 2 * w * i, 2 * w), new[] { 1, 2 * w, frames });
            var sum = _engine.TensorAdd(input, gl);
            var acts = _engine.TensorMultiply(_engine.Tanh(_engine.TensorNarrow(sum, 1, 0, w)), _engine.Sigmoid(_engine.TensorNarrow(sum, 1, w, w)));
            acts = Drop(acts, c.WavenetDropout, training, random);
            var rs = layer.ResidualSkip.Forward(acts);
            if (i < Wavenet.Count - 1)
            {
                y = _engine.TensorAdd(y, _engine.TensorNarrow(rs, 1, 0, w));
                output = _engine.TensorAdd(output, _engine.TensorNarrow(rs, 1, w, w));
            }
            else
            {
                output = _engine.TensorAdd(output, rs);
            }
        }
        var z = _engine.TensorAdd(_engine.TensorTranspose(_engine.Reshape(output, new[] { w, frames })), ResidualProjection.Forward(residual));
        var modulation = FinalModulation.Forward(_engine.Swish(t1));
        var shift = _engine.TensorBroadcastTo(_engine.TensorNarrow(modulation, 1, 0, w), new[] { frames, w });
        var scale = _engine.TensorBroadcastTo(_engine.TensorNarrow(modulation, 1, w, w), new[] { frames, w });
        z = _engine.TensorAdd(_engine.TensorMultiply(MegaTts3Ops.PlainLayerNorm(_engine, z, 1e-6), _engine.TensorAddScalar(scale, NumOps.One)), shift);
        z = IndexTtsS2MelOps.PerRow(_engine, FinalLinear, z);
        return IndexTtsS2MelOps.PerRow(_engine, Conv2, z);
    }

    public void Load(Func<string, int[], double[]> read, Func<string, bool> has, string prefix)
    {
        var c = Configuration;
        int d = c.Hidden, w = c.WavenetHidden;
        Time.Load(read, prefix + "t_embedder.");
        IndexTtsOps.LoadLinear(_engine, read, ContentProjection, prefix + "cond_projection", c.RegulatorChannels, d);
        IndexTtsOps.LoadLinear(_engine, read, Merge, prefix + "cond_x_merge_linear", d + 2 * c.MelChannels + c.StyleDim, d);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}transformer.layers.{i}.";
            IndexTtsOps.LoadLinear(_engine, read, b.AttentionModulation, p + "attention_norm.project_layer", d, 2 * d);
            TorchParameters.Linear(b.AttentionNorm, 1, d, read(p + "attention_norm.norm.weight", new[] { d }));
            IndexTtsOps.LoadLinear(read, b.QueryKeyValue, p + "attention.wqkv", d, 3 * d);
            IndexTtsOps.LoadLinear(read, b.Output, p + "attention.wo", d, d);
            IndexTtsOps.LoadLinear(_engine, read, b.FeedForwardModulation, p + "ffn_norm.project_layer", d, 2 * d);
            TorchParameters.Linear(b.FeedForwardNorm, 1, d, read(p + "ffn_norm.norm.weight", new[] { d }));
            IndexTtsOps.LoadLinear(read, b.W1, p + "feed_forward.w1", d, Intermediate);
            IndexTtsOps.LoadLinear(read, b.W3, p + "feed_forward.w3", d, Intermediate);
            IndexTtsOps.LoadLinear(read, b.W2, p + "feed_forward.w2", Intermediate, d);
            if (b.SkipIn is not null) IndexTtsOps.LoadLinear(_engine, read, b.SkipIn, p + "skip_in_linear", 2 * d, d);
        }
        IndexTtsOps.LoadLinear(_engine, read, NormModulation, prefix + "transformer.norm.project_layer", d, 2 * d);
        TorchParameters.Linear(NormWeight, 1, d, read(prefix + "transformer.norm.norm.weight", new[] { d }));
        IndexTtsOps.LoadLinear(_engine, read, SkipLinear, prefix + "skip_linear", d + c.MelChannels, d);
        IndexTtsOps.LoadLinear(_engine, read, Conv1, prefix + "conv1", d, w);
        Time2.Load(read, prefix + "t_embedder2.");
        IndexTtsS2MelOps.LoadWeightNorm(read, has, WavenetCondition, prefix + "wavenet.cond_layer.conv.conv", 2 * w * c.WavenetLayers);
        for (int i = 0; i < Wavenet.Count; i++)
        {
            IndexTtsS2MelOps.LoadWeightNorm(read, has, Wavenet[i].In, $"{prefix}wavenet.in_layers.{i}.conv.conv", 2 * w);
            IndexTtsS2MelOps.LoadWeightNorm(read, has, Wavenet[i].ResidualSkip, $"{prefix}wavenet.res_skip_layers.{i}.conv.conv",
                i < Wavenet.Count - 1 ? 2 * w : w);
        }
        IndexTtsOps.LoadLinear(_engine, read, ResidualProjection, prefix + "res_projection", d, w);
        IndexTtsOps.LoadLinear(_engine, read, FinalModulation, prefix + "final_layer.adaLN_modulation.1", w, 2 * w);
        IndexTtsS2MelOps.LoadWeightNorm(read, has, FinalLinear, prefix + "final_layer.linear", w);
        IndexTtsOps.LoadConv(read, Conv2, prefix + "conv2", c.MelChannels);
    }
}

/// <summary>
/// IndexTTS2's semantic-to-mel module (paper § Semantic-to-Mel; the release's <c>MyModel</c> with
/// <c>use_gpt_latent</c>): the GPT latent MLP that fuses the text-to-semantic model's last hidden states into the
/// semantic features, the length regulator, and conditional flow matching (σ_min = 1e-6) with the DiT estimator solved by
/// Euler steps under classifier-free guidance.
/// </summary>
internal sealed class IndexTtsSemanticToMel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const double SigmaMin = 1e-6;
    private readonly IEngine _engine;

    public IndexTtsSemanticToMel(IEngine engine, List<LayerBase<T>> layers, IndexTtsSemanticToMelConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Regulator = new IndexTtsLengthRegulator<T>(engine, layers, random, c.SemanticDim, c.RegulatorChannels, c.RegulatorStages);
        Estimator = new IndexTtsDiT<T>(engine, layers, c, random);
        Gpt1 = IndexTtsOps.Linear(engine, layers, random, c.GptDim, c.GptBottleneck1);
        Gpt2 = IndexTtsOps.Linear(engine, layers, random, c.GptBottleneck1, c.GptBottleneck2);
        Gpt3 = IndexTtsOps.Linear(engine, layers, random, c.GptBottleneck2, c.SemanticDim);
    }

    public IndexTtsSemanticToMelConfiguration Configuration { get; }
    public IndexTtsLengthRegulator<T> Regulator { get; }
    public IndexTtsDiT<T> Estimator { get; }
    public DenseLayer<T> Gpt1 { get; }
    public DenseLayer<T> Gpt2 { get; }
    public DenseLayer<T> Gpt3 { get; }

    /// <summary>The GPT latent enhancement: <c>Q_sem + MLP(H_GPT)</c> of quantized semantic features <c>[n, 1024]</c> and
    /// the text-to-semantic latents <c>[n, d]</c> (three linear maps, as the release's <c>gpt_layer</c>).</summary>
    public Tensor<T> Fuse(Tensor<T> semantic, Tensor<T> latents) =>
        _engine.TensorAdd(semantic, Gpt3.Forward(Gpt2.Forward(Gpt1.Forward(latents))));

    /// <summary>
    /// The mel <c>[frames, mel]</c> after the prompt: Euler steps of the flow from noise <c>[P + frames, mel]</c>
    /// conditioned on <c>[prompt condition; target condition]</c>, the prompt mel <c>[P, mel]</c> and the style, guided
    /// by <c>(1 + r)·v − r·v_uncond</c>; the prompt region of the state stays zero.
    /// </summary>
    public Tensor<T> Solve(Tensor<T> condition, Tensor<T> promptMel, Tensor<T> style, Tensor<T> noise, int steps, double guidance)
    {
        int total = condition.Shape[0], prompt = promptMel.Shape[0], mels = Configuration.MelChannels;
        var promptX = _engine.TensorConcatenate(new[] { promptMel, new Tensor<T>(new[] { total - prompt, mels }) }, 0);
        var keep = new Tensor<T>(new[] { total, mels });
        for (int i = prompt; i < total; i++)
            for (int m = 0; m < mels; m++) keep[i, m] = NumOps.One;
        var x = _engine.TensorMultiply(noise, keep);
        var zeroPrompt = new Tensor<T>(promptX.Shape.ToArray());
        var zeroStyle = new Tensor<T>(style.Shape.ToArray());
        var zeroCondition = new Tensor<T>(condition.Shape.ToArray());
        for (int step = 1; step <= steps; step++)
        {
            double t = (double)(step - 1) / steps, dt = (double)step / steps - t;
            var v = Estimator.Forward(x, promptX, t, style, condition, false);
            if (guidance > 0)
            {
                // The unconditional pass zeroes the prompt, the style and the content before the estimator (so the
                // content projection's bias remains), unlike the class dropout, which zeroes the merged inputs.
                var unconditional = Estimator.Forward(x, zeroPrompt, t, zeroStyle, zeroCondition, false);
                v = _engine.TensorSubtract(_engine.TensorMultiplyScalar(v, NumOps.FromDouble(1 + guidance)),
                    _engine.TensorMultiplyScalar(unconditional, NumOps.FromDouble(guidance)));
            }
            x = _engine.TensorMultiply(_engine.TensorAdd(x, _engine.TensorMultiplyScalar(v, NumOps.FromDouble(dt))), keep);
        }
        return _engine.TensorNarrow(x, 0, prompt, total - prompt);
    }

    /// <summary>
    /// The flow-matching loss (the release's <c>CFM.forward</c>): with t ~ U(0, 1) and z ~ N(0, I), the state
    /// <c>(1 − (1 − σ)t) z + t x₁</c> with the prompt frames zeroed, regressed by L1 onto <c>x₁ − (1 − σ) z</c> over the
    /// target frames (Eq. 2); the conditioning is dropped with the class dropout probability.
    /// </summary>
    public Tensor<T> Loss(Tensor<T> mel, int prompt, Tensor<T> style, Tensor<T> condition, Random random)
    {
        var c = Configuration;
        int total = mel.Shape[0], mels = c.MelChannels;
        double t = random.NextDouble();
        var z = new Tensor<T>(new[] { total, mels });
        for (int i = 0; i < z.Length; i++) z[i] = NumOps.FromDouble(IndexTtsOps.Gaussian(random));
        var target = new Tensor<T>(new[] { total, mels });
        for (int i = prompt; i < total; i++)
            for (int m = 0; m < mels; m++) target[i, m] = NumOps.One;
        var y = _engine.TensorAdd(_engine.TensorMultiplyScalar(z, NumOps.FromDouble(1 - (1 - SigmaMin) * t)), _engine.TensorMultiplyScalar(mel, NumOps.FromDouble(t)));
        y = _engine.TensorMultiply(y, target);
        var u = _engine.TensorSubtract(mel, _engine.TensorMultiplyScalar(z, NumOps.FromDouble(1 - SigmaMin)));
        var promptX = _engine.TensorMultiply(mel, _engine.TensorSubtract(_engine.TensorAddScalar(new Tensor<T>(target.Shape.ToArray()), NumOps.One), target));
        bool drop = random.NextDouble() < c.ClassDropout;
        var v = Estimator.Forward(y, promptX, t, style, condition, drop, training: true, random: random);
        var error = _engine.TensorMultiply(_engine.TensorAbs(_engine.TensorSubtract(v, u)), target);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(error, null, false), NumOps.FromDouble(1.0 / ((total - prompt) * mels)));
    }

    /// <summary>Loads the release's <c>s2mel.pth</c> modules (<c>cfm</c>, <c>length_regulator</c>, <c>gpt_layer</c>).</summary>
    public void Load(Func<string, int[], double[]> read, Func<string, bool> has)
    {
        var c = Configuration;
        Regulator.Load(_engine, read, "length_regulator.");
        Estimator.Load(read, has, "cfm.estimator.");
        IndexTtsOps.LoadLinear(_engine, read, Gpt1, "gpt_layer.0", c.GptDim, c.GptBottleneck1);
        IndexTtsOps.LoadLinear(_engine, read, Gpt2, "gpt_layer.1", c.GptBottleneck1, c.GptBottleneck2);
        IndexTtsOps.LoadLinear(_engine, read, Gpt3, "gpt_layer.2", c.GptBottleneck2, c.SemanticDim);
    }
}
