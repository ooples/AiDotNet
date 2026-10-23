using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Shared operations of MeloTTS's modules (myshell-ai/MeloTTS, after Bert-VITS2) on <c>[1, C, T]</c> tensors.</summary>
internal static class MeloOps
{
    /// <summary>A plain <c>nn.Conv1d</c> (no weight reparameterization), as MeloTTS's modules build most convolutions.</summary>
    public static NormedConv1DLayer<T> Conv<T>(List<LayerBase<T>> layers, int input, int output, int kernel, int dilation = 1,
        int padding = 0, int groups = 1, bool bias = true)
    {
        var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, dilation, groups, padding, false, ConvolutionNormalization.None, useBias: bias);
        layers.Add(conv);
        return conv;
    }

    /// <summary>A weight-normalized <c>nn.Conv1d</c> (WaveNet's layers).</summary>
    public static NormedConv1DLayer<T> WeightNormConv<T>(List<LayerBase<T>> layers, int input, int output, int kernel, int dilation = 1, int padding = 0)
    {
        var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, dilation, 1, padding, false, ConvolutionNormalization.Weight);
        layers.Add(conv);
        return conv;
    }

    /// <summary>A channel LayerNorm (<c>modules.LayerNorm</c>: layer_norm over the channels of <c>[1, C, T]</c>, ε 1e-5).</summary>
    public static LayerNormalizationLayer<T> Norm<T>(List<LayerBase<T>> layers, int channels)
    {
        var norm = new LayerNormalizationLayer<T>(channels, 1e-5);
        layers.Add(norm);
        return norm;
    }

    public static Tensor<T> ChannelNorm<T>(IEngine engine, LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var rows = engine.TensorTranspose(engine.Reshape(x, new[] { c, t }));
        return engine.Reshape(engine.TensorTranspose(norm.Forward(rows)), new[] { 1, c, t });
    }

    public static Tensor<T> Channels<T>(IEngine engine, Tensor<T> x, int start, int count) =>
        engine.TensorSlice(x, new[] { 0, start, 0 }, new[] { 1, count, x.Shape[2] });

    /// <summary>A stop-gradient copy (<c>torch.detach</c>).</summary>
    public static Tensor<T> Detach<T>(Tensor<T> x) => new(x._shape, x.ToVector());

    /// <summary><c>torch.flip(x, [1])</c> over the channels.</summary>
    public static Tensor<T> Flip<T>(IEngine engine, Tensor<T> x)
    {
        int c = x.Shape[1];
        var parts = new Tensor<T>[c];
        for (int i = 0; i < c; i++) parts[i] = Channels(engine, x, c - 1 - i, 1);
        return c == 1 ? x : engine.TensorConcatenate(parts, 1);
    }

    /// <summary>A <c>[1, C, 1]</c> vector repeated over <paramref name="frames"/>.</summary>
    public static Tensor<T> OverTime<T>(IEngine engine, Tensor<T> g, int frames) => engine.TensorTile(g, new[] { 1, 1, frames });

    public static void LoadConv<T>(Func<string, double[]> read, NormedConv1DLayer<T> conv, string name, bool bias = true)
    {
        if (conv.Normalization == ConvolutionNormalization.Weight)
            conv.LoadTorchWeights(read(name + ".weight_v"), read(name + ".weight_g"), read(name + ".bias"));
        else
            conv.LoadTorchWeights(read(name + ".weight"), null, bias ? read(name + ".bias") : new double[conv.OutputChannels]);
    }

    public static void LoadNorm<T>(IEngine engine, Func<string, double[]> read, LayerNormalizationLayer<T> norm, string name, string weight = "gamma", string bias = "beta")
        => AiDotNet.TextToSpeech.CodecBased.TorchParameters.LayerNorm(engine, norm, read($"{name}.{weight}"), read($"{name}.{bias}"));

    public static void LoadLinear<T>(IEngine engine, Func<string, double[]> read, DenseLayer<T> dense, int inputs, int outputs, string name)
        => AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(engine, dense, inputs, outputs, read(name + ".weight"), read(name + ".bias"));
}

/// <summary>
/// VITS's relative-position Transformer encoder as MeloTTS uses it (<c>attentions.Encoder</c>): post-norm blocks of
/// multi-head self-attention with learned relative key and value embeddings (window 4, shared by the heads) and a
/// two-convolution ReLU feed-forward; a speaker embedding, projected by a linear layer, is added before block
/// <c>condLayer</c> (2).
/// </summary>
internal sealed class MeloAttentionEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _hidden, _heads, _window, _kernel, _condLayer;
    private readonly double _dropout;

    internal sealed record Block(NormedConv1DLayer<T> Query, NormedConv1DLayer<T> Key, NormedConv1DLayer<T> Value, NormedConv1DLayer<T> Output,
        TiedEmbeddingLayer<T> RelativeKeys, TiedEmbeddingLayer<T> RelativeValues, LayerNormalizationLayer<T> Norm1,
        NormedConv1DLayer<T> Ffn1, NormedConv1DLayer<T> Ffn2, LayerNormalizationLayer<T> Norm2);

    public MeloAttentionEncoder(IEngine engine, List<LayerBase<T>> layers, int hidden, int filter, int heads, int blocks, int kernel,
        double dropout, int speakerChannels, int window = 4, int condLayer = 2)
    {
        _engine = engine;
        _hidden = hidden;
        _heads = heads;
        _window = window;
        _kernel = kernel;
        _dropout = dropout;
        int dk = hidden / heads;
        for (int i = 0; i < blocks; i++)
        {
            var q = MeloOps.Conv(layers, hidden, hidden, 1);
            var k = MeloOps.Conv(layers, hidden, hidden, 1);
            var v = MeloOps.Conv(layers, hidden, hidden, 1);
            var o = MeloOps.Conv(layers, hidden, hidden, 1);
            var relK = new TiedEmbeddingLayer<T>(2 * window + 1, dk);
            var relV = new TiedEmbeddingLayer<T>(2 * window + 1, dk);
            layers.Add(relK);
            layers.Add(relV);
            var n1 = MeloOps.Norm(layers, hidden);
            // FFN pads (k − 1) / 2 on the left and k / 2 on the right ("same"); the convolution itself has none.
            var f1 = MeloOps.Conv(layers, hidden, filter, kernel);
            var f2 = MeloOps.Conv(layers, filter, hidden, kernel);
            var n2 = MeloOps.Norm(layers, hidden);
            Blocks.Add(new Block(q, k, v, o, relK, relV, n1, f1, f2, n2));
        }
        _condLayer = blocks;
        if (speakerChannels > 0)
        {
            if (condLayer >= blocks) throw new ArgumentException("The speaker's layer must be below the block count.", nameof(condLayer));
            _condLayer = condLayer;
            SpeakerProjection = new DenseLayer<T>(hidden, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(SpeakerProjection);
        }
        SpeakerChannels = speakerChannels;
    }

    public List<Block> Blocks { get; } = new();
    public DenseLayer<T>? SpeakerProjection { get; }
    public int SpeakerChannels { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && _dropout > 0 ? AiDotNet.TextToSpeech.CodecBased.T5Seq2Seq<T>.Dropout(_engine, x, _dropout, random) : x;

    // F.pad "same" for a kernel of size k along time.
    private Tensor<T> PadSame(Tensor<T> x)
    {
        if (_kernel == 1) return x;
        int left = (_kernel - 1) / 2, right = _kernel / 2, c = x.Shape[1], t = x.Shape[2];
        var parts = new List<Tensor<T>>();
        if (left > 0) parts.Add(new Tensor<T>(new[] { 1, c, left }));
        parts.Add(x);
        if (right > 0) parts.Add(new Tensor<T>(new[] { 1, c, right }));
        return _engine.TensorConcatenate(parts.ToArray(), 2);
    }

    // _get_relative_embeddings: the 2L − 1 relative positions −(L − 1) … L − 1 of the 2w + 1 learned ones, zero beyond ±w.
    private Tensor<T> Relative(TiedEmbeddingLayer<T> table, int length)
    {
        int dk = table.Dimension, padding = Math.Max(length - (_window + 1), 0), start = Math.Max(_window + 1 - length, 0);
        var used = table.Table;
        if (padding > 0)
            used = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { padding, dk }), used, new Tensor<T>(new[] { padding, dk }) }, 0);
        return _engine.TensorSlice(used, new[] { start, 0 }, new[] { 2 * length - 1, dk });
    }

    private Tensor<T> Attention(Block b, Tensor<T> x, bool training, Random random)
    {
        int length = x.Shape[2], h = _heads, dk = _hidden / h;
        Tensor<T> Heads(Tensor<T> y) => _engine.TensorPermute(_engine.Reshape(y, new[] { h, dk, length }), new[] { 0, 2, 1 });   // [h, L, dk]
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(dk));
        var q = _engine.TensorMultiplyScalar(Heads(b.Query.Forward(x)), scale);
        var k = Heads(b.Key.Forward(x));
        var v = Heads(b.Value.Forward(x));
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                         // [h, L, L]
        // Relative logits [h, L, 2L − 1] to absolute positions: score[i, j] += rel[i, L − 1 + j − i].
        var relKeys = Relative(b.RelativeKeys, length);                                                            // [2L − 1, dk]
        var relLogits = _engine.Reshape(_engine.TensorMatMul(_engine.Reshape(q, new[] { h * length, dk }), _engine.TensorTranspose(relKeys)),
            new[] { h, length * (2 * length - 1) });
        var toAbsolute = new Tensor<int>(new[] { length * length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++) toAbsolute[i * length + j] = i * (2 * length - 1) + (length - 1 + j - i);
        scores = _engine.TensorAdd(scores, _engine.Reshape(_engine.TensorIndexSelect(relLogits, toAbsolute, 1), new[] { h, length, length }));
        var p = Drop(_engine.Softmax(scores, axis: 2), training, random);
        var output = _engine.BatchMatMul(p, v);                                                                    // [h, L, dk]
        // Absolute weights to relative positions: rel[i, r] = p[i, i + r − (L − 1)] where that is a position, else 0.
        var padded = _engine.TensorConcatenate(new[] { _engine.Reshape(p, new[] { h, length * length }), new Tensor<T>(new[] { h, 1 }) }, 1);
        var toRelative = new Tensor<int>(new[] { length * (2 * length - 1) });
        for (int i = 0; i < length; i++)
            for (int r = 0; r < 2 * length - 1; r++)
            {
                int j = i + r - (length - 1);
                toRelative[i * (2 * length - 1) + r] = j >= 0 && j < length ? i * length + j : length * length;
            }
        var relWeights = _engine.Reshape(_engine.TensorIndexSelect(padded, toRelative, 1), new[] { h * length, 2 * length - 1 });
        var relValues = Relative(b.RelativeValues, length);
        output = _engine.TensorAdd(output, _engine.Reshape(_engine.TensorMatMul(relWeights, relValues), new[] { h, length, dk }));
        var merged = _engine.Reshape(_engine.TensorPermute(output, new[] { 0, 2, 1 }), new[] { 1, _hidden, length });
        return b.Output.Forward(merged);
    }

    /// <summary>Encodes <paramref name="x"/> <c>[1, hidden, T]</c>, with the speaker <paramref name="g"/> <c>[1, gin, 1]</c> when given.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? g, bool training, Random random)
    {
        int length = x.Shape[2];
        for (int i = 0; i < Blocks.Count; i++)
        {
            if (i == _condLayer && g is not null && SpeakerProjection is not null)
            {
                var projected = SpeakerProjection.Forward(_engine.Reshape(g, new[] { 1, SpeakerChannels }));
                x = _engine.TensorAdd(x, MeloOps.OverTime(_engine, _engine.Reshape(projected, new[] { 1, _hidden, 1 }), length));
            }
            var b = Blocks[i];
            var y = Drop(Attention(b, x, training, random), training, random);
            x = MeloOps.ChannelNorm(_engine, b.Norm1, _engine.TensorAdd(x, y));
            y = b.Ffn1.Forward(PadSame(x));
            y = Drop(_engine.ReLU(y), training, random);
            y = Drop(b.Ffn2.Forward(PadSame(y)), training, random);
            x = MeloOps.ChannelNorm(_engine, b.Norm2, _engine.TensorAdd(x, y));
        }
        return x;
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string a = $"{prefix}attn_layers.{i}";
            MeloOps.LoadConv(read, b.Query, a + ".conv_q");
            MeloOps.LoadConv(read, b.Key, a + ".conv_k");
            MeloOps.LoadConv(read, b.Value, a + ".conv_v");
            MeloOps.LoadConv(read, b.Output, a + ".conv_o");
            b.RelativeKeys.LoadTable(read(a + ".emb_rel_k"));
            b.RelativeValues.LoadTable(read(a + ".emb_rel_v"));
            MeloOps.LoadNorm(engine, read, b.Norm1, $"{prefix}norm_layers_1.{i}");
            MeloOps.LoadConv(read, b.Ffn1, $"{prefix}ffn_layers.{i}.conv_1");
            MeloOps.LoadConv(read, b.Ffn2, $"{prefix}ffn_layers.{i}.conv_2");
            MeloOps.LoadNorm(engine, read, b.Norm2, $"{prefix}norm_layers_2.{i}");
        }
        if (SpeakerProjection is not null)
            MeloOps.LoadLinear(engine, read, SpeakerProjection, SpeakerChannels, _hidden, prefix + "spk_emb_linear");
    }
}

/// <summary>
/// VITS's dilated depth-separable convolution stack (<c>modules.DDSConv</c>): per layer a depthwise convolution with
/// dilation kⁱ, channel LayerNorm, exact GELU, a 1×1 convolution, LayerNorm, GELU and dropout, added residually.
/// </summary>
internal sealed class MeloDdsConv<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;
    private readonly List<(NormedConv1DLayer<T> Separable, LayerNormalizationLayer<T> Norm1, NormedConv1DLayer<T> Pointwise, LayerNormalizationLayer<T> Norm2)> _layers = new();

    public MeloDdsConv(IEngine engine, List<LayerBase<T>> layers, int channels, int kernel, int count, double dropout)
    {
        _engine = engine;
        _dropout = dropout;
        for (int i = 0; i < count; i++)
        {
            int dilation = (int)Math.Pow(kernel, i);
            var sep = MeloOps.Conv(layers, channels, channels, kernel, dilation, (kernel * dilation - dilation) / 2, groups: channels);
            var n1 = MeloOps.Norm(layers, channels);
            var pw = MeloOps.Conv(layers, channels, channels, 1);
            var n2 = MeloOps.Norm(layers, channels);
            _layers.Add((sep, n1, pw, n2));
        }
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? g, bool training, Random random)
    {
        if (g is not null) x = _engine.TensorAdd(x, g);
        foreach (var (sep, n1, pw, n2) in _layers)
        {
            var y = AiDotNet.Helpers.ActivationHelper.ExactGelu(MeloOps.ChannelNorm(_engine, n1, sep.Forward(x)), _engine);
            y = AiDotNet.Helpers.ActivationHelper.ExactGelu(MeloOps.ChannelNorm(_engine, n2, pw.Forward(y)), _engine);
            if (training && _dropout > 0) y = AiDotNet.TextToSpeech.CodecBased.T5Seq2Seq<T>.Dropout(_engine, y, _dropout, random);
            x = _engine.TensorAdd(x, y);
        }
        return x;
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        for (int i = 0; i < _layers.Count; i++)
        {
            MeloOps.LoadConv(read, _layers[i].Separable, $"{prefix}convs_sep.{i}");
            MeloOps.LoadNorm(engine, read, _layers[i].Norm1, $"{prefix}norms_1.{i}");
            MeloOps.LoadConv(read, _layers[i].Pointwise, $"{prefix}convs_1x1.{i}");
            MeloOps.LoadNorm(engine, read, _layers[i].Norm2, $"{prefix}norms_2.{i}");
        }
    }
}

/// <summary>
/// VITS's neural spline flow layer (<c>modules.ConvFlow</c>): the first channel conditions a rational-quadratic spline
/// (10 bins, linear tails beyond ±5) of the second.
/// </summary>
internal sealed class MeloConvFlow<T>
{
    private const int Bins = 10;
    private const double TailBound = 5.0;
    private readonly IEngine _engine;
    private readonly int _filters;
    private readonly NormedConv1DLayer<T> _pre, _proj;
    private readonly MeloDdsConv<T> _convs;

    public MeloConvFlow(IEngine engine, List<LayerBase<T>> layers, int filters, int kernel)
    {
        _engine = engine;
        _filters = filters;
        _pre = MeloOps.Conv(layers, 1, filters, 1);
        _convs = new MeloDdsConv<T>(engine, layers, filters, kernel, 3, 0.0);
        _proj = MeloOps.Conv(layers, filters, 3 * Bins - 1, 1);
    }

    public (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> x, Tensor<T> g, bool reverse)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int t = x.Shape[2];
        var x0 = MeloOps.Channels(_engine, x, 0, 1);
        var x1 = MeloOps.Channels(_engine, x, 1, 1);
        var h = _proj.Forward(_convs.Forward(_pre.Forward(x0), g, false, null!));
        var rows = _engine.TensorTranspose(_engine.Reshape(h, new[] { 3 * Bins - 1, t }));                       // [T, 29]
        var scale = ops.FromDouble(1.0 / Math.Sqrt(_filters));
        var widths = _engine.TensorMultiplyScalar(_engine.TensorSlice(rows, new[] { 0, 0 }, new[] { t, Bins }), scale);
        var heights = _engine.TensorMultiplyScalar(_engine.TensorSlice(rows, new[] { 0, Bins }, new[] { t, Bins }), scale);
        var derivatives = _engine.TensorSlice(rows, new[] { 0, 2 * Bins }, new[] { t, Bins - 1 });
        var (y1, logAbsDet) = RationalQuadraticSpline.Apply(_engine, _engine.Reshape(x1, new[] { t }), widths, heights, derivatives, reverse, TailBound);
        var output = _engine.TensorConcatenate(new[] { x0, _engine.Reshape(y1, new[] { 1, 1, t }) }, 1);
        return (output, reverse ? null : _engine.ReduceSum(logAbsDet, new[] { 0 }, keepDims: false));
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        MeloOps.LoadConv(read, _pre, prefix + "pre");
        _convs.LoadTorch(engine, read, prefix + "convs.");
        MeloOps.LoadConv(read, _proj, prefix + "proj");
    }
}

/// <summary>VITS's <c>modules.ElementwiseAffine</c>: y = m + e^{logs}·x per channel, both zero-initialized.</summary>
internal sealed class MeloAffine<T>
{
    private readonly IEngine _engine;

    public MeloAffine(IEngine engine, List<LayerBase<T>> layers, int channels)
    {
        _engine = engine;
        Mean = new TiedEmbeddingLayer<T>(channels, 1);
        LogScale = new TiedEmbeddingLayer<T>(channels, 1);
        Mean.Reinitialize(() => 0.0);
        LogScale.Reinitialize(() => 0.0);
        layers.Add(Mean);
        layers.Add(LogScale);
    }

    public TiedEmbeddingLayer<T> Mean { get; }
    public TiedEmbeddingLayer<T> LogScale { get; }

    public (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> x, bool reverse)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var m = MeloOps.OverTime(_engine, _engine.Reshape(Mean.Table, new[] { 1, c, 1 }), t);
        var logs = MeloOps.OverTime(_engine, _engine.Reshape(LogScale.Table, new[] { 1, c, 1 }), t);
        if (!reverse)
            return (_engine.TensorAdd(m, _engine.TensorMultiply(_engine.TensorExp(logs), x)), _engine.ReduceSum(logs, new[] { 0, 1, 2 }, keepDims: false));
        return (_engine.TensorMultiply(_engine.TensorSubtract(x, m), _engine.TensorExp(_engine.TensorNegate(logs))), null);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        Mean.LoadTable(read(prefix + "m"));
        LogScale.LoadTable(read(prefix + "logs"));
    }
}
