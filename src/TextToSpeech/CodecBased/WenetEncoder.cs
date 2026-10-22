using System.Collections.Generic;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>How a WeNet encoder projects its input before the blocks (WeNet's <c>input_layer</c>).</summary>
internal enum WenetInputLayer
{
    /// <summary><c>linear</c>: Linear → LayerNorm(ε 1e-5) → Dropout.</summary>
    Linear,

    /// <summary><c>linear_legacy</c>: Linear → LayerNorm(ε 1e-5) → Dropout → ReLU.</summary>
    LinearLegacy,
}

/// <summary>
/// A WeNet encoder (CosyVoice's <c>cosyvoice/transformer/encoder.py</c>) with ESPnet relative positional encoding
/// (<c>rel_pos_espnet</c>) and relative-position self-attention (<c>rel_selfattn</c>), pre-norm blocks, no convolution
/// module and no macaron feed-forward: the configuration every CosyVoice text encoder, language model and flow encoder
/// uses. <c>TransformerEncoder</c> and <c>ConformerEncoder(use_cnn_module=False, macaron_style=False)</c> compute the
/// same block; they differ only in the feed-forward activation (ReLU and Swish by default).
/// </summary>
internal sealed record WenetEncoderConfiguration(
    int InputSize, int OutputSize, int Heads, int LinearUnits, int Blocks,
    double Dropout = 0.1, double PositionalDropout = 0.1, double AttentionDropout = 0.0,
    WenetInputLayer InputLayer = WenetInputLayer.Linear, bool SwishActivation = false,
    int StaticChunkSize = 0, bool KeyBias = true);

/// <summary>One WeNet relative-position attention block: pre-norm self-attention, then a pre-norm feed-forward.</summary>
internal sealed class WenetEncoderBlock<T>
{
    private readonly IEngine _engine;
    private readonly WenetEncoderConfiguration _c;
    private readonly int _dk;

    public WenetEncoderBlock(IEngine engine, List<LayerBase<T>> layers, WenetEncoderConfiguration c)
    {
        _engine = engine;
        _c = c;
        if (c.OutputSize % c.Heads != 0) throw new ArgumentException($"{c.OutputSize} is not divisible by {c.Heads} heads.");
        _dk = c.OutputSize / c.Heads;
        AttentionNorm = Own(layers, new LayerNormalizationLayer<T>(c.OutputSize, 1e-12));
        Query = Own(layers, new DenseLayer<T>(c.OutputSize));
        Key = c.KeyBias ? Own(layers, new DenseLayer<T>(c.OutputSize)) : null;
        KeyNoBias = c.KeyBias ? null : Own(layers, new BiasFreeLinearLayer<T>(c.OutputSize, c.OutputSize));
        Value = Own(layers, new DenseLayer<T>(c.OutputSize));
        Output = Own(layers, new DenseLayer<T>(c.OutputSize));
        Position = Own(layers, new BiasFreeLinearLayer<T>(c.OutputSize, c.OutputSize));
        BiasU = Own(layers, new TiedEmbeddingLayer<T>(c.Heads, _dk));
        BiasV = Own(layers, new TiedEmbeddingLayer<T>(c.Heads, _dk));
        FeedForwardNorm = Own(layers, new LayerNormalizationLayer<T>(c.OutputSize, 1e-12));
        FeedForwardIn = Own(layers, new DenseLayer<T>(c.LinearUnits));
        FeedForwardOut = Own(layers, new DenseLayer<T>(c.OutputSize));
    }

    public LayerNormalizationLayer<T> AttentionNorm { get; }
    public DenseLayer<T> Query { get; }
    public DenseLayer<T>? Key { get; }
    public BiasFreeLinearLayer<T>? KeyNoBias { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> Output { get; }
    public BiasFreeLinearLayer<T> Position { get; }
    public TiedEmbeddingLayer<T> BiasU { get; }
    public TiedEmbeddingLayer<T> BiasV { get; }
    public LayerNormalizationLayer<T> FeedForwardNorm { get; }
    public DenseLayer<T> FeedForwardIn { get; }
    public DenseLayer<T> FeedForwardOut { get; }

    internal static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    /// <summary>
    /// <paramref name="x"/> <c>[T, D]</c>, <paramref name="positions"/> the relative position table <c>[2T − 1, D]</c>
    /// and <paramref name="maskBias"/> <c>[T, T]</c> (0, or a large negative value on keys a query may not attend).
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> positions, Tensor<T>? maskBias, bool training, Random random)
    {
        var normed = AttentionNorm.Forward(x);
        x = _engine.TensorAdd(x, Drop(Attention(normed, positions, maskBias, training, random), _c.Dropout, training, random));
        var hidden = FeedForwardIn.Forward(FeedForwardNorm.Forward(x));
        hidden = _c.SwishActivation ? _engine.Swish(hidden) : _engine.ReLU(hidden);
        hidden = Drop(hidden, _c.Dropout, training, random);
        return _engine.TensorAdd(x, Drop(FeedForwardOut.Forward(hidden), _c.Dropout, training, random));
    }

    // RelPositionMultiHeadedAttention: scores = ((q + u)·kᵀ + rel_shift((q + v)·pᵀ)) / √d_k.
    private Tensor<T> Attention(Tensor<T> x, Tensor<T> positions, Tensor<T>? maskBias, bool training, Random random)
    {
        int t = x.Shape[0], h = _c.Heads, dk = _dk, p = positions.Shape[0];
        Tensor<T> Heads(Tensor<T> y, int length) =>
            _engine.TensorPermute(_engine.Reshape(y, new[] { length, h, dk }), new[] { 1, 0, 2 });          // [H, L, dk]
        var q = _engine.Reshape(Query.Forward(x), new[] { t, h, dk });
        var k = Heads(Key is not null ? Key.Forward(x) : KeyNoBias!.Forward(x), t);
        var v = Heads(Value.Forward(x), t);
        var pos = Heads(Position.Forward(positions), p);                                                    // [H, P, dk]
        var qu = _engine.TensorPermute(_engine.TensorAdd(q, _engine.Reshape(BiasU.Table, new[] { 1, h, dk })), new[] { 1, 0, 2 });
        var qv = _engine.TensorPermute(_engine.TensorAdd(q, _engine.Reshape(BiasV.Table, new[] { 1, h, dk })), new[] { 1, 0, 2 });
        var ac = _engine.BatchMatMul(qu, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                       // [H, T, T]
        var bd = _engine.BatchMatMul(qv, _engine.TensorPermute(pos, new[] { 0, 2, 1 }));                     // [H, T, P]
        if (p != t) bd = RelativeShift(bd, h, t, p);
        var scores = _engine.TensorMultiplyScalar(_engine.TensorAdd(ac, bd), MathHelper.GetNumericOperations<T>().FromDouble(1.0 / Math.Sqrt(dk)));
        if (maskBias is not null) scores = _engine.TensorAdd(scores, _engine.Reshape(maskBias, new[] { 1, t, t }));
        var weights = _engine.Softmax(scores, axis: 2);
        weights = Drop(weights, _c.AttentionDropout, training, random);
        var context = _engine.BatchMatMul(weights, v);                                                      // [H, T, dk]
        var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { t, h * dk });
        return Output.Forward(merged);
    }

    // WeNet's rel_shift: pad a zero column on the left, view [H, P + 1, T], drop the first row, view back as [H, T, P]
    // and keep the first P / 2 + 1 columns.
    private Tensor<T> RelativeShift(Tensor<T> x, int h, int t, int p)
    {
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { h, t, 1 }), x }, 2);           // [H, T, P + 1]
        var viewed = _engine.Reshape(padded, new[] { h, p + 1, t });
        var dropped = _engine.TensorSlice(viewed, new[] { 0, 1, 0 }, new[] { h, p, t });
        var back = _engine.Reshape(dropped, new[] { h, t, p });
        return _engine.TensorSlice(back, new[] { 0, 0, 0 }, new[] { h, t, p / 2 + 1 });
    }

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random random) =>
        training && rate > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, rate, random) : x;
}

/// <summary>
/// A WeNet encoder over one unpadded sequence <c>[T, input]</c> → <c>[T, output]</c> (see
/// <see cref="WenetEncoderConfiguration"/>): the input layer, ×√d and ESPnet's relative positional encoding, the blocks
/// and a final LayerNorm (ε 1e-5). A positive <see cref="WenetEncoderConfiguration.StaticChunkSize"/> restricts each frame
/// to its own chunk and the chunks before it (chunk size 1 is causal attention).
/// </summary>
internal sealed class WenetEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public WenetEncoder(IEngine engine, List<LayerBase<T>> layers, WenetEncoderConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        InputProjection = WenetEncoderBlock<T>.Own(layers, new DenseLayer<T>(c.OutputSize));
        InputNorm = WenetEncoderBlock<T>.Own(layers, new LayerNormalizationLayer<T>(c.OutputSize, 1e-5));
        for (int i = 0; i < c.Blocks; i++) Blocks.Add(new WenetEncoderBlock<T>(engine, layers, c));
        FinalNorm = WenetEncoderBlock<T>.Own(layers, new LayerNormalizationLayer<T>(c.OutputSize, 1e-5));
    }

    public WenetEncoderConfiguration Configuration { get; }
    public DenseLayer<T> InputProjection { get; }
    public LayerNormalizationLayer<T> InputNorm { get; }
    public List<WenetEncoderBlock<T>> Blocks { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }

    public Tensor<T> Forward(Tensor<T> x, bool training, Random random)
    {
        var c = Configuration;
        int t = x.Shape[0];
        var h = InputNorm.Forward(InputProjection.Forward(x));
        h = Drop(h, c.Dropout, training, random);
        if (c.InputLayer == WenetInputLayer.LinearLegacy) h = _engine.ReLU(h);
        h = _engine.TensorMultiplyScalar(h, NumOps.FromDouble(Math.Sqrt(c.OutputSize)));
        h = Drop(h, c.PositionalDropout, training, random);
        var positions = Drop(RelativePositions(t, c.OutputSize), c.PositionalDropout, training, random);
        var mask = c.StaticChunkSize > 0 ? ChunkMaskBias(t, c.StaticChunkSize) : null;
        foreach (var block in Blocks) h = block.Forward(h, positions, mask, training, random);
        return FinalNorm.Forward(h);
    }

    /// <summary>
    /// ESPnet's relative positional encoding for a length-<paramref name="t"/> sequence: <c>[2t − 1, d]</c>, row k at
    /// relative position <c>t − 1 − k</c>, sines at even and cosines at odd columns.
    /// </summary>
    internal static Tensor<T> RelativePositions(int t, int d)
    {
        var table = new Tensor<T>(new[] { 2 * t - 1, d });
        for (int k = 0; k < 2 * t - 1; k++)
        {
            double position = t - 1 - k;
            for (int i = 0; i < d; i += 2)
            {
                double angle = position * Math.Exp(i * -(Math.Log(10000.0) / d));
                table[k, i] = NumOps.FromDouble(Math.Sin(angle));
                if (i + 1 < d) table[k, i + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        }
        return table;
    }

    // WeNet's subsequent_chunk_mask: frame i attends to j < (i / chunk + 1) · chunk.
    internal static Tensor<T> ChunkMaskBias(int t, int chunk)
    {
        var bias = new Tensor<T>(new[] { t, t });
        var blocked = NumOps.FromDouble(-1e30);
        for (int i = 0; i < t; i++)
        {
            int limit = (i / chunk + 1) * chunk;
            for (int j = limit; j < t; j++) bias[i, j] = blocked;
        }
        return bias;
    }

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random random) =>
        training && rate > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, rate, random) : x;

    /// <summary>
    /// PyTorch's initialization of these modules: <c>nn.Linear</c> weights and biases uniform on ±1/√fan_in, LayerNorm
    /// ones and zeros, and the attention's <c>pos_bias_u</c>/<c>pos_bias_v</c> Xavier-uniform.
    /// </summary>
    public void InitializeLikeTorch(Random random)
    {
        var c = Configuration;
        CosyVoiceInit.Linear(_engine, random, InputProjection, c.InputSize, c.OutputSize);
        foreach (var b in Blocks)
        {
            int d = c.OutputSize, dk = d / c.Heads;
            CosyVoiceInit.Linear(_engine, random, b.Query, d, d);
            if (b.Key is not null) CosyVoiceInit.Linear(_engine, random, b.Key, d, d);
            else CosyVoiceInit.Linear(random, b.KeyNoBias!, d, d);
            CosyVoiceInit.Linear(_engine, random, b.Value, d, d);
            CosyVoiceInit.Linear(_engine, random, b.Output, d, d);
            CosyVoiceInit.Linear(random, b.Position, d, d);
            double bound = Math.Sqrt(6.0 / (c.Heads + dk));
            b.BiasU.Reinitialize(() => (random.NextDouble() * 2 - 1) * bound);
            b.BiasV.Reinitialize(() => (random.NextDouble() * 2 - 1) * bound);
            CosyVoiceInit.Linear(_engine, random, b.FeedForwardIn, d, c.LinearUnits);
            CosyVoiceInit.Linear(_engine, random, b.FeedForwardOut, c.LinearUnits, d);
        }
    }

    /// <summary>Loads a WeNet encoder's state dict (PyTorch names under <paramref name="prefix"/>, row-major).</summary>
    public void LoadTorch(IReadOnlyDictionary<string, double[]> state, string prefix)
    {
        var c = Configuration;
        int d = c.OutputSize;
        TorchParameters.Linear(_engine, InputProjection, c.InputSize, d, state[prefix + "embed.out.0.weight"], state[prefix + "embed.out.0.bias"]);
        TorchParameters.LayerNorm(_engine, InputNorm, state[prefix + "embed.out.1.weight"], state[prefix + "embed.out.1.bias"]);
        // A TransformerEncoderLayer names its norms norm1/norm2; a ConformerEncoderLayer norm_mha/norm_ff.
        bool conformerNames = state.ContainsKey(prefix + "encoders.0.norm_mha.weight");
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}encoders.{i}.";
            TorchParameters.LayerNorm(_engine, b.AttentionNorm, state[p + (conformerNames ? "norm_mha" : "norm1") + ".weight"], state[p + (conformerNames ? "norm_mha" : "norm1") + ".bias"]);
            TorchParameters.Linear(_engine, b.Query, d, d, state[p + "self_attn.linear_q.weight"], state[p + "self_attn.linear_q.bias"]);
            if (b.Key is not null)
                TorchParameters.Linear(_engine, b.Key, d, d, state[p + "self_attn.linear_k.weight"], state[p + "self_attn.linear_k.bias"]);
            else
                TorchParameters.Linear(b.KeyNoBias!, d, d, state[p + "self_attn.linear_k.weight"]);
            TorchParameters.Linear(_engine, b.Value, d, d, state[p + "self_attn.linear_v.weight"], state[p + "self_attn.linear_v.bias"]);
            TorchParameters.Linear(_engine, b.Output, d, d, state[p + "self_attn.linear_out.weight"], state[p + "self_attn.linear_out.bias"]);
            TorchParameters.Linear(b.Position, d, d, state[p + "self_attn.linear_pos.weight"]);
            b.BiasU.LoadTable(state[p + "self_attn.pos_bias_u"]);
            b.BiasV.LoadTable(state[p + "self_attn.pos_bias_v"]);
            TorchParameters.LayerNorm(_engine, b.FeedForwardNorm, state[p + (conformerNames ? "norm_ff" : "norm2") + ".weight"], state[p + (conformerNames ? "norm_ff" : "norm2") + ".bias"]);
            TorchParameters.Linear(_engine, b.FeedForwardIn, d, c.LinearUnits, state[p + "feed_forward.w_1.weight"], state[p + "feed_forward.w_1.bias"]);
            TorchParameters.Linear(_engine, b.FeedForwardOut, c.LinearUnits, d, state[p + "feed_forward.w_2.weight"], state[p + "feed_forward.w_2.bias"]);
        }
        TorchParameters.LayerNorm(_engine, FinalNorm, state[prefix + "after_norm.weight"], state[prefix + "after_norm.bias"]);
    }
}

/// <summary>PyTorch's default initializations, drawn from a given generator.</summary>
internal static class CosyVoiceInit
{
    /// <summary><c>nn.Linear</c>: weight and bias uniform on ±1/√fan_in.</summary>
    public static void Linear<T>(IEngine engine, Random random, DenseLayer<T> layer, int inputs, int outputs)
    {
        double bound = 1.0 / Math.Sqrt(inputs);
        TorchParameters.Linear(engine, layer, inputs, outputs, Uniform(random, outputs * inputs, bound), Uniform(random, outputs, bound));
    }

    /// <summary><c>nn.Linear(bias=False)</c>: weight uniform on ±1/√fan_in.</summary>
    public static void Linear<T>(Random random, BiasFreeLinearLayer<T> layer, int inputs, int outputs)
        => TorchParameters.Linear(layer, inputs, outputs, Uniform(random, outputs * inputs, 1.0 / Math.Sqrt(inputs)));

    /// <summary><c>nn.Embedding</c>: N(0, 1).</summary>
    public static void Embedding<T>(Random random, TiedEmbeddingLayer<T> table)
    {
        table.Reinitialize(() =>
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        });
    }

    public static double[] Uniform(Random random, int count, double bound)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (random.NextDouble() * 2 - 1) * bound;
        return values;
    }
}
