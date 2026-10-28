using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The shape of Dia's encoder-decoder (<c>dia/config.py</c>); the defaults are Dia 1.6B.</summary>
internal sealed record DiaConfiguration
{
    public int TextVocab { get; init; } = 256;
    public int EncoderHidden { get; init; } = 1024;
    public int EncoderLayers { get; init; } = 12;
    public int EncoderHeads { get; init; } = 16;
    public int EncoderKeyValueHeads { get; init; } = 16;
    public int EncoderHeadDim { get; init; } = 128;
    public int EncoderIntermediate { get; init; } = 4096;
    public int DecoderHidden { get; init; } = 2048;
    public int DecoderLayers { get; init; } = 18;
    public int DecoderHeads { get; init; } = 16;
    public int DecoderKeyValueHeads { get; init; } = 4;
    public int DecoderHeadDim { get; init; } = 128;
    public int CrossHeads { get; init; } = 16;
    public int CrossKeyValueHeads { get; init; } = 16;
    public int CrossHeadDim { get; init; } = 128;
    public int DecoderIntermediate { get; init; } = 8192;
    public int AudioVocab { get; init; } = 1028;
    public int Channels { get; init; } = 9;
    public double NormEpsilon { get; init; } = 1e-5;
    public double RopeTheta { get; init; } = 10000;
}

/// <summary>A decoding's state: each decoder layer's self-attention keys and values so far (after the rotary embedding)
/// and its cross-attention keys and values of the encoder's output.</summary>
internal sealed class DiaCache<T>
{
    internal readonly List<(Tensor<T> Keys, Tensor<T> Values)?> Self = new();
    internal readonly List<(Tensor<T> Keys, Tensor<T> Values)> Cross = new();

    /// <summary>Decoder positions read so far.</summary>
    public int Length { get; internal set; }
}

/// <summary>
/// Dia's encoder-decoder (nari-labs/dia <c>dia/layers.py</c>): a pre-norm Transformer encoder over byte embeddings, and a
/// pre-norm decoder over the sum of each channel's code embedding with causal grouped-query self-attention, cross-attention
/// to the encoder (no rotary embedding), and a head giving every channel's logits. Every block is RMSNorm, attention or a
/// SwiGLU feed-forward, bias-free; attention is unscaled (the release trains with scale 1); the rotary embedding rotates
/// each head's halves by position / θ^(2i/d).
/// </summary>
/// <remarks>Weights are held in transformers' layout (<c>DiaForConditionalGeneration</c>: each projection <c>[out, in]</c>,
/// the channels' tables stacked into one, the head <c>[channels · vocabulary, width]</c>); <see cref="Dia{T}"/> converts the
/// release's <c>DenseGeneral</c> kernels to it.</remarks>
internal sealed class DiaNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Attention(BiasFreeLinearLayer<T> Query, BiasFreeLinearLayer<T> Key, BiasFreeLinearLayer<T> Value, BiasFreeLinearLayer<T> Output,
        int Heads, int KeyValueHeads, int HeadDim);

    internal sealed record FeedForward(BiasFreeLinearLayer<T> Gate, BiasFreeLinearLayer<T> Up, BiasFreeLinearLayer<T> Down);

    internal sealed record EncoderLayer(BiasFreeLinearLayer<T> PreNorm, Attention SelfAttention, BiasFreeLinearLayer<T> PostNorm, FeedForward Mlp);

    internal sealed record DecoderLayer(BiasFreeLinearLayer<T> SelfNorm, Attention SelfAttention, BiasFreeLinearLayer<T> CrossNorm, Attention CrossAttention,
        BiasFreeLinearLayer<T> MlpNorm, FeedForward Mlp);

    public DiaNetwork(IEngine engine, List<LayerBase<T>> layers, DiaConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.EncoderHeads % c.EncoderKeyValueHeads != 0 || c.DecoderHeads % c.DecoderKeyValueHeads != 0 || c.CrossHeads % c.CrossKeyValueHeads != 0)
            throw new ArgumentException("Query heads must divide by key-value heads.", nameof(configuration));
        if (c.EncoderHeadDim % 2 != 0 || c.DecoderHeadDim % 2 != 0) throw new ArgumentException("Rotary embeddings need an even head width.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Linear(int inputs, int outputs) => Add(new BiasFreeLinearLayer<T>(inputs, outputs));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        Attention Attend(int queryWidth, int keyWidth, int heads, int kvHeads, int headDim, int outWidth) =>
            new(Linear(queryWidth, heads * headDim), Linear(keyWidth, kvHeads * headDim), Linear(keyWidth, kvHeads * headDim),
                Linear(heads * headDim, outWidth), heads, kvHeads, headDim);
        FeedForward Mlp(int width, int intermediate) => new(Linear(width, intermediate), Linear(width, intermediate), Linear(intermediate, width));

        TextEmbeddings = Values(c.TextVocab * c.EncoderHidden);
        for (int i = 0; i < c.EncoderLayers; i++)
            Encoder.Add(new EncoderLayer(Values(c.EncoderHidden),
                Attend(c.EncoderHidden, c.EncoderHidden, c.EncoderHeads, c.EncoderKeyValueHeads, c.EncoderHeadDim, c.EncoderHidden),
                Values(c.EncoderHidden), Mlp(c.EncoderHidden, c.EncoderIntermediate)));
        EncoderNorm = Values(c.EncoderHidden);
        AudioEmbeddings = Values(c.Channels * c.AudioVocab * c.DecoderHidden);
        for (int i = 0; i < c.DecoderLayers; i++)
            Decoder.Add(new DecoderLayer(Values(c.DecoderHidden),
                Attend(c.DecoderHidden, c.DecoderHidden, c.DecoderHeads, c.DecoderKeyValueHeads, c.DecoderHeadDim, c.DecoderHidden),
                Values(c.DecoderHidden),
                Attend(c.DecoderHidden, c.EncoderHidden, c.CrossHeads, c.CrossKeyValueHeads, c.CrossHeadDim, c.DecoderHidden),
                Values(c.DecoderHidden), Mlp(c.DecoderHidden, c.DecoderIntermediate)));
        DecoderNorm = Values(c.DecoderHidden);
        Head = Linear(c.DecoderHidden, c.Channels * c.AudioVocab);
    }

    public DiaConfiguration Configuration { get; }

    public BiasFreeLinearLayer<T> TextEmbeddings { get; }
    public List<EncoderLayer> Encoder { get; } = new();
    public BiasFreeLinearLayer<T> EncoderNorm { get; }
    public BiasFreeLinearLayer<T> AudioEmbeddings { get; }
    public List<DecoderLayer> Decoder { get; } = new();
    public BiasFreeLinearLayer<T> DecoderNorm { get; }
    public BiasFreeLinearLayer<T> Head { get; }

    private static Tensor<T> Value(BiasFreeLinearLayer<T> holder) => LlamaNetwork<T>.Paged(holder.Weights);

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    private Tensor<T> RmsNorm(BiasFreeLinearLayer<T> weight, Tensor<T> x)
    {
        var shape = x._shape;
        int rank = shape.Length;
        var meanSquare = _engine.ReduceMean(_engine.TensorMultiply(x, x), new[] { rank - 1 }, true);
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(Configuration.NormEpsilon)), NumOps.FromDouble(-0.5));
        var w = _engine.Reshape(Value(weight), Enumerable.Repeat(1, rank - 1).Concat(new[] { shape[rank - 1] }).ToArray());
        return _engine.TensorMultiply(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(inverse, shape)), _engine.TensorBroadcastTo(w, shape));
    }

    private Tensor<T> Lookup(BiasFreeLinearLayer<T> table, int width, int[] ids, int[] shape)
    {
        int count = table.Weights.Length / width;
        var index = new Tensor<int>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] >= count) throw new ArgumentOutOfRangeException(nameof(ids), ids[i], $"Ids lie in [0, {count}).");
            index[i] = ids[i];
        }
        return _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(Value(table), new[] { count, width }), index, 0), shape);
    }

    // The rotary tables [1, 1, L, d] of positions start..start+L-1: the angle position / θ^(2i/d) for both halves.
    private (Tensor<T> Cos, Tensor<T> Sin) Rotary(int start, int length, int headDim)
    {
        int half = headDim / 2;
        var cos = new Tensor<T>(new[] { 1, 1, length, headDim });
        var sin = new Tensor<T>(new[] { 1, 1, length, headDim });
        for (int j = 0; j < length; j++)
            for (int k = 0; k < half; k++)
            {
                double angle = (start + j) / Math.Pow(Configuration.RopeTheta, 2.0 * k / headDim);
                cos[0, 0, j, k] = cos[0, 0, j, k + half] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, 0, j, k] = sin[0, 0, j, k + half] = NumOps.FromDouble(Math.Sin(angle));
            }
        return (cos, sin);
    }

    private Tensor<T> Rotate(Tensor<T> x, (Tensor<T> Cos, Tensor<T> Sin) table)
    {
        var shape = x.Shape.ToArray();
        int half = shape[3] / 2;
        var first = _engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { shape[0], shape[1], shape[2], half });
        var second = _engine.TensorSlice(x, new[] { 0, 0, 0, half }, new[] { shape[0], shape[1], shape[2], half });
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 3);
        return _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(table.Cos, shape)),
            _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(table.Sin, shape)));
    }

    // [B, L, heads · d] as [B, heads, L, d].
    private Tensor<T> Split(Tensor<T> x, int heads, int headDim)
    {
        int b = x.Shape[0], l = x.Shape[1];
        return _engine.TensorPermute(_engine.Reshape(x, new[] { b, l, heads, headDim }), new[] { 0, 2, 1, 3 });
    }

    // Unscaled attention of queries [B, H, L, d] over keys and values [B, KV, S, d]; with causalPast set, query j sees keys
    // up to causalPast + j.
    private Tensor<T> Attend(Attention a, Tensor<T> q, Tensor<T> k, Tensor<T> v, int? causalPast)
    {
        int b = q.Shape[0], l = q.Shape[2], s = k.Shape[2], d = a.HeadDim, groups = a.Heads / a.KeyValueHeads;
        if (groups > 1)
        {
            Tensor<T> Repeat(Tensor<T> t) => _engine.Reshape(_engine.TensorBroadcastTo(
                _engine.Reshape(t, new[] { b, a.KeyValueHeads, 1, s, d }), new[] { b, a.KeyValueHeads, groups, s, d }), new[] { b, a.Heads, s, d });
            k = Repeat(k);
            v = Repeat(v);
        }
        var scores = _engine.BatchMatMul(_engine.Reshape(q, new[] { b * a.Heads, l, d }), _engine.TensorPermute(_engine.Reshape(k, new[] { b * a.Heads, s, d }), new[] { 0, 2, 1 }));
        if (causalPast is int past)
        {
            var bias = new Tensor<T>(new[] { 1, l, s });
            var blocked = NumOps.FromDouble(double.NegativeInfinity);
            for (int i = 0; i < l; i++)
                for (int j = past + i + 1; j < s; j++) bias[0, i, j] = blocked;
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(bias, new[] { b * a.Heads, l, s }));
        }
        var probabilities = _engine.Softmax(scores, axis: 2);
        var context = _engine.Reshape(_engine.BatchMatMul(probabilities, _engine.Reshape(v, new[] { b * a.Heads, s, d })), new[] { b, a.Heads, l, d });
        return Rows(a.Output, _engine.Reshape(_engine.TensorPermute(context, new[] { 0, 2, 1, 3 }), new[] { b, l, a.Heads * d }));
    }

    private Tensor<T> Mlp(FeedForward f, Tensor<T> x) =>
        Rows(f.Down, _engine.TensorMultiply(_engine.Swish(Rows(f.Gate, x)), Rows(f.Up, x)));

    /// <summary>The encoder's output <c>[B, S, width]</c> of byte ids <c>[B, S]</c> (every position real: the release's
    /// padding never reaches a real position).</summary>
    public Tensor<T> Encode(int[,] ids)
    {
        var c = Configuration;
        int b = ids.GetLength(0), s = ids.GetLength(1);
        var flat = new int[b * s];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < s; j++) flat[i * s + j] = ids[i, j];
        var x = Lookup(TextEmbeddings, c.EncoderHidden, flat, new[] { b, s, c.EncoderHidden });
        var rotary = Rotary(0, s, c.EncoderHeadDim);
        foreach (var layer in Encoder)
        {
            var a = layer.SelfAttention;
            var normed = RmsNorm(layer.PreNorm, x);
            var q = Rotate(Split(Rows(a.Query, normed), a.Heads, a.HeadDim), rotary);
            var k = Rotate(Split(Rows(a.Key, normed), a.KeyValueHeads, a.HeadDim), rotary);
            var v = Split(Rows(a.Value, normed), a.KeyValueHeads, a.HeadDim);
            x = _engine.TensorAdd(x, Attend(a, q, k, v, null));
            x = _engine.TensorAdd(x, Mlp(layer.Mlp, RmsNorm(layer.PostNorm, x)));
        }
        return RmsNorm(EncoderNorm, x);
    }

    /// <summary>A decoding of the encoder's output <c>[B, S, width]</c>: each layer's cross-attention keys and values.</summary>
    public DiaCache<T> StartDecoding(Tensor<T> encoded)
    {
        var cache = new DiaCache<T>();
        foreach (var layer in Decoder)
        {
            var a = layer.CrossAttention;
            cache.Cross.Add((Split(Rows(a.Key, encoded), a.KeyValueHeads, a.HeadDim), Split(Rows(a.Value, encoded), a.KeyValueHeads, a.HeadDim)));
            cache.Self.Add(null);
        }
        return cache;
    }

    /// <summary>
    /// The decoder's logits <c>[B, L, channels, vocabulary]</c> for the next <c>L</c> steps' tokens <c>[B, L, channels]</c>
    /// of <paramref name="cache"/>'s decoding, which takes their keys and values.
    /// </summary>
    public Tensor<T> Decode(int[,,] tokens, DiaCache<T> cache)
    {
        var c = Configuration;
        int b = tokens.GetLength(0), l = tokens.GetLength(1), channels = tokens.GetLength(2), past = cache.Length;
        if (channels != c.Channels) throw new ArgumentException($"Dia's decoder reads {c.Channels} channels.", nameof(tokens));
        var ids = new int[b * l * channels];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
                for (int k = 0; k < channels; k++)
                {
                    int token = tokens[i, j, k];
                    if (token < 0 || token >= c.AudioVocab) throw new ArgumentOutOfRangeException(nameof(tokens), token, $"Tokens lie in [0, {c.AudioVocab}).");
                    ids[(i * l + j) * channels + k] = k * c.AudioVocab + token;
                }
        // Each channel's embedding, summed.
        var x = _engine.ReduceSum(Lookup(AudioEmbeddings, c.DecoderHidden, ids, new[] { b, l, channels, c.DecoderHidden }), new[] { 2 }, keepDims: false);
        var rotary = Rotary(past, l, c.DecoderHeadDim);
        for (int i = 0; i < Decoder.Count; i++)
        {
            var layer = Decoder[i];
            var a = layer.SelfAttention;
            var normed = RmsNorm(layer.SelfNorm, x);
            var q = Rotate(Split(Rows(a.Query, normed), a.Heads, a.HeadDim), rotary);
            var k = Rotate(Split(Rows(a.Key, normed), a.KeyValueHeads, a.HeadDim), rotary);
            var v = Split(Rows(a.Value, normed), a.KeyValueHeads, a.HeadDim);
            if (cache.Self[i] is { } held)
            {
                k = _engine.TensorConcatenate(new[] { held.Keys, k }, 2);
                v = _engine.TensorConcatenate(new[] { held.Values, v }, 2);
            }
            cache.Self[i] = (k, v);
            x = _engine.TensorAdd(x, Attend(a, q, k, v, past));
            var cross = layer.CrossAttention;
            var crossQuery = Split(Rows(cross.Query, RmsNorm(layer.CrossNorm, x)), cross.Heads, cross.HeadDim);
            x = _engine.TensorAdd(x, Attend(cross, crossQuery, cache.Cross[i].Keys, cache.Cross[i].Values, null));
            x = _engine.TensorAdd(x, Mlp(layer.Mlp, RmsNorm(layer.MlpNorm, x)));
        }
        cache.Length = past + l;
        return _engine.Reshape(Rows(Head, RmsNorm(DecoderNorm, x)), new[] { b, l, channels, c.AudioVocab });
    }

    /// <summary>Loads a state dict in transformers' names (<c>DiaForConditionalGeneration</c>, after <paramref name="prefix"/>
    /// for the model and with the head at <c>logits_dense.weight</c>); returns the names read.</summary>
    public HashSet<string> LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "model.")
    {
        var c = Configuration;
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[] Read(string name, int[] shape)
        {
            used.Add(name);
            return read(name, shape);
        }
        void Linear(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(layer, inputs, outputs, Read(name, new[] { outputs, inputs }));
        void Vector(BiasFreeLinearLayer<T> holder, string name, int[] shape)
            => TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, s) => a * s), Read(name, shape));
        void Attention(Attention a, string name, int queryWidth, int keyWidth, int outWidth)
        {
            Linear(a.Query, $"{name}.q_proj.weight", queryWidth, a.Heads * a.HeadDim);
            Linear(a.Key, $"{name}.k_proj.weight", keyWidth, a.KeyValueHeads * a.HeadDim);
            Linear(a.Value, $"{name}.v_proj.weight", keyWidth, a.KeyValueHeads * a.HeadDim);
            Linear(a.Output, $"{name}.o_proj.weight", a.Heads * a.HeadDim, outWidth);
        }
        void Mlp(FeedForward f, string name, int width, int intermediate)
        {
            // gate_up_proj [2 · intermediate, width]: the gate's rows, then the up projection's.
            var fused = Read($"{name}.gate_up_proj.weight", new[] { 2 * intermediate, width });
            TorchParameters.Linear(f.Gate, width, intermediate, fused.Take(intermediate * width).ToArray());
            TorchParameters.Linear(f.Up, width, intermediate, fused.Skip(intermediate * width).ToArray());
            Linear(f.Down, $"{name}.down_proj.weight", intermediate, width);
        }
        Vector(TextEmbeddings, $"{prefix}encoder.embedding.weight", new[] { c.TextVocab, c.EncoderHidden });
        for (int i = 0; i < Encoder.Count; i++)
        {
            var layer = Encoder[i];
            string n = $"{prefix}encoder.layers.{i}";
            Vector(layer.PreNorm, $"{n}.pre_sa_norm.weight", new[] { c.EncoderHidden });
            Attention(layer.SelfAttention, $"{n}.self_attention", c.EncoderHidden, c.EncoderHidden, c.EncoderHidden);
            Vector(layer.PostNorm, $"{n}.post_sa_norm.weight", new[] { c.EncoderHidden });
            Mlp(layer.Mlp, $"{n}.mlp", c.EncoderHidden, c.EncoderIntermediate);
        }
        Vector(EncoderNorm, $"{prefix}encoder.norm.weight", new[] { c.EncoderHidden });
        Vector(AudioEmbeddings, $"{prefix}decoder.embeddings.embed.weight", new[] { c.Channels * c.AudioVocab, c.DecoderHidden });
        for (int i = 0; i < Decoder.Count; i++)
        {
            var layer = Decoder[i];
            string n = $"{prefix}decoder.layers.{i}";
            Vector(layer.SelfNorm, $"{n}.pre_sa_norm.weight", new[] { c.DecoderHidden });
            Attention(layer.SelfAttention, $"{n}.self_attention", c.DecoderHidden, c.DecoderHidden, c.DecoderHidden);
            Vector(layer.CrossNorm, $"{n}.pre_ca_norm.weight", new[] { c.DecoderHidden });
            Attention(layer.CrossAttention, $"{n}.cross_attention", c.DecoderHidden, c.EncoderHidden, c.DecoderHidden);
            Vector(layer.MlpNorm, $"{n}.pre_mlp_norm.weight", new[] { c.DecoderHidden });
            Mlp(layer.Mlp, $"{n}.mlp", c.DecoderHidden, c.DecoderIntermediate);
        }
        Vector(DecoderNorm, $"{prefix}decoder.norm.weight", new[] { c.DecoderHidden });
        Linear(Head, "logits_dense.weight", c.DecoderHidden, c.Channels * c.AudioVocab);
        return used;
    }

    /// <summary>transformers' initialization (<c>PreTrainedModel._init_weights</c> with initializer_range 0.02): linear
    /// weights and embeddings N(0, 0.02²), RMSNorm weights 1.</summary>
    public void InitializeLikeReference(Random random)
    {
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            if (name.EndsWith("norm.weight", StringComparison.Ordinal)) return Enumerable.Repeat(1.0, count).ToArray();
            return TorchParameters.Gaussian<double>(random, count, 0.02).ToArray();
        });
    }
}
