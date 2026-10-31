using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The shape of Zonos's backbone (<c>zonos/config.py</c>'s BackboneConfig, with mamba_ssm 2.2.4's Mamba2 defaults);
/// the defaults are Zonos-v0.1-transformer.</summary>
internal sealed record ZonosConfiguration
{
    public int Hidden { get; init; } = 2048;
    public int Layers { get; init; } = 26;

    /// <summary>The layers that attend; the others are Mamba2 layers. Null: every layer attends.</summary>
    public int[]? AttentionLayers { get; init; }

    public int Heads { get; init; } = 16;
    public int KeyValueHeads { get; init; } = 4;

    /// <summary>The attention layers' gated feed-forward width (<c>attn_mlp_d_intermediate</c>).</summary>
    public int AttentionIntermediate { get; init; } = 8192;

    /// <summary>The Mamba2 layers' gated feed-forward width (<c>d_intermediate</c>; 0: none).</summary>
    public int MambaIntermediate { get; init; }

    /// <summary>Dimensions of each head the rotary embedding turns (<c>rotary_emb_dim</c>).</summary>
    public int RotaryDim { get; init; } = 128;

    /// <summary>Whether the rotary embedding turns adjacent pairs (the transformer) or the two halves (the hybrid).</summary>
    public bool RotaryInterleaved { get; init; } = true;

    public double RotaryBase { get; init; } = 10000;
    public double NormEpsilon { get; init; } = 1e-5;

    public int StateDim { get; init; } = 128;
    public int ConvWidth { get; init; } = 4;
    public int Expand { get; init; } = 2;
    public int MambaHeadDim { get; init; } = 64;
    public int Groups { get; init; } = 1;
    public int ChunkSize { get; init; } = 256;
    public double MambaNormEpsilon { get; init; } = 1e-5;

    public int Codebooks { get; init; } = 9;

    /// <summary>Entries in each codebook's table (1026: 1024 codes, end, mask).</summary>
    public int CodeVocab { get; init; } = 1026;

    /// <summary>Logits of each head (1025: 1024 codes and end; the release pads to 1026 and masks the padding).</summary>
    public int HeadVocab { get; init; } = 1025;

    internal bool Attends(int layer) => AttentionLayers is null || AttentionLayers.Contains(layer);
    internal int HeadDim => Hidden / Heads;
    internal int Inner => Expand * Hidden;
    internal int MambaHeads => Inner / MambaHeadDim;
    internal int ConvChannels => Inner + 2 * Groups * StateDim;
}

/// <summary>A decoding's state: each attention layer's keys and values (after the rotary embedding), each Mamba2 layer's
/// last inputs to its convolution and its SSM state.</summary>
internal sealed class ZonosCache<T>
{
    internal readonly Dictionary<int, (Tensor<T> Keys, Tensor<T> Values)> Attention = new();
    internal readonly Dictionary<int, (Tensor<T> ConvTail, Tensor<T> State)> Mamba = new();

    /// <summary>Positions read so far.</summary>
    public int Length { get; internal set; }
}

/// <summary>
/// Zonos's backbone and code tables (Zyphra/Zonos <c>zonos/backbone</c>, <c>zonos/model.py</c>): pre-LayerNorm residual
/// layers, each an attention layer (grouped-query, rotary, causal, then a gated feed-forward) or a Mamba2 layer
/// (mamba_ssm's <c>Mamba2</c>: a projection to the gate, the convolved x, B, C and the step sizes; a causal depthwise
/// convolution; the selective state-space recurrence; a gated RMSNorm; an output projection), a final LayerNorm; the sum of
/// the nine codebooks' embeddings in, one head per codebook out.
/// </summary>
/// <remarks>The state-space recurrence is computed with the SSD algorithm (Dao and Gu 2024, Listing 1; mamba_ssm's
/// <c>ssd_minimal_discrete</c>): within each chunk as a masked product, across chunks by passing the state. It is the same
/// recurrence the release's kernels compute, and a decoding passes the final state on.</remarks>
internal sealed class ZonosNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Norm(BiasFreeLinearLayer<T> Weight, BiasFreeLinearLayer<T> Bias);

    internal sealed record GatedMlp(BiasFreeLinearLayer<T> Fc1, BiasFreeLinearLayer<T> Fc2, int Intermediate);

    internal sealed record Mamba(BiasFreeLinearLayer<T> InProj, BiasFreeLinearLayer<T> ConvWeight, BiasFreeLinearLayer<T> ConvBias,
        BiasFreeLinearLayer<T> DtBias, BiasFreeLinearLayer<T> ALog, BiasFreeLinearLayer<T> D, BiasFreeLinearLayer<T> NormWeight,
        BiasFreeLinearLayer<T> OutProj);

    internal sealed record Attention(BiasFreeLinearLayer<T> InProj, BiasFreeLinearLayer<T> OutProj);

    internal sealed record Layer(Norm Norm, Attention? SelfAttention, Mamba? Ssm, Norm? Norm2, GatedMlp? Mlp);

    public ZonosNetwork(IEngine engine, List<LayerBase<T>> layers, ZonosConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Hidden % c.Heads != 0 || c.Heads % c.KeyValueHeads != 0) throw new ArgumentException("The width must divide by the heads, and the heads by the key-value heads.", nameof(configuration));
        if (c.RotaryDim % 2 != 0 || c.RotaryDim > c.HeadDim) throw new ArgumentException("The rotary dimensions are even and fit in a head.", nameof(configuration));
        if (c.Inner % c.MambaHeadDim != 0 || c.MambaHeads % c.Groups != 0) throw new ArgumentException("Mamba2's inner width must divide into heads, and its heads into groups.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Linear(int inputs, int outputs) => Add(new BiasFreeLinearLayer<T>(inputs, outputs));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        Norm LayerNorm() => new(Values(c.Hidden), Values(c.Hidden));
        for (int i = 0; i < c.Layers; i++)
        {
            var norm = LayerNorm();
            if (c.Attends(i))
            {
                int qkv = (c.Heads + 2 * c.KeyValueHeads) * c.HeadDim;
                var attention = new Attention(Linear(c.Hidden, qkv), Linear(c.Heads * c.HeadDim, c.Hidden));
                Layers.Add(new Layer(norm, attention, null, LayerNorm(), new GatedMlp(Linear(c.Hidden, 2 * c.AttentionIntermediate),
                    Linear(c.AttentionIntermediate, c.Hidden), c.AttentionIntermediate)));
            }
            else
            {
                int inProj = 2 * c.Inner + 2 * c.Groups * c.StateDim + c.MambaHeads;
                var ssm = new Mamba(Linear(c.Hidden, inProj), Values(c.ConvChannels * c.ConvWidth), Values(c.ConvChannels), Values(c.MambaHeads),
                    Values(c.MambaHeads), Values(c.MambaHeads), Values(c.Inner), Linear(c.Inner, c.Hidden));
                GatedMlp? mlp = c.MambaIntermediate > 0
                    ? new GatedMlp(Linear(c.Hidden, 2 * c.MambaIntermediate), Linear(c.MambaIntermediate, c.Hidden), c.MambaIntermediate) : null;
                Layers.Add(new Layer(norm, null, ssm, mlp is null ? null : LayerNorm(), mlp));
            }
        }
        FinalNorm = LayerNorm();
        CodeTables = Values(c.Codebooks * c.CodeVocab * c.Hidden);
        Heads = Linear(c.Hidden, c.Codebooks * c.HeadVocab);
    }

    public ZonosConfiguration Configuration { get; }
    public List<Layer> Layers { get; } = new();
    public Norm FinalNorm { get; }

    /// <summary>The nine codebooks' tables, stacked <c>[codebooks · 1026, width]</c>.</summary>
    public BiasFreeLinearLayer<T> CodeTables { get; }

    /// <summary>The nine heads, stacked <c>[codebooks · 1025, width]</c>.</summary>
    public BiasFreeLinearLayer<T> Heads { get; }

    private static Tensor<T> Value(BiasFreeLinearLayer<T> holder) => LlamaNetwork<T>.Paged(holder.Weights);

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    private Tensor<T> Broadcast(Tensor<T> row, int[] shape) =>
        _engine.TensorBroadcastTo(_engine.Reshape(row, Enumerable.Repeat(1, shape.Length - 1).Concat(new[] { shape[^1] }).ToArray()), shape);

    private Tensor<T> Slice(Tensor<T> x, int axis, int start, int length)
    {
        var begin = new int[x.Rank];
        var size = x.Shape.ToArray();
        begin[axis] = start;
        size[axis] = length;
        return _engine.TensorSlice(x, begin, size);
    }

    /// <summary>nn.LayerNorm over the last axis: (x − mean) / √(var + ε) · w + b.</summary>
    internal Tensor<T> LayerNorm(Norm norm, Tensor<T> x, double epsilon)
    {
        var shape = x._shape;
        int rank = shape.Length;
        var centered = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(_engine.ReduceMean(x, new[] { rank - 1 }, true), shape));
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centered, centered), new[] { rank - 1 }, true);
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(variance, NumOps.FromDouble(epsilon)), NumOps.FromDouble(-0.5));
        var normed = _engine.TensorMultiply(centered, _engine.TensorBroadcastTo(inverse, shape));
        return _engine.TensorAdd(_engine.TensorMultiply(normed, Broadcast(Value(norm.Weight), shape)), Broadcast(Value(norm.Bias), shape));
    }

    /// <summary>The embeddings <c>[B, L, width]</c> of codes <c>[B, codebooks, L]</c>: each codebook's table row, summed.</summary>
    public Tensor<T> EmbedCodes(int[,,] codes)
    {
        var c = Configuration;
        int b = codes.GetLength(0), books = codes.GetLength(1), l = codes.GetLength(2);
        if (books != c.Codebooks) throw new ArgumentException($"Zonos reads {c.Codebooks} codebooks.", nameof(codes));
        var index = new Tensor<int>(new[] { b * l * books });
        for (int i = 0; i < b; i++)
            for (int t = 0; t < l; t++)
                for (int k = 0; k < books; k++)
                {
                    int code = codes[i, k, t];
                    if (code < 0 || code >= c.CodeVocab) throw new ArgumentOutOfRangeException(nameof(codes), code, $"Codes lie in [0, {c.CodeVocab}).");
                    index[(i * l + t) * books + k] = k * c.CodeVocab + code;
                }
        var table = _engine.Reshape(Value(CodeTables), new[] { c.Codebooks * c.CodeVocab, c.Hidden });
        var rows = _engine.Reshape(_engine.TensorIndexSelect(table, index, 0), new[] { b, l, books, c.Hidden });
        return _engine.ReduceSum(rows, new[] { 2 }, keepDims: false);
    }

    /// <summary>The heads' logits <c>[B, L, codebooks, 1025]</c> of hidden states <c>[B, L, width]</c>.</summary>
    public Tensor<T> ApplyHeads(Tensor<T> hidden)
    {
        var c = Configuration;
        int b = hidden.Shape[0], l = hidden.Shape[1];
        return _engine.Reshape(Rows(Heads, hidden), new[] { b, l, c.Codebooks, c.HeadVocab });
    }

    /// <summary>
    /// The backbone on inputs <c>[B, L, width]</c>, continuing <paramref name="cache"/>'s positions when given (and adding
    /// these to it): the final LayerNorm's output <c>[B, L, width]</c>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, ZonosCache<T>? cache)
    {
        var c = Configuration;
        int past = cache?.Length ?? 0;
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var normed = LayerNorm(layer.Norm, x, c.NormEpsilon);
            x = _engine.TensorAdd(x, layer.SelfAttention is not null ? Attend(layer.SelfAttention, normed, cache, i, past) : RunMamba(layer.Ssm!, normed, cache, i));
            if (layer.Mlp is not null) x = _engine.TensorAdd(x, Mlp(layer.Mlp, LayerNorm(layer.Norm2!, x, c.NormEpsilon)));
        }
        if (cache is not null) cache.Length = past + x.Shape[1];
        return LayerNorm(FinalNorm, x, c.NormEpsilon);
    }

    // GatedMLP: fc1 gives (y, gate); y · silu(gate) through fc2.
    private Tensor<T> Mlp(GatedMlp mlp, Tensor<T> x)
    {
        var both = Rows(mlp.Fc1, x);
        int last = both.Rank - 1;
        var y = Slice(both, last, 0, mlp.Intermediate);
        var gate = Slice(both, last, mlp.Intermediate, mlp.Intermediate);
        return Rows(mlp.Fc2, _engine.TensorMultiply(y, _engine.Swish(gate)));
    }

    // ------------------------------------------------------------ attention

    // Rotary tables [1, 1, L, rotary/2] of positions start..start+L-1: position · base^(−2i/d).
    private (Tensor<T> Cos, Tensor<T> Sin) Rotary(int start, int length)
    {
        var c = Configuration;
        int half = c.RotaryDim / 2;
        var cos = new Tensor<T>(new[] { 1, 1, length, half });
        var sin = new Tensor<T>(new[] { 1, 1, length, half });
        for (int j = 0; j < length; j++)
            for (int k = 0; k < half; k++)
            {
                double angle = (start + j) * (1.0 / Math.Pow(c.RotaryBase, 2.0 * k / c.RotaryDim));
                cos[0, 0, j, k] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, 0, j, k] = NumOps.FromDouble(Math.Sin(angle));
            }
        return (cos, sin);
    }

    // The rotary embedding on heads [B, H, L, d]: the first RotaryDim dimensions turned, as adjacent pairs (interleaved) or
    // as the two halves.
    private Tensor<T> Rotate(Tensor<T> x, (Tensor<T> Cos, Tensor<T> Sin) table)
    {
        var c = Configuration;
        var shape = x._shape;
        int b = shape[0], h = shape[1], l = shape[2], d = shape[3], half = c.RotaryDim / 2;
        var rotary = c.RotaryDim == d ? x : Slice(x, 3, 0, c.RotaryDim);
        var halfShape = new[] { b, h, l, half };
        var cos = _engine.TensorBroadcastTo(table.Cos, halfShape);
        var sin = _engine.TensorBroadcastTo(table.Sin, halfShape);
        Tensor<T> first, second;
        if (c.RotaryInterleaved)
        {
            var pairs = _engine.Reshape(rotary, new[] { b, h, l, half, 2 });
            first = _engine.Reshape(Slice(pairs, 4, 0, 1), halfShape);
            second = _engine.Reshape(Slice(pairs, 4, 1, 1), halfShape);
        }
        else
        {
            first = Slice(rotary, 3, 0, half);
            second = Slice(rotary, 3, half, half);
        }
        var outFirst = _engine.TensorSubtract(_engine.TensorMultiply(first, cos), _engine.TensorMultiply(second, sin));
        var outSecond = _engine.TensorAdd(_engine.TensorMultiply(second, cos), _engine.TensorMultiply(first, sin));
        Tensor<T> turned = c.RotaryInterleaved
            ? _engine.Reshape(_engine.TensorConcatenate(new[] { _engine.Reshape(outFirst, new[] { b, h, l, half, 1 }), _engine.Reshape(outSecond, new[] { b, h, l, half, 1 }) }, 4),
                new[] { b, h, l, c.RotaryDim })
            : _engine.TensorConcatenate(new[] { outFirst, outSecond }, 3);
        return c.RotaryDim == d ? turned : _engine.TensorConcatenate(new[] { turned, Slice(x, 3, c.RotaryDim, d - c.RotaryDim) }, 3);
    }

    private Tensor<T> Attend(Attention a, Tensor<T> x, ZonosCache<T>? cache, int index, int past)
    {
        var c = Configuration;
        int b = x.Shape[0], l = x.Shape[1], d = c.HeadDim, groups = c.Heads / c.KeyValueHeads;
        var qkv = Rows(a.InProj, x);
        Tensor<T> Heads(int start, int heads) =>
            _engine.TensorPermute(_engine.Reshape(Slice(qkv, 2, start, heads * d), new[] { b, l, heads, d }), new[] { 0, 2, 1, 3 });
        var rotary = Rotary(past, l);
        var q = Rotate(Heads(0, c.Heads), rotary);
        var k = Rotate(Heads(c.Heads * d, c.KeyValueHeads), rotary);
        var v = Heads((c.Heads + c.KeyValueHeads) * d, c.KeyValueHeads);
        if (cache is not null)
        {
            if (cache.Attention.TryGetValue(index, out var held))
            {
                k = _engine.TensorConcatenate(new[] { held.Keys, k }, 2);
                v = _engine.TensorConcatenate(new[] { held.Values, v }, 2);
            }
            cache.Attention[index] = (k, v);
        }
        int keys = k.Shape[2];
        if (groups > 1)
        {
            Tensor<T> Repeat(Tensor<T> t) => _engine.Reshape(_engine.TensorBroadcastTo(
                _engine.Reshape(t, new[] { b, c.KeyValueHeads, 1, keys, d }), new[] { b, c.KeyValueHeads, groups, keys, d }), new[] { b, c.Heads, keys, d });
            k = Repeat(k);
            v = Repeat(v);
        }
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(_engine.Reshape(q, new[] { b * c.Heads, l, d }),
            _engine.TensorPermute(_engine.Reshape(k, new[] { b * c.Heads, keys, d }), new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(d)));
        var bias = new Tensor<T>(new[] { 1, l, keys });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        int offset = keys - l;
        for (int i = 0; i < l; i++)
            for (int j = offset + i + 1; j < keys; j++) bias[0, i, j] = blocked;
        scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(bias, new[] { b * c.Heads, l, keys }));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), _engine.Reshape(v, new[] { b * c.Heads, keys, d }));
        context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { b, c.Heads, l, d }), new[] { 0, 2, 1, 3 }), new[] { b, l, c.Heads * d });
        return Rows(a.OutProj, context);
    }

    // ------------------------------------------------------------ Mamba2

    private Tensor<T> RunMamba(Mamba m, Tensor<T> u, ZonosCache<T>? cache, int index)
    {
        var c = Configuration;
        int b = u.Shape[0], l = u.Shape[1], w = c.ConvWidth, channels = c.ConvChannels, heads = c.MambaHeads, p = c.MambaHeadDim, n = c.StateDim;
        var projected = Rows(m.InProj, u);                                                         // [B, L, 2·inner + 2·G·N + H]
        var z = Slice(projected, 2, 0, c.Inner);
        var xbc = Slice(projected, 2, c.Inner, channels);
        var dtRaw = Slice(projected, 2, c.Inner + channels, heads);

        // The causal depthwise convolution over the last w − 1 inputs (zeros at the start) and these.
        Tensor<T>? convTail = null, state = null;
        if (cache is not null && cache.Mamba.TryGetValue(index, out var held)) (convTail, state) = held;
        convTail ??= new Tensor<T>(new[] { b, w - 1, channels });
        var full = _engine.TensorConcatenate(new[] { convTail, xbc }, 1);                           // [B, w − 1 + L, channels]
        var kernel = _engine.Reshape(Value(m.ConvWeight), new[] { channels, w });
        var convolved = Broadcast(Value(m.ConvBias), new[] { b, l, channels });
        for (int j = 0; j < w; j++)
        {
            var tap = _engine.Reshape(_engine.TensorSlice(kernel, new[] { 0, j }, new[] { channels, 1 }), new[] { channels });
            convolved = _engine.TensorAdd(convolved, _engine.TensorMultiply(Slice(full, 1, j, l), Broadcast(tap, new[] { b, l, channels })));
        }
        var newTail = Slice(full, 1, l, w - 1);
        convolved = _engine.Swish(convolved);
        var x = _engine.Reshape(Slice(convolved, 2, 0, c.Inner), new[] { b, l, heads, p });
        var bGroups = _engine.Reshape(Slice(convolved, 2, c.Inner, c.Groups * n), new[] { b, l, c.Groups, n });
        var cGroups = _engine.Reshape(Slice(convolved, 2, c.Inner + c.Groups * n, c.Groups * n), new[] { b, l, c.Groups, n });
        // Each group's B and C serve its heads.
        int perGroup = heads / c.Groups;
        Tensor<T> PerHead(Tensor<T> g) => _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(g, new[] { b, l, c.Groups, 1, n }),
            new[] { b, l, c.Groups, perGroup, n }), new[] { b, l, heads, n });
        var bHeads = PerHead(bGroups);
        var cHeads = PerHead(cGroups);

        var dt = _engine.Softplus(_engine.TensorAdd(dtRaw, Broadcast(Value(m.DtBias), new[] { b, l, heads })));      // [B, L, H]
        var a = _engine.TensorNegate(_engine.TensorExp(Value(m.ALog)));                                              // [H]
        var (y, finalState) = Scan(x, dt, a, bHeads, cHeads, state);                                                  // [B, L, H, P]
        var dHeads = _engine.Reshape(Value(m.D), new[] { 1, 1, heads, 1 });
        y = _engine.TensorAdd(y, _engine.TensorMultiply(x, _engine.TensorBroadcastTo(dHeads, new[] { b, l, heads, p })));
        if (cache is not null) cache.Mamba[index] = (newTail, finalState);

        // The gated RMSNorm (norm after the gate), per group of channels.
        var gated = _engine.TensorMultiply(_engine.Reshape(y, new[] { b, l, c.Inner }), _engine.Swish(z));
        int groupSize = c.Inner / c.Groups;
        var grouped = _engine.Reshape(gated, new[] { b, l, c.Groups, groupSize });
        var meanSquare = _engine.ReduceMean(_engine.TensorMultiply(grouped, grouped), new[] { 3 }, true);
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(c.MambaNormEpsilon)), NumOps.FromDouble(-0.5));
        var normed = _engine.Reshape(_engine.TensorMultiply(grouped, _engine.TensorBroadcastTo(inverse, grouped._shape)), new[] { b, l, c.Inner });
        normed = _engine.TensorMultiply(normed, Broadcast(Value(m.NormWeight), new[] { b, l, c.Inner }));
        return Rows(m.OutProj, normed);
    }

    /// <summary>
    /// The selective state-space recurrence h_t = exp(Δ_t A) h_{t−1} + Δ_t x_t ⊗ B_t, y_t = h_t C_t over inputs
    /// <c>[B, L, H, P]</c>, steps <c>[B, L, H]</c>, A <c>[H]</c>, B and C <c>[B, L, H, N]</c>, from <paramref name="initial"/>
    /// (<c>[B, H, P, N]</c>, zeros when null), by the SSD algorithm: returns y <c>[B, L, H, P]</c> and the final state.
    /// </summary>
    internal (Tensor<T> Y, Tensor<T> State) Scan(Tensor<T> x, Tensor<T> dt, Tensor<T> a, Tensor<T> bm, Tensor<T> cm, Tensor<T>? initial)
    {
        int b = x.Shape[0], l = x.Shape[1], h = x.Shape[2], p = x.Shape[3], n = bm.Shape[3];
        int q = Math.Min(Configuration.ChunkSize, l), chunks = (l + q - 1) / q, padded = chunks * q;
        // Padding steps have Δ = 0: they neither decay the state nor add to it.
        Tensor<T> Pad(Tensor<T> t)
        {
            if (padded == l) return t;
            var shape = t.Shape.ToArray();
            shape[1] = padded - l;
            return _engine.TensorConcatenate(new[] { t, new Tensor<T>(shape) }, 1);
        }
        var xd = Pad(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(_engine.Reshape(dt, new[] { b, l, h, 1 }), new[] { b, l, h, p })));
        var ad = Pad(_engine.TensorMultiply(dt, _engine.TensorBroadcastTo(_engine.Reshape(a, new[] { 1, 1, h }), new[] { b, l, h })));
        var bp = Pad(bm);
        var cp = Pad(cm);

        // Per head and chunk: [B, H, chunks, Q, …].
        var xc = _engine.TensorPermute(_engine.Reshape(xd, new[] { b, chunks, q, h, p }), new[] { 0, 3, 1, 2, 4 });   // [B, H, c, Q, P]
        var bc = _engine.TensorPermute(_engine.Reshape(bp, new[] { b, chunks, q, h, n }), new[] { 0, 3, 1, 2, 4 });   // [B, H, c, Q, N]
        var cc = _engine.TensorPermute(_engine.Reshape(cp, new[] { b, chunks, q, h, n }), new[] { 0, 3, 1, 2, 4 });   // [B, H, c, Q, N]
        var cumulative = _engine.TensorCumSum(_engine.TensorPermute(_engine.Reshape(ad, new[] { b, chunks, q, h }), new[] { 0, 3, 1, 2 }), 3); // [B, H, c, Q]

        int batches = b * h * chunks;
        // 1. Within each chunk: y_l = Σ_{s ≤ l} (C_l · B_s) exp(cum_l − cum_s) x_s.
        var difference = _engine.TensorSubtract(
            _engine.TensorBroadcastTo(_engine.Reshape(cumulative, new[] { b, h, chunks, q, 1 }), new[] { b, h, chunks, q, q }),
            _engine.TensorBroadcastTo(_engine.Reshape(cumulative, new[] { b, h, chunks, 1, q }), new[] { b, h, chunks, q, q }));
        var future = new Tensor<T>(new[] { 1, 1, 1, q, q });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < q; i++)
            for (int j = i + 1; j < q; j++) future[0, 0, 0, i, j] = blocked;
        var decay = _engine.TensorExp(_engine.TensorAdd(difference, _engine.TensorBroadcastTo(future, new[] { b, h, chunks, q, q })));
        var scores = _engine.BatchMatMul(_engine.Reshape(cc, new[] { batches, q, n }), _engine.TensorPermute(_engine.Reshape(bc, new[] { batches, q, n }), new[] { 0, 2, 1 }));
        var mixing = _engine.TensorMultiply(scores, _engine.Reshape(decay, new[] { batches, q, q }));
        var yDiagonal = _engine.BatchMatMul(mixing, _engine.Reshape(xc, new[] { batches, q, p }));                          // [B·H·c, Q, P]

        // 2. Each chunk's own contribution to the state at its end: Σ_l exp(cum_last − cum_l) x_l ⊗ B_l.
        var last = _engine.TensorBroadcastTo(Slice(cumulative, 3, q - 1, 1), new[] { b, h, chunks, q });
        var toEnd = _engine.TensorExp(_engine.TensorSubtract(last, cumulative));
        var weighted = _engine.TensorMultiply(xc, _engine.TensorBroadcastTo(_engine.Reshape(toEnd, new[] { b, h, chunks, q, 1 }), new[] { b, h, chunks, q, p }));
        var chunkStates = _engine.Reshape(_engine.BatchMatMul(_engine.TensorPermute(_engine.Reshape(weighted, new[] { batches, q, p }), new[] { 0, 2, 1 }),
            _engine.Reshape(bc, new[] { batches, q, n })), new[] { b, h, chunks, p, n });

        // 3–4. Across chunks: the state entering each chunk reaches its outputs as exp(cum_l) C_l · h, then decays by
        // exp(cum_last) and gains the chunk's contribution.
        var state = initial ?? new Tensor<T>(new[] { b, h, p, n });
        var offDiagonal = new List<Tensor<T>>(chunks);
        for (int k = 0; k < chunks; k++)
        {
            var ck = _engine.Reshape(Slice(cc, 2, k, 1), new[] { b * h, q, n });
            var reach = _engine.Reshape(_engine.BatchMatMul(ck, _engine.TensorPermute(_engine.Reshape(state, new[] { b * h, p, n }), new[] { 0, 2, 1 })), new[] { b, h, q, p });
            var entering = _engine.TensorExp(_engine.Reshape(Slice(cumulative, 2, k, 1), new[] { b, h, q, 1 }));
            offDiagonal.Add(_engine.Reshape(_engine.TensorMultiply(reach, _engine.TensorBroadcastTo(entering, new[] { b, h, q, p })), new[] { b, h, 1, q, p }));
            var through = _engine.TensorExp(_engine.Reshape(Slice(Slice(cumulative, 2, k, 1), 3, q - 1, 1), new[] { b, h, 1, 1 }));
            state = _engine.TensorAdd(_engine.TensorMultiply(state, _engine.TensorBroadcastTo(through, new[] { b, h, p, n })),
                _engine.Reshape(Slice(chunkStates, 2, k, 1), new[] { b, h, p, n }));
        }
        var yOff = chunks == 1 ? offDiagonal[0] : _engine.TensorConcatenate(offDiagonal.ToArray(), 2);                   // [B, H, c, Q, P]
        var y = _engine.TensorAdd(_engine.Reshape(yDiagonal, new[] { b, h, chunks, q, p }), yOff);
        y = _engine.Reshape(_engine.TensorPermute(y, new[] { 0, 2, 3, 1, 4 }), new[] { b, padded, h, p });
        return (padded == l ? y : Slice(y, 1, 0, l), state);
    }

    // ------------------------------------------------------------ weights

    /// <summary>Loads the release's backbone, tables and heads (names after the model: <c>backbone.layers.{i}.*</c>,
    /// <c>backbone.norm_f</c>, <c>embeddings.{k}</c>, <c>heads.{k}</c>); returns the names read.</summary>
    public HashSet<string> LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[] Read(string name, int[] shape)
        {
            used.Add(name);
            return read(name, shape);
        }
        void Linear(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(layer, inputs, outputs, Read(name, new[] { outputs, inputs }));
        void Vector(BiasFreeLinearLayer<T> holder, string name, int[] shape) =>
            TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, s) => a * s), Read(name, shape));
        void LoadNorm(Norm norm, string name)
        {
            Vector(norm.Weight, $"{name}.weight", new[] { c.Hidden });
            Vector(norm.Bias, $"{name}.bias", new[] { c.Hidden });
        }
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string n = $"backbone.layers.{i}";
            LoadNorm(layer.Norm, $"{n}.norm");
            if (layer.SelfAttention is { } a)
            {
                Linear(a.InProj, $"{n}.mixer.in_proj.weight", c.Hidden, (c.Heads + 2 * c.KeyValueHeads) * c.HeadDim);
                Linear(a.OutProj, $"{n}.mixer.out_proj.weight", c.Heads * c.HeadDim, c.Hidden);
            }
            else
            {
                var m = layer.Ssm!;
                Linear(m.InProj, $"{n}.mixer.in_proj.weight", c.Hidden, 2 * c.Inner + 2 * c.Groups * c.StateDim + c.MambaHeads);
                Vector(m.ConvWeight, $"{n}.mixer.conv1d.weight", new[] { c.ConvChannels, 1, c.ConvWidth });
                Vector(m.ConvBias, $"{n}.mixer.conv1d.bias", new[] { c.ConvChannels });
                Vector(m.DtBias, $"{n}.mixer.dt_bias", new[] { c.MambaHeads });
                Vector(m.ALog, $"{n}.mixer.A_log", new[] { c.MambaHeads });
                Vector(m.D, $"{n}.mixer.D", new[] { c.MambaHeads });
                Vector(m.NormWeight, $"{n}.mixer.norm.weight", new[] { c.Inner });
                Linear(m.OutProj, $"{n}.mixer.out_proj.weight", c.Inner, c.Hidden);
            }
            if (layer.Mlp is { } mlp)
            {
                LoadNorm(layer.Norm2!, $"{n}.norm2");
                Linear(mlp.Fc1, $"{n}.mlp.fc1.weight", c.Hidden, 2 * mlp.Intermediate);
                Linear(mlp.Fc2, $"{n}.mlp.fc2.weight", mlp.Intermediate, c.Hidden);
            }
        }
        LoadNorm(FinalNorm, "backbone.norm_f");
        TorchParameters.Linear(CodeTables, 1, c.Codebooks * c.CodeVocab * c.Hidden,
            Enumerable.Range(0, c.Codebooks).SelectMany(k => Read($"embeddings.{k}.weight", new[] { c.CodeVocab, c.Hidden })).ToArray());
        TorchParameters.Linear(Heads, c.Hidden, c.Codebooks * c.HeadVocab,
            Enumerable.Range(0, c.Codebooks).SelectMany(k => Read($"heads.{k}.weight", new[] { c.HeadVocab, c.Hidden })).ToArray());
        return used;
    }
}
