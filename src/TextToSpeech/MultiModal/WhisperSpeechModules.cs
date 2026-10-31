using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Shared pieces of WhisperSpeech's networks (collabora/WhisperSpeech <c>whisperspeech/modules.py</c>).</summary>
internal static class WhisperSpeechOps
{
    /// <summary><c>sinusoids(length, channels)</c>: sines then cosines at timescales 10000^(−k/(channels/2 − 1)).</summary>
    public static double[,] Sinusoids(int length, int channels)
    {
        int half = channels / 2;
        double increment = Math.Log(10000) / (half - 1);
        var table = new double[length, channels];
        for (int t = 0; t < length; t++)
            for (int k = 0; k < half; k++)
            {
                double scaled = t * Math.Exp(-increment * k);
                table[t, k] = Math.Sin(scaled);
                table[t, half + k] = Math.Cos(scaled);
            }
        return table;
    }

    /// <summary>Truncated normal at ±3σ (torch's <c>trunc_normal_</c>), by rejection.</summary>
    public static double TruncatedNormal(Random random, double std)
    {
        while (true)
        {
            double z = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
            if (Math.Abs(z) <= 3) return z * std;
        }
    }
}

/// <summary>
/// WhisperSpeech's <c>MultiHeadAttention</c>: a biased query (<c>QueryHead</c>), a bias-free key, a biased value and
/// output; queries and keys each scaled by √qk_scale before the usual 1/√d; optionally rotary positions on queries and keys
/// (positions times each side's subsampling, <c>Rotary</c> base 10000); a key-value cache for generation.
/// </summary>
internal sealed class WhisperSpeechAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _width, _heads;
    private readonly double _sqrtScale;
    private readonly bool _rope;

    public WhisperSpeechAttention(IEngine engine, List<LayerBase<T>> layers, int width, int heads, double qkScale, bool rope)
    {
        _engine = engine;
        _width = width;
        _heads = heads;
        _sqrtScale = Math.Sqrt(qkScale);
        _rope = rope;
        DenseLayer<T> Dense() { var l = new DenseLayer<T>(width, (IActivationFunction<T>)new IdentityActivation<T>()); layers.Add(l); return l; }
        Query = Dense();
        Key = new BiasFreeLinearLayer<T>(width, width);
        layers.Add(Key);
        Value = Dense();
        Output = Dense();
    }

    public DenseLayer<T> Query { get; }
    public BiasFreeLinearLayer<T> Key { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> Output { get; }

    /// <summary>Key positions are multiplied by this before the rotary embedding (the S2A decoder's cross-attention: 3).</summary>
    public int KeySubsampling { get; set; } = 1;

    /// <summary>Cached keys and values <c>[heads, positions, head width]</c> (generation).</summary>
    public sealed class Cache
    {
        public Tensor<T>? Keys, Values;
    }

    private Tensor<T> Heads(Tensor<T> x, int[] positions, bool rope, int subsampling)
    {
        int length = x.Shape[0], dk = _width / _heads;
        var h = _engine.TensorPermute(_engine.Reshape(x, new[] { length, _heads, dk }), new[] { 1, 0, 2 });   // [H, L, dk]
        if (!rope) return h;
        // Rotary: inv_freq 10000^(−2i/dk), angles position · inv_freq duplicated for both halves; x·cos + rotate_half(x)·sin.
        int half = dk / 2;
        var cos = new Tensor<T>(new[] { 1, length, dk });
        var sin = new Tensor<T>(new[] { 1, length, dk });
        for (int t = 0; t < length; t++)
            for (int i = 0; i < half; i++)
            {
                double angle = positions[t] * subsampling * (1.0 / Math.Pow(10000, 2.0 * i / dk));
                cos[0, t, i] = cos[0, t, i + half] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, t, i] = sin[0, t, i + half] = NumOps.FromDouble(Math.Sin(angle));
            }
        var shape = h.Shape.ToArray();
        var first = _engine.TensorSlice(h, new[] { 0, 0, 0 }, new[] { _heads, length, half });
        var second = _engine.TensorSlice(h, new[] { 0, 0, half }, new[] { _heads, length, half });
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 2);
        return _engine.TensorAdd(_engine.TensorMultiply(h, _engine.TensorBroadcastTo(cos, shape)),
            _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(sin, shape)));
    }

    /// <summary>
    /// Attention of queries <c>[L, width]</c> at <paramref name="queryPositions"/> over <paramref name="keyValueInput"/>
    /// (null with a cache that already holds the cross-attention's keys) at <paramref name="keyPositions"/>, causal when
    /// <paramref name="causal"/> (query position ≥ key position). A self-attention cache appends this input's keys.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> queryInput, int[] queryPositions, Tensor<T>? keyValueInput, int[] keyPositions, bool causal, Cache? cache = null,
        bool appendToCache = true)
    {
        int length = queryInput.Shape[0], dk = _width / _heads;
        var scale = NumOps.FromDouble(_sqrtScale);
        var q = Heads(_engine.TensorMultiplyScalar(Query.Forward(queryInput), scale), queryPositions, _rope, 1);
        Tensor<T> k, v;
        if (keyValueInput is not null)
        {
            k = Heads(_engine.TensorMultiplyScalar(Key.Forward(keyValueInput), scale), keyPositions, _rope, KeySubsampling);
            v = Heads(Value.Forward(keyValueInput), keyPositions, false, 1);
            if (cache is not null)
            {
                if (appendToCache && cache.Keys is not null)
                {
                    k = _engine.TensorConcatenate(new[] { cache.Keys, k }, 1);
                    v = _engine.TensorConcatenate(new[] { cache.Values!, v }, 1);
                }
                cache.Keys = k;
                cache.Values = v;
            }
        }
        else
        {
            k = cache?.Keys ?? throw new InvalidOperationException("No keys: give the key input or a filled cache.");
            v = cache.Values!;
        }
        int keys = k.Shape[1];
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
        if (causal)
        {
            // Key j is visible to the query at position p when j ≤ p (the cache's keys are positions 0 … keys − 1).
            var mask = new Tensor<T>(new[] { 1, length, keys });
            var blocked = NumOps.FromDouble(double.NegativeInfinity);
            for (int i = 0; i < length; i++)
                for (int j = 0; j < keys; j++)
                    if (j > queryPositions[i]) mask[0, i, j] = blocked;
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(mask, new[] { _heads, length, keys }));
        }
        var weights = _engine.Softmax(scores, axis: 2);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, _width });
        return Output.Forward(context);
    }

    public void LoadTorchWeights(Func<string, double[]?> read, string prefix)
    {
        double[] Read(string name) => read(prefix + name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{prefix}{name}'.");
        TorchParameters.Linear(_engine, Query, _width, _width, Read("query.weight"), Read("query.bias"));
        TorchParameters.Linear(Key, _width, _width, Read("key.weight"));
        TorchParameters.Linear(_engine, Value, _width, _width, Read("value.weight"), Read("value.bias"));
        TorchParameters.Linear(_engine, Output, _width, _width, Read("out.weight"), Read("out.bias"));
    }
}

/// <summary>WhisperSpeech's <c>ResidualAttentionBlock</c>: pre-norm self-attention, optional pre-norm cross-attention,
/// pre-norm GELU feed-forward, each residual.</summary>
internal sealed class WhisperSpeechBlock<T>
{
    private readonly IEngine _engine;
    private readonly int _width, _hidden;

    public WhisperSpeechBlock(IEngine engine, List<LayerBase<T>> layers, int width, int heads, double qkScale, int ffnMult, bool cross, bool rope)
    {
        _engine = engine;
        _width = width;
        _hidden = width * ffnMult;
        LayerNormalizationLayer<T> Norm() { var l = new LayerNormalizationLayer<T>(width, 1e-5); layers.Add(l); return l; }
        DenseLayer<T> Dense(int outputs) { var l = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()); layers.Add(l); return l; }
        AttentionNorm = Norm();
        Attention = new WhisperSpeechAttention<T>(engine, layers, width, heads, qkScale, rope);
        if (cross)
        {
            CrossNorm = Norm();
            CrossAttention = new WhisperSpeechAttention<T>(engine, layers, width, heads, qkScale, rope);
        }
        MlpNorm = Norm();
        Mlp1 = Dense(_hidden);
        Mlp2 = Dense(width);
    }

    public LayerNormalizationLayer<T> AttentionNorm { get; }
    public WhisperSpeechAttention<T> Attention { get; }
    public LayerNormalizationLayer<T>? CrossNorm { get; }
    public WhisperSpeechAttention<T>? CrossAttention { get; }
    public LayerNormalizationLayer<T> MlpNorm { get; }
    public DenseLayer<T> Mlp1 { get; }
    public DenseLayer<T> Mlp2 { get; }

    public Tensor<T> Forward(Tensor<T> x, int[] positions, Tensor<T>? xa, int[]? xaPositions, bool causal,
        WhisperSpeechAttention<T>.Cache? selfCache = null, WhisperSpeechAttention<T>.Cache? crossCache = null)
    {
        var normed = AttentionNorm.Forward(x);
        x = _engine.TensorAdd(x, Attention.Forward(normed, positions, normed, positions, causal, selfCache));
        if (CrossAttention is not null)
        {
            var crossNormed = CrossNorm!.Forward(x);
            bool cached = crossCache?.Keys is not null;
            x = _engine.TensorAdd(x, CrossAttention.Forward(crossNormed, positions, cached ? null : xa, xaPositions ?? Array.Empty<int>(), false,
                crossCache, appendToCache: false));
        }
        var h = Mlp2.Forward(ActivationHelper.ExactGelu(Mlp1.Forward(MlpNorm.Forward(x)), _engine));
        return _engine.TensorAdd(x, h);
    }

    public void LoadTorchWeights(Func<string, double[]?> read, string prefix)
    {
        double[] Read(string name) => read(prefix + name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{prefix}{name}'.");
        TorchParameters.LayerNorm(_engine, AttentionNorm, Read("attn_ln.weight"), Read("attn_ln.bias"));
        Attention.LoadTorchWeights(read, prefix + "attn.");
        if (CrossAttention is not null)
        {
            TorchParameters.LayerNorm(_engine, CrossNorm!, Read("cross_attn_ln.weight"), Read("cross_attn_ln.bias"));
            CrossAttention.LoadTorchWeights(read, prefix + "cross_attn.");
        }
        TorchParameters.LayerNorm(_engine, MlpNorm, Read("mlp_ln.weight"), Read("mlp_ln.bias"));
        TorchParameters.Linear(_engine, Mlp1, _width, _hidden, Read("mlp.0.weight"), Read("mlp.0.bias"));
        TorchParameters.Linear(_engine, Mlp2, _hidden, _width, Read("mlp.2.weight"), Read("mlp.2.bias"));
    }
}

/// <summary>
/// WhisperSpeech's <c>FlexEmbeddings</c>: a table of <c>codes</c> rows in the frozen width (the semantic or acoustic
/// codebook's), projected to the model width (<c>emb_to_hidden</c>) when they differ; ids at or above <c>codes</c> read a
/// separate table of special rows in the model width. Unembedding projects back (<c>hidden_to_emb</c>) and scores against
/// the main rows, then appends the special rows' scores.
/// </summary>
internal sealed class WhisperSpeechEmbeddings<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public WhisperSpeechEmbeddings(IEngine engine, List<LayerBase<T>> layers, int codes, int width, int specialCodes, int frozenWidth, bool unembed,
        WhisperSpeechEmbeddings<T>? sharedSpecial = null)
    {
        _engine = engine;
        Codes = codes;
        Width = width;
        FrozenWidth = frozenWidth;
        SpecialCodes = specialCodes;
        Main = new BiasFreeLinearLayer<T>(1, codes * frozenWidth);
        layers.Add(Main);
        if (frozenWidth != width)
        {
            EmbedToHidden = new DenseLayer<T>(width, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(EmbedToHidden);
            if (unembed)
            {
                HiddenToEmbed = new DenseLayer<T>(frozenWidth, (IActivationFunction<T>)new IdentityActivation<T>());
                layers.Add(HiddenToEmbed);
            }
        }
        if (specialCodes > 0)
        {
            Special = sharedSpecial?.Special ?? new BiasFreeLinearLayer<T>(1, specialCodes * width);
            if (sharedSpecial?.Special is null) layers.Add(Special);
        }
    }

    public int Codes { get; }
    public int Width { get; }
    public int FrozenWidth { get; }
    public int SpecialCodes { get; }
    public BiasFreeLinearLayer<T> Main { get; }
    public DenseLayer<T>? EmbedToHidden { get; }
    public DenseLayer<T>? HiddenToEmbed { get; }
    public BiasFreeLinearLayer<T>? Special { get; }

    private Tensor<T> Rows(BiasFreeLinearLayer<T> table, int rows, int dim, IReadOnlyList<int> ids)
    {
        var index = new Tensor<int>(new[] { ids.Count }, new Vector<int>(ids.ToArray()));
        return _engine.TensorIndexSelect(_engine.Reshape(table.Weights, new[] { rows, dim }), index, 0);
    }

    /// <summary>Embeddings <c>[n, width]</c> of ids.</summary>
    public Tensor<T> Embed(IReadOnlyList<int> ids)
    {
        var main = Rows(Main, Codes, FrozenWidth, ids.Select(id => id >= Codes ? 0 : id).ToArray());
        if (EmbedToHidden is not null) main = EmbedToHidden.Forward(main);
        if (Special is null || ids.All(id => id < Codes)) return main;
        var special = Rows(Special, SpecialCodes, Width, ids.Select(id => id >= Codes ? id - Codes : 0).ToArray());
        var isSpecial = new Tensor<T>(new[] { ids.Count, 1 });
        var isMain = new Tensor<T>(new[] { ids.Count, 1 });
        for (int i = 0; i < ids.Count; i++)
        {
            isSpecial[i, 0] = ids[i] >= Codes ? NumOps.One : NumOps.Zero;
            isMain[i, 0] = ids[i] >= Codes ? NumOps.Zero : NumOps.One;
        }
        var shape = new[] { ids.Count, Width };
        return _engine.TensorAdd(_engine.TensorMultiply(main, _engine.TensorBroadcastTo(isMain, shape)),
            _engine.TensorMultiply(special, _engine.TensorBroadcastTo(isSpecial, shape)));
    }

    /// <summary>Logits <c>[n, codes + special]</c> of hidden states <c>[n, width]</c>.</summary>
    public Tensor<T> Unembed(Tensor<T> x)
    {
        var projected = HiddenToEmbed is not null ? HiddenToEmbed.Forward(x) : x;
        var main = _engine.TensorMatMul(projected, _engine.TensorTranspose(_engine.Reshape(Main.Weights, new[] { Codes, FrozenWidth })));
        if (Special is null) return main;
        var special = _engine.TensorMatMul(x, _engine.TensorTranspose(_engine.Reshape(Special.Weights, new[] { SpecialCodes, Width })));
        return _engine.TensorConcatenate(new[] { main, special }, 1);
    }

    public void LoadTorchWeights(Func<string, double[]?> read, string prefix, bool loadSpecial = true)
    {
        double[] Read(string name) => read(prefix + name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{prefix}{name}'.");
        TorchParameters.Linear(Main, 1, Codes * FrozenWidth, Read("main.weight"));
        if (EmbedToHidden is not null) TorchParameters.Linear(_engine, EmbedToHidden, FrozenWidth, Width, Read("emb_to_hidden.weight"), Read("emb_to_hidden.bias"));
        if (HiddenToEmbed is not null) TorchParameters.Linear(_engine, HiddenToEmbed, Width, FrozenWidth, Read("hidden_to_emb.weight"), Read("hidden_to_emb.bias"));
        if (Special is not null && loadSpecial) TorchParameters.Linear(Special, 1, SpecialCodes * Width, Read("special.weight"));
    }
}
