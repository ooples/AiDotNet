using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.NeuralNetworks;

/// <summary>The shape of an LFM2 decoder (Liquid AI); the defaults are LFM2-350M, which KaniTTS builds on.</summary>
internal sealed record Lfm2Configuration
{
    public int VocabSize { get; init; } = 80539;

    public int Hidden { get; init; } = 1024;

    /// <summary>Which layers are attention layers (<c>layer_types == "full_attention"</c>); the rest are short
    /// convolutions. LFM2-350M's 16 layers attend at 2, 5, 8, 10, 12 and 14.</summary>
    public bool[] AttentionLayers { get; init; } =
        { false, false, true, false, false, true, false, false, true, false, true, false, true, false, true, false };

    public int Heads { get; init; } = 16;

    /// <summary>Key and value heads; fewer than <see cref="Heads"/> is grouped-query attention.</summary>
    public int KeyValueHeads { get; init; } = 8;

    /// <summary>The feed-forward width as configured (<c>intermediate_size</c>, <c>block_ff_dim</c>).</summary>
    public int Intermediate { get; init; } = 6656;

    /// <summary><c>block_auto_adjust_ff_dim</c>: the width becomes 2/3 of <see cref="Intermediate"/>, times
    /// <see cref="FeedForwardMultiplier"/>, rounded up to a multiple of <see cref="FeedForwardMultipleOf"/>.</summary>
    public bool AutoAdjustFeedForward { get; init; } = true;

    /// <summary><c>block_ffn_dim_multiplier</c>; 0 for none (the width is then not rounded either).</summary>
    public double FeedForwardMultiplier { get; init; } = 1.0;

    /// <summary><c>block_multiple_of</c>.</summary>
    public int FeedForwardMultipleOf { get; init; } = 256;

    public double NormEpsilon { get; init; } = 1e-5;

    public double RopeTheta { get; init; } = 1000000;

    /// <summary>The padding token (<c>pad_token_id</c>), whose embedding starts at zero; -1 for none.</summary>
    public int PadTokenId { get; init; }

    /// <summary>The short convolution's kernel (<c>conv_L_cache</c>).</summary>
    public int ConvKernel { get; init; } = 3;

    /// <summary>Biases on the short convolution and its projections (<c>conv_bias</c>).</summary>
    public bool ConvBias { get; init; }

    /// <summary>Whether the language-model head is the token table (<c>tie_embedding</c>, true for LFM2).</summary>
    public bool TieWordEmbeddings { get; init; } = true;

    /// <summary>LoRA rank on <see cref="LoraTargets"/>; 0 for none.</summary>
    public int LoraRank { get; init; }

    /// <summary>LoRA alpha: the update is scaled by alpha / rank, or alpha / √rank with <see cref="RankStabilizedLora"/>.</summary>
    public double LoraAlpha { get; init; } = 16;

    /// <summary>Dropout on the LoRA branch's input.</summary>
    public double LoraDropout { get; init; } = 0.1;

    /// <summary>peft's <c>use_rslora</c>: scale by alpha / √rank (Kalajdzievski 2023).</summary>
    public bool RankStabilizedLora { get; init; } = true;

    /// <summary>The modules LoRA adapts, by peft's suffix names: <c>q_proj</c>, <c>k_proj</c>, <c>v_proj</c>,
    /// <c>out_proj</c> (both the attention's and the convolution's), <c>in_proj</c>, <c>w1</c>, <c>w2</c>, <c>w3</c>.</summary>
    public string[] LoraTargets { get; init; } = { "q_proj", "w1", "w2" };

    /// <summary>The width of one attention head.</summary>
    internal int HeadDim => Hidden / Heads;

    /// <summary>The feed-forward width the model builds.</summary>
    internal int ResolvedIntermediate
    {
        get
        {
            if (!AutoAdjustFeedForward) return Intermediate;
            int width = (int)(2 * Intermediate / 3.0);
            if (FeedForwardMultiplier == 0) return width;
            width = (int)(FeedForwardMultiplier * width);
            return FeedForwardMultipleOf * ((width + FeedForwardMultipleOf - 1) / FeedForwardMultipleOf);
        }
    }

    /// <summary>The scale of a LoRA update.</summary>
    internal double LoraScale => RankStabilizedLora ? LoraAlpha / Math.Sqrt(LoraRank) : LoraAlpha / LoraRank;
}

/// <summary>What an <see cref="Lfm2Network{T}"/> has read of a sequence, per layer: an attention layer's keys and values
/// (<c>[B, key-value heads, positions, head width]</c>, after the rotary embedding), or a convolution layer's last
/// <c>kernel − 1</c> gated inputs (<c>[B, kernel − 1, Hidden]</c>), for generating one token at a time.</summary>
internal sealed class Lfm2Cache<T>
{
    internal readonly List<(Tensor<T> Keys, Tensor<T> Values)?> Attention = new();
    internal readonly List<Tensor<T>?> Convolution = new();

    /// <summary>Positions read so far.</summary>
    public int Length { get; internal set; }
}

/// <summary>
/// LFM2 (Liquid AI 2025) as Hugging Face's <c>Lfm2ForCausalLM</c> computes it: token embeddings, pre-norm layers that
/// each mix tokens by either a gated short convolution (<c>Lfm2ShortConv</c>: in_proj to B, C and x; a causal depthwise
/// convolution of B·x; C times it; out_proj) or grouped-query attention with per-head query and key RMSNorm and rotary
/// embeddings, each followed by a SwiGLU feed-forward block; a final RMSNorm (<c>embedding_norm</c>) and a head tied to
/// the token table. Optional LoRA adapters follow peft (with rsLoRA's scaling).
/// </summary>
/// <remarks>Parameter names follow the Hugging Face checkpoints (<c>model.embed_tokens</c>, <c>model.layers.{i}.*</c>,
/// <c>model.embedding_norm</c>). A batch's sequences share one length (no padding mask).</remarks>
internal sealed class Lfm2Network<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    /// <summary>A projection and its LoRA branch: down projection A, up projection B, and dropout on its input.</summary>
    internal sealed record Projection(BiasFreeLinearLayer<T> Weight, BiasFreeLinearLayer<T>? Bias, BiasFreeLinearLayer<T>? LoraA,
        BiasFreeLinearLayer<T>? LoraB, DropoutLayer<T>? LoraDropout);

    /// <summary>One layer: its token mixer (attention or short convolution) and feed-forward block.</summary>
    internal sealed record Layer(bool IsAttention, BiasFreeLinearLayer<T> OperatorNorm, BiasFreeLinearLayer<T> FeedForwardNorm,
        Projection W1, Projection W2, Projection W3)
    {
        public Projection? Query { get; init; }
        public Projection? Key { get; init; }
        public Projection? Value { get; init; }
        public Projection? Output { get; init; }
        public BiasFreeLinearLayer<T>? QueryNorm { get; init; }
        public BiasFreeLinearLayer<T>? KeyNorm { get; init; }
        public Projection? ConvIn { get; init; }
        public Projection? ConvOut { get; init; }

        /// <summary>The depthwise kernel <c>[Hidden, kernel]</c> (<c>conv.conv.weight [Hidden, 1, kernel]</c>).</summary>
        public BiasFreeLinearLayer<T>? ConvKernel { get; init; }
        public BiasFreeLinearLayer<T>? ConvKernelBias { get; init; }
    }

    public Lfm2Network(IEngine engine, List<LayerBase<T>> layers, Lfm2Configuration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Hidden % c.Heads != 0 || c.Heads % c.KeyValueHeads != 0)
            throw new ArgumentException("The width must divide by the heads, and the heads by the key-value heads.", nameof(configuration));
        if (c.HeadDim % 2 != 0) throw new ArgumentException("Rotary embeddings need an even head width.", nameof(configuration));
        if (c.VocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(configuration), "The vocabulary must be positive.");
        if (c.ConvKernel < 1) throw new ArgumentOutOfRangeException(nameof(configuration), "The convolution kernel must be positive.");
        if (c.LoraRank < 0) throw new ArgumentOutOfRangeException(nameof(configuration), "The LoRA rank is 0 (none) or positive.");
        var known = new[] { "q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3" };
        foreach (var target in c.LoraTargets)
            if (!known.Contains(target)) throw new ArgumentException($"Unknown LoRA target '{target}'.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Linear(int inputs, int outputs) => Add(new BiasFreeLinearLayer<T>(inputs, outputs));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        Projection Project(string name, int inputs, int outputs, bool bias)
        {
            var weight = Linear(inputs, outputs);
            var b = bias ? Values(outputs) : null;
            if (c.LoraRank <= 0 || !c.LoraTargets.Contains(name)) return new Projection(weight, b, null, null, null);
            DropoutLayer<T>? dropout = null;
            if (c.LoraDropout > 0)
            {
                dropout = Add(new DropoutLayer<T>(c.LoraDropout));
                dropout.SetTrainingMode(false);
                _dropouts.Add(dropout);
            }
            return new Projection(weight, b, Linear(inputs, c.LoraRank), Linear(c.LoraRank, outputs), dropout);
        }
        int h = c.Hidden, hd = c.HeadDim, ff = c.ResolvedIntermediate;
        Embeddings = Values(c.VocabSize * h);
        foreach (bool attention in c.AttentionLayers)
        {
            var operatorNorm = Values(h);
            Layer layer;
            if (attention)
            {
                var query = Project("q_proj", h, c.Heads * hd, false);
                var key = Project("k_proj", h, c.KeyValueHeads * hd, false);
                var value = Project("v_proj", h, c.KeyValueHeads * hd, false);
                var output = Project("out_proj", c.Heads * hd, h, false);
                var queryNorm = Values(hd);
                var keyNorm = Values(hd);
                layer = new Layer(true, operatorNorm, Values(h), Project("w1", h, ff, false), Project("w2", ff, h, false), Project("w3", h, ff, false))
                {
                    Query = query, Key = key, Value = value, Output = output, QueryNorm = queryNorm, KeyNorm = keyNorm,
                };
            }
            else
            {
                var kernel = Values(h * c.ConvKernel);
                var kernelBias = c.ConvBias ? Values(h) : null;
                var convIn = Project("in_proj", h, 3 * h, c.ConvBias);
                var convOut = Project("out_proj", h, h, c.ConvBias);
                layer = new Layer(false, operatorNorm, Values(h), Project("w1", h, ff, false), Project("w2", ff, h, false), Project("w3", h, ff, false))
                {
                    ConvKernel = kernel, ConvKernelBias = kernelBias, ConvIn = convIn, ConvOut = convOut,
                };
            }
            Layers.Add(layer);
        }
        FinalNorm = Values(h);
        LmHead = c.TieWordEmbeddings ? null : Linear(h, c.VocabSize);
    }

    public Lfm2Configuration Configuration { get; }

    private readonly List<DropoutLayer<T>> _dropouts = new();

    /// <summary>Whether the LoRA dropout is active. A new network starts in inference mode.</summary>
    public bool Training
    {
        get => _training;
        set
        {
            _training = value;
            foreach (var dropout in _dropouts) dropout.SetTrainingMode(value);
        }
    }

    private bool _training;

    /// <summary>The token table <c>[VocabSize · Hidden]</c>.</summary>
    public BiasFreeLinearLayer<T> Embeddings { get; }

    public List<Layer> Layers { get; } = new();

    /// <summary>The final RMSNorm (<c>embedding_norm</c>).</summary>
    public BiasFreeLinearLayer<T> FinalNorm { get; }

    /// <summary>The head's own weights; null when it is the token table.</summary>
    public BiasFreeLinearLayer<T>? LmHead { get; }

    /// <summary>The LoRA adapters' layers, empty without LoRA.</summary>
    public IEnumerable<BiasFreeLinearLayer<T>> LoraLayers => Layers
        .SelectMany(l => new[] { l.Query, l.Key, l.Value, l.Output, l.ConvIn, l.ConvOut, l.W1, l.W2, l.W3 })
        .Where(p => p?.LoraA is not null).SelectMany(p => new[] { p!.LoraA!, p.LoraB! });

    // A value holder's parameter [1, N], read directly (the gradient tape records it as the layer's own forward does).
    private static Tensor<T> Value(BiasFreeLinearLayer<T> holder) => holder.Weights;

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

    // W x (+ b) (+ scale · B A dropout(x)), as peft's LoRA Linear computes it.
    private Tensor<T> Apply(Projection p, Tensor<T> x)
    {
        var y = Rows(p.Weight, x);
        if (p.Bias is not null) y = _engine.TensorAdd(y, Broadcast(Value(p.Bias), y.Shape.ToArray()));
        if (p.LoraA is null) return y;
        var input = p.LoraDropout is null ? x : p.LoraDropout.Forward(x);
        return _engine.TensorAdd(y, _engine.TensorMultiplyScalar(Rows(p.LoraB!, Rows(p.LoraA, input)), NumOps.FromDouble(Configuration.LoraScale)));
    }

    /// <summary>The token embeddings <c>[B, L, Hidden]</c> of ids <c>[B, L]</c>.</summary>
    public Tensor<T> Embed(int[,] ids)
    {
        var c = Configuration;
        int b = ids.GetLength(0), l = ids.GetLength(1);
        var index = new Tensor<int>(new[] { b * l });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                int id = ids[i, j];
                if (id < 0 || id >= c.VocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Id {id} is outside [0, {c.VocabSize}).");
                index[i * l + j] = id;
            }
        var table = _engine.Reshape(Value(Embeddings), new[] { c.VocabSize, c.Hidden });
        return _engine.Reshape(_engine.TensorIndexSelect(table, index, 0), new[] { b, l, c.Hidden });
    }

    /// <summary>RMSNorm: <c>w · x / sqrt(mean(x²) + ε)</c> over the last axis.</summary>
    private Tensor<T> RmsNorm(BiasFreeLinearLayer<T> weight, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int rank = shape.Length;
        var meanSquare = _engine.ReduceMean(_engine.TensorMultiply(x, x), new[] { rank - 1 }, true);
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(Configuration.NormEpsilon)), NumOps.FromDouble(-0.5));
        return _engine.TensorMultiply(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(inverse, shape)), Broadcast(Value(weight), shape));
    }

    /// <summary>The logits <c>[B, L, VocabSize]</c> of final hidden states <c>[B, L, Hidden]</c>.</summary>
    public Tensor<T> Head(Tensor<T> hidden)
    {
        var c = Configuration;
        if (LmHead is not null) return Rows(LmHead, hidden);
        var table = _engine.Reshape(Value(Embeddings), new[] { c.VocabSize, c.Hidden });
        var shape = hidden.Shape.ToArray();
        var logits = _engine.TensorMatMulTransposed(_engine.Reshape(hidden, new[] { hidden.Length / c.Hidden, c.Hidden }), table);
        shape[^1] = c.VocabSize;
        return _engine.Reshape(logits, shape);
    }

    /// <summary>
    /// The decoder on the next positions of a sequence whose earlier positions <paramref name="cache"/> holds (none, or
    /// an empty cache, for a new one): the final normalized hidden states <c>[B, L, Hidden]</c> and the logits — of every
    /// position, or with <paramref name="lastLogitsOnly"/> only the last (<c>[B, 1, VocabSize]</c>), as generation reads
    /// them. The cache takes what these positions add.
    /// </summary>
    public (Tensor<T> Hidden, Tensor<T> Logits) Forward(Tensor<T> embeddings, Lfm2Cache<T>? cache = null, bool lastLogitsOnly = false)
    {
        var c = Configuration;
        int b = embeddings.Shape[0], l = embeddings.Shape[1], hd = c.HeadDim, half = hd / 2;
        int past = cache?.Length ?? 0, total = past + l;

        // Rotary tables per position: cos and sin of position times the inverse frequencies, repeated for both halves.
        var cos = new Tensor<T>(new[] { 1, 1, l, hd });
        var sin = new Tensor<T>(new[] { 1, 1, l, hd });
        for (int j = 0; j < l; j++)
            for (int k = 0; k < half; k++)
            {
                double angle = (past + j) * (1.0 / Math.Pow(c.RopeTheta, 2.0 * k / hd));
                cos[0, 0, j, k] = cos[0, 0, j, k + half] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, 0, j, k] = sin[0, 0, j, k + half] = NumOps.FromDouble(Math.Sin(angle));
            }
        // The causal mask as torch.finfo(dtype).min added to the scores.
        var bias = new Tensor<T>(new[] { 1, 1, l, total });
        var blocked = NumOps.FromDouble(typeof(T) == typeof(float) ? float.MinValue : double.MinValue);
        for (int q = 0; q < l; q++)
            for (int k = past + q + 1; k < total; k++) bias[0, 0, q, k] = blocked;

        if (cache is not null)
        {
            while (cache.Attention.Count < Layers.Count) cache.Attention.Add(null);
            while (cache.Convolution.Count < Layers.Count) cache.Convolution.Add(null);
        }
        var x = embeddings;
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var normed = RmsNorm(layer.OperatorNorm, x);
            var mixed = layer.IsAttention ? Attend(layer, normed, cos, sin, bias, cache, i) : Convolve(layer, normed, cache, i);
            x = _engine.TensorAdd(mixed, x);
            var post = RmsNorm(layer.FeedForwardNorm, x);
            var gated = _engine.TensorMultiply(_engine.Swish(Apply(layer.W1, post)), Apply(layer.W3, post));
            x = _engine.TensorAdd(x, Apply(layer.W2, gated));
        }
        if (cache is not null) cache.Length = total;
        var hidden = RmsNorm(FinalNorm, x);
        var read = lastLogitsOnly ? _engine.TensorSlice(hidden, new[] { 0, l - 1, 0 }, new[] { b, 1, c.Hidden }) : hidden;
        return (hidden, Head(read));
    }

    private Tensor<T> Rotate(Tensor<T> x, Tensor<T> cos, Tensor<T> sin)
    {
        var shape = x.Shape.ToArray();
        int hd = shape[3], half = hd / 2;
        var first = _engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { shape[0], shape[1], shape[2], half });
        var second = _engine.TensorSlice(x, new[] { 0, 0, 0, half }, new[] { shape[0], shape[1], shape[2], half });
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 3);
        return _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(cos, shape)),
            _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(sin, shape)));
    }

    // Lfm2Attention: per-head RMSNorm of the queries and keys, the rotary embedding, grouped-query attention, out_proj.
    private Tensor<T> Attend(Layer layer, Tensor<T> x, Tensor<T> cos, Tensor<T> sin, Tensor<T> bias, Lfm2Cache<T>? cache, int index)
    {
        var c = Configuration;
        int b = x.Shape[0], l = x.Shape[1], hd = c.HeadDim, groups = c.Heads / c.KeyValueHeads;
        Tensor<T> Heads(Tensor<T> projected, int heads, BiasFreeLinearLayer<T>? norm)
        {
            var split = _engine.Reshape(projected, new[] { b, l, heads, hd });
            if (norm is not null) split = RmsNorm(norm, split);
            return _engine.TensorPermute(split, new[] { 0, 2, 1, 3 });
        }
        var q = Rotate(Heads(Apply(layer.Query!, x), c.Heads, layer.QueryNorm), cos, sin);
        var k = Rotate(Heads(Apply(layer.Key!, x), c.KeyValueHeads, layer.KeyNorm), cos, sin);
        var v = Heads(Apply(layer.Value!, x), c.KeyValueHeads, null);
        if (cache is not null)
        {
            if (cache.Attention[index] is { } held)
            {
                k = _engine.TensorConcatenate(new[] { held.Keys, k }, 2);
                v = _engine.TensorConcatenate(new[] { held.Values, v }, 2);
            }
            cache.Attention[index] = (k, v);
        }
        int keys = k.Shape[2];
        if (groups > 1)
        {
            // repeat_kv: each key-value head serves `groups` consecutive query heads.
            Tensor<T> Repeat(Tensor<T> t) => _engine.Reshape(_engine.TensorBroadcastTo(
                _engine.Reshape(t, new[] { b, c.KeyValueHeads, 1, keys, hd }), new[] { b, c.KeyValueHeads, groups, keys, hd }), new[] { b, c.Heads, keys, hd });
            k = Repeat(k);
            v = Repeat(v);
        }
        var scores = _engine.TensorMultiplyScalar(
            _engine.BatchMatMul(_engine.Reshape(q, new[] { b * c.Heads, l, hd }), _engine.TensorPermute(_engine.Reshape(k, new[] { b * c.Heads, keys, hd }), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        scores = _engine.TensorAdd(_engine.Reshape(scores, new[] { b, c.Heads, l, keys }), _engine.TensorBroadcastTo(bias, new[] { b, c.Heads, l, keys }));
        var probabilities = _engine.Reshape(_engine.Softmax(scores, axis: 3), new[] { b * c.Heads, l, keys });
        var context = _engine.Reshape(_engine.TensorPermute(
            _engine.Reshape(_engine.BatchMatMul(probabilities, _engine.Reshape(v, new[] { b * c.Heads, keys, hd })), new[] { b, c.Heads, l, hd }),
            new[] { 0, 2, 1, 3 }), new[] { b, l, c.Heads * hd });
        return Apply(layer.Output!, context);
    }

    // Lfm2ShortConv: in_proj to B, C and x; the causal depthwise convolution of B·x over the time axis (the cache holds
    // the previous kernel − 1 inputs, zeros at the start); C times it; out_proj.
    private Tensor<T> Convolve(Layer layer, Tensor<T> x, Lfm2Cache<T>? cache, int index)
    {
        var c = Configuration;
        int b = x.Shape[0], l = x.Shape[1], h = c.Hidden, kernel = c.ConvKernel;
        var projected = Apply(layer.ConvIn!, x);
        Tensor<T> Part(int i) => _engine.TensorSlice(projected, new[] { 0, 0, i * h }, new[] { b, l, h });
        var gate = Part(0);
        var output = Part(1);
        var gated = _engine.TensorMultiply(gate, Part(2));
        var previous = cache?.Convolution[index] ?? new Tensor<T>(new[] { b, kernel - 1, h });
        var padded = kernel > 1 ? _engine.TensorConcatenate(new[] { previous, gated }, 1) : gated;
        if (cache is not null && kernel > 1)
            cache.Convolution[index] = _engine.TensorSlice(padded, new[] { 0, l, 0 }, new[] { b, kernel - 1, h });
        var weights = _engine.Reshape(Value(layer.ConvKernel!), new[] { h, kernel });
        Tensor<T>? convolved = null;
        for (int k = 0; k < kernel; k++)
        {
            var tap = _engine.TensorSlice(weights, new[] { 0, k }, new[] { h, 1 });
            var term = _engine.TensorMultiply(_engine.TensorSlice(padded, new[] { 0, k, 0 }, new[] { b, l, h }), Broadcast(tap, new[] { b, l, h }));
            convolved = convolved is null ? term : _engine.TensorAdd(convolved, term);
        }
        if (layer.ConvKernelBias is not null) convolved = _engine.TensorAdd(convolved!, Broadcast(Value(layer.ConvKernelBias), new[] { b, l, h }));
        return Apply(layer.ConvOut!, _engine.TensorMultiply(output, convolved!));
    }

    /// <summary>The causal language-model loss (Hugging Face's <c>ForCausalLMLoss</c>): each position predicts the next
    /// label, cross-entropy averaged over the labels that are not <paramref name="ignoreIndex"/>.</summary>
    /// <param name="logits">Logits <c>[B, L, V]</c>.</param>
    /// <param name="labels">Labels <c>[B, L]</c>, aligned with the inputs (position t's logits score label t + 1).</param>
    public Tensor<T> CausalLmLoss(Tensor<T> logits, int[,] labels, int ignoreIndex = -100) =>
        CausalLanguageModelLoss.Compute(_engine, logits, labels, ignoreIndex);

    /// <summary>Loads a Hugging Face <c>Lfm2ForCausalLM</c> state dict (with peft's <c>lora_A</c> and <c>lora_B</c>
    /// names after each adapted module, without the adapter name).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int h = c.Hidden, hd = c.HeadDim, ff = c.ResolvedIntermediate;
        void Vector(BiasFreeLinearLayer<T> holder, string name, int[] shape)
            => TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, s) => a * s), read(name, shape));
        void Project(Projection p, string name, int inputs, int outputs)
        {
            TorchParameters.Linear(p.Weight, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }));
            if (p.Bias is not null) Vector(p.Bias, $"{name}.bias", new[] { outputs });
            if (p.LoraA is null) return;
            TorchParameters.Linear(p.LoraA, inputs, c.LoraRank, read($"{name}.lora_A.weight", new[] { c.LoraRank, inputs }));
            TorchParameters.Linear(p.LoraB!, c.LoraRank, outputs, read($"{name}.lora_B.weight", new[] { outputs, c.LoraRank }));
        }
        Vector(Embeddings, "model.embed_tokens.weight", new[] { c.VocabSize, h });
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string n = $"model.layers.{i}";
            Vector(layer.OperatorNorm, $"{n}.operator_norm.weight", new[] { h });
            Vector(layer.FeedForwardNorm, $"{n}.ffn_norm.weight", new[] { h });
            if (layer.IsAttention)
            {
                Project(layer.Query!, $"{n}.self_attn.q_proj", h, c.Heads * hd);
                Project(layer.Key!, $"{n}.self_attn.k_proj", h, c.KeyValueHeads * hd);
                Project(layer.Value!, $"{n}.self_attn.v_proj", h, c.KeyValueHeads * hd);
                Project(layer.Output!, $"{n}.self_attn.out_proj", c.Heads * hd, h);
                Vector(layer.QueryNorm!, $"{n}.self_attn.q_layernorm.weight", new[] { hd });
                Vector(layer.KeyNorm!, $"{n}.self_attn.k_layernorm.weight", new[] { hd });
            }
            else
            {
                Vector(layer.ConvKernel!, $"{n}.conv.conv.weight", new[] { h, 1, c.ConvKernel });
                if (layer.ConvKernelBias is not null) Vector(layer.ConvKernelBias, $"{n}.conv.conv.bias", new[] { h });
                Project(layer.ConvIn!, $"{n}.conv.in_proj", h, 3 * h);
                Project(layer.ConvOut!, $"{n}.conv.out_proj", h, h);
            }
            Project(layer.W1, $"{n}.feed_forward.w1", h, ff);
            Project(layer.W2, $"{n}.feed_forward.w2", ff, h);
            Project(layer.W3, $"{n}.feed_forward.w3", h, ff);
        }
        Vector(FinalNorm, "model.embedding_norm.weight", new[] { h });
        if (LmHead is not null) TorchParameters.Linear(LmHead, h, c.VocabSize, read("lm_head.weight", new[] { c.VocabSize, h }));
    }

    /// <summary>
    /// The reference initialization (Hugging Face's <c>PreTrainedModel._init_weights</c>, which LFM2 keeps): linear and
    /// convolution weights and embeddings N(0, 0.02²) with the padding token's row zero, biases zero, RMSNorm weights 1;
    /// and peft's for LoRA: A uniform in ±1/√inputs (Kaiming with a = √5), B zero, so a new adapter leaves the model
    /// unchanged.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            if (name.EndsWith("lora_B.weight", StringComparison.Ordinal) || name.EndsWith(".bias", StringComparison.Ordinal)) return values;
            if (name.EndsWith("lora_A.weight", StringComparison.Ordinal))
            {
                double bound = 1.0 / Math.Sqrt(shape[^1]);
                for (int i = 0; i < count; i++) values[i] = bound * (2 * random.NextDouble() - 1);
                return values;
            }
            bool norm = name.EndsWith("norm.weight", StringComparison.Ordinal) || name.EndsWith("layernorm.weight", StringComparison.Ordinal);
            for (int i = 0; i < count; i++) values[i] = norm ? 1.0 : 0.02 * Normal();
            if (name == "model.embed_tokens.weight" && Configuration.PadTokenId >= 0)
                Array.Clear(values, Configuration.PadTokenId * shape[1], shape[1]);
            return values;
        });
    }
}
