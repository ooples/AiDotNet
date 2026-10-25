using AiDotNet.Helpers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks;

/// <summary>The shape of a LLaMA decoder; the defaults are LLaMA 2 7B (Vicuna-7B v1.5).</summary>
internal sealed record LlamaConfiguration
{
    /// <summary>Token vocabulary; 0 builds no token table and no <c>lm_head</c>, a bare <c>LlamaModel</c> whose input
    /// embeddings come from elsewhere.</summary>
    public int VocabSize { get; init; } = 32000;

    public int Hidden { get; init; } = 4096;

    public int Layers { get; init; } = 32;

    public int Heads { get; init; } = 32;

    /// <summary>Key and value heads; fewer than <see cref="Heads"/> is grouped-query attention.</summary>
    public int KeyValueHeads { get; init; } = 32;

    public int Intermediate { get; init; } = 11008;

    public double RmsEpsilon { get; init; } = 1e-5;

    public double RopeTheta { get; init; } = 10000;

    /// <summary>Biases on the query, key and value projections (Qwen2's <c>q_proj</c>, <c>k_proj</c>, <c>v_proj</c>;
    /// LLaMA has none).</summary>
    public bool AttentionBias { get; init; }

    /// <summary>Whether a token table comes with a language-model head; false for a decoder whose output hidden
    /// states another head reads (CosyVoice 2's Qwen2 reads text through the table and speaks through its own head).</summary>
    public bool IncludeHead { get; init; } = true;

    /// <summary>LoRA rank on the query and value projections (Hu et al. 2022); 0 for none.</summary>
    public int LoraRank { get; init; }

    /// <summary>LoRA alpha; the update is scaled by alpha / rank.</summary>
    public double LoraAlpha { get; init; } = 16;

    /// <summary>Dropout on the LoRA branch's input.</summary>
    public double LoraDropout { get; init; } = 0.05;

    /// <summary>The width of one attention head; 0 for <see cref="Hidden"/> / <see cref="Heads"/> (Qwen3-0.6B sets 128
    /// with 16 heads over a width of 1024).</summary>
    public int HeadDim { get; init; }

    /// <summary>Whether the language-model head is the token table (<c>tie_word_embeddings</c>); the head then has no
    /// weights of its own.</summary>
    public bool TieWordEmbeddings { get; init; }

    /// <summary>Qwen3's RMSNorm on each query and key head (<c>q_norm</c>, <c>k_norm</c>) before the rotary embedding.</summary>
    public bool QueryKeyNorm { get; init; }

    /// <summary>Llama 3's rotary frequency scaling (<c>rope_type = "llama3"</c>): the factor, 0 for none.</summary>
    public double RopeScalingFactor { get; init; }

    /// <summary>Llama 3's low-frequency factor (1).</summary>
    public double RopeLowFrequencyFactor { get; init; } = 1;

    /// <summary>Llama 3's high-frequency factor (4).</summary>
    public double RopeHighFrequencyFactor { get; init; } = 4;

    /// <summary>Llama 3's original context length (8192).</summary>
    public int RopeOriginalMaxPositions { get; init; } = 8192;

    /// <summary>The resolved head width.</summary>
    internal int ResolvedHeadDim => HeadDim > 0 ? HeadDim : Hidden / Heads;
}

/// <summary>The keys and values of the positions a <see cref="LlamaNetwork{T}"/> has already read, per layer
/// (<c>[B, key-value heads, positions, head width]</c>, after the rotary embedding), for generating one token at a time.</summary>
internal sealed class LlamaCache<T>
{
    internal readonly List<(Tensor<T> Keys, Tensor<T> Values)?> Layers = new();

    /// <summary>Positions read so far.</summary>
    public int Length { get; internal set; }
}

/// <summary>
/// LLaMA (Touvron et al. 2023) as Hugging Face's <c>LlamaForCausalLM</c> computes it: token embeddings, pre-norm decoder
/// layers of RMSNorm, causal self-attention with rotary position embeddings (no biases) and a SwiGLU feed-forward block,
/// a final RMSNorm and an untied language-model head. With <see cref="LlamaConfiguration.AttentionBias"/> it is Qwen2
/// (<c>Qwen2ForCausalLM</c>), which adds biases to the query, key and value projections.
/// </summary>
/// <remarks>Parameter names follow the Hugging Face checkpoints (<c>model.embed_tokens</c>, <c>model.layers.{i}.*</c>,
/// <c>model.norm</c>, <c>lm_head</c>).</remarks>
internal sealed class LlamaNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Layer(BiasFreeLinearLayer<T> InputNorm, BiasFreeLinearLayer<T> Query, BiasFreeLinearLayer<T> Key,
        BiasFreeLinearLayer<T> Value, BiasFreeLinearLayer<T> Output, BiasFreeLinearLayer<T> PostNorm, BiasFreeLinearLayer<T> Gate,
        BiasFreeLinearLayer<T> Up, BiasFreeLinearLayer<T> Down, Lora? QueryLora, Lora? ValueLora)
    {
        /// <summary>The query, key and value biases, null without <see cref="LlamaConfiguration.AttentionBias"/>.</summary>
        public BiasFreeLinearLayer<T>? QueryBias { get; init; }
        public BiasFreeLinearLayer<T>? KeyBias { get; init; }
        public BiasFreeLinearLayer<T>? ValueBias { get; init; }

        /// <summary>Qwen3's per-head query and key RMSNorm weights, null without <see cref="LlamaConfiguration.QueryKeyNorm"/>.</summary>
        public BiasFreeLinearLayer<T>? QueryNorm { get; init; }
        public BiasFreeLinearLayer<T>? KeyNorm { get; init; }
    }

    /// <summary>A LoRA branch: down projection A, up projection B, and dropout on its input.</summary>
    internal sealed record Lora(BiasFreeLinearLayer<T> A, BiasFreeLinearLayer<T> B, DropoutLayer<T>? Dropout);

    public LlamaNetwork(IEngine engine, List<LayerBase<T>> layers, LlamaConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if ((c.HeadDim <= 0 && c.Hidden % c.Heads != 0) || c.Heads % c.KeyValueHeads != 0)
            throw new ArgumentException("The width must divide by the heads, and the heads by the key-value heads.", nameof(configuration));
        if (c.ResolvedHeadDim % 2 != 0) throw new ArgumentException("Rotary embeddings need an even head width.", nameof(configuration));
        if (c.TieWordEmbeddings && (c.VocabSize <= 0 || !c.IncludeHead))
            throw new ArgumentException("A tied head needs a token table and a head.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Linear(int inputs, int outputs) => Add(new BiasFreeLinearLayer<T>(inputs, outputs));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        int hd = c.ResolvedHeadDim, kv = c.KeyValueHeads * hd, qd = c.Heads * hd;
        Lora? Adapter(int outputs)
        {
            if (c.LoraRank <= 0) return null;
            DropoutLayer<T>? dropout = null;
            if (c.LoraDropout > 0)
            {
                dropout = Add(new DropoutLayer<T>(c.LoraDropout));
                dropout.SetTrainingMode(false);
                _dropouts.Add(dropout);
            }
            return new Lora(Linear(c.Hidden, c.LoraRank), Linear(c.LoraRank, outputs), dropout);
        }
        if (c.VocabSize < 0) throw new ArgumentOutOfRangeException(nameof(configuration), "The vocabulary is 0 (none) or positive.");
        Embeddings = c.VocabSize > 0 ? Values(c.VocabSize * c.Hidden) : null;
        for (int i = 0; i < c.Layers; i++)
        {
            var input = Values(c.Hidden);
            var query = Linear(c.Hidden, qd);
            var queryLora = Adapter(qd);
            var key = Linear(c.Hidden, kv);
            var value = Linear(c.Hidden, kv);
            var valueLora = Adapter(kv);
            var queryBias = c.AttentionBias ? Values(qd) : null;
            var keyBias = c.AttentionBias ? Values(kv) : null;
            var valueBias = c.AttentionBias ? Values(kv) : null;
            var output = Linear(qd, c.Hidden);
            var queryNorm = c.QueryKeyNorm ? Values(hd) : null;
            var keyNorm = c.QueryKeyNorm ? Values(hd) : null;
            Layers.Add(new Layer(input, query, key, value, output, Values(c.Hidden), Linear(c.Hidden, c.Intermediate),
                Linear(c.Hidden, c.Intermediate), Linear(c.Intermediate, c.Hidden), queryLora, valueLora)
            {
                QueryBias = queryBias, KeyBias = keyBias, ValueBias = valueBias, QueryNorm = queryNorm, KeyNorm = keyNorm,
            });
        }
        FinalNorm = Values(c.Hidden);
        LmHead = c.VocabSize > 0 && c.IncludeHead && !c.TieWordEmbeddings ? Linear(c.Hidden, c.VocabSize) : null;
    }

    public LlamaConfiguration Configuration { get; }

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

    /// <summary>The token table; null for a bare decoder (<see cref="LlamaConfiguration.VocabSize"/> 0).</summary>
    public BiasFreeLinearLayer<T>? Embeddings { get; }

    public List<Layer> Layers { get; } = new();

    public BiasFreeLinearLayer<T> FinalNorm { get; }

    /// <summary>The language-model head's own weights; null for a bare decoder or a head tied to the token table.</summary>
    public BiasFreeLinearLayer<T>? LmHead { get; }

    /// <summary>Whether the decoder produces logits (its own head, or one tied to the token table).</summary>
    public bool HasHead => LmHead is not null || Configuration.TieWordEmbeddings;

    // The logits of hidden states: the head, or the token table transposed.
    private Tensor<T>? Head(Tensor<T> hidden)
    {
        if (LmHead is not null) return Rows(LmHead, hidden);
        if (!Configuration.TieWordEmbeddings) return null;
        var c = Configuration;
        var table = _engine.Reshape(Value(Embeddings!), new[] { c.VocabSize, c.Hidden });
        var shape = hidden.Shape.ToArray();
        int rows = hidden.Length / c.Hidden;
        var logits = _engine.TensorMatMulTransposed(_engine.Reshape(hidden, new[] { rows, c.Hidden }), table);
        shape[^1] = c.VocabSize;
        return _engine.Reshape(logits, shape);
    }

    // A value holder's parameter [1, N], read directly (the gradient tape records it as the layer's own forward does)
    // rather than as a product with a one, which copies the whole token table on every call.
    private static Tensor<T> Value(BiasFreeLinearLayer<T> holder) => Paged(holder.Weights);

    // A streamed weight has its shape but no storage until paged in; compute ops page it in, view ops do not.
    internal static Tensor<T> Paged(Tensor<T> weight)
    {
        if (weight.Lifetime == WeightLifetime.Streaming) WeightRegistry.Materialize(weight);
        return weight;
    }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
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
        var embeddings = Embeddings ?? throw new InvalidOperationException("This decoder has no token table; pass input embeddings to Forward.");
        var table = _engine.Reshape(Value(embeddings), new[] { c.VocabSize, c.Hidden });
        return _engine.Reshape(_engine.TensorIndexSelect(table, index, 0), new[] { b, l, c.Hidden });
    }

    /// <summary>RMSNorm: <c>w · x / sqrt(mean(x²) + ε)</c> over the last axis.</summary>
    private Tensor<T> RmsNorm(BiasFreeLinearLayer<T> weight, Tensor<T> x)
    {
        var shape = x._shape;
        int rank = shape.Length;
        var meanSquare = _engine.ReduceMean(_engine.TensorMultiply(x, x), new[] { rank - 1 }, true);
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(Configuration.RmsEpsilon)), NumOps.FromDouble(-0.5));
        var w = _engine.Reshape(Value(weight), Enumerable.Repeat(1, rank - 1).Concat(new[] { shape[rank - 1] }).ToArray());
        return _engine.TensorMultiply(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(inverse, shape)), _engine.TensorBroadcastTo(w, shape));
    }

    /// <summary>
    /// The decoder on input embeddings <c>[B, L, Hidden]</c>: the final normalized hidden states <c>[B, L, Hidden]</c>
    /// (Hugging Face's last <c>hidden_states</c> entry) and the logits <c>[B, L, VocabSize]</c>, null for a bare decoder.
    /// </summary>
    /// <param name="embeddings">The input embeddings (token embeddings, or a multimodal sequence built from them).</param>
    /// <param name="attentionMask">False for padding keys, <c>[B, L]</c>; null attends to every earlier position.</param>
    /// <param name="positionIds">Rotary positions <c>[B, L]</c>; null for 0..L-1.</param>
    public (Tensor<T> Hidden, Tensor<T>? Logits) Forward(Tensor<T> embeddings, bool[,]? attentionMask = null, int[,]? positionIds = null)
        => Forward(embeddings, attentionMask, positionIds, cache: null, lastLogitsOnly: false);

    /// <summary>
    /// The decoder on the next positions of a sequence whose earlier positions <paramref name="cache"/> holds (empty for
    /// a new one): their hidden states and logits, as the full forward computes them for the whole sequence. The cache
    /// takes their keys and values. With <paramref name="lastLogitsOnly"/> only the last position's logits are computed
    /// (<c>[B, 1, VocabSize]</c>), as generation reads them.
    /// </summary>
    /// <param name="attentionMask">False for padding keys over every position read so far and these, <c>[B, past + L]</c>.</param>
    public (Tensor<T> Hidden, Tensor<T>? Logits) Forward(Tensor<T> embeddings, bool[,]? attentionMask, int[,]? positionIds, LlamaCache<T>? cache,
        bool lastLogitsOnly)
    {
        var c = Configuration;
        int b = embeddings.Shape[0], l = embeddings.Shape[1], hd = c.ResolvedHeadDim, half = hd / 2;
        int past = cache?.Length ?? 0, total = past + l;
        if (positionIds is null)
        {
            positionIds = new int[b, l];
            for (int i = 0; i < b; i++) for (int j = 0; j < l; j++) positionIds[i, j] = past + j;
        }

        // Rotary tables per (batch, position): cos and sin of position times the inverse frequencies, repeated for both halves.
        var frequencies = InverseFrequencies();
        var cos = new Tensor<T>(new[] { b, 1, l, hd });
        var sin = new Tensor<T>(new[] { b, 1, l, hd });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
                for (int k = 0; k < half; k++)
                {
                    double angle = positionIds[i, j] * frequencies[k];
                    cos[i, 0, j, k] = cos[i, 0, j, k + half] = NumOps.FromDouble(Math.Cos(angle));
                    sin[i, 0, j, k] = sin[i, 0, j, k + half] = NumOps.FromDouble(Math.Sin(angle));
                }

        // Causal mask plus key padding, as torch.finfo(dtype).min added to the scores.
        var bias = new Tensor<T>(new[] { b, 1, l, total });
        var blocked = NumOps.FromDouble(typeof(T) == typeof(float) ? float.MinValue : double.MinValue);
        for (int i = 0; i < b; i++)
            for (int q = 0; q < l; q++)
                for (int k = 0; k < total; k++)
                    if (k > past + q || (attentionMask is not null && !attentionMask[i, k])) bias[i, 0, q, k] = blocked;

        if (cache is not null)
            while (cache.Layers.Count < Layers.Count) cache.Layers.Add(null);
        var x = embeddings;
        for (int i = 0; i < Layers.Count; i++) x = Run(Layers[i], x, cos, sin, bias, cache, i);
        if (cache is not null) cache.Length = total;
        var hidden = RmsNorm(FinalNorm, x);
        var read = lastLogitsOnly ? _engine.TensorSlice(hidden, new[] { 0, l - 1, 0 }, new[] { b, 1, c.Hidden }) : hidden;
        return (hidden, Head(read));
    }

    /// <summary>The rotary inverse frequencies, with Llama 3's scaling when it is set (Hugging Face's
    /// <c>_compute_llama3_parameters</c>: long wavelengths divided by the factor, short ones kept, the band between
    /// interpolated).</summary>
    internal double[] InverseFrequencies()
    {
        var c = Configuration;
        int hd = c.ResolvedHeadDim, half = hd / 2;
        var frequencies = new double[half];
        for (int k = 0; k < half; k++) frequencies[k] = 1.0 / Math.Pow(c.RopeTheta, 2.0 * k / hd);
        if (c.RopeScalingFactor <= 0) return frequencies;
        double lowWavelength = c.RopeOriginalMaxPositions / c.RopeLowFrequencyFactor;
        double highWavelength = c.RopeOriginalMaxPositions / c.RopeHighFrequencyFactor;
        for (int k = 0; k < half; k++)
        {
            double f = frequencies[k], wavelength = 2 * Math.PI / f;
            double scaled = wavelength > lowWavelength ? f / c.RopeScalingFactor : f;
            if (!(wavelength < highWavelength) && !(wavelength > lowWavelength))
            {
                double smooth = (c.RopeOriginalMaxPositions / wavelength - c.RopeLowFrequencyFactor) / (c.RopeHighFrequencyFactor - c.RopeLowFrequencyFactor);
                scaled = (1 - smooth) * scaled / c.RopeScalingFactor + smooth * scaled;
            }
            frequencies[k] = scaled;
        }
        return frequencies;
    }

    /// <summary><see cref="Forward"/> for a decoder with a head: the hidden states and the logits, never null.</summary>
    public (Tensor<T> Hidden, Tensor<T> Logits) ForwardWithLogits(Tensor<T> embeddings, bool[,]? attentionMask = null, int[,]? positionIds = null)
    {
        if (!HasHead) throw new InvalidOperationException("This decoder has no language-model head.");
        var (hidden, logits) = Forward(embeddings, attentionMask, positionIds);
        return (hidden, logits ?? throw new InvalidOperationException("The head produced no logits."));
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

    private Tensor<T> Run(Layer layer, Tensor<T> x, Tensor<T> cos, Tensor<T> sin, Tensor<T> bias, LlamaCache<T>? cache, int index)
    {
        var c = Configuration;
        int b = x.Shape[0], l = x.Shape[1], hd = c.ResolvedHeadDim, groups = c.Heads / c.KeyValueHeads;
        var normed = RmsNorm(layer.InputNorm, x);
        Tensor<T> Project(BiasFreeLinearLayer<T> projection, Lora? lora, BiasFreeLinearLayer<T>? projectionBias)
        {
            var y = Rows(projection, normed);
            if (projectionBias is not null)
            {
                var shape = y.Shape.ToArray();
                var row = _engine.Reshape(Value(projectionBias), Enumerable.Repeat(1, shape.Length - 1).Concat(new[] { shape[^1] }).ToArray());
                y = _engine.TensorAdd(y, _engine.TensorBroadcastTo(row, shape));
            }
            if (lora is null) return y;
            // peft's LoRA: W x + (alpha / r) B A dropout(x).
            var x = lora.Dropout is null ? normed : lora.Dropout.Forward(normed);
            return _engine.TensorAdd(y, _engine.TensorMultiplyScalar(Rows(lora.B, Rows(lora.A, x)), NumOps.FromDouble(c.LoraAlpha / c.LoraRank)));
        }
        // Heads [B, heads, L, hd]; Qwen3 normalizes each query and key head (q_norm, k_norm) before the rotary embedding.
        Tensor<T> Heads(Tensor<T> projected, int heads, BiasFreeLinearLayer<T>? norm)
        {
            var split = _engine.Reshape(projected, new[] { b, l, heads, hd });
            if (norm is not null) split = RmsNorm(norm, split);
            return _engine.TensorPermute(split, new[] { 0, 2, 1, 3 });
        }
        var q = Rotate(Heads(Project(layer.Query, layer.QueryLora, layer.QueryBias), c.Heads, layer.QueryNorm), cos, sin);
        var k = Rotate(Heads(Project(layer.Key, null, layer.KeyBias), c.KeyValueHeads, layer.KeyNorm), cos, sin);
        var v = Heads(Project(layer.Value, layer.ValueLora, layer.ValueBias), c.KeyValueHeads, null);
        if (cache is not null)
        {
            if (cache.Layers[index] is { } held)
            {
                k = _engine.TensorConcatenate(new[] { held.Keys, k }, 2);
                v = _engine.TensorConcatenate(new[] { held.Values, v }, 2);
            }
            cache.Layers[index] = (k, v);
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
        x = _engine.TensorAdd(x, Rows(layer.Output, context));
        var post = RmsNorm(layer.PostNorm, x);
        var gated = _engine.TensorMultiply(_engine.Swish(Rows(layer.Gate, post)), Rows(layer.Up, post));
        return _engine.TensorAdd(x, Rows(layer.Down, gated));
    }

    /// <summary>
    /// The causal language-model loss (Hugging Face's <c>ForCausalLMLoss</c>): each position predicts the next label,
    /// cross-entropy averaged over the labels that are not <paramref name="ignoreIndex"/>.
    /// </summary>
    /// <param name="logits">Logits <c>[B, L, V]</c>.</param>
    /// <param name="labels">Labels <c>[B, L]</c>, aligned with the inputs (position t's logits score label t + 1).</param>
    /// <param name="ignoreIndex">The label that is not scored (-100).</param>
    public Tensor<T> CausalLmLoss(Tensor<T> logits, int[,] labels, int ignoreIndex = -100) =>
        CausalLanguageModelLoss.Compute(_engine, logits, labels, ignoreIndex);

    /// <summary>Loads a Hugging Face <c>LlamaForCausalLM</c> state dict (names after an optional <paramref name="prefix"/>
    /// for the decoder; the head is read as <paramref name="headName"/>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "model.", string headName = "lm_head.weight")
    {
        var c = Configuration;
        int h = c.Hidden, hd = c.ResolvedHeadDim, kv = c.KeyValueHeads * hd, qd = c.Heads * hd;
        void Linear(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(layer, inputs, outputs, read(name, new[] { outputs, inputs }));
        void Vector(BiasFreeLinearLayer<T> holder, string name, int[] shape)
            => TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, s) => a * s), read(name, shape));
        if (Embeddings is not null) Vector(Embeddings, $"{prefix}embed_tokens.weight", new[] { c.VocabSize, h });
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string n = $"{prefix}layers.{i}";
            Vector(layer.InputNorm, $"{n}.input_layernorm.weight", new[] { h });
            Linear(layer.Query, $"{n}.self_attn.q_proj.weight", h, qd);
            Linear(layer.Key, $"{n}.self_attn.k_proj.weight", h, kv);
            Linear(layer.Value, $"{n}.self_attn.v_proj.weight", h, kv);
            Linear(layer.Output, $"{n}.self_attn.o_proj.weight", qd, h);
            if (layer.QueryBias is not null) Vector(layer.QueryBias, $"{n}.self_attn.q_proj.bias", new[] { qd });
            if (layer.QueryNorm is not null) Vector(layer.QueryNorm, $"{n}.self_attn.q_norm.weight", new[] { hd });
            if (layer.KeyNorm is not null) Vector(layer.KeyNorm, $"{n}.self_attn.k_norm.weight", new[] { hd });
            if (layer.KeyBias is not null) Vector(layer.KeyBias, $"{n}.self_attn.k_proj.bias", new[] { kv });
            if (layer.ValueBias is not null) Vector(layer.ValueBias, $"{n}.self_attn.v_proj.bias", new[] { kv });
            Vector(layer.PostNorm, $"{n}.post_attention_layernorm.weight", new[] { h });
            Linear(layer.Gate, $"{n}.mlp.gate_proj.weight", h, c.Intermediate);
            Linear(layer.Up, $"{n}.mlp.up_proj.weight", h, c.Intermediate);
            Linear(layer.Down, $"{n}.mlp.down_proj.weight", c.Intermediate, h);
            if (layer.QueryLora is not null)
            {
                Linear(layer.QueryLora.A, $"{n}.self_attn.q_proj.lora_A.weight", h, c.LoraRank);
                Linear(layer.QueryLora.B, $"{n}.self_attn.q_proj.lora_B.weight", c.LoraRank, qd);
            }
            if (layer.ValueLora is not null)
            {
                Linear(layer.ValueLora.A, $"{n}.self_attn.v_proj.lora_A.weight", h, c.LoraRank);
                Linear(layer.ValueLora.B, $"{n}.self_attn.v_proj.lora_B.weight", c.LoraRank, kv);
            }
        }
        Vector(FinalNorm, $"{prefix}norm.weight", new[] { h });
        if (LmHead is not null) Linear(LmHead, headName, h, c.VocabSize);
    }

    /// <summary>The reference initialization (Hugging Face <c>LlamaPreTrainedModel._init_weights</c>): linear weights and
    /// embeddings N(0, 0.02²), RMSNorm weights 1; and peft's for LoRA: A uniform in ±1/√Hidden (Kaiming with a = √5), B
    /// zero, so a new adapter leaves the model unchanged.</summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            // lora_B and the projections' biases start at zero.
            if (name.EndsWith("lora_B.weight", StringComparison.Ordinal) || name.EndsWith(".bias", StringComparison.Ordinal)) return values;
            if (name.EndsWith("lora_A.weight", StringComparison.Ordinal))
            {
                double bound = 1.0 / Math.Sqrt(shape[1]);
                for (int i = 0; i < count; i++) values[i] = bound * (2 * random.NextDouble() - 1);
                return values;
            }
            bool norm = name.EndsWith("layernorm.weight", StringComparison.Ordinal) || name.EndsWith("norm.weight", StringComparison.Ordinal);
            for (int i = 0; i < count; i++) values[i] = norm ? 1.0 : 0.02 * Normal();
            return values;
        });
    }
}
