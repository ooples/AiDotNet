using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Which decoder <see cref="PalmDecoder{T}"/> builds.</summary>
public enum PalmDecoderKind
{
    /// <summary>PaLM (Chowdhery et al. 2022 §2): parallel attention and SwiGLU MLP behind one bias-free LayerNorm,
    /// multi-query attention, rotary positions, no biases, input and output embeddings shared, the logits scaled by
    /// 1/√d.</summary>
    Palm,

    /// <summary>Gemma (Gemma Team 2024): sequential attention and GeGLU MLP each behind an RMSNorm scaled by (1 + w),
    /// the embeddings scaled by √d, multi-query (2B) or multi-head attention, rotary positions, shared embeddings.</summary>
    Gemma,
}

/// <summary>The shape of a <see cref="PalmDecoder{T}"/>.</summary>
internal sealed record PalmConfiguration
{
    public PalmDecoderKind Kind { get; init; } = PalmDecoderKind.Palm;
    public int VocabSize { get; init; } = 256000;
    public int Hidden { get; init; } = 4096;
    public int Layers { get; init; } = 32;
    public int Heads { get; init; } = 16;
    public int KeyValueHeads { get; init; } = 1;
    public int HeadDim { get; init; } = 256;
    public int Intermediate { get; init; } = 16384;
    public double NormEpsilon { get; init; } = 1e-5;
    public double RopeTheta { get; init; } = 10000;
    public double Dropout { get; init; }
}

/// <summary>
/// A PaLM-family decoder-only Transformer: PaLM's own block (Chowdhery et al. 2022 §2; lucidrains/PaLM-pytorch for the
/// layout the paper leaves open: one LayerNorm per block, a final LayerNorm, rotary positions on half-rotated pairs) or
/// Gemma's, which loads public pretrained weights. Token ids <c>[L]</c> in, hidden states and logits out, with a
/// key/value cache for generation.
/// </summary>
internal sealed class PalmDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly PalmConfiguration _c;

    internal sealed class Block
    {
        public BiasFreeLinearLayer<T> AttentionNorm = null!;     // PaLM's single norm; Gemma's input_layernorm
        public BiasFreeLinearLayer<T>? MlpNorm;                  // Gemma's post_attention_layernorm
        public BiasFreeLinearLayer<T> Query = null!, Key = null!, Value = null!, Output = null!;
        public BiasFreeLinearLayer<T> Gate = null!, Up = null!, Down = null!;
    }

    public PalmDecoder(IEngine engine, List<LayerBase<T>> layers, PalmConfiguration configuration, Random random)
    {
        _engine = engine;
        _c = configuration;
        var c = configuration;
        if (c.HeadDim % 2 != 0) throw new ArgumentException("Rotary positions need an even head width.", nameof(configuration));
        if (c.Heads % c.KeyValueHeads != 0) throw new ArgumentException("Heads must be a multiple of KeyValueHeads.", nameof(configuration));
        // PaLM §5: E ~ N(0, 1) (no norm on the embeddings), kernels N(0, 1/n_in), norm scales 1.
        Embedding = new TiedEmbeddingLayer<T>(c.VocabSize, c.Hidden);
        layers.Add(Embedding);
        Embedding.Reinitialize(() => IndexTtsOps.Gaussian(random));
        BiasFreeLinearLayer<T> Kernel(int inputs, int outputs)
        {
            var layer = new BiasFreeLinearLayer<T>(inputs, outputs);
            layers.Add(layer);
            TorchParameters.Linear(layer, inputs, outputs, IndexTtsOps.Normal(random, inputs * outputs, 1.0 / Math.Sqrt(inputs)));
            return layer;
        }
        double normInit = c.Kind == PalmDecoderKind.Palm ? 1.0 : 0.0;   // Gemma's scale is (1 + w), so w starts at 0
        for (int i = 0; i < c.Layers; i++)
        {
            var block = new Block { AttentionNorm = IndexTtsOps.Values(layers, Enumerable.Repeat(normInit, c.Hidden).ToArray()) };
            if (c.Kind == PalmDecoderKind.Gemma) block.MlpNorm = IndexTtsOps.Values(layers, new double[c.Hidden]);
            block.Query = Kernel(c.Hidden, c.Heads * c.HeadDim);
            block.Key = Kernel(c.Hidden, c.KeyValueHeads * c.HeadDim);
            block.Value = Kernel(c.Hidden, c.KeyValueHeads * c.HeadDim);
            block.Output = Kernel(c.Heads * c.HeadDim, c.Hidden);
            block.Gate = Kernel(c.Hidden, c.Intermediate);
            block.Up = Kernel(c.Hidden, c.Intermediate);
            block.Down = Kernel(c.Intermediate, c.Hidden);
            Blocks.Add(block);
        }
        FinalNorm = IndexTtsOps.Values(layers, Enumerable.Repeat(normInit, c.Hidden).ToArray());
    }

    public PalmConfiguration Configuration => _c;
    public TiedEmbeddingLayer<T> Embedding { get; }
    public List<Block> Blocks { get; } = new();
    public BiasFreeLinearLayer<T> FinalNorm { get; }

    /// <summary>A generation's keys and values per block (<c>[KV heads, S, head]</c>) and the positions so far.</summary>
    internal sealed class Cache
    {
        public List<Tensor<T>?> Keys { get; } = new();
        public List<Tensor<T>?> Values { get; } = new();
        public int Length { get; set; }
    }

    public Cache NewCache()
    {
        var cache = new Cache();
        for (int i = 0; i < Blocks.Count; i++)
        {
            cache.Keys.Add(null);
            cache.Values.Add(null);
        }
        return cache;
    }

    // ---------------------------------------------------------------- pieces

    private Tensor<T> Norm(Tensor<T> x, BiasFreeLinearLayer<T> scale)
    {
        int length = x.Shape[0], d = x.Shape[1];
        var eps = NumOps.FromDouble(_c.NormEpsilon);
        Tensor<T> normed;
        Tensor<T> gain;
        if (_c.Kind == PalmDecoderKind.Palm)
        {
            // LayerNorm without a bias (the beta of lucidrains' LayerNorm is a zero buffer).
            var mean = _engine.TensorBroadcastTo(_engine.ReduceMean(x, new[] { 1 }, true), new[] { length, d });
            var centered = _engine.TensorSubtract(x, mean);
            var variance = _engine.TensorBroadcastTo(_engine.ReduceMean(_engine.TensorMultiply(centered, centered), new[] { 1 }, true), new[] { length, d });
            normed = _engine.TensorDivide(centered, _engine.TensorSqrt(_engine.TensorAddScalar(variance, eps)));
            gain = IndexTtsOps.Value(scale);
        }
        else
        {
            // GemmaRMSNorm: x · rsqrt(mean(x²) + ε) · (1 + w).
            var meanSquare = _engine.TensorBroadcastTo(_engine.ReduceMean(_engine.TensorMultiply(x, x), new[] { 1 }, true), new[] { length, d });
            normed = _engine.TensorDivide(x, _engine.TensorSqrt(_engine.TensorAddScalar(meanSquare, eps)));
            gain = _engine.TensorAddScalar(IndexTtsOps.Value(scale), NumOps.One);
        }
        return _engine.TensorMultiply(normed, _engine.TensorBroadcastTo(_engine.Reshape(gain, new[] { 1, d }), new[] { length, d }));
    }

    /// <summary>Rotates <c>[heads, L, head]</c> by positions from <paramref name="start"/>: x·cos + rotate_half(x)·sin
    /// with the frequencies repeated over both halves.</summary>
    private Tensor<T> Rotate(Tensor<T> x, int start)
    {
        int heads = x.Shape[0], length = x.Shape[1], hd = x.Shape[2], half = hd / 2;
        var cos = new Tensor<T>(new[] { 1, length, hd });
        var sin = new Tensor<T>(new[] { 1, length, hd });
        for (int t = 0; t < length; t++)
            for (int k = 0; k < half; k++)
            {
                double angle = (start + t) / Math.Pow(_c.RopeTheta, 2.0 * k / hd);
                cos[0, t, k] = cos[0, t, k + half] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, t, k] = sin[0, t, k + half] = NumOps.FromDouble(Math.Sin(angle));
            }
        var shape = new[] { heads, length, hd };
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(_engine.TensorNarrow(x, 2, half, half)), _engine.TensorNarrow(x, 2, 0, half) }, 2);
        return _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(cos, shape)),
            _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(sin, shape)));
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random? random) =>
        training && _c.Dropout > 0 && random is not null ? T5Seq2Seq<T>.Dropout(_engine, x, _c.Dropout, random) : x;

    private Tensor<T> Attention(Block block, Tensor<T> normed, Cache? cache, int index, int start)
    {
        int length = normed.Shape[0], hd = _c.HeadDim, heads = _c.Heads, kvHeads = _c.KeyValueHeads;
        var q = Rotate(IndexTtsOps.SplitHeads(_engine, block.Query.Forward(normed), heads), start);
        var k = Rotate(IndexTtsOps.SplitHeads(_engine, block.Key.Forward(normed), kvHeads), start);
        var v = IndexTtsOps.SplitHeads(_engine, block.Value.Forward(normed), kvHeads);
        if (cache is not null)
        {
            if (cache.Keys[index] is { } pastK) k = _engine.TensorConcatenate(new[] { pastK, k }, 1);
            if (cache.Values[index] is { } pastV) v = _engine.TensorConcatenate(new[] { pastV, v }, 1);
            cache.Keys[index] = k;
            cache.Values[index] = v;
        }
        int total = k.Shape[1], group = heads / kvHeads;
        if (group > 1)
        {
            // Each key/value head serves `group` consecutive query heads.
            k = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(k, new[] { kvHeads, 1, total, hd }), new[] { kvHeads, group, total, hd }), new[] { heads, total, hd });
            v = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(v, new[] { kvHeads, 1, total, hd }), new[] { kvHeads, group, total, hd }), new[] { heads, total, hd });
        }
        var mask = new Tensor<T>(new[] { length, total });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < length; i++)
            for (int j = start + i + 1; j < total; j++) mask[i, j] = blocked;
        var context = IndexTtsOps.Attend(_engine, q, k, v, 1.0 / Math.Sqrt(hd), mask);
        return block.Output.Forward(IndexTtsOps.MergeHeads(_engine, context));
    }

    private Tensor<T> Mlp(Block block, Tensor<T> normed)
    {
        var gate = block.Gate.Forward(normed);
        var activated = _c.Kind == PalmDecoderKind.Palm ? _engine.Swish(gate) : IndexTtsOps.NewGelu(_engine, gate);
        return block.Down.Forward(_engine.TensorMultiply(activated, block.Up.Forward(normed)));
    }

    // ---------------------------------------------------------------- forward

    /// <summary>The input embeddings <c>[L, d]</c> of token ids (Gemma's scaled by √d).</summary>
    public Tensor<T> Embed(IReadOnlyList<int> ids)
    {
        var x = Embedding.Forward(IndexTtsOps.Ids<T>(ids));
        return _c.Kind == PalmDecoderKind.Gemma ? _engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.Sqrt(_c.Hidden))) : x;
    }

    /// <summary>
    /// The final hidden states <c>[L, d]</c> of embeddings <c>[L, d]</c>, continuing <paramref name="cache"/> when given.
    /// Fine-tuning's dropout (rate <see cref="PalmConfiguration.Dropout"/>) falls on the embeddings and on each
    /// branch's output before its residual sum.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> embeddings, Cache? cache = null, bool training = false, Random? random = null)
    {
        int start = cache?.Length ?? 0;
        var x = Drop(embeddings, training, random);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var block = Blocks[i];
            if (_c.Kind == PalmDecoderKind.Palm)
            {
                // y = x + Attention(LN(x)) + MLP(LN(x)), one LN shared by both branches.
                var normed = Norm(x, block.AttentionNorm);
                var attention = Drop(Attention(block, normed, cache, i, start), training, random);
                var mlp = Drop(Mlp(block, normed), training, random);
                x = _engine.TensorAdd(x, _engine.TensorAdd(attention, mlp));
            }
            else
            {
                x = _engine.TensorAdd(x, Drop(Attention(block, Norm(x, block.AttentionNorm), cache, i, start), training, random));
                x = _engine.TensorAdd(x, Drop(Mlp(block, Norm(x, block.MlpNorm!)), training, random));
            }
        }
        if (cache is not null) cache.Length += embeddings.Shape[0];
        return Norm(x, FinalNorm);
    }

    /// <summary>The logits <c>[L, V]</c> of hidden states through the shared embedding (PaLM's scaled by 1/√d, §5).</summary>
    public Tensor<T> Logits(Tensor<T> hidden)
    {
        var logits = Embedding.Logits(hidden);
        return _c.Kind == PalmDecoderKind.Palm ? _engine.TensorMultiplyScalar(logits, NumOps.FromDouble(1.0 / Math.Sqrt(_c.Hidden))) : logits;
    }

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads lucidrains/PaLM-pytorch's state (<c>0.weight</c>, <c>{i}.fn.norm.gamma</c>, the fused
    /// <c>{i}.fn.fused_attn_ff_proj.weight</c> [q, k, v, ff (value half, gate half)], <c>{i}.fn.attn_out.weight</c>,
    /// <c>{i}.fn.ff_out.1.weight</c>, <c>{L+1}.gamma</c>); <paramref name="rows"/> embedding rows are read (the rest of
    /// the table, AudioPaLM's audio rows, stays as it is).
    /// </summary>
    public void LoadPalmPytorch(Func<string, int[], double[]> read, int? rows = null)
    {
        var c = _c;
        if (c.Kind != PalmDecoderKind.Palm || c.KeyValueHeads != 1) throw new InvalidOperationException("PaLM-pytorch holds a multi-query PaLM block.");
        LoadEmbedding(read("0.weight", new[] { rows ?? c.VocabSize, c.Hidden }), rows ?? c.VocabSize);
        int d = c.Hidden, attention = c.Heads * c.HeadDim, hd = c.HeadDim, ff = c.Intermediate;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{i + 1}.fn.";
            TorchParameters.Linear(b.AttentionNorm, 1, d, read(p + "norm.gamma", new[] { d }));
            var fused = read(p + "fused_attn_ff_proj.weight", new[] { attention + 2 * hd + 2 * ff, d });
            double[] Rows(int from, int count) => fused.Skip(from * d).Take(count * d).ToArray();
            TorchParameters.Linear(b.Query, d, attention, Rows(0, attention));
            TorchParameters.Linear(b.Key, d, hd, Rows(attention, hd));
            TorchParameters.Linear(b.Value, d, hd, Rows(attention + hd, hd));
            // SwiGLU: x, gate = ff.chunk(2); silu(gate) · x.
            TorchParameters.Linear(b.Up, d, ff, Rows(attention + 2 * hd, ff));
            TorchParameters.Linear(b.Gate, d, ff, Rows(attention + 2 * hd + ff, ff));
            TorchParameters.Linear(b.Output, attention, d, read(p + "attn_out.weight", new[] { d, attention }));
            TorchParameters.Linear(b.Down, ff, d, read(p + "ff_out.1.weight", new[] { d, ff }));
        }
        TorchParameters.Linear(FinalNorm, 1, d, read($"{Blocks.Count + 1}.gamma", new[] { d }));
    }

    /// <summary>Loads a Hugging Face <c>GemmaForCausalLM</c> state (<c>model.*</c>; the head is the shared table);
    /// <paramref name="rows"/> embedding rows are read.</summary>
    public void LoadGemma(Func<string, int[], double[]> read, int? rows = null)
    {
        var c = _c;
        if (c.Kind != PalmDecoderKind.Gemma) throw new InvalidOperationException("Gemma weights need a Gemma decoder.");
        LoadEmbedding(read("model.embed_tokens.weight", new[] { rows ?? c.VocabSize, c.Hidden }), rows ?? c.VocabSize);
        int d = c.Hidden, q = c.Heads * c.HeadDim, kv = c.KeyValueHeads * c.HeadDim, ff = c.Intermediate;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"model.layers.{i}.";
            TorchParameters.Linear(b.AttentionNorm, 1, d, read(p + "input_layernorm.weight", new[] { d }));
            TorchParameters.Linear(b.MlpNorm!, 1, d, read(p + "post_attention_layernorm.weight", new[] { d }));
            TorchParameters.Linear(b.Query, d, q, read(p + "self_attn.q_proj.weight", new[] { q, d }));
            TorchParameters.Linear(b.Key, d, kv, read(p + "self_attn.k_proj.weight", new[] { kv, d }));
            TorchParameters.Linear(b.Value, d, kv, read(p + "self_attn.v_proj.weight", new[] { kv, d }));
            TorchParameters.Linear(b.Output, q, d, read(p + "self_attn.o_proj.weight", new[] { d, q }));
            TorchParameters.Linear(b.Gate, d, ff, read(p + "mlp.gate_proj.weight", new[] { ff, d }));
            TorchParameters.Linear(b.Up, d, ff, read(p + "mlp.up_proj.weight", new[] { ff, d }));
            TorchParameters.Linear(b.Down, ff, d, read(p + "mlp.down_proj.weight", new[] { d, ff }));
        }
        TorchParameters.Linear(FinalNorm, 1, d, read("model.norm.weight", new[] { d }));
    }

    /// <summary>Overwrites the first <paramref name="rows"/> rows of the token table; with fewer rows than the table, the
    /// rest (AudioPaLM's audio tokens) are zeroed, as §3.2's checkpoint extension initializes them.</summary>
    private void LoadEmbedding(double[] values, int rows)
    {
        int d = _c.Hidden;
        if (rows > _c.VocabSize) throw new InvalidDataException($"The checkpoint's {rows} tokens exceed the decoder's {_c.VocabSize}.");
        var table = new double[_c.VocabSize * d];
        Array.Copy(values, table, rows * d);
        Embedding.LoadTable(table);
    }
}
