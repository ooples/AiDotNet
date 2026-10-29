using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The shape of a <see cref="GlmDecoder{T}"/>; the defaults are GLM-4-Voice-9B's.</summary>
internal sealed record GlmConfiguration
{
    public int VocabSize { get; init; } = 168960;
    public int Hidden { get; init; } = 4096;
    public int Layers { get; init; } = 40;
    public int Heads { get; init; } = 32;
    /// <summary>Key/value heads (<c>multi_query_group_num</c>, 2).</summary>
    public int KeyValueGroups { get; init; } = 2;
    /// <summary>Width of each head (<c>kv_channels</c>, 128).</summary>
    public int HeadDim { get; init; } = 128;
    /// <summary>The SwiGLU inner width (<c>ffn_hidden_size</c>, 13696).</summary>
    public int FeedForward { get; init; } = 13696;
    public double NormEpsilon { get; init; } = 3.90625e-08;
    public double RopeRatio { get; init; } = 1;
    public double Dropout { get; init; }
}

/// <summary>
/// GLM-4's decoder as THUDM's <c>modeling_chatglm.py</c> builds it (GLM-4-Voice-9B): token embeddings, pre-norm blocks
/// of multi-query attention (2 key/value groups, biased query/key/value projection, rotary positions on the first half
/// of each head in interleaved pairs) and a SwiGLU MLP, each with RMSNorm, a final RMSNorm and an untied output layer;
/// with a key/value cache for generation.
/// </summary>
internal sealed class GlmDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly GlmConfiguration _c;

    internal sealed class Block
    {
        public BiasFreeLinearLayer<T> InputNorm = null!, PostAttentionNorm = null!;
        public DenseLayer<T> QueryKeyValue = null!;
        public BiasFreeLinearLayer<T> Dense = null!, HiddenToFour = null!, FourToHidden = null!;
    }

    public GlmDecoder(IEngine engine, List<LayerBase<T>> layers, GlmConfiguration configuration, Random random)
    {
        _engine = engine;
        _c = configuration;
        var c = configuration;
        if (c.Heads % c.KeyValueGroups != 0) throw new ArgumentException("Heads must be a multiple of KeyValueGroups.", nameof(configuration));
        if (c.HeadDim % 4 != 0) throw new ArgumentException("Rotary positions on half of each head need a width divisible by 4.", nameof(configuration));
        // ChatGLM's _init_weights: N(0, 0.02) everywhere (init_method_std), biases zero, norms one.
        Embedding = new TiedEmbeddingLayer<T>(c.VocabSize, c.Hidden);
        layers.Add(Embedding);
        Embedding.Reinitialize(() => 0.02 * IndexTtsOps.Gaussian(random));
        BiasFreeLinearLayer<T> Kernel(int inputs, int outputs)
        {
            var layer = new BiasFreeLinearLayer<T>(inputs, outputs);
            layers.Add(layer);
            TorchParameters.Linear(layer, inputs, outputs, IndexTtsOps.Normal(random, inputs * outputs, 0.02));
            return layer;
        }
        int qkv = c.Heads * c.HeadDim + 2 * c.KeyValueGroups * c.HeadDim;
        for (int i = 0; i < c.Layers; i++)
        {
            var block = new Block { InputNorm = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, c.Hidden).ToArray()) };
            block.QueryKeyValue = IndexTtsOps.Linear(engine, layers, random, c.Hidden, qkv);
            TorchParameters.Linear(engine, block.QueryKeyValue, c.Hidden, qkv, IndexTtsOps.Normal(random, c.Hidden * qkv, 0.02), new double[qkv]);
            block.Dense = Kernel(c.Heads * c.HeadDim, c.Hidden);
            block.PostAttentionNorm = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, c.Hidden).ToArray());
            block.HiddenToFour = Kernel(c.Hidden, 2 * c.FeedForward);
            block.FourToHidden = Kernel(c.FeedForward, c.Hidden);
            Blocks.Add(block);
        }
        FinalNorm = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, c.Hidden).ToArray());
        Output = Kernel(c.Hidden, c.VocabSize);
    }

    public GlmConfiguration Configuration => _c;
    public TiedEmbeddingLayer<T> Embedding { get; }
    public List<Block> Blocks { get; } = new();
    public BiasFreeLinearLayer<T> FinalNorm { get; }
    public BiasFreeLinearLayer<T> Output { get; }

    /// <summary>A generation's keys and values per block (<c>[groups, S, head]</c>) and its length.</summary>
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

    private Tensor<T> Norm(Tensor<T> x, BiasFreeLinearLayer<T> scale)
    {
        int length = x.Shape[0], d = x.Shape[1];
        var meanSquare = _engine.TensorBroadcastTo(_engine.ReduceMean(_engine.TensorMultiply(x, x), new[] { 1 }, true), new[] { length, d });
        var normed = _engine.TensorDivide(x, _engine.TensorSqrt(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(_c.NormEpsilon))));
        return _engine.TensorMultiply(normed, _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(scale), new[] { 1, d }), new[] { length, d }));
    }

    /// <summary>ChatGLM's rotary positions on <c>[heads, L, head]</c> from <paramref name="start"/>: the first half of
    /// each head rotated in interleaved pairs by θᵢ = (10000·ratio)^(−2i / (head/2)), the second half passed through.</summary>
    private Tensor<T> Rotate(Tensor<T> x, int start)
    {
        int heads = x.Shape[0], length = x.Shape[1], hd = x.Shape[2], rot = hd / 2, pairs = rot / 2;
        var cos = new Tensor<T>(new[] { 1, length, pairs, 1 });
        var sin = new Tensor<T>(new[] { 1, length, pairs, 1 });
        double theBase = 10000 * _c.RopeRatio;
        for (int t = 0; t < length; t++)
            for (int i = 0; i < pairs; i++)
            {
                double angle = (start + t) / Math.Pow(theBase, 2.0 * i / rot);
                cos[0, t, i, 0] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, t, i, 0] = NumOps.FromDouble(Math.Sin(angle));
            }
        var rotated = _engine.Reshape(_engine.TensorNarrow(x, 2, 0, rot), new[] { heads, length, pairs, 2 });
        var even = _engine.TensorNarrow(rotated, 3, 0, 1);
        var odd = _engine.TensorNarrow(rotated, 3, 1, 1);
        var shape = new[] { heads, length, pairs, 1 };
        var c = _engine.TensorBroadcastTo(cos, shape);
        var s = _engine.TensorBroadcastTo(sin, shape);
        var first = _engine.TensorSubtract(_engine.TensorMultiply(even, c), _engine.TensorMultiply(odd, s));
        var second = _engine.TensorAdd(_engine.TensorMultiply(odd, c), _engine.TensorMultiply(even, s));
        var interleaved = _engine.Reshape(_engine.TensorConcatenate(new[] { first, second }, 3), new[] { heads, length, rot });
        return _engine.TensorConcatenate(new[] { interleaved, _engine.TensorNarrow(x, 2, rot, hd - rot) }, 2);
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random? random) =>
        training && _c.Dropout > 0 && random is not null ? T5Seq2Seq<T>.Dropout(_engine, x, _c.Dropout, random) : x;

    /// <summary>The token embeddings <c>[L, d]</c> of ids.</summary>
    public Tensor<T> Embed(IReadOnlyList<int> ids) => Embedding.Forward(IndexTtsOps.Ids<T>(ids));

    /// <summary>The final hidden states <c>[L, d]</c> of embeddings <c>[L, d]</c>, continuing <paramref name="cache"/>
    /// when given; <paramref name="training"/> applies the hidden dropout after each branch, as ChatGLM's
    /// <c>hidden_dropout</c> does.</summary>
    public Tensor<T> Forward(Tensor<T> embeddings, Cache? cache = null, bool training = false, Random? random = null)
    {
        var c = _c;
        int length = embeddings.Shape[0], start = cache?.Length ?? 0, hd = c.HeadDim, heads = c.Heads, groups = c.KeyValueGroups;
        var x = embeddings;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            var qkv = b.QueryKeyValue.Forward(Norm(x, b.InputNorm));
            var q = Rotate(IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(qkv, 1, 0, heads * hd), heads), start);
            var k = Rotate(IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(qkv, 1, heads * hd, groups * hd), groups), start);
            var v = IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(qkv, 1, heads * hd + groups * hd, groups * hd), groups);
            if (cache is not null)
            {
                if (cache.Keys[i] is { } pastK) k = _engine.TensorConcatenate(new[] { pastK, k }, 1);
                if (cache.Values[i] is { } pastV) v = _engine.TensorConcatenate(new[] { pastV, v }, 1);
                cache.Keys[i] = k;
                cache.Values[i] = v;
            }
            int total = k.Shape[1], per = heads / groups;
            if (per > 1)
            {
                k = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(k, new[] { groups, 1, total, hd }), new[] { groups, per, total, hd }), new[] { heads, total, hd });
                v = _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(v, new[] { groups, 1, total, hd }), new[] { groups, per, total, hd }), new[] { heads, total, hd });
            }
            var mask = new Tensor<T>(new[] { length, total });
            var blocked = NumOps.FromDouble(double.NegativeInfinity);
            for (int r = 0; r < length; r++)
                for (int col = start + r + 1; col < total; col++) mask[r, col] = blocked;
            var context = IndexTtsOps.MergeHeads(_engine, IndexTtsOps.Attend(_engine, q, k, v, 1.0 / Math.Sqrt(hd), mask));
            x = _engine.TensorAdd(x, Drop(b.Dense.Forward(context), training, random));
            var inner = b.HiddenToFour.Forward(Norm(x, b.PostAttentionNorm));
            // swiglu: silu(first half) · second half.
            var gated = _engine.TensorMultiply(_engine.Swish(_engine.TensorNarrow(inner, 1, 0, c.FeedForward)), _engine.TensorNarrow(inner, 1, c.FeedForward, c.FeedForward));
            x = _engine.TensorAdd(x, Drop(b.FourToHidden.Forward(gated), training, random));
        }
        if (cache is not null) cache.Length += length;
        return Norm(x, FinalNorm);
    }

    /// <summary>The logits <c>[L, V]</c> of hidden states (the untied output layer).</summary>
    public Tensor<T> Logits(Tensor<T> hidden) => Output.Forward(hidden);

    /// <summary>Loads a Hugging Face <c>ChatGLMForConditionalGeneration</c> state (<c>transformer.*</c>; the rotary
    /// buffer <c>inv_freq</c> is recomputed and not read).</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = _c;
        int d = c.Hidden, qkv = c.Heads * c.HeadDim + 2 * c.KeyValueGroups * c.HeadDim;
        Embedding.LoadTable(read("transformer.embedding.word_embeddings.weight", new[] { c.VocabSize, d }));
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"transformer.encoder.layers.{i}.";
            TorchParameters.Linear(b.InputNorm, 1, d, read(p + "input_layernorm.weight", new[] { d }));
            TorchParameters.Linear(_engine, b.QueryKeyValue, d, qkv, read(p + "self_attention.query_key_value.weight", new[] { qkv, d }),
                read(p + "self_attention.query_key_value.bias", new[] { qkv }));
            TorchParameters.Linear(b.Dense, c.Heads * c.HeadDim, d, read(p + "self_attention.dense.weight", new[] { d, c.Heads * c.HeadDim }));
            TorchParameters.Linear(b.PostAttentionNorm, 1, d, read(p + "post_attention_layernorm.weight", new[] { d }));
            TorchParameters.Linear(b.HiddenToFour, d, 2 * c.FeedForward, read(p + "mlp.dense_h_to_4h.weight", new[] { 2 * c.FeedForward, d }));
            TorchParameters.Linear(b.FourToHidden, c.FeedForward, d, read(p + "mlp.dense_4h_to_h.weight", new[] { d, c.FeedForward }));
        }
        TorchParameters.Linear(FinalNorm, 1, d, read("transformer.encoder.final_layernorm.weight", new[] { d }));
        TorchParameters.Linear(Output, d, c.VocabSize, read("transformer.output_layer.weight", new[] { c.VocabSize, d }));
    }
}
