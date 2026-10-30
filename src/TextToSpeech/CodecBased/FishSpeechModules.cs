using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of Fish Speech's Dual-AR transformer (the release's <c>DualARModelArgs</c>).</summary>
internal sealed record FishDualArConfiguration(int Vocabulary, int Dim, int Layers, int Heads, int LocalHeads, int Intermediate, int FastLayers,
    int CodebookSize, int Codebooks, int MaxSequence, double RopeBase, double NormEpsilon, double Dropout);

/// <summary>
/// Fish Speech's Dual-AR transformer (Liao et al. 2024, §3.1; the release's <c>DualARTransformer</c>, after gpt-fast): the
/// slow transformer reads each position's token embedding plus, at semantic positions, the sum of its codebooks' embeddings,
/// and predicts the next token; the fast transformer reads the slow transformer's hidden state followed by the frame's
/// codebook embeddings and predicts the codebooks in turn. Both are pre-norm Llama blocks: RMSNorm, grouped-query attention
/// with a fused query-key-value projection and interleaved rotary positions (the release's table is stored in bfloat16),
/// and a SwiGLU feed-forward.
/// </summary>
internal sealed class FishDualAr<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly double[,] _cos, _sin, _fastCos, _fastSin;

    internal sealed record Block(BiasFreeLinearLayer<T> AttentionNorm, BiasFreeLinearLayer<T> Qkv, BiasFreeLinearLayer<T> Output,
        BiasFreeLinearLayer<T> FeedForwardNorm, BiasFreeLinearLayer<T> Gate, BiasFreeLinearLayer<T> Up, BiasFreeLinearLayer<T> Down);

    public FishDualAr(IEngine engine, List<LayerBase<T>> layers, FishDualArConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.Heads != 0 || c.Heads % c.LocalHeads != 0) throw new ArgumentException("The width divides into heads and the heads into key-value groups.", nameof(configuration));
        // transformers-style initialization (the release's _init_weights): N(0, 0.02) linear weights and embeddings.
        BiasFreeLinearLayer<T> Linear(int inputs, int outputs)
        {
            var layer = new BiasFreeLinearLayer<T>(inputs, outputs);
            layers.Add(layer);
            TorchParameters.Linear(layer, inputs, outputs, IndexTtsOps.Normal(random, inputs * outputs, 0.02));
            return layer;
        }
        BiasFreeLinearLayer<T> Norm(int width) => IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, width).ToArray());
        Block NewBlock()
        {
            int hd = c.Dim / c.Heads;
            return new Block(Norm(c.Dim), Linear(c.Dim, (c.Heads + 2 * c.LocalHeads) * hd), Linear(c.Dim, c.Dim), Norm(c.Dim),
                Linear(c.Dim, c.Intermediate), Linear(c.Dim, c.Intermediate), Linear(c.Intermediate, c.Dim));
        }
        Embeddings = IndexTtsOps.Embedding(layers, random, c.Vocabulary, c.Dim, 0.02);
        CodebookEmbeddings = IndexTtsOps.Embedding(layers, random, c.CodebookSize * c.Codebooks, c.Dim, 0.02);
        for (int i = 0; i < c.Layers; i++) Slow.Add(NewBlock());
        SlowNorm = Norm(c.Dim);
        TokenOutput = Linear(c.Dim, c.Vocabulary);
        FastEmbeddings = IndexTtsOps.Embedding(layers, random, c.CodebookSize, c.Dim, 0.02);
        for (int i = 0; i < c.FastLayers; i++) Fast.Add(NewBlock());
        FastNorm = Norm(c.Dim);
        FastOutput = Linear(c.Dim, c.CodebookSize);
        (_cos, _sin) = RotaryTable(c.MaxSequence, c.Dim / c.Heads, c.RopeBase);
        (_fastCos, _fastSin) = RotaryTable(c.Codebooks, c.Dim / c.Heads, c.RopeBase);
    }

    public FishDualArConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> Embeddings { get; }
    public TiedEmbeddingLayer<T> CodebookEmbeddings { get; }
    public List<Block> Slow { get; } = new();
    public BiasFreeLinearLayer<T> SlowNorm { get; }
    public BiasFreeLinearLayer<T> TokenOutput { get; }
    public TiedEmbeddingLayer<T> FastEmbeddings { get; }
    public List<Block> Fast { get; } = new();
    public BiasFreeLinearLayer<T> FastNorm { get; }
    public BiasFreeLinearLayer<T> FastOutput { get; }

    /// <summary>
    /// <c>precompute_freqs_cis</c>: frequencies <c>1 / base^(2i/d)</c> and angles in float32, cos and sin stored in
    /// bfloat16 (round to nearest even), as the release keeps its table.
    /// </summary>
    internal static (double[,] Cos, double[,] Sin) RotaryTable(int positions, int headDim, double ropeBase)
    {
        int half = headDim / 2;
        var cos = new double[positions, half];
        var sin = new double[positions, half];
        for (int i = 0; i < half; i++)
        {
            float exponent = (float)(2 * i) / headDim;
            float frequency = (float)(1.0 / (float)Math.Pow(ropeBase, exponent));
            for (int p = 0; p < positions; p++)
            {
                float angle = p * frequency;
                cos[p, i] = BFloat16((float)Math.Cos(angle));
                sin[p, i] = BFloat16((float)Math.Sin(angle));
            }
        }
        return (cos, sin);
    }

    /// <summary>float32 to bfloat16 by rounding to the nearest even (torch's conversion).</summary>
    internal static double BFloat16(float value)
    {
        uint bits = BitConverter.ToUInt32(BitConverter.GetBytes(value), 0);
        uint rounding = 0x7FFF + ((bits >> 16) & 1);
        uint rounded = (bits + rounding) & 0xFFFF0000;
        return BitConverter.ToSingle(BitConverter.GetBytes(rounded), 0);
    }

    private Tensor<T> RmsNorm(BiasFreeLinearLayer<T> weight, Tensor<T> x)
    {
        int rows = x.Shape[0], d = x.Shape[1];
        var meanSquare = _engine.TensorDivideScalar(_engine.ReduceSum(_engine.TensorSquare(x), new[] { 1 }, keepDims: true), NumOps.FromDouble(d));
        var inverse = _engine.TensorReciprocal(_engine.TensorSqrt(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(Configuration.NormEpsilon))));
        var normed = _engine.TensorMultiply(x, _engine.TensorBroadcastTo(inverse, new[] { rows, d }));
        return _engine.TensorMultiply(normed, _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(weight), new[] { 1, d }), new[] { rows, d }));
    }

    // Interleaved rotation of [heads, L, hd] at the given positions: pairs (2i, 2i+1).
    private Tensor<T> Rotate(Tensor<T> x, IReadOnlyList<int> positions, double[,] cos, double[,] sin)
    {
        int heads = x.Shape[0], l = x.Shape[1], hd = x.Shape[2], half = hd / 2;
        var pairs = _engine.Reshape(x, new[] { heads, l, half, 2 });
        var even = _engine.Reshape(_engine.TensorNarrow(pairs, 3, 0, 1), new[] { heads, l, half });
        var odd = _engine.Reshape(_engine.TensorNarrow(pairs, 3, 1, 1), new[] { heads, l, half });
        var c = new Tensor<T>(new[] { 1, l, half });
        var s = new Tensor<T>(new[] { 1, l, half });
        for (int j = 0; j < l; j++)
            for (int i = 0; i < half; i++)
            {
                c[0, j, i] = NumOps.FromDouble(cos[positions[j], i]);
                s[0, j, i] = NumOps.FromDouble(sin[positions[j], i]);
            }
        var shape = new[] { heads, l, half };
        var cb = _engine.TensorBroadcastTo(c, shape);
        var sb = _engine.TensorBroadcastTo(s, shape);
        var first = _engine.TensorSubtract(_engine.TensorMultiply(even, cb), _engine.TensorMultiply(odd, sb));
        var second = _engine.TensorAdd(_engine.TensorMultiply(odd, cb), _engine.TensorMultiply(even, sb));
        return _engine.Reshape(_engine.TensorConcatenate(new[] { _engine.Reshape(first, new[] { heads, l, half, 1 }),
            _engine.Reshape(second, new[] { heads, l, half, 1 }) }, 3), new[] { heads, l, hd });
    }

    /// <summary>Keys and values cached per layer, <c>[localHeads, past, hd]</c>.</summary>
    internal sealed class Cache
    {
        public readonly List<(Tensor<T> Keys, Tensor<T> Values)?> Layers = new();
        public int Length;
    }

    private Tensor<T> RunBlock(Block b, Tensor<T> x, IReadOnlyList<int> positions, double[,] cos, double[,] sin, Cache? cache, int layer,
        bool training, Random? random)
    {
        var c = Configuration;
        int l = x.Shape[0], hd = c.Dim / c.Heads, local = c.LocalHeads, heads = c.Heads;
        var qkv = b.Qkv.Forward(RmsNorm(b.AttentionNorm, x));
        Tensor<T> Heads(int start, int count) =>
            _engine.TensorPermute(_engine.Reshape(_engine.TensorNarrow(qkv, 1, start, count * hd), new[] { l, count, hd }), new[] { 1, 0, 2 });
        var q = Rotate(Heads(0, heads), positions, cos, sin);
        var k = Rotate(Heads(heads * hd, local), positions, cos, sin);
        var v = Heads((heads + local) * hd, local);
        int past = 0;
        if (cache is not null)
        {
            while (cache.Layers.Count <= layer) cache.Layers.Add(null);
            if (cache.Layers[layer] is { } stored)
            {
                past = stored.Keys.Shape[1];
                k = _engine.TensorConcatenate(new[] { stored.Keys, k }, 1);
                v = _engine.TensorConcatenate(new[] { stored.Values, v }, 1);
            }
            cache.Layers[layer] = (k, v);
        }
        int total = k.Shape[1], group = heads / local;
        // repeat_interleave: query head h reads key-value head h / group.
        var index = new Tensor<int>(new[] { heads });
        for (int head = 0; head < heads; head++) index[head] = head / group;
        Tensor<T> Expand(Tensor<T> t) => _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(t, new[] { local, total * hd }), index, 0), new[] { heads, total, hd });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(Expand(k), new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var bias = new Tensor<T>(new[] { 1, l, total });
        var blocked = NumOps.FromDouble(typeof(T) == typeof(float) ? float.MinValue : double.MinValue);
        for (int i = 0; i < l; i++)
            for (int j = past + i + 1; j < total; j++) bias[0, i, j] = blocked;
        scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(bias, new[] { heads, l, total }));
        var weights = _engine.Softmax(scores, axis: 2);
        if (training && c.Dropout > 0 && random is not null) weights = T5Seq2Seq<T>.Dropout(_engine, weights, c.Dropout, random);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, Expand(v)), new[] { 1, 0, 2 }), new[] { l, heads * hd });
        var h = _engine.TensorAdd(x, b.Output.Forward(context));
        var normed = RmsNorm(b.FeedForwardNorm, h);
        var gate = b.Gate.Forward(normed);
        var ff = b.Down.Forward(_engine.TensorMultiply(_engine.TensorMultiply(gate, _engine.Sigmoid(gate)), b.Up.Forward(normed)));
        return _engine.TensorAdd(h, ff);
    }

    /// <summary>
    /// The input embeddings <c>[L, dim]</c> of rows <c>[1 + codebooks, L]</c>: the token's embedding plus, where
    /// <paramref name="isSemantic"/> holds for the token, the codebooks' embeddings (codebook i offset by i · size).
    /// </summary>
    public Tensor<T> Embed(int[,] rows, Func<int, bool> isSemantic)
    {
        var c = Configuration;
        int l = rows.GetLength(1);
        var tokens = new int[l];
        for (int t = 0; t < l; t++) tokens[t] = rows[0, t];
        var x = Embeddings.Forward(IndexTtsOps.Ids<T>(tokens));
        var codes = new List<int>();
        var mask = new Tensor<T>(new[] { l, 1 });
        for (int t = 0; t < l; t++)
        {
            bool semantic = isSemantic(tokens[t]);
            mask[t, 0] = semantic ? NumOps.One : NumOps.Zero;
        }
        for (int i = 0; i < c.Codebooks; i++)
        {
            var ids = new int[l];
            for (int t = 0; t < l; t++) ids[t] = rows[i + 1, t] + i * c.CodebookSize;
            var e = CodebookEmbeddings.Forward(IndexTtsOps.Ids<T>(ids));
            x = _engine.TensorAdd(x, _engine.TensorMultiply(e, _engine.TensorBroadcastTo(mask, new[] { l, c.Dim })));
        }
        return x;
    }

    /// <summary>The slow transformer over embeddings <c>[L, dim]</c> at <paramref name="positions"/>: the hidden states
    /// before the final norm.</summary>
    public Tensor<T> SlowHidden(Tensor<T> x, IReadOnlyList<int> positions, Cache? cache, bool training = false, Random? random = null)
    {
        for (int i = 0; i < Slow.Count; i++) x = RunBlock(Slow[i], x, positions, _cos, _sin, cache, i, training, random);
        if (cache is not null) cache.Length += x.Shape[0];
        return x;
    }

    /// <summary>The token logits <c>[L, vocabulary]</c> of slow hidden states.</summary>
    public Tensor<T> TokenLogits(Tensor<T> hidden) => TokenOutput.Forward(RmsNorm(SlowNorm, hidden));

    /// <summary>The fast transformer over <c>[n, dim]</c> (the hidden state then codebook embeddings) at positions
    /// 0 … n − 1: codebook logits <c>[n, size]</c>.</summary>
    public Tensor<T> FastLogits(Tensor<T> x, bool training = false, Random? random = null)
    {
        var positions = Enumerable.Range(0, x.Shape[0]).ToArray();
        for (int i = 0; i < Fast.Count; i++) x = RunBlock(Fast[i], x, positions, _fastCos, _fastSin, null, i, training, random);
        return FastOutput.Forward(RmsNorm(FastNorm, x));
    }

    /// <summary>
    /// The teacher-forced pass (<c>forward</c>) over rows <c>[1 + codebooks, L]</c>: token logits <c>[L, vocabulary]</c> and
    /// codebook logits per position <c>[L, codebooks, size]</c> — the fast transformer reads each position's hidden state
    /// and the next position's codebooks but the last; positions whose next codebooks are all zero (text) are skipped and
    /// left at zero.
    /// </summary>
    public (Tensor<T> Tokens, Tensor<T> Codebooks) Forward(int[,] rows, Func<int, bool> isSemantic, bool training = false, Random? random = null)
    {
        var c = Configuration;
        int l = rows.GetLength(1), n = c.Codebooks;
        var hidden = SlowHidden(Embed(rows, isSemantic), Enumerable.Range(0, l).ToArray(), null, training, random);
        var tokens = TokenLogits(hidden);
        var frames = new List<Tensor<T>>();
        var zero = new Tensor<T>(new[] { 1, n, c.CodebookSize });
        for (int t = 0; t < l; t++)
        {
            var next = new int[n - 1];
            bool any = false;
            for (int i = 0; i < n - 1; i++)
            {
                next[i] = t + 1 < l ? rows[i + 1, t + 1] : 0;
                any |= next[i] != 0;
            }
            if (!any)
            {
                frames.Add(zero);
                continue;
            }
            var input = _engine.TensorConcatenate(new[] { _engine.TensorNarrow(hidden, 0, t, 1), FastEmbeddings.Forward(IndexTtsOps.Ids<T>(next)) }, 0);
            frames.Add(_engine.Reshape(FastLogits(input, training, random), new[] { 1, n, c.CodebookSize }));
        }
        // The release keeps at least eight positions when every one is text (codebook_mask[:8] = False).
        if (frames.All(f => ReferenceEquals(f, zero)))
            for (int t = 0; t < Math.Min(8, l); t++)
            {
                var input = _engine.TensorConcatenate(new[] { _engine.TensorNarrow(hidden, 0, t, 1), FastEmbeddings.Forward(IndexTtsOps.Ids<T>(new int[n - 1])) }, 0);
                frames[t] = _engine.Reshape(FastLogits(input, training, random), new[] { 1, n, c.CodebookSize });
            }
        return (tokens, _engine.TensorConcatenate(frames.ToArray(), 0));
    }

    /// <summary>Loads the release's <c>model.pth</c> names.</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int hd = c.Dim / c.Heads;
        void Linear(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }));
        void Vector(BiasFreeLinearLayer<T> holder, string name, int width) => TorchParameters.Linear(holder, 1, width, read(name, new[] { width }));
        void Blocks(List<Block> blocks, string prefix)
        {
            for (int i = 0; i < blocks.Count; i++)
            {
                var b = blocks[i];
                string p = $"{prefix}.{i}.";
                Vector(b.AttentionNorm, p + "attention_norm.weight", c.Dim);
                Linear(b.Qkv, p + "attention.wqkv", c.Dim, (c.Heads + 2 * c.LocalHeads) * hd);
                Linear(b.Output, p + "attention.wo", c.Dim, c.Dim);
                Vector(b.FeedForwardNorm, p + "ffn_norm.weight", c.Dim);
                Linear(b.Gate, p + "feed_forward.w1", c.Dim, c.Intermediate);
                Linear(b.Up, p + "feed_forward.w3", c.Dim, c.Intermediate);
                Linear(b.Down, p + "feed_forward.w2", c.Intermediate, c.Dim);
            }
        }
        Embeddings.LoadTable(read("embeddings.weight", new[] { c.Vocabulary, c.Dim }));
        CodebookEmbeddings.LoadTable(read("codebook_embeddings.weight", new[] { c.CodebookSize * c.Codebooks, c.Dim }));
        Blocks(Slow, "layers");
        Vector(SlowNorm, "norm.weight", c.Dim);
        Linear(TokenOutput, "output", c.Dim, c.Vocabulary);
        FastEmbeddings.LoadTable(read("fast_embeddings.weight", new[] { c.CodebookSize, c.Dim }));
        Blocks(Fast, "fast_layers");
        Vector(FastNorm, "fast_norm.weight", c.Dim);
        Linear(FastOutput, "fast_output", c.Dim, c.CodebookSize);
    }
}
