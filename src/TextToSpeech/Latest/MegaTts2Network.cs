using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>How the CVQ-VAE quantizer of <see cref="MegaTTS2{T}"/> finds a vector's code.</summary>
public enum MegaTts2CodeDistance
{
    /// <summary>The largest cosine similarity (CVQ-VAE's default, <c>--distance cos</c>).</summary>
    Cosine,

    /// <summary>The smallest Euclidean distance (the original VQ-VAE's).</summary>
    Euclidean,
}

/// <summary>Where CVQ-VAE's dynamic initialization draws the anchors that move rarely used codes.</summary>
public enum MegaTts2CodeAnchor
{
    /// <summary>Each code's closest feature in the step (CVQ-VAE's default, <c>--anchor closest</c>).</summary>
    Closest,

    /// <summary>A feature drawn for each code with probabilities from the softmax of its similarities
    /// (<c>probrandom</c>).</summary>
    ProbabilisticRandom,

    /// <summary>Features from a pool of earlier steps' features (<c>random</c>).</summary>
    Random,
}

/// <summary>Mega-TTS 2's sizes (Jiang et al. 2024, Appendix A.4 and Table 5).</summary>
internal sealed record MegaTts2Configuration
{
    public int PhoneVocabulary { get; init; } = 200;
    public int Hidden { get; init; } = 512;
    public int ContentLayers { get; init; } = 8;
    public int ContentHeads { get; init; } = 2;
    public int ContentKernel { get; init; } = 5;
    public int ContentFilter { get; init; } = 1024;
    public int ProsodyHidden { get; init; } = 384;
    public int ProsodyKernel { get; init; } = 5;
    public int ProsodyBlocks { get; init; } = 3;
    public int Compression { get; init; } = 8;
    public int Codes { get; init; } = 1024;
    public int CodeDimension { get; init; } = 256;
    public MegaTts2CodeDistance Distance { get; init; } = MegaTts2CodeDistance.Cosine;
    public MegaTts2CodeAnchor Anchor { get; init; } = MegaTts2CodeAnchor.Closest;
    public double CommitmentWeight { get; init; } = 0.25;
    public double UsageDecay { get; init; } = 0.99;
    public int TimbreHidden { get; init; } = 256;
    public int TimbreKernel { get; init; } = 3;
    public int TimbreBlocks { get; init; } = 5;
    public int Downsampling { get; init; } = 16;
    public int TimbreHeads { get; init; } = 2;
    public int DecoderBlocks { get; init; } = 4;
    public int DecoderKernel { get; init; } = 5;
    public int Mels { get; init; } = 80;
    public int LmLayers { get; init; } = 12;
    public int LmHidden { get; init; } = 1024;
    public int LmHeads { get; init; } = 16;
    public int DurationLayers { get; init; } = 8;
    public int DurationHidden { get; init; } = 512;
    public int DurationHeads { get; init; } = 8;
    public int DurationCodes { get; init; } = 128;
    public int[] DiscriminatorWindows { get; init; } = { 32, 64, 128 };
    public int DiscriminatorHidden { get; init; } = 192;
}

/// <summary>
/// CVQ-VAE's quantizer (Zheng and Vedaldi 2023, <c>quantise.VectorQuantiser</c>), which Mega-TTS 2 adopts against codebook
/// collapse (Appendix A.3): the code of each vector by cosine similarity (or Euclidean distance), the codebook vector
/// passed straight through, the loss β·‖sg[z_q] − z‖² + ‖z_q − sg[z]‖² (means), and in training the dynamic
/// initialization: each code's average usage as a running mean (decay 0.99), and each code moved toward an anchor feature
/// by <c>exp(−usage · K · 10 / (1 − decay) − 1e-3)</c>, so rarely used codes move most.
/// </summary>
internal sealed class MegaTts2Quantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private double[]? _pool;
    private int _pooled;

    public MegaTts2Quantizer(IEngine engine, List<LayerBase<T>> layers, int codes, int dimension, MegaTts2CodeDistance distance,
        MegaTts2CodeAnchor anchor, double commitment, double decay)
    {
        _engine = engine;
        Codes = codes;
        Dimension = dimension;
        Distance = distance;
        Anchor = anchor;
        Commitment = commitment;
        Decay = decay;
        Codebook = new TiedEmbeddingLayer<T>(codes, dimension);
        layers.Add(Codebook);
        Usage = new double[codes];
    }

    public int Codes { get; }
    public int Dimension { get; }
    public MegaTts2CodeDistance Distance { get; }
    public MegaTts2CodeAnchor Anchor { get; }
    public double Commitment { get; }
    public double Decay { get; }
    public TiedEmbeddingLayer<T> Codebook { get; }

    /// <summary>The running mean of each code's share of the quantized vectors (<c>embed_prob</c>).</summary>
    public double[] Usage { get; }

    /// <summary>Starts the codebook as CVQ-VAE does, uniform in ±1/K, with no usage.</summary>
    public void Reset(Random random)
    {
        Codebook.Reinitialize(() => (2 * random.NextDouble() - 1) / Codes);
        Array.Clear(Usage, 0, Usage.Length);
        _pool = null;
        _pooled = 0;
    }

    private double[] Table()
    {
        Tensor<T> table;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            table = Codebook.Forward(new Tensor<T>(new[] { Codes }, new Vector<T>(Enumerable.Range(0, Codes).Select(i => NumOps.FromDouble(i)).ToArray())));
        return table.ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
    }

    // F.normalize: x / max(‖x‖, 1e-12).
    private static void Normalize(double[] v, int offset, int length, double[] into, int at)
    {
        double norm = 0;
        for (int d = 0; d < length; d++) norm += v[offset + d] * v[offset + d];
        norm = Math.Max(Math.Sqrt(norm), 1e-12);
        for (int d = 0; d < length; d++) into[at + d] = v[offset + d] / norm;
    }

    /// <summary>The similarities <c>[N, K]</c> of vectors <c>[N·d]</c> to the codes: cosines, or −‖z − e‖² written as
    /// the reference expands it.</summary>
    public double[,] Similarities(double[] z, int n)
    {
        var table = Table();
        int k = Codes, dim = Dimension;
        var d = new double[n, k];
        if (Distance == MegaTts2CodeDistance.Cosine)
        {
            var nz = new double[n * dim];
            var ne = new double[k * dim];
            for (int i = 0; i < n; i++) Normalize(z, i * dim, dim, nz, i * dim);
            for (int c = 0; c < k; c++) Normalize(table, c * dim, dim, ne, c * dim);
            for (int i = 0; i < n; i++)
                for (int c = 0; c < k; c++)
                {
                    double s = 0;
                    for (int j = 0; j < dim; j++) s += nz[i * dim + j] * ne[c * dim + j];
                    d[i, c] = s;
                }
            return d;
        }
        for (int i = 0; i < n; i++)
        {
            double zz = 0;
            for (int j = 0; j < dim; j++) zz += z[i * dim + j] * z[i * dim + j];
            for (int c = 0; c < k; c++)
            {
                double ee = 0, ze = 0;
                for (int j = 0; j < dim; j++)
                {
                    ee += table[c * dim + j] * table[c * dim + j];
                    ze += z[i * dim + j] * table[c * dim + j];
                }
                d[i, c] = -zz - ee + 2 * ze;
            }
        }
        return d;
    }

    /// <summary>The code of each row of similarities: the largest (the last of equal ones, as the reference's ascending
    /// sort takes).</summary>
    public static int[] Nearest(double[,] similarities)
    {
        int n = similarities.GetLength(0), k = similarities.GetLength(1);
        var codes = new int[n];
        for (int i = 0; i < n; i++)
        {
            int best = 0;
            for (int c = 1; c < k; c++)
                if (similarities[i, c] >= similarities[i, best]) best = c;
            codes[i] = best;
        }
        return codes;
    }

    /// <summary>The codes <c>[N]</c> of vectors <c>[1, N, d]</c>.</summary>
    public int[] Encode(Tensor<T> x)
    {
        int n = x.Shape[1];
        return Nearest(Similarities(x.ToArray().Select(v => NumOps.ToDouble(v)).ToArray(), n));
    }

    /// <summary>The codebook vectors <c>[1, N, d]</c> of codes.</summary>
    public Tensor<T> Lookup(IReadOnlyList<int> codes)
    {
        var ids = new Tensor<T>(new[] { codes.Count }, new Vector<T>(codes.Select(c => NumOps.FromDouble(c)).ToArray()));
        return _engine.Reshape(Codebook.Forward(ids), new[] { 1, codes.Count, Dimension });
    }

    /// <summary>Quantizes <c>[1, N, d]</c>: the straight-through vectors, the codes, the loss <c>[1]</c>; in training also
    /// the dynamic initialization of the codebook (after the lookup, before the step, as the reference orders it).</summary>
    public (Tensor<T> Quantized, int[] Codes, Tensor<T> Loss) Forward(Tensor<T> x, bool training, Random random)
    {
        int n = x.Shape[1];
        var z = x.ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
        var similarities = Similarities(z, n);
        var codes = Nearest(similarities);
        var zq = Lookup(codes);
        var stopX = _engine.StopGradient(x);
        var stopQ = _engine.StopGradient(zq);
        Tensor<T> Mean(Tensor<T> a, Tensor<T> b)
        {
            var d = _engine.TensorSubtract(a, b);
            var sum = _engine.ReduceSum(_engine.TensorMultiply(d, d), new[] { 0, 1, 2 }, false);
            return _engine.TensorMultiplyScalar(_engine.Reshape(sum, new[] { 1 }), NumOps.FromDouble(1.0 / d.Length));
        }
        var loss = _engine.TensorAdd(_engine.TensorMultiplyScalar(Mean(stopQ, x), NumOps.FromDouble(Commitment)), Mean(zq, stopX));
        var quantized = _engine.TensorAdd(x, _engine.StopGradient(_engine.TensorSubtract(zq, x)));
        if (training) Reinitialize(z, n, similarities, codes, random);
        return (quantized, codes, loss);
    }

    /// <summary>CVQ-VAE's online clustered reinitialization of the codebook from a step's features <c>[N·d]</c>.</summary>
    internal void Reinitialize(double[] z, int n, double[,] similarities, int[] codes, Random random)
    {
        int k = Codes, dim = Dimension;
        var share = new double[k];
        foreach (int c in codes) share[c] += 1.0 / n;
        for (int c = 0; c < k; c++) Usage[c] = Usage[c] * Decay + share[c] * (1 - Decay);
        var anchors = new double[k * dim];
        if (Anchor == MegaTts2CodeAnchor.Closest)
        {
            for (int c = 0; c < k; c++)
            {
                int best = 0;
                for (int i = 1; i < n; i++)
                    if (similarities[i, c] >= similarities[best, c]) best = i;
                Array.Copy(z, best * dim, anchors, c * dim, dim);
            }
        }
        else if (Anchor == MegaTts2CodeAnchor.ProbabilisticRandom)
        {
            for (int c = 0; c < k; c++)
            {
                double max = double.NegativeInfinity;
                for (int i = 0; i < n; i++) max = Math.Max(max, similarities[i, c]);
                var weights = new double[n];
                for (int i = 0; i < n; i++) weights[i] = Math.Exp(similarities[i, c] - max);
                double u = random.NextDouble() * weights.Sum();
                int pick = n - 1;
                for (int i = 0; i < n; i++)
                {
                    u -= weights[i];
                    if (u < 0) { pick = i; break; }
                }
                Array.Copy(z, pick * dim, anchors, c * dim, dim);
            }
        }
        else
        {
            anchors = Query(z, n, random);
        }
        var table = Table();
        for (int c = 0; c < k; c++)
        {
            double move = Math.Exp(-(Usage[c] * k * 10) / (1 - Decay) - 1e-3);
            for (int d = 0; d < dim; d++) table[c * dim + d] = table[c * dim + d] * (1 - move) + anchors[c * dim + d] * move;
        }
        Codebook.LoadTable(table);
    }

    // CVQ-VAE's FeaturePool: K stored features, uniform in ±1/K until filled; a step with more features than K replaces
    // them all by K random draws, otherwise its features fill the free slots, then random slots.
    private double[] Query(double[] z, int n, Random random)
    {
        int k = Codes, dim = Dimension;
        if (_pool is null)
        {
            _pool = new double[k * dim];
            for (int i = 0; i < _pool.Length; i++) _pool[i] = (random.NextDouble() * 2 - 1) / k;
        }
        if (n > k)
        {
            for (int c = 0; c < k; c++) Array.Copy(z, random.Next(n) * dim, _pool, c * dim, dim);
            _pooled = k;
        }
        else if (_pooled < k)
        {
            int count = Math.Min(n, k - _pooled);
            Array.Copy(z, 0, _pool, _pooled * dim, count * dim);
            _pooled += count;
        }
        else
        {
            var slots = Enumerable.Range(0, k).OrderBy(_ => random.Next()).Take(n).ToArray();
            for (int i = 0; i < n; i++) Array.Copy(z, i * dim, _pool, slots[i] * dim, dim);
        }
        return (double[])_pool.Clone();
    }
}

/// <summary>
/// fairseq's multi-head attention between widths, as NATSpeech's <c>MultiheadAttention(embed_dim, heads, kdim, vdim)</c>
/// builds it: separate query, key and value projections sharing one bias, queries scaled by the head width's inverse
/// square root, and an output projection with a bias.
/// </summary>
internal sealed class MegaTts2CrossAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTts2CrossAttention(IEngine engine, List<LayerBase<T>> layers, int width, int keyWidth, int heads)
    {
        if (width % heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(heads));
        _engine = engine;
        Width = width;
        KeyWidth = keyWidth;
        Heads = heads;
        Query = MegaTts3Ops.Linear(layers, width);
        Key = MegaTts3Ops.Linear(layers, width);
        Value = MegaTts3Ops.Linear(layers, width);
        Output = MegaTts3Ops.Linear(layers, width);
    }

    public int Width { get; }
    public int KeyWidth { get; }
    public int Heads { get; }
    public DenseLayer<T> Query { get; }
    public DenseLayer<T> Key { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> Output { get; }

    /// <summary>Queries <c>[1, L, width]</c> attending to keys and values <c>[1, S, keyWidth]</c>: <c>[1, L, width]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> query, Tensor<T> memory)
    {
        int l = query.Shape[1], s = memory.Shape[1], h = Width, hd = h / Heads;
        Tensor<T> Split(Tensor<T> x, int t) => _engine.TensorPermute(_engine.Reshape(x, new[] { t, Heads, hd }), new[] { 1, 0, 2 });
        var q = _engine.TensorMultiplyScalar(Split(MegaTts3Ops.Rows(_engine, Query, query), l), NumOps.FromDouble(Math.Pow(hd, -0.5)));
        var k = Split(MegaTts3Ops.Rows(_engine, Key, memory), s);
        var v = Split(MegaTts3Ops.Rows(_engine, Value, memory), s);
        var context = _engine.BatchMatMul(_engine.Softmax(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), axis: 2), v);
        var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { 1, l, h });
        return MegaTts3Ops.Rows(_engine, Output, merged);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        int h = Width, kw = KeyWidth;
        var bias = read($"{prefix}.in_proj_bias");
        TorchParameters.Linear(_engine, Query, h, h, read($"{prefix}.q_proj_weight"), bias.Take(h).ToArray());
        TorchParameters.Linear(_engine, Key, kw, h, read($"{prefix}.k_proj_weight"), bias.Skip(h).Take(h).ToArray());
        TorchParameters.Linear(_engine, Value, kw, h, read($"{prefix}.v_proj_weight"), bias.Skip(2 * h).ToArray());
        MegaTts3Ops.LoadDense(_engine, read, Output, $"{prefix}.out_proj", h, h);
    }
}

/// <summary>
/// The decoder-only Transformer of the authors' <c>CodePredictor</c> (MegaTTS 3's release, <c>ar_dur_predictor.py</c>),
/// which Mega-TTS 2's P-LLM and autoregressive duration model share (§3.3): token embeddings (the padding row 0 left at
/// zero) plus the projected linguistic features, pre-norm layers of causal self-attention with rotary positions (packed
/// projections without biases) and a GELU feed-forward of 4× width, a final layer norm and an output layer. Positions
/// restart and attention stays within each speaker's sequence (the paper's speaker-level attention masks).
/// </summary>
internal sealed class MegaTts2CodeDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Layer(LayerNormalizationLayer<T> Norm1, BiasFreeLinearLayer<T> InProjection, BiasFreeLinearLayer<T> OutProjection,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> Ffn1, DenseLayer<T> Ffn2);

    public MegaTts2CodeDecoder(IEngine engine, List<LayerBase<T>> layers, int vocabulary, int linguistic, int width, int heads, int layerCount,
        int outputs, bool regression)
    {
        if (width % heads != 0 || (width / heads) % 2 != 0)
            throw new ArgumentException("The width must divide into heads of even width.", nameof(heads));
        _engine = engine;
        Vocabulary = vocabulary;
        Linguistic = linguistic;
        Width = width;
        Heads = heads;
        Outputs = outputs;
        Regression = regression;
        Embedding = new TiedEmbeddingLayer<T>(vocabulary, width);
        layers.Add(Embedding);
        Projection = MegaTts3Ops.Linear(layers, width);
        for (int i = 0; i < layerCount; i++)
        {
            var n1 = new LayerNormalizationLayer<T>(width, 1e-5);
            var inProjection = new BiasFreeLinearLayer<T>(width, 3 * width);
            var outProjection = new BiasFreeLinearLayer<T>(width, width);
            var n2 = new LayerNormalizationLayer<T>(width, 1e-5);
            layers.Add(n1);
            layers.Add(inProjection);
            layers.Add(outProjection);
            layers.Add(n2);
            Layers.Add(new Layer(n1, inProjection, outProjection, n2, MegaTts3Ops.Linear(layers, 4 * width), MegaTts3Ops.Linear(layers, width)));
        }
        FinalNorm = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(FinalNorm);
        Output = MegaTts3Ops.Linear(layers, outputs);
    }

    public int Vocabulary { get; }
    public int Linguistic { get; }
    public int Width { get; }
    public int Heads { get; }
    public int Outputs { get; }
    public bool Regression { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public DenseLayer<T> Projection { get; }
    public List<Layer> Layers { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public DenseLayer<T> Output { get; }

    // The decoder's inputs [L, width]: each token's embedding (zero for the padding token 0) plus the projected features.
    private Tensor<T> Inputs(IReadOnlyList<int> tokens, Tensor<T> linguistic)
    {
        int l = tokens.Count;
        var ids = new Tensor<T>(new[] { l });
        var keep = new Tensor<T>(new[] { l, 1 });
        for (int i = 0; i < l; i++)
        {
            if (tokens[i] < 0 || tokens[i] >= Vocabulary) throw new ArgumentOutOfRangeException(nameof(tokens), $"Tokens lie in [0, {Vocabulary}).");
            ids[i] = NumOps.FromDouble(tokens[i]);
            if (tokens[i] != 0) keep[i, 0] = NumOps.One;
        }
        var embedded = _engine.TensorMultiply(Embedding.Forward(ids), _engine.TensorBroadcastTo(keep, new[] { l, Width }));
        return _engine.TensorAdd(embedded, Projection.Forward(linguistic));
    }

    // RotaryEmbeddings: rotate-half with cos and sin of position · 10000^(−2k/width), the angles repeated for both halves.
    private Tensor<T> Rotate(Tensor<T> x, IReadOnlyList<int> positions)
    {
        int heads = x.Shape[0], l = x.Shape[1], hd = x.Shape[2], half = hd / 2;
        var cos = new Tensor<T>(new[] { 1, l, hd });
        var sin = new Tensor<T>(new[] { 1, l, hd });
        for (int j = 0; j < l; j++)
            for (int k = 0; k < half; k++)
            {
                double angle = positions[j] * Math.Pow(10000.0, -2.0 * k / hd);
                cos[0, j, k] = cos[0, j, k + half] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, j, k] = sin[0, j, k + half] = NumOps.FromDouble(Math.Sin(angle));
            }
        var first = _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { heads, l, half });
        var second = _engine.TensorSlice(x, new[] { 0, 0, half }, new[] { heads, l, half });
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 2);
        var shape = new[] { heads, l, hd };
        return _engine.TensorAdd(_engine.TensorMultiply(_engine.TensorBroadcastTo(cos, shape), x), _engine.TensorMultiply(_engine.TensorBroadcastTo(sin, shape), rotated));
    }

    // The layers over inputs [L, width]. Without a cache the keys are the queries; with one they are every cached position
    // and the new ones, the new tokens at the positions after the cached ones. A key later than its query, or of another
    // group, is masked (−1e8, as the reference clamps it).
    private Tensor<T> Decode(Tensor<T> x, IReadOnlyList<int> positions, IReadOnlyList<int>? groups, MegaTts3DurationCache<T>? cache)
    {
        int l = x.Shape[0], h = Width, heads = Heads, hd = h / heads;
        if (cache is not null)
            while (cache.Layers.Count < Layers.Count) cache.Layers.Add(null);
        int cached = cache?.Layers[0]?.Keys.Shape[1] ?? 0, keyCount = cached + l;
        Tensor<T>? mask = null;
        if (l > 1)
        {
            mask = new Tensor<T>(new[] { 1, l, keyCount });
            for (int a = 0; a < l; a++)
                for (int b = 0; b < keyCount; b++)
                    if (b > cached + a || (groups is not null && groups[a] != groups[b])) mask[0, a, b] = NumOps.FromDouble(-1e8);
        }
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var qkv = layer.InProjection.Forward(layer.Norm1.Forward(x));
            Tensor<T> Part(int p) => _engine.TensorPermute(_engine.Reshape(_engine.TensorSlice(qkv, new[] { 0, p * h }, new[] { l, h }),
                new[] { l, heads, hd }), new[] { 1, 0, 2 });
            var q = Rotate(_engine.TensorMultiplyScalar(Part(0), NumOps.FromDouble(1.0 / Math.Sqrt(hd))), positions);
            var k = Part(1);
            var v = Part(2);
            IReadOnlyList<int> keyPositions = positions;
            if (cache is not null)
            {
                if (cache.Layers[i] is { } held)
                {
                    k = _engine.TensorConcatenate(new[] { held.Keys, k }, 1);
                    v = _engine.TensorConcatenate(new[] { held.Values, v }, 1);
                }
                cache.Layers[i] = (k, v);
                keyPositions = Enumerable.Range(0, k.Shape[1]).ToArray();
            }
            int keys = k.Shape[1];
            var scores = _engine.BatchMatMul(q, _engine.TensorPermute(Rotate(k, keyPositions), new[] { 0, 2, 1 }));
            if (mask is not null) scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(mask, new[] { heads, l, keys }));
            var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
            x = _engine.TensorAdd(x, layer.OutProjection.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { l, h })));
            x = _engine.TensorAdd(x, layer.Ffn2.Forward(ActivationHelper.ExactGelu(layer.Ffn1.Forward(layer.Norm2.Forward(x)), _engine)));
        }
        var y = Output.Forward(FinalNorm.Forward(x));
        return Regression ? _engine.Reshape(_engine.Softplus(y), new[] { l }) : y;
    }

    /// <summary>The outputs (<c>[L, outputs]</c>, or the durations <c>[L]</c>) for input tokens and their features
    /// <c>[L, linguistic]</c> at rotary positions, attending within each group.</summary>
    public Tensor<T> Forward(IReadOnlyList<int> tokens, Tensor<T> linguistic, IReadOnlyList<int> positions, IReadOnlyList<int>? groups) =>
        Decode(Inputs(tokens, linguistic), positions, groups, null);

    /// <summary>Continues an incremental state (positions 0, 1, … in order) with tokens and their features
    /// <c>[n, linguistic]</c>: their outputs, each attending to the state and the tokens before it.</summary>
    public Tensor<T> Extend(IReadOnlyList<int> tokens, Tensor<T> linguistic, MegaTts3DurationCache<T> cache)
    {
        int cached = cache.Layers.Count > 0 ? cache.Layers[0]?.Keys.Shape[1] ?? 0 : 0;
        return Decode(Inputs(tokens, linguistic), Enumerable.Range(cached, tokens.Count).ToArray(), null, cache);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        int h = Width;
        Embedding.LoadTable(read($"{prefix}.code_emb.weight"));
        MegaTts3Ops.LoadDense(_engine, read, Projection, $"{prefix}.enc_proj", Linguistic, h);
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string p = $"{prefix}.layers.{i}.op";
            TorchParameters.LayerNorm(_engine, layer.Norm1, read($"{p}.layer_norm1.weight"), read($"{p}.layer_norm1.bias"));
            TorchParameters.Linear(layer.InProjection, h, 3 * h, read($"{p}.self_attn.in_proj_weight"));
            TorchParameters.Linear(layer.OutProjection, h, h, read($"{p}.self_attn.out_proj.weight"));
            TorchParameters.LayerNorm(_engine, layer.Norm2, read($"{p}.layer_norm2.weight"), read($"{p}.layer_norm2.bias"));
            MegaTts3Ops.LoadDense(_engine, read, layer.Ffn1, $"{p}.ffn.ffn_1.1", h, 4 * h);
            MegaTts3Ops.LoadDense(_engine, read, layer.Ffn2, $"{p}.ffn.ffn_2", 4 * h, h);
        }
        TorchParameters.LayerNorm(_engine, FinalNorm, read($"{prefix}.layer_norm.weight"), read($"{prefix}.layer_norm.bias"));
        MegaTts3Ops.LoadDense(_engine, read, Output, Regression ? $"{prefix}.project_out_dim.0" : $"{prefix}.project_out_dim", h, Outputs);
    }
}

/// <summary>
/// Mega-TTS 2 (Jiang et al. 2024): the compressive acoustic autoencoder (a content encoder; a VQ prosody encoder that
/// pools the mel spectrogram r = 8 times into CVQ-VAE codes; the multi-reference timbre encoder, whose convolutions
/// downsample the joined prompt mel spectrograms d = 16 times for the content's attention; a mel decoder; the multi-length
/// discriminator), and the second stage's prosody latent language model and autoregressive duration model.
/// </summary>
internal sealed class MegaTts2Network<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTts2Network(IEngine engine, List<LayerBase<T>> acoustic, List<LayerBase<T>> lm, List<LayerBase<T>> duration,
        List<LayerBase<T>> discriminator, MegaTts2Configuration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Content = new MegaTtsContentEncoder<T>(engine, acoustic, c.PhoneVocabulary, c.Hidden, c.ContentLayers, c.ContentHeads, c.ContentKernel, c.ContentFilter);
        ProsodyPrenet = MegaTts3Ops.Linear(acoustic, c.ProsodyHidden);
        ProsodyFrames = new MegaTtsConvBlocks<T>(engine, acoustic, c.ProsodyHidden, c.ProsodyHidden, c.ProsodyKernel, c.ProsodyBlocks);
        ProsodyGroups = new MegaTtsConvBlocks<T>(engine, acoustic, c.ProsodyHidden, c.CodeDimension, c.ProsodyKernel, c.ProsodyBlocks);
        Quantizer = new MegaTts2Quantizer<T>(engine, acoustic, c.Codes, c.CodeDimension, c.Distance, c.Anchor, c.CommitmentWeight, c.UsageDecay);
        ProsodyOut = MegaTts3Ops.Linear(acoustic, c.Hidden);
        TimbrePrenet = MegaTts3Ops.Linear(acoustic, c.TimbreHidden);
        TimbreFrames = new MegaTtsConvBlocks<T>(engine, acoustic, c.TimbreHidden, c.TimbreHidden, c.TimbreKernel, c.TimbreBlocks);
        TimbreGroups = new MegaTtsConvBlocks<T>(engine, acoustic, c.TimbreHidden, c.TimbreHidden, c.TimbreKernel, c.TimbreBlocks);
        TimbreAttention = new MegaTts2CrossAttention<T>(engine, acoustic, c.Hidden, c.TimbreHidden, c.TimbreHeads);
        Decoder = new MegaTtsConvBlocks<T>(engine, acoustic, c.Hidden, c.Mels, c.DecoderKernel, c.DecoderBlocks);
        ProsodyModel = new MegaTts2CodeDecoder<T>(engine, lm, c.Codes + 2, c.Hidden, c.LmHidden, c.LmHeads, c.LmLayers, c.Codes + 1, regression: false);
        DurationModel = new MegaTts2CodeDecoder<T>(engine, duration, c.DurationCodes + 2, c.Hidden, c.DurationHidden, c.DurationHeads, c.DurationLayers, 1,
            regression: true);
        Discriminator = new MegaTtsMelDiscriminator<T>(engine, discriminator, c.DiscriminatorWindows, c.Mels, c.DiscriminatorHidden);
    }

    public MegaTts2Configuration Configuration { get; }
    public MegaTtsContentEncoder<T> Content { get; }
    public DenseLayer<T> ProsodyPrenet { get; }
    public MegaTtsConvBlocks<T> ProsodyFrames { get; }
    public MegaTtsConvBlocks<T> ProsodyGroups { get; }
    public MegaTts2Quantizer<T> Quantizer { get; }
    public DenseLayer<T> ProsodyOut { get; }
    public DenseLayer<T> TimbrePrenet { get; }
    public MegaTtsConvBlocks<T> TimbreFrames { get; }
    public MegaTtsConvBlocks<T> TimbreGroups { get; }
    public MegaTts2CrossAttention<T> TimbreAttention { get; }
    public MegaTtsConvBlocks<T> Decoder { get; }
    public MegaTts2CodeDecoder<T> ProsodyModel { get; }
    public MegaTts2CodeDecoder<T> DurationModel { get; }
    public MegaTtsMelDiscriminator<T> Discriminator { get; }

    /// <summary>The P-LLM's start token; code c is token c + 1, and 0 the end of a sentence.</summary>
    public int StartToken => Configuration.Codes + 1;

    /// <summary>The duration model's start token; a duration d is token min(d, <see cref="MegaTts2Configuration.DurationCodes"/>).</summary>
    public int DurationStart => Configuration.DurationCodes + 1;

    private static Tensor<T> Ones(int frames)
    {
        var ones = new Tensor<T>(new[] { 1, frames, 1 });
        for (int f = 0; f < frames; f++) ones[0, f, 0] = NumOps.One;
        return ones;
    }

    /// <summary>Max pooling over time of <c>[1, T, C]</c> in windows of <paramref name="r"/>, the last window partial
    /// (<c>max_pool1d(ceil_mode=True)</c>): <c>[1, ⌈T/r⌉, C]</c>.</summary>
    public Tensor<T> MaxPool(Tensor<T> x, int r)
    {
        int t = x.Shape[1], c = x.Shape[2], n = (t + r - 1) / r;
        if (n * r > t)
        {
            // Repeating the last frame leaves each window's maximum that of its frames.
            var last = _engine.TensorSlice(x, new[] { 0, t - 1, 0 }, new[] { 1, 1, c });
            x = _engine.TensorConcatenate(new[] { x, _engine.TensorBroadcastTo(last, new[] { 1, n * r - t, c }) }, 1);
        }
        return _engine.ReduceMax(_engine.Reshape(x, new[] { 1, n, r, c }), new[] { 2 }, false);
    }

    /// <summary>The VQ encoder's hidden states <c>[1, ⌈F/r⌉, codeDim]</c> of a mel spectrogram <c>[F, mels]</c>: a linear
    /// prenet, the first stack, max pooling by r, the second stack.</summary>
    public Tensor<T> ProsodyHidden(Tensor<T> mel)
    {
        var c = Configuration;
        int frames = mel.Shape[0];
        var x = MegaTts3Ops.Rows(_engine, ProsodyPrenet, _engine.Reshape(mel, new[] { 1, frames, c.Mels }));
        var pooled = MaxPool(ProsodyFrames.Forward(x, Ones(frames)), c.Compression);
        return ProsodyGroups.Forward(pooled, Ones(pooled.Shape[1]));
    }

    /// <summary>The prosody of quantized vectors <c>[1, n, codeDim]</c> at the frames: projected, each repeated r times,
    /// cut to <paramref name="frames"/>: <c>[1, F, h]</c>.</summary>
    public Tensor<T> ProsodyAtFrames(Tensor<T> quantized, int frames)
    {
        int n = quantized.Shape[1], r = Configuration.Compression;
        var repeat = new Tensor<T>(new[] { 1, frames, n });
        for (int f = 0; f < frames; f++) repeat[0, f, f / r] = NumOps.One;
        return _engine.BatchMatMul(repeat, MegaTts3Ops.Rows(_engine, ProsodyOut, quantized));
    }

    /// <summary>The MRTE's acoustic states <c>[1, ⌈F/d⌉, timbre]</c> of the joined prompt mel spectrograms <c>[F, mels]</c>.</summary>
    public Tensor<T> TimbreStates(Tensor<T> referenceMel)
    {
        var c = Configuration;
        int frames = referenceMel.Shape[0];
        var x = MegaTts3Ops.Rows(_engine, TimbrePrenet, _engine.Reshape(referenceMel, new[] { 1, frames, c.Mels }));
        var pooled = MaxPool(TimbreFrames.Forward(x, Ones(frames)), c.Downsampling);
        return TimbreGroups.Forward(pooled, Ones(pooled.Shape[1]));
    }

    /// <summary>The timbre-to-content attention: the content <c>[1, L, h]</c> attending to the acoustic states.</summary>
    public Tensor<T> Timbre(Tensor<T> content, Tensor<T> timbreStates) => TimbreAttention.Forward(content, timbreStates);

    /// <summary>The frame expansion <c>[1, F, L]</c> of a 1-based frame map (frames of no phoneme read nothing).</summary>
    public static Tensor<T> Expansion(int[] mel2ph, int phones)
    {
        var expand = new Tensor<T>(new[] { 1, mel2ph.Length, phones });
        for (int f = 0; f < mel2ph.Length; f++)
            if (mel2ph[f] > 0) expand[0, f, mel2ph[f] - 1] = NumOps.One;
        return expand;
    }

    /// <summary>The decoder's mel spectrogram <c>[F, mels]</c>: the content and the MRTE's timbre expanded to the frames by
    /// the length regulator, plus the prosody at the frames.</summary>
    public Tensor<T> DecodeMel(Tensor<T> content, Tensor<T> timbre, Tensor<T> prosody, int[] mel2ph)
    {
        int frames = mel2ph.Length;
        var x = _engine.TensorAdd(_engine.BatchMatMul(Expansion(mel2ph, content.Shape[1]), _engine.TensorAdd(content, timbre)), prosody);
        return _engine.Reshape(Decoder.Forward(x, Ones(frames)), new[] { frames, Configuration.Mels });
    }

    /// <summary>The P-LLM's content <c>[⌈F/r⌉, h]</c>: the content expanded to the frames and max pooled by r (§3.3).</summary>
    public Tensor<T> PooledContent(Tensor<T> content, int[] mel2ph)
    {
        var frames = _engine.BatchMatMul(Expansion(mel2ph, content.Shape[1]), content);
        var pooled = MaxPool(frames, Configuration.Compression);
        return _engine.Reshape(pooled, new[] { pooled.Shape[1], Configuration.Hidden });
    }

    /// <summary>Loads a state dict in the module names of <c>tools/reference-data/megatts2_model_reference.py</c>.</summary>
    public void LoadTorch(Func<string, double[]> read)
    {
        var c = Configuration;
        Content.LoadTorch(_engine, read, "content");
        MegaTts3Ops.LoadDense(_engine, read, ProsodyPrenet, "prosody_prenet", c.Mels, c.ProsodyHidden);
        ProsodyFrames.LoadTorch(_engine, read, "prosody_frames");
        ProsodyGroups.LoadTorch(_engine, read, "prosody_groups");
        Quantizer.Codebook.LoadTable(read("codebook.weight"));
        MegaTts3Ops.LoadDense(_engine, read, ProsodyOut, "prosody_out", c.CodeDimension, c.Hidden);
        MegaTts3Ops.LoadDense(_engine, read, TimbrePrenet, "timbre_prenet", c.Mels, c.TimbreHidden);
        TimbreFrames.LoadTorch(_engine, read, "timbre_frames");
        TimbreGroups.LoadTorch(_engine, read, "timbre_groups");
        TimbreAttention.LoadTorch(read, "timbre_attention");
        Decoder.LoadTorch(_engine, read, "decoder");
        ProsodyModel.LoadTorch(read, "plm");
        DurationModel.LoadTorch(read, "dur");
        Discriminator.LoadTorch(read, "disc");
    }
}
