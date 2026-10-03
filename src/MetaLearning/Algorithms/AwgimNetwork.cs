using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// The AWGIM weight generator (Guo &amp; Cheung, CVPR 2020) over one episode's embeddings, shared by the
/// algorithm's training objective and the adapted model's prediction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Notation: p is the embedding width, d the latent width, S the support rows, Q the query rows and c the
/// classes. The structure follows the authors' model.py:
/// </para>
/// <list type="number">
/// <item>Two linear encoders p→d: CA for the attentive path (support and query) and SA for the contextual
/// path (support).</item>
/// <item>Contextual path: self-attention over the SA support codes gives each support row a task-aware
/// code.</item>
/// <item>Attentive path: self-attention over the CA support codes, averaged per class and spread back to
/// that class's rows, is the value; the CA support codes are the keys; each CA query code attends to
/// them.</item>
/// <item>For every (query, support row) pair the decoder maps [contextual code; attentive code] to a mean
/// and a scale of a p-wide weight vector, std = exp(u) − (1 − √(2/(c+p))). Training samples, evaluation
/// takes the mean. Averaging over each class's rows gives every query its own c×p classifier.</item>
/// </list>
/// <para>
/// Attention blocks follow the reference: per-head projections whose outputs are summed (equivalently, a
/// concatenation projected by one matrix), a residual and LayerNorm, then a ReLU FFN of width 2d, a residual
/// and LayerNorm. LayerNorm normalizes each row over its features, as the paper's transformer blocks
/// specify. Dropout applies to every MLP input in training.
/// </para>
/// </remarks>
internal sealed class AwgimNetwork<T>
{
    private const double LayerNormEpsilon = 1e-12;
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly int _embedding;
    private readonly int _latent;
    private readonly int _heads;
    private readonly int _decoderLayers;
    private readonly List<(string Name, int Rows, int Columns)> _shapes = new();

    public AwgimNetwork(int embeddingDimension, int latentDimension, int heads, int decoderLayers)
    {
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        if (latentDimension <= 0) throw new ArgumentOutOfRangeException(nameof(latentDimension));
        if (heads <= 0 || latentDimension % heads != 0)
            throw new ArgumentException($"LatentDimension ({latentDimension}) must be divisible by NumHeads ({heads}).", nameof(heads));
        if (decoderLayers < 1) throw new ArgumentOutOfRangeException(nameof(decoderLayers));

        _embedding = embeddingDimension;
        _latent = latentDimension;
        _heads = heads;
        _decoderLayers = decoderLayers;

        int d = latentDimension, p = embeddingDimension;
        Declare("encoder.ca", d, p);
        Declare("encoder.sa", d, p);
        foreach (var block in new[] { "context_sa", "context_ca", "query_ca" })
        {
            Declare(block + ".q", d, d);
            Declare(block + ".k", d, d);
            Declare(block + ".v", d, d);
            Declare(block + ".o", d, d);
            Declare(block + ".ln1.gamma", 1, d);
            Declare(block + ".ln1.beta", 1, d);
            Declare(block + ".ffn1", 2 * d, d);
            Declare(block + ".ffn2", d, 2 * d);
            Declare(block + ".ln2.gamma", 1, d);
            Declare(block + ".ln2.beta", 1, d);
        }

        DeclareMlp("decoder", 2 * d, 2 * p);
        DeclareMlp("reconstruct.context", p, d);
        DeclareMlp("reconstruct.query", p, d);
    }

    /// <summary>The number of trainable values.</summary>
    public int ParameterCount => _shapes.Sum(s => s.Rows * s.Columns);

    /// <summary>
    /// Initial weights: Glorot-uniform dense layers, N(0, 1/d) attention projections, LayerNorm gain 1 and
    /// bias 0, matching the reference's TensorFlow defaults.
    /// </summary>
    public Vector<T> Initialize(Random random)
    {
        var weights = new Vector<T>(ParameterCount);
        int offset = 0;
        foreach (var (name, rows, columns) in _shapes)
        {
            int count = rows * columns;
            for (int i = 0; i < count; i++)
            {
                double value;
                if (name.EndsWith(".gamma", StringComparison.Ordinal)) value = 1.0;
                else if (name.EndsWith(".beta", StringComparison.Ordinal)) value = 0.0;
                else if (name.EndsWith(".q", StringComparison.Ordinal) || name.EndsWith(".k", StringComparison.Ordinal)
                    || name.EndsWith(".v", StringComparison.Ordinal) || name.EndsWith(".o", StringComparison.Ordinal))
                    value = random.NextGaussian() / Math.Sqrt(_latent);
                else
                {
                    double bound = Math.Sqrt(6.0 / (rows + columns));
                    value = (2.0 * random.NextDouble() - 1.0) * bound;
                }

                weights[offset + i] = Ops.FromDouble(value);
            }

            offset += count;
        }

        return weights;
    }

    /// <summary>The flat-vector range of every declared tensor whose name starts with <paramref name="prefix"/>.</summary>
    internal (int Offset, int Length) RangeOf(string prefix)
    {
        int offset = 0, start = -1, length = 0;
        foreach (var (name, rows, columns) in _shapes)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
            {
                if (start < 0) start = offset;
                length += rows * columns;
            }

            offset += rows * columns;
        }

        if (start < 0) throw new ArgumentException($"No AWGIM weights are named '{prefix}*'.", nameof(prefix));
        return (start, length);
    }

    /// <summary>Whether a parameter is a dense kernel (the tensors the reference's L2 fallback covers).</summary>
    public bool[] KernelMask()
    {
        var mask = new bool[ParameterCount];
        int offset = 0;
        foreach (var (name, rows, columns) in _shapes)
        {
            bool kernel = !name.EndsWith(".gamma", StringComparison.Ordinal) && !name.EndsWith(".beta", StringComparison.Ordinal);
            for (int i = 0; i < rows * columns; i++) mask[offset + i] = kernel;
            offset += rows * columns;
        }

        return mask;
    }

    /// <summary>Splits a flat weight vector into tape leaves, one per declared tensor.</summary>
    public Leaves CreateLeaves(Vector<T> weights)
    {
        if (weights.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} AWGIM weights, got {weights.Length}.", nameof(weights));
        var leaves = new Leaves();
        int offset = 0;
        foreach (var (name, rows, columns) in _shapes)
        {
            var leaf = new Tensor<T>(rows == 1 ? new[] { columns } : new[] { rows, columns });
            for (int i = 0; i < leaf.Length; i++) leaf[i] = weights[offset + i];
            leaves.Add(name, leaf, offset);
            offset += rows * columns;
        }

        return leaves;
    }

    /// <summary>
    /// Generates every query's classifier. Returns the per-query class weights <c>[Q, c, p]</c>, the sampled
    /// per-pair weights <c>[Q·S, p]</c>, and the two codes they were generated from, tiled per pair.
    /// </summary>
    public Generation Generate(Leaves w, Tensor<T> support, Tensor<T> query, Tensor<T> membership,
        Tensor<T> classAverager, Noise noise)
    {
        var engine = AiDotNetEngine.Current;
        int supportRows = support.Shape[0], queryRows = query.Shape[0], classes = classAverager.Shape[0];
        int d = _latent, p = _embedding;

        var supportCa = Linear(engine, noise.Drop(support), w["encoder.ca"]);
        var queryCa = Linear(engine, noise.Drop(query), w["encoder.ca"]);
        var supportSa = Linear(engine, noise.Drop(support), w["encoder.sa"]);

        // Contextual path: the support set attending to itself.
        var context = Attention(engine, w, "context_sa", supportSa, supportSa, supportSa, noise);

        // Attentive path: the class-averaged, self-attended support codes are the values each query reads.
        var supportValues = Attention(engine, w, "context_ca", supportCa, supportCa, supportCa, noise);
        var classValues = engine.TensorMatMul(classAverager, supportValues);
        var valuesPerRow = engine.TensorMatMul(engine.TensorTranspose(membership), classValues);
        var queryCode = Attention(engine, w, "query_ca", queryCa, supportCa, valuesPerRow, noise);

        // Every (query, support row) pair: [contextual code; attentive code] -> a weight distribution.
        var contextPairs = engine.TensorBroadcastTo(engine.Reshape(context, new[] { 1, supportRows, d }), new[] { queryRows, supportRows, d });
        var queryPairs = engine.TensorBroadcastTo(engine.Reshape(queryCode, new[] { queryRows, 1, d }), new[] { queryRows, supportRows, d });
        var contextRows = engine.Reshape(contextPairs, new[] { queryRows * supportRows, d });
        var queryRowsCode = engine.Reshape(queryPairs, new[] { queryRows * supportRows, d });
        var pairs = engine.TensorConcatenate(new[] { contextRows, queryRowsCode }, axis: 1);

        var decoded = Mlp(engine, w, "decoder", pairs, noise);
        var mean = engine.TensorNarrow(decoded, dim: 1, start: 0, length: p);
        var unscaled = engine.TensorNarrow(decoded, dim: 1, start: p, length: p);
        double offset = Math.Sqrt(2.0 / (classes + p));
        var scale = engine.TensorClamp(
            engine.TensorAddScalar(engine.TensorExp(unscaled), Ops.FromDouble(offset - 1.0)),
            Ops.FromDouble(1e-10), Ops.MaxValue);
        var sampled = noise.WeightNoise is null
            ? mean
            : engine.TensorAdd(mean, engine.TensorMultiply(scale, noise.Gaussian(queryRows * supportRows, p)));

        // Each query's classifier: its pair weights averaged over each class's support rows.
        var perQuery = engine.Reshape(sampled, new[] { queryRows, supportRows, p });
        var averager = engine.TensorBroadcastTo(engine.Reshape(classAverager, new[] { 1, classes, supportRows }), new[] { queryRows, classes, supportRows });
        var classWeights = engine.BatchMatMul(averager, perQuery);

        return new Generation(classWeights, sampled, contextRows, queryRowsCode);
    }

    /// <summary>Each query row scored against its own classifier: <c>[Q, c]</c>.</summary>
    public static Tensor<T> QueryLogits(IEngine engine, Tensor<T> query, Tensor<T> classWeights)
    {
        int queryRows = query.Shape[0], p = query.Shape[1], classes = classWeights.Shape[1];
        var rows = engine.Reshape(query, new[] { queryRows, 1, p });
        var logits = engine.BatchMatMul(rows, engine.TensorPermute(classWeights, new[] { 0, 2, 1 }));
        return engine.Reshape(logits, new[] { queryRows, classes });
    }

    /// <summary>Every support row scored by every query's classifier: <c>[Q·S, c]</c>.</summary>
    public static Tensor<T> SupportLogits(IEngine engine, Tensor<T> support, Tensor<T> classWeights)
    {
        int queryRows = classWeights.Shape[0], classes = classWeights.Shape[1];
        int supportRows = support.Shape[0], p = support.Shape[1];
        var rows = engine.TensorBroadcastTo(engine.Reshape(support, new[] { 1, supportRows, p }), new[] { queryRows, supportRows, p });
        var logits = engine.BatchMatMul(rows, engine.TensorPermute(classWeights, new[] { 0, 2, 1 }));
        return engine.Reshape(logits, new[] { queryRows * supportRows, classes });
    }

    /// <summary>
    /// The information-maximization surrogate: reconstruct a stop-gradient code from the sampled weights,
    /// squared error summed over features and averaged over pairs.
    /// </summary>
    public Tensor<T> Reconstruction(Leaves w, string network, Tensor<T> sampled, Tensor<T> code, Noise noise)
    {
        var engine = AiDotNetEngine.Current;
        var reconstructed = Mlp(engine, w, network, sampled, noise);
        var gap = engine.TensorSubtract(engine.StopGradient(code), reconstructed);
        var perPair = engine.ReduceSum(engine.TensorMultiply(gap, gap), new[] { 1 }, keepDims: false);
        return engine.Reshape(engine.ReduceMean(perPair, new[] { 0 }, keepDims: false), new[] { 1 });
    }

    private Tensor<T> Attention(IEngine engine, Leaves w, string block, Tensor<T> queries, Tensor<T> keys, Tensor<T> values, Noise noise)
    {
        int headWidth = _latent / _heads, nq = queries.Shape[0], nk = keys.Shape[0];
        var q = SplitHeads(engine, Linear(engine, queries, w[block + ".q"]), nq, headWidth);
        var k = SplitHeads(engine, Linear(engine, keys, w[block + ".k"]), nk, headWidth);
        var v = SplitHeads(engine, Linear(engine, values, w[block + ".v"]), nk, headWidth);

        var scores = engine.TensorMultiplyScalar(engine.BatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 })),
            Ops.FromDouble(1.0 / Math.Sqrt(headWidth)));
        var attended = engine.BatchMatMul(engine.TensorSoftmax(scores, axis: 2), v);
        var merged = engine.Reshape(engine.TensorPermute(attended, new[] { 1, 0, 2 }), new[] { nq, _latent });

        // Summed per-head output projections = one projection of the concatenated heads.
        var x = LayerNorm(engine, engine.TensorAdd(queries, Linear(engine, merged, w[block + ".o"])),
            w[block + ".ln1.gamma"], w[block + ".ln1.beta"]);
        var hidden = engine.ReLU(Linear(engine, noise.Drop(x), w[block + ".ffn1"]));
        var ffn = Linear(engine, noise.Drop(hidden), w[block + ".ffn2"]);
        return LayerNorm(engine, engine.TensorAdd(x, ffn), w[block + ".ln2.gamma"], w[block + ".ln2.beta"]);
    }

    private Tensor<T> SplitHeads(IEngine engine, Tensor<T> rows, int count, int headWidth)
        => engine.TensorPermute(engine.Reshape(rows, new[] { count, _heads, headWidth }), new[] { 1, 0, 2 });

    private Tensor<T> Mlp(IEngine engine, Leaves w, string network, Tensor<T> input, Noise noise)
    {
        var x = input;
        for (int layer = 0; layer < _decoderLayers; layer++)
        {
            x = Linear(engine, noise.Drop(x), w[$"{network}.{layer}"]);
            if (layer < _decoderLayers - 1) x = engine.ReLU(x);
        }

        return x;
    }

    private static Tensor<T> Linear(IEngine engine, Tensor<T> rows, Tensor<T> kernel)
        => engine.TensorMatMul(rows, engine.TensorTranspose(kernel));

    private static Tensor<T> LayerNorm(IEngine engine, Tensor<T> rows, Tensor<T> gamma, Tensor<T> beta)
    {
        var mean = engine.ReduceMean(rows, new[] { 1 }, keepDims: true);
        var centered = engine.TensorSubtract(rows, mean);
        var variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 1 }, keepDims: true);
        var normalized = engine.TensorDivide(centered,
            engine.TensorSqrt(engine.TensorAddScalar(variance, Ops.FromDouble(LayerNormEpsilon))));
        var shape = rows._shape;
        return engine.TensorAdd(
            engine.TensorMultiply(normalized, engine.TensorBroadcastTo(engine.Reshape(gamma, new[] { 1, shape[1] }), shape)),
            engine.TensorBroadcastTo(engine.Reshape(beta, new[] { 1, shape[1] }), shape));
    }

    private void Declare(string name, int rows, int columns) => _shapes.Add((name, rows, columns));

    private void DeclareMlp(string name, int input, int output)
    {
        int width = input;
        for (int layer = 0; layer < _decoderLayers; layer++)
        {
            int next = layer < _decoderLayers - 1 ? 2 * _latent : output;
            Declare($"{name}.{layer}", next, width);
            width = next;
        }
    }

    /// <summary>One episode's tape leaves, by name, with their offsets in the flat weight vector.</summary>
    internal sealed class Leaves
    {
        private readonly Dictionary<string, Tensor<T>> _byName = new(StringComparer.Ordinal);
        private readonly List<(Tensor<T> Leaf, int Offset)> _ordered = new();

        public Tensor<T> this[string name] => _byName[name];

        public IReadOnlyList<Tensor<T>> All => _ordered.Select(entry => entry.Leaf).ToList();

        public void Add(string name, Tensor<T> leaf, int offset)
        {
            _byName.Add(name, leaf);
            _ordered.Add((leaf, offset));
        }

        /// <summary>Writes each leaf's gradient into its slice of a flat gradient vector.</summary>
        public Vector<T> FlattenGradients(Dictionary<Tensor<T>, Tensor<T>> gradients, int length)
        {
            var flat = new Vector<T>(length);
            foreach (var (leaf, offset) in _ordered)
            {
                if (!gradients.TryGetValue(leaf, out var gradient)) continue;
                for (int i = 0; i < leaf.Length; i++) flat[offset + i] = gradient[i];
            }

            return flat;
        }

        /// <summary>The flat-vector ranges of each leaf, for per-variable norm clipping.</summary>
        public IEnumerable<(int Offset, int Length)> Ranges => _ordered.Select(entry => (entry.Offset, entry.Leaf.Length));
    }

    /// <summary>The generated classifiers and the codes they came from.</summary>
    internal sealed class Generation
    {
        public Generation(Tensor<T> classWeights, Tensor<T> sampled, Tensor<T> contextCode, Tensor<T> queryCode)
        {
            ClassWeights = classWeights;
            Sampled = sampled;
            ContextCode = contextCode;
            QueryCode = queryCode;
        }

        /// <summary>Each query's classifier, <c>[Q, c, p]</c>.</summary>
        public Tensor<T> ClassWeights { get; }

        /// <summary>The per-pair weights, sampled in training, <c>[Q·S, p]</c>.</summary>
        public Tensor<T> Sampled { get; }

        /// <summary>The contextual code of each pair's support row, <c>[Q·S, d]</c>.</summary>
        public Tensor<T> ContextCode { get; }

        /// <summary>The attentive code of each pair's query row, <c>[Q·S, d]</c>.</summary>
        public Tensor<T> QueryCode { get; }
    }

    /// <summary>Training randomness: inverted-dropout masks drawn per use, and the weight-sampling noise.</summary>
    internal sealed class Noise
    {
        private readonly Random? _random;
        private readonly double _dropout;

        private Noise(Random? random, double dropout, bool sampleWeights)
        {
            _random = random;
            _dropout = dropout;
            WeightNoise = sampleWeights ? random : null;
        }

        /// <summary>No dropout and the mean weights: evaluation and adapted-model prediction.</summary>
        public static Noise None { get; } = new Noise(null, 0, sampleWeights: false);

        /// <summary>Training: dropout at the given rate and sampled weights.</summary>
        public static Noise Training(Random random, double dropout) => new Noise(random, dropout, sampleWeights: true);

        /// <summary>The weight-sampling source, or null when weights are the mean.</summary>
        public Random? WeightNoise { get; }

        public Tensor<T> Drop(Tensor<T> rows)
        {
            if (_random is null || _dropout <= 0) return rows;
            var mask = new Tensor<T>(rows._shape);
            T keep = Ops.FromDouble(1.0 / (1.0 - _dropout));
            for (int i = 0; i < mask.Length; i++) mask[i] = _random.NextDouble() >= _dropout ? keep : Ops.Zero;
            return AiDotNetEngine.Current.TensorMultiply(rows, mask);
        }

        public Tensor<T> Gaussian(int rows, int columns)
        {
            var random = WeightNoise ?? throw new InvalidOperationException("Weights are not sampled outside training.");
            var noise = new Tensor<T>(new[] { rows, columns });
            for (int i = 0; i < noise.Length; i++) noise[i] = Ops.FromDouble(random.NextGaussian());
            return noise;
        }
    }
}
