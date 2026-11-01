using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Shared pieces of NaturalSpeech 2's networks: torch's initializations, layout changes and embeddings.</summary>
internal static class Ns2Ops
{
    private static readonly INumericOperations<double> D = MathHelper.GetNumericOperations<double>();

    /// <summary>N(0, std²) draws.</summary>
    public static double[] Normal(Random random, int count, double std) => IndexTtsOps.Normal(random, count, std);

    /// <summary><c>xavier_uniform_</c>: U(±√(6 / (fan_in + fan_out))).</summary>
    public static double[] Xavier(Random random, int count, int fanIn, int fanOut)
    {
        double bound = Math.Sqrt(6.0 / (fanIn + fanOut));
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (2 * random.NextDouble() - 1) * bound;
        return values;
    }

    /// <summary><c>kaiming_normal_</c> (fan_in, ReLU gain): N(0, 2 / fan_in).</summary>
    public static double[] KaimingNormal(Random random, int count, int fanIn) => Normal(random, count, Math.Sqrt(2.0 / fanIn));

    /// <summary>An <c>nn.Linear</c> with torch's default initialization, or the given weight (and bias).</summary>
    public static DenseLayer<T> Linear<T>(IEngine engine, List<LayerBase<T>> layers, Random random, int inputs, int outputs,
        double[]? weight = null, double[]? bias = null)
    {
        var layer = IndexTtsOps.Linear(engine, layers, random, inputs, outputs);
        if (weight is not null || bias is not null)
        {
            var w = weight ?? IndexTtsOps.Uniform(random, inputs * outputs, inputs);
            var b = bias ?? IndexTtsOps.Uniform(random, outputs, inputs);
            CodecBased.TorchParameters.Linear(engine, layer, inputs, outputs, w, b);
        }
        return layer;
    }

    /// <summary>An <c>nn.Conv1d</c> (torch's default initialization unless <paramref name="weight"/> is given) padded by
    /// <paramref name="padding"/> on both sides.</summary>
    public static NormedConv1DLayer<T> Conv<T>(List<LayerBase<T>> layers, Random random, int inputs, int outputs, int kernel, int dilation = 1,
        int padding = 0, double[]? weight = null)
    {
        var conv = MeloOps.Conv(layers, inputs, outputs, kernel, dilation, padding);
        int fanIn = inputs * kernel;
        SetConv(conv, weight ?? IndexTtsOps.Uniform(random, outputs * fanIn, fanIn), IndexTtsOps.Uniform(random, outputs, fanIn));
        return conv;
    }

    public static void SetConv<T>(NormedConv1DLayer<T> conv, double[] weight, double[] bias) =>
        conv.LoadTorchWeights(weight, null, bias);

    /// <summary><c>[1, C, T]</c> → rows <c>[T, C]</c>.</summary>
    public static Tensor<T> Rows<T>(IEngine engine, Tensor<T> x) => engine.TensorTranspose(engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }));

    /// <summary>Rows <c>[T, C]</c> → <c>[1, C, T]</c>.</summary>
    public static Tensor<T> Channels<T>(IEngine engine, Tensor<T> rows) =>
        engine.Reshape(engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    /// <summary><c>nn.LayerNorm(elementwise_affine=False)</c> over the last axis.</summary>
    public static Tensor<T> NormalizeLastAxis<T>(IEngine engine, Tensor<T> x, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int last = x.Rank - 1;
        var mean = engine.TensorMultiplyScalar(engine.ReduceSum(x, new[] { last }, keepDims: true), ops.FromDouble(1.0 / x.Shape[last]));
        var centred = engine.TensorSubtract(x, mean);
        var variance = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(centred, centred), new[] { last }, keepDims: true),
            ops.FromDouble(1.0 / x.Shape[last]));
        return engine.TensorDivide(centred, engine.TensorSqrt(engine.TensorAddScalar(variance, ops.FromDouble(epsilon))));
    }

    /// <summary>
    /// The Transformer's sinusoidal table (Amphion's <c>PositionalEncoding</c>): <c>pe[p, 2i] = sin(p·e^{−2i ln 10⁴ / d})</c>
    /// and <c>pe[p, 2i+1]</c> its cosine, rows <c>[length, d]</c> from position <paramref name="first"/>.
    /// </summary>
    public static Tensor<T> Sinusoids<T>(int first, int length, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var table = new Tensor<T>(new[] { length, dim });
        for (int p = 0; p < length; p++)
            for (int i = 0; 2 * i < dim; i++)
            {
                double angle = (first + p) * Math.Exp(2 * i * (-Math.Log(10000.0) / dim));
                table[p, 2 * i] = ops.FromDouble(Math.Sin(angle));
                if (2 * i + 1 < dim) table[p, 2 * i + 1] = ops.FromDouble(Math.Cos(angle));
            }
        return table;
    }

    /// <summary>
    /// The denoiser's diffusion-step embedding (<c>SinusoidalPosEmb</c>): <c>[sin(t·f), cos(t·f)]</c> with
    /// <c>f_k = e^{−k ln 10⁴ / (d/2 − 1)}</c>, as <c>[1, d]</c>.
    /// </summary>
    public static Tensor<T> StepEmbedding<T>(double t, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = dim / 2;
        double scale = Math.Log(10000.0) / (half - 1);
        var embedding = new Tensor<T>(new[] { 1, dim });
        for (int k = 0; k < half; k++)
        {
            double angle = t * Math.Exp(-k * scale);
            embedding[0, k] = ops.FromDouble(Math.Sin(angle));
            embedding[0, half + k] = ops.FromDouble(Math.Cos(angle));
        }
        return embedding;
    }

    public static Tensor<T> Dropout<T>(IEngine engine, Tensor<T> x, double rate, bool training, Random? random) =>
        training && rate > 0 && random is not null ? T5Seq2Seq<T>.Dropout(engine, x, rate, random) : x;

    public static Func<string, double[]> Required(Func<string, double[]?> read, string model) =>
        name => read(name) ?? throw new KeyNotFoundException($"{model}'s checkpoint has no '{name}'.");
}

/// <summary>
/// <c>nn.MultiheadAttention</c> (batch first) with packed projections: <c>in_proj_weight [3d, d]</c> (Xavier-uniform,
/// zero bias) and <c>out_proj</c> (zero bias), and dropout on the attention weights while training.
/// </summary>
internal sealed class Ns2Attention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim, _heads;
    private readonly double _dropout;

    public Ns2Attention(IEngine engine, List<LayerBase<T>> layers, Random random, int dim, int heads, double dropout = 0)
    {
        if (dim % heads != 0) throw new ArgumentException($"The attention width {dim} must divide by its {heads} heads.");
        _engine = engine;
        _dim = dim;
        _heads = heads;
        _dropout = dropout;
        InProjection = Ns2Ops.Linear(engine, layers, random, dim, 3 * dim, Ns2Ops.Xavier(random, 3 * dim * dim, dim, 3 * dim), new double[3 * dim]);
        OutProjection = Ns2Ops.Linear(engine, layers, random, dim, dim, IndexTtsOps.Uniform(random, dim * dim, dim), new double[dim]);
    }

    public DenseLayer<T> InProjection { get; }
    public DenseLayer<T> OutProjection { get; }

    /// <summary>Queries <c>[Lq, d]</c> attend to keys and values <c>[Lk, d]</c>; returns <c>[Lq, d]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> query, Tensor<T> keys, bool training = false, Random? random = null)
    {
        var q = _engine.TensorNarrow(InProjection.Forward(query), 1, 0, _dim);
        var kv = InProjection.Forward(keys);
        var k = IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(kv, 1, _dim, _dim), _heads);
        var v = IndexTtsOps.SplitHeads(_engine, _engine.TensorNarrow(kv, 1, 2 * _dim, _dim), _heads);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(IndexTtsOps.SplitHeads(_engine, q, _heads), _engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(_dim / _heads)));
        var weights = Ns2Ops.Dropout(_engine, _engine.Softmax(scores, axis: 2), _dropout, training, random);
        return OutProjection.Forward(IndexTtsOps.MergeHeads(_engine, _engine.BatchMatMul(weights, v)));
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        TorchParameters.Linear(_engine, InProjection, _dim, 3 * _dim, read(prefix + "in_proj_weight"), read(prefix + "in_proj_bias"));
        TorchParameters.Linear(_engine, OutProjection, _dim, _dim, read(prefix + "out_proj.weight"), read(prefix + "out_proj.bias"));
    }
}

/// <summary>
/// A LayerNorm over the last axis: the plain affine one, or the style-adaptive one (Amphion's
/// <c>StyleAdaptiveLayerNorm</c>): a non-affine norm scaled and shifted by <c>style(mean_t(condition))</c>, the style
/// projection's bias initialized to gains of 1 and shifts of 0.
/// </summary>
internal sealed class Ns2Norm<T>
{
    private readonly IEngine _engine;
    private readonly int _dim;

    public Ns2Norm(IEngine engine, List<LayerBase<T>> layers, Random random, int dim, bool styleAdaptive)
    {
        _engine = engine;
        _dim = dim;
        if (styleAdaptive)
        {
            var bias = new double[2 * dim];
            for (int i = 0; i < dim; i++) bias[i] = 1;
            Style = Ns2Ops.Linear(engine, layers, random, dim, 2 * dim, IndexTtsOps.Uniform(random, 2 * dim * dim, dim), bias);
        }
        else Plain = IndexTtsOps.LayerNorm(layers, dim);
    }

    public LayerNormalizationLayer<T>? Plain { get; }
    public DenseLayer<T>? Style { get; }

    /// <summary>Rows <c>[T, d]</c>, conditioned (style-adaptive) on rows <c>[T′, d]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? condition)
    {
        if (Plain is not null) return Plain.Forward(x);
        if (condition is null) throw new ArgumentNullException(nameof(condition), "A style-adaptive norm reads the prompt.");
        var ops = MathHelper.GetNumericOperations<T>();
        var mean = _engine.TensorMultiplyScalar(_engine.ReduceSum(condition, new[] { 0 }, keepDims: true), ops.FromDouble(1.0 / condition.Shape[0]));
        var style = Style!.Forward(mean);                                                                   // [1, 2d]
        var gamma = _engine.TensorNarrow(style, 1, 0, _dim);
        var beta = _engine.TensorNarrow(style, 1, _dim, _dim);
        var normed = Ns2Ops.NormalizeLastAxis(_engine, x, 1e-5);
        return _engine.TensorAdd(_engine.TensorMultiply(normed, _engine.TensorBroadcastTo(gamma, normed._shape)), _engine.TensorBroadcastTo(beta, normed._shape));
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        if (Plain is not null) TorchParameters.LayerNorm(_engine, Plain, read(prefix + "weight"), read(prefix + "bias"));
        else TorchParameters.Linear(_engine, Style!, _dim, 2 * _dim, read(prefix + "style.weight"), read(prefix + "style.bias"));
    }
}
