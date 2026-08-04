using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Stockformer's Dual-Frequency Spatiotemporal Encoder and its adaptive cross-band fusion.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Ma, Xue, Lu and Chen, arXiv:2401.06139. Transcribed from <c>dualEncoder</c>,
/// <c>sparseSpatialAttention</c>, <c>temporalConvNet</c> and <c>adaptiveFusion</c> in the reference
/// implementation (github.com/Eric991005/Multitask-Stockformer,
/// <c>Stockformermodel/Multitask_Stockformer_models.py</c>).
/// </para>
/// <para>
/// <b>The bands take DIFFERENT temporal operators, and that asymmetry is the contribution.</b> The
/// name "dual-frequency encoder" does not say which branch gets what, and getting it backwards
/// produces a model that trains and means nothing:
/// </para>
/// <code>
///   low  band  ->  temporalAttention   (self-attention across time; long-range trend)
///   high band  ->  temporalConvNet     (causal TCN; local, short-horizon structure)
///   both bands ->  their OWN sparseSpatialAttention over the stock graph, added residually
/// </code>
/// <para>
/// Each band owns a SEPARATE spatial-attention instance (<c>ssal</c>, <c>ssah</c> in the reference) —
/// they are not a shared module applied twice, so the two bands learn different cross-stock structure.
/// </para>
/// <para>
/// <b>Fusion is asymmetric.</b> <c>adaptiveFusion</c> takes its query from the LOW band and its
/// key/value from the HIGH band (through <c>relu</c>), so the trend representation attends to the
/// fluctuation representation and not the reverse. The temporal embedding is added to both bands
/// before the projections.
/// </para>
/// <para><b>For Beginners:</b> The slow-moving part of a price series and its fast jitter carry
/// different information, so this analyses them with different tools, lets each look across all the
/// other stocks, and then lets the slow view consult the fast one.</para>
/// </remarks>
public sealed class StockformerDualEncoder<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly int _features;
    private readonly int _samples;

    // Low band: temporal self-attention projections.
    private readonly Matrix<T> _lowQuery;
    private readonly Matrix<T> _lowKey;
    private readonly Matrix<T> _lowValue;

    // High band: causal temporal convolution kernel [features, features, kernelWidth].
    private readonly T[][][] _highKernel;
    private readonly int _kernelWidth;

    // Per-band spatial attention (separate instances, per the reference).
    private readonly Matrix<T> _spatialLow;
    private readonly Matrix<T> _spatialHigh;

    // Adaptive fusion: query from low, key/value from high.
    private readonly Matrix<T> _fusionQuery;
    private readonly Matrix<T> _fusionKey;
    private readonly Matrix<T> _fusionValue;

    /// <summary>Gets the model width.</summary>
    public int Features => _features;

    /// <summary>Gets the sparsity parameter of the spatial attention (<c>samples</c>).</summary>
    public int SpatialSamples => _samples;

    /// <summary>
    /// Creates an encoder.
    /// </summary>
    /// <param name="features">Model width (<c>dims</c>, 128 in the reference config).</param>
    /// <param name="spatialSamples">Sparsity parameter (<c>samples</c>, 1 in the reference).</param>
    /// <param name="kernelWidth">Causal TCN kernel width for the high band.</param>
    /// <param name="random">RNG for initialization.</param>
    public StockformerDualEncoder(int features, int spatialSamples, int kernelWidth, Random random)
    {
        if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features), features, "Width must be positive.");
        if (spatialSamples <= 0) throw new ArgumentOutOfRangeException(nameof(spatialSamples), spatialSamples, "Sparsity must be positive.");
        if (kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelWidth), kernelWidth, "Kernel width must be positive.");
        if (random is null) throw new ArgumentNullException(nameof(random));

        _features = features;
        _samples = spatialSamples;
        _kernelWidth = kernelWidth;

        double scale = 1.0 / Math.Sqrt(features);
        _lowQuery = RandomMatrix(features, features, scale, random);
        _lowKey = RandomMatrix(features, features, scale, random);
        _lowValue = RandomMatrix(features, features, scale, random);
        _spatialLow = RandomMatrix(features, features, scale, random);
        _spatialHigh = RandomMatrix(features, features, scale, random);
        _fusionQuery = RandomMatrix(features, features, scale, random);
        _fusionKey = RandomMatrix(features, features, scale, random);
        _fusionValue = RandomMatrix(features, features, scale, random);

        _highKernel = new T[features][][];
        for (int o = 0; o < features; o++)
        {
            _highKernel[o] = new T[features][];
            for (int i = 0; i < features; i++)
            {
                var taps = new T[kernelWidth];
                for (int k = 0; k < kernelWidth; k++)
                    taps[k] = Ops.FromDouble((random.NextDouble() * 2.0 - 1.0) * scale);
                _highKernel[o][i] = taps;
            }
        }
    }

    /// <summary>
    /// Encodes both bands and fuses them.
    /// </summary>
    /// <param name="low">Low band, <c>[stocks, time, features]</c>.</param>
    /// <param name="high">High band, same shape.</param>
    /// <param name="temporalEmbedding">Per-timestep embedding, <c>[time, features]</c>.</param>
    /// <param name="adjacency">Stock graph, <c>[stocks, stocks]</c> (struc2vec-derived).</param>
    /// <returns>
    /// The fused representation and the encoded low band — the two representations the paper's output
    /// heads are each applied to.
    /// </returns>
    public (Tensor<T> Fused, Tensor<T> LowEncoded) Encode(
        Tensor<T> low, Tensor<T> high, Matrix<T> temporalEmbedding, Matrix<T> adjacency)
    {
        ValidateShapes(low, high, temporalEmbedding, adjacency);

        int stocks = low.Shape[0];
        int time = low.Shape[1];

        // Low band: attention across time. High band: causal convolution across time.
        var lowTemporal = TemporalAttention(low, stocks, time);
        var highTemporal = CausalConvolution(high, stocks, time);

        // Each band through its OWN spatial attention, added residually (reference: xl = ssal(xl) + xl).
        var lowEncoded = AddInPlace(SpatialAttention(lowTemporal, adjacency, _spatialLow, stocks, time), lowTemporal);
        var highEncoded = AddInPlace(SpatialAttention(highTemporal, adjacency, _spatialHigh, stocks, time), highTemporal);

        var fused = AdaptiveFusion(lowEncoded, highEncoded, temporalEmbedding, stocks, time);
        return (fused, lowEncoded);
    }

    private void ValidateShapes(Tensor<T> low, Tensor<T> high, Matrix<T> te, Matrix<T> adjacency)
    {
        if (low is null) throw new ArgumentNullException(nameof(low));
        if (high is null) throw new ArgumentNullException(nameof(high));
        if (te is null) throw new ArgumentNullException(nameof(te));
        if (adjacency is null) throw new ArgumentNullException(nameof(adjacency));

        if (low.Shape.Length != 3)
            throw new ArgumentException($"Expected [stocks, time, features]; got rank {low.Shape.Length}.", nameof(low));
        for (int d = 0; d < 3; d++)
        {
            if (high.Shape[d] != low.Shape[d])
                throw new ArgumentException("The two bands must have identical shapes — they come from one split.", nameof(high));
        }
        if (low.Shape[2] != _features)
            throw new ArgumentException($"Band width {low.Shape[2]} does not match encoder width {_features}.", nameof(low));
        if (adjacency.Rows != low.Shape[0] || adjacency.Columns != low.Shape[0])
            throw new ArgumentException(
                $"Adjacency must be [stocks, stocks] = [{low.Shape[0]}, {low.Shape[0]}]; got " +
                $"[{adjacency.Rows}, {adjacency.Columns}].", nameof(adjacency));
        if (te.Rows != low.Shape[1] || te.Columns != _features)
            throw new ArgumentException(
                $"Temporal embedding must be [time, features] = [{low.Shape[1]}, {_features}]; got " +
                $"[{te.Rows}, {te.Columns}].", nameof(te));
    }

    /// <summary>Masked self-attention across time, applied per stock. The LOW band's operator.</summary>
    private Tensor<T> TemporalAttention(Tensor<T> x, int stocks, int time)
    {
        var output = new Tensor<T>(new[] { stocks, time, _features });
        double norm = 1.0 / Math.Sqrt(_features);

        var q = new double[_features];
        var k = new double[_features];
        var scores = new double[time];

        for (int s = 0; s < stocks; s++)
        {
            for (int t = 0; t < time; t++)
            {
                Project(x, s, t, _lowQuery, q);

                // Causal mask: a timestep may not attend to its future.
                double max = double.NegativeInfinity;
                for (int u = 0; u <= t; u++)
                {
                    Project(x, s, u, _lowKey, k);
                    double dot = 0.0;
                    for (int f = 0; f < _features; f++) dot += q[f] * k[f];
                    scores[u] = dot * norm;
                    if (scores[u] > max) max = scores[u];
                }

                double sum = 0.0;
                for (int u = 0; u <= t; u++) { scores[u] = Math.Exp(scores[u] - max); sum += scores[u]; }
                if (sum <= 0.0) sum = 1.0;

                for (int u = 0; u <= t; u++)
                {
                    double w = scores[u] / sum;
                    for (int f = 0; f < _features; f++)
                    {
                        double v = 0.0;
                        for (int g = 0; g < _features; g++)
                            v += Ops.ToDouble(x[Index(s, u, g, time)]) * Ops.ToDouble(_lowValue[g, f]);
                        int oi = Index(s, t, f, time);
                        output[oi] = Ops.FromDouble(Ops.ToDouble(output[oi]) + (w * v));
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Causal 1-D convolution across time, applied per stock. The HIGH band's operator.</summary>
    /// <remarks>
    /// Causal: output at t reads only t-k+1..t, which is the reference's <c>Chomp1d</c> trimming the
    /// right-hand padding so no future information leaks backwards.
    /// </remarks>
    private Tensor<T> CausalConvolution(Tensor<T> x, int stocks, int time)
    {
        var output = new Tensor<T>(new[] { stocks, time, _features });
        for (int s = 0; s < stocks; s++)
        {
            for (int t = 0; t < time; t++)
            {
                for (int o = 0; o < _features; o++)
                {
                    double acc = 0.0;
                    for (int k = 0; k < _kernelWidth; k++)
                    {
                        int src = t - k;
                        if (src < 0) break;
                        for (int i = 0; i < _features; i++)
                            acc += Ops.ToDouble(x[Index(s, src, i, time)]) * Ops.ToDouble(_highKernel[o][i][k]);
                    }
                    output[Index(s, t, o, time)] = Ops.FromDouble(Math.Max(0.0, acc));
                }
            }
        }
        return output;
    }

    /// <summary>Attention across STOCKS at each timestep, gated by the graph.</summary>
    /// <remarks>
    /// The adjacency gates which stocks may influence which — a zero entry removes the edge entirely
    /// rather than merely down-weighting it, so an unconnected pair cannot exchange information.
    /// </remarks>
    private Tensor<T> SpatialAttention(Tensor<T> x, Matrix<T> adjacency, Matrix<T> projection, int stocks, int time)
    {
        var output = new Tensor<T>(new[] { stocks, time, _features });
        var weights = new double[stocks];

        for (int t = 0; t < time; t++)
        {
            for (int s = 0; s < stocks; s++)
            {
                double max = double.NegativeInfinity;
                for (int n = 0; n < stocks; n++)
                {
                    double edge = Ops.ToDouble(adjacency[s, n]);
                    if (edge == 0.0) { weights[n] = double.NegativeInfinity; continue; }

                    double dot = 0.0;
                    for (int f = 0; f < _features; f++)
                        dot += Ops.ToDouble(x[Index(s, t, f, time)]) * Ops.ToDouble(x[Index(n, t, f, time)]);
                    weights[n] = (dot / Math.Sqrt(_features)) * edge;
                    if (weights[n] > max) max = weights[n];
                }

                if (double.IsNegativeInfinity(max))
                {
                    // Isolated stock: no neighbours at all. Leave its slice zero so the residual add
                    // upstream simply passes the unattended representation through.
                    continue;
                }

                double sum = 0.0;
                for (int n = 0; n < stocks; n++)
                {
                    weights[n] = double.IsNegativeInfinity(weights[n]) ? 0.0 : Math.Exp(weights[n] - max);
                    sum += weights[n];
                }
                if (sum <= 0.0) sum = 1.0;

                for (int f = 0; f < _features; f++)
                {
                    double acc = 0.0;
                    for (int n = 0; n < stocks; n++)
                    {
                        if (weights[n] == 0.0) continue;
                        double v = 0.0;
                        for (int g = 0; g < _features; g++)
                            v += Ops.ToDouble(x[Index(n, t, g, time)]) * Ops.ToDouble(projection[g, f]);
                        acc += (weights[n] / sum) * v;
                    }
                    output[Index(s, t, f, time)] = Ops.FromDouble(acc);
                }
            }
        }
        return output;
    }

    /// <summary>Cross-band attention: query from the LOW band, key/value from the HIGH band.</summary>
    private Tensor<T> AdaptiveFusion(Tensor<T> low, Tensor<T> high, Matrix<T> te, int stocks, int time)
    {
        // The reference adds the temporal embedding to both bands before projecting.
        var lowPlus = AddEmbedding(low, te, stocks, time);
        var highPlus = AddEmbedding(high, te, stocks, time);

        var output = new Tensor<T>(new[] { stocks, time, _features });
        double norm = 1.0 / Math.Sqrt(_features);

        var q = new double[_features];
        var k = new double[_features];
        var scores = new double[time];

        for (int s = 0; s < stocks; s++)
        {
            for (int t = 0; t < time; t++)
            {
                Project(lowPlus, s, t, _fusionQuery, q);

                double max = double.NegativeInfinity;
                for (int u = 0; u < time; u++)
                {
                    ProjectRelu(highPlus, s, u, _fusionKey, k);
                    double dot = 0.0;
                    for (int f = 0; f < _features; f++) dot += q[f] * k[f];
                    scores[u] = dot * norm;
                    if (scores[u] > max) max = scores[u];
                }

                double sum = 0.0;
                for (int u = 0; u < time; u++) { scores[u] = Math.Exp(scores[u] - max); sum += scores[u]; }
                if (sum <= 0.0) sum = 1.0;

                for (int u = 0; u < time; u++)
                {
                    double w = scores[u] / sum;
                    if (w == 0.0) continue;
                    for (int f = 0; f < _features; f++)
                    {
                        double v = 0.0;
                        for (int g = 0; g < _features; g++)
                            v += Ops.ToDouble(highPlus[Index(s, u, g, time)]) * Ops.ToDouble(_fusionValue[g, f]);
                        v = Math.Max(0.0, v);
                        int oi = Index(s, t, f, time);
                        output[oi] = Ops.FromDouble(Ops.ToDouble(output[oi]) + (w * v));
                    }
                }
            }
        }
        return output;
    }

    private int Index(int stock, int t, int feature, int time) => ((stock * time) + t) * _features + feature;

    private void Project(Tensor<T> x, int s, int t, Matrix<T> w, double[] dest)
    {
        int time = x.Shape[1];
        for (int f = 0; f < _features; f++)
        {
            double acc = 0.0;
            for (int g = 0; g < _features; g++)
                acc += Ops.ToDouble(x[Index(s, t, g, time)]) * Ops.ToDouble(w[g, f]);
            dest[f] = acc;
        }
    }

    private void ProjectRelu(Tensor<T> x, int s, int t, Matrix<T> w, double[] dest)
    {
        Project(x, s, t, w, dest);
        for (int f = 0; f < _features; f++) dest[f] = Math.Max(0.0, dest[f]);
    }

    private Tensor<T> AddEmbedding(Tensor<T> x, Matrix<T> te, int stocks, int time)
    {
        var result = new Tensor<T>(new[] { stocks, time, _features });
        for (int s = 0; s < stocks; s++)
            for (int t = 0; t < time; t++)
                for (int f = 0; f < _features; f++)
                    result[Index(s, t, f, time)] = Ops.Add(x[Index(s, t, f, time)], te[t, f]);
        return result;
    }

    private static Tensor<T> AddInPlace(Tensor<T> target, Tensor<T> addend)
    {
        for (int i = 0; i < target.Length; i++) target[i] = Ops.Add(target[i], addend[i]);
        return target;
    }

    private static Matrix<T> RandomMatrix(int rows, int columns, double scale, Random random)
    {
        var m = new Matrix<T>(rows, columns);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
                m[r, c] = Ops.FromDouble((random.NextDouble() * 2.0 - 1.0) * scale);
        return m;
    }
}
