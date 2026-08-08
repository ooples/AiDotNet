using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Stockformer's Dual-Frequency Spatiotemporal Encoder and its fusion decoder, following the paper's
/// equations rather than the shape of its prose.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Ma, Xue, Lu and Chen, arXiv:2401.06139, equations 7, 10 and 11.
/// </para>
/// <code>
///   Eq. 7   temporal (low band, per asset, over time)   ta_n = Att(X_l, X_l, X_l)
///   Eq. 10  spatial  (per timestep, over assets)        sa_t = Att(X~, X~, X~)
///                                                        X~  = X + rho^spa + rho^tem
///   Eq. 11  fusion                                       fa  = Att(Y_l, Y_l, Y_l)
///                                                            + Att(Y_l, Y_h, Y_h)
/// </code>
/// <para>
/// <b>Three corrections against an earlier revision of this class</b>, each found by reading the
/// method section rather than only the reference code:
/// </para>
/// <list type="number">
/// <item><description>Attention is REAL scaled dot-product, not a fixed causal average. The average
/// left the model with no learnable temporal weighting, so longer training made the objective WORSE
/// instead of better.</description></item>
/// <item><description>The graph enters as an ADDITIVE EMBEDDING summed into the features
/// (<c>rho^spa</c>), after which plain self-attention runs over assets. It is NOT a row-normalized
/// adjacency matmul: the paper lets attention LEARN asset relationships with the graph as a prior,
/// whereas a fixed mixing matrix imposes weights the model cannot override.</description></item>
/// <item><description>Fusion is TWO summed attention terms — self-attention on the low band plus
/// cross-attention with low as query and high as key/value — not a single projection.</description></item>
/// </list>
/// <para>
/// The high band's temporal operator is a causal convolution rather than attention. Eq. 7 specifies
/// attention for the LOW band only; the reference implementation uses <c>temporalConvNet</c> for the
/// high band, so that choice is justified by the code, not the paper.
/// </para>
/// <para><b>For Beginners:</b> The slow and fast parts of each price series are analysed separately,
/// each is allowed to look across the other assets — with a graph embedding telling it which assets are
/// similar — and then the slow view consults the fast one.</para>
/// </remarks>
public sealed class StockformerDualEncoder<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// One stacked encoder layer's blocks. The paper stacks L of these; the reference config sets
    /// <c>layers = 2</c>.
    /// </summary>
    /// <param name="LowTemporal">Eq. 7 self-attention over time, low band.</param>
    /// <param name="LowSpatial">Eq. 10 self-attention over assets, low band.</param>
    /// <param name="HighSpatial">Eq. 10 self-attention over assets, high band.</param>
    /// <param name="HighTemporalProjection">
    /// Projection after the high band's causal convolution — its temporal operator is convolutional,
    /// per the reference, so it needs no Q/K/V triple.
    /// </param>
    public sealed record Layer(
        StockformerAttention<T> LowTemporal,
        StockformerAttention<T> LowSpatial,
        StockformerAttention<T> HighSpatial,
        ILayer<T> HighTemporalProjection);

    private readonly int _features;
    private readonly int _kernelWidth;
    private readonly IReadOnlyList<Layer> _layers;
    private readonly StockformerAttention<T> _fusionSelf;
    private readonly StockformerAttention<T> _fusionCross;
    private readonly ILayer<T> _fusionNorm;

    /// <summary>Gets the model width.</summary>
    public int Features => _features;

    /// <summary>Gets the number of stacked encoder layers (L).</summary>
    public int LayerCount => _layers.Count;

    /// <summary>
    /// Creates an encoder over caller-owned layers. This type holds no parameters of its own, so every
    /// weight is registered in the owning model and visible to both the optimizer and the tape.
    /// </summary>
    /// <param name="features">Model width.</param>
    /// <param name="kernelWidth">Causal window for the high band's convolution.</param>
    /// <param name="layers">The L stacked encoder layers, applied in order.</param>
    /// <param name="fusionSelf">Eq. 11's first term: self-attention on the low band.</param>
    /// <param name="fusionCross">Eq. 11's second term: low queries the high band.</param>
    /// <param name="fusionNorm">
    /// Normalization on the fused output. Present in the reference's <c>adaptiveFusion</c>
    /// (<c>nn.LayerNorm</c>); the paper's method section does not mention it, so it is justified by the
    /// code rather than the paper.
    /// </param>
    public StockformerDualEncoder(
        int features, int kernelWidth,
        IReadOnlyList<Layer> layers,
        StockformerAttention<T> fusionSelf,
        StockformerAttention<T> fusionCross,
        ILayer<T> fusionNorm)
    {
        if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features), features, "Width must be positive.");
        if (kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelWidth), kernelWidth, "Kernel width must be positive.");
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        if (layers.Count == 0)
            throw new ArgumentException(
                "At least one encoder layer is required. The paper stacks L of them and the reference " +
                "config sets layers = 2; an empty stack would leave the bands unencoded.", nameof(layers));

        _features = features;
        _kernelWidth = kernelWidth;
        _layers = layers;
        _fusionSelf = fusionSelf ?? throw new ArgumentNullException(nameof(fusionSelf));
        _fusionCross = fusionCross ?? throw new ArgumentNullException(nameof(fusionCross));
        _fusionNorm = fusionNorm ?? throw new ArgumentNullException(nameof(fusionNorm));
    }

    /// <summary>
    /// Encodes both bands through the stack and fuses them.
    /// </summary>
    /// <param name="low">Low band, <c>[assets, time, features]</c>.</param>
    /// <param name="high">High band, same shape.</param>
    /// <param name="spatialEmbedding">
    /// <c>rho^spa</c>: per-asset structural embedding, <c>[assets, features]</c>, broadcast over time.
    /// </param>
    /// <param name="temporalEmbedding">
    /// <c>rho^tem</c>: per-timestep embedding, <c>[time, features]</c>, broadcast over assets.
    /// </param>
    /// <returns>The fused representation and the encoded low band.</returns>
    public (Tensor<T> Fused, Tensor<T> LowEncoded) Encode(
        Tensor<T> low, Tensor<T> high, Matrix<T> spatialEmbedding, Matrix<T> temporalEmbedding)
    {
        Validate(low, high, spatialEmbedding, temporalEmbedding);

        int assets = low.Shape[0];
        int time = low.Shape[1];

        var currentLow = low;
        var currentHigh = high;

        foreach (var layer in _layers)
        {
            // Eq. 7: temporal self-attention on the low band, per asset, causal over time.
            var lowTemporal = OverTime(currentLow, assets, time, layer.LowTemporal);

            // High band: causal convolution then its projection (the reference's temporalConvNet).
            var highTemporal = Project(
                CausalWindow(currentHigh, assets, time), layer.HighTemporalProjection, assets, time);

            // Eq. 10: X~ = X + rho^spa + rho^tem, then self-attention over assets per timestep.
            var lowSpatial = OverAssets(
                AddEmbeddings(lowTemporal, spatialEmbedding, temporalEmbedding, assets, time),
                assets, time, layer.LowSpatial);
            var highSpatial = OverAssets(
                AddEmbeddings(highTemporal, spatialEmbedding, temporalEmbedding, assets, time),
                assets, time, layer.HighSpatial);

            // Residual, so stacking layers cannot destroy what earlier ones found.
            currentLow = Engine.TensorAdd(lowSpatial, lowTemporal);
            currentHigh = Engine.TensorAdd(highSpatial, highTemporal);
        }

        // Eq. 11: Att(Y_l, Y_l, Y_l) + Att(Y_l, Y_h, Y_h) — self plus cross, summed.
        var fusedSelf = OverTime(currentLow, assets, time, _fusionSelf);
        var fusedCross = OverTimeCross(currentLow, currentHigh, assets, time, _fusionCross);
        var fused = Project(Engine.TensorAdd(fusedSelf, fusedCross), _fusionNorm, assets, time);

        return (fused, currentLow);
    }

    private void Validate(Tensor<T> low, Tensor<T> high, Matrix<T> spatial, Matrix<T> temporal)
    {
        if (low is null) throw new ArgumentNullException(nameof(low));
        if (high is null) throw new ArgumentNullException(nameof(high));
        if (spatial is null) throw new ArgumentNullException(nameof(spatial));
        if (temporal is null) throw new ArgumentNullException(nameof(temporal));

        if (low.Shape.Length != 3)
            throw new ArgumentException($"Expected [assets, time, features]; got rank {low.Shape.Length}.", nameof(low));
        // high's rank is checked BEFORE its dimensions are read. The loop below indexes
        // high.Shape[d] up to d = 2, so a rank-1 or rank-2 high threw IndexOutOfRangeException from
        // the shape access -- a message that tells the caller nothing -- instead of this one.
        if (high.Shape.Length != 3)
            throw new ArgumentException($"Expected [assets, time, features]; got rank {high.Shape.Length}.", nameof(high));
        for (int d = 0; d < 3; d++)
        {
            if (high.Shape[d] != low.Shape[d])
                throw new ArgumentException("The two bands must have identical shapes — they come from one split.", nameof(high));
        }
        if (low.Shape[2] != _features)
            throw new ArgumentException($"Band width {low.Shape[2]} does not match encoder width {_features}.", nameof(low));
        if (spatial.Rows != low.Shape[0] || spatial.Columns != _features)
            throw new ArgumentException(
                $"rho^spa must be [assets, features] = [{low.Shape[0]}, {_features}]; got " +
                $"[{spatial.Rows}, {spatial.Columns}].", nameof(spatial));
        if (temporal.Rows != low.Shape[1] || temporal.Columns != _features)
            throw new ArgumentException(
                $"rho^tem must be [time, features] = [{low.Shape[1]}, {_features}]; got " +
                $"[{temporal.Rows}, {temporal.Columns}].", nameof(temporal));
    }

    /// <summary>Attention along the TIME axis, independently per asset.</summary>
    private Tensor<T> OverTime(Tensor<T> x, int assets, int time, StockformerAttention<T> attention)
    {
        var perAsset = new Tensor<T>[assets];
        for (int a = 0; a < assets; a++)
        {
            var slice = AssetSlice(x, a, assets, time);
            perAsset[a] = Engine.Reshape(attention.Apply(slice, slice, slice), new[] { 1, time, _features });
        }
        return Engine.Concat(perAsset, 0);
    }

    /// <summary>Cross-attention along TIME: query from <paramref name="q"/>, key/value from <paramref name="kv"/>.</summary>
    private Tensor<T> OverTimeCross(
        Tensor<T> q, Tensor<T> kv, int assets, int time, StockformerAttention<T> attention)
    {
        var perAsset = new Tensor<T>[assets];
        for (int a = 0; a < assets; a++)
        {
            var qSlice = AssetSlice(q, a, assets, time);
            var kvSlice = AssetSlice(kv, a, assets, time);
            perAsset[a] = Engine.Reshape(
                attention.Apply(qSlice, kvSlice, kvSlice), new[] { 1, time, _features });
        }
        return Engine.Concat(perAsset, 0);
    }

    /// <summary>
    /// Attention along the ASSET axis, independently per timestep. No causal mask — assets are unordered.
    /// </summary>
    private Tensor<T> OverAssets(Tensor<T> x, int assets, int time, StockformerAttention<T> attention)
    {
        var perStep = new Tensor<T>[time];
        for (int t = 0; t < time; t++)
        {
            var slice = TimeSlice(x, t, assets, time);
            perStep[t] = Engine.Reshape(attention.Apply(slice, slice, slice), new[] { assets, 1, _features });
        }
        return Engine.Concat(perStep, 1);
    }

    /// <summary>Eq. 10's <c>X~ = X + rho^spa + rho^tem</c>, both broadcast.</summary>
    private Tensor<T> AddEmbeddings(
        Tensor<T> x, Matrix<T> spatial, Matrix<T> temporal, int assets, int time)
    {
        var bias = new Tensor<T>(new[] { assets, time, _features });
        for (int a = 0; a < assets; a++)
        {
            for (int t = 0; t < time; t++)
            {
                int at = ((a * time) + t) * _features;
                for (int f = 0; f < _features; f++)
                    bias[at + f] = Ops.Add(spatial[a, f], temporal[t, f]);
            }
        }
        // The bias is DATA, so building it by index costs no gradient; the add itself is recorded.
        return Engine.TensorAdd(x, bias);
    }

    /// <summary>
    /// The causal local-mean operator for <paramref name="time"/> steps, built once per time length.
    /// </summary>
    /// <remarks>
    /// The operator depends only on the position count and the kernel width, both fixed for an
    /// encoder instance, but it was allocated and filled again for every asset, every layer and every
    /// training step. Instance-scoped rather than static because _kernelWidth is per-encoder; the
    /// tensor is never mutated after construction, so one instance serves every call at that length.
    /// </remarks>
    private readonly Dictionary<int, Tensor<T>> _causalWindowOperators = new();

    private Tensor<T> CausalWindowOperator(int time)
    {
        if (_causalWindowOperators.TryGetValue(time, out var cached)) return cached;

        var op = new Tensor<T>(new[] { time, time });
        for (int t = 0; t < time; t++)
        {
            int first = Math.Max(0, t - _kernelWidth + 1);
            var weight = Ops.FromDouble(1.0 / (t - first + 1));
            for (int u = first; u <= t; u++) op[(t * time) + u] = weight;
        }

        _causalWindowOperators[time] = op;
        return op;
    }

    /// <summary>Causal local mean over the last <c>kernelWidth</c> steps, as a matmul so it stays on the tape.</summary>
    private Tensor<T> CausalWindow(Tensor<T> x, int assets, int time)
    {
        var op = CausalWindowOperator(time);


        var timeFirst = Engine.Reshape(
            Engine.TensorPermute(x, new[] { 1, 0, 2 }), new[] { time, assets * _features });
        return Engine.TensorPermute(
            Engine.Reshape(Engine.TensorMatMul(op, timeFirst), new[] { time, assets, _features }),
            new[] { 1, 0, 2 });
    }

    private Tensor<T> Project(Tensor<T> x, ILayer<T> layer, int assets, int time)
        => Engine.Reshape(
            layer.Forward(Engine.Reshape(x, new[] { assets * time, _features })),
            new[] { assets, time, _features });

    /// <summary>One asset's <c>[time, features]</c> slice, via a one-hot matmul so the tape follows it.</summary>
    private Tensor<T> AssetSlice(Tensor<T> x, int asset, int assets, int time)
    {
        var selector = new Tensor<T>(new[] { 1, assets });
        selector[asset] = Ops.One;
        var flat = Engine.Reshape(x, new[] { assets, time * _features });
        return Engine.Reshape(Engine.TensorMatMul(selector, flat), new[] { time, _features });
    }

    /// <summary>One timestep's <c>[assets, features]</c> slice, likewise by one-hot matmul.</summary>
    private Tensor<T> TimeSlice(Tensor<T> x, int step, int assets, int time)
    {
        var selector = new Tensor<T>(new[] { 1, time });
        selector[step] = Ops.One;
        var timeFirst = Engine.Reshape(
            Engine.TensorPermute(x, new[] { 1, 0, 2 }), new[] { time, assets * _features });
        return Engine.Reshape(Engine.TensorMatMul(selector, timeFirst), new[] { assets, _features });
    }
}
