using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Stockformer's Dual-Frequency Spatiotemporal Encoder: the routing that gives each frequency band its
/// own temporal operator, mixes each across the asset graph, and fuses them asymmetrically.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Ma, Xue, Lu and Chen, arXiv:2401.06139. Structure transcribed from <c>dualEncoder</c>,
/// <c>sparseSpatialAttention</c>, <c>temporalConvNet</c> and <c>adaptiveFusion</c> in the reference
/// implementation (github.com/Eric991005/Multitask-Stockformer).
/// </para>
/// <para>
/// <b>The bands take DIFFERENT operators, and the direction is the contribution.</b> "Dual-frequency
/// encoder" does not say which branch gets what, and inverting it yields a model that trains fine and
/// means nothing:
/// </para>
/// <code>
///   low  band  ->  temporal mixing over the causal window   (trend)
///   high band  ->  causal local convolution + ReLU          (short-horizon structure)
///   both bands ->  their OWN graph-mixing projection, added residually
///   fusion     ->  asymmetric: the low band is the base, the high band is projected in
/// </code>
/// <para>
/// Each band owns a SEPARATE spatial projection (<c>ssal</c>/<c>ssah</c> in the reference) rather than
/// sharing one module applied twice, so the bands learn different cross-asset structure.
/// </para>
/// <para>
/// <b>Every weight is owned by a layer supplied by the caller.</b> This type holds no parameters of its
/// own. That is deliberate: weights in bare matrices are invisible to the parameter vector and to the
/// gradient tape, so a model built that way cannot actually be trained through the standard path — it
/// reports zero parameters and zero gradient. Projections go through <c>ILayer.Forward</c> and
/// everything else through <c>Engine</c>, so the whole encoder is differentiable.
/// </para>
/// <para>
/// <b>Documented simplification.</b> The reference's temporal and fusion stages are full scaled
/// dot-product attention. Here temporal mixing is a causal average over the window and fusion is a
/// learned projection of the high band onto the low, both differentiable and both preserving the
/// asymmetry and the causality that matter. Rank-order behaviour of the bands is preserved; the
/// attention weights are not learned per pair. Stated rather than implied, because it is a real
/// departure from the reference even though the routing is faithful.
/// </para>
/// <para><b>For Beginners:</b> The slow and fast parts of a price series carry different information,
/// so each is processed differently, each is allowed to look across the other assets, and then the slow
/// view absorbs what the fast view found.</para>
/// </remarks>
public sealed class StockformerDualEncoder<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _features;
    private readonly int _kernelWidth;

    private readonly ILayer<T> _lowTemporal;
    private readonly ILayer<T> _highTemporal;
    private readonly ILayer<T> _spatialLow;
    private readonly ILayer<T> _spatialHigh;
    private readonly ILayer<T> _fusion;
    private readonly ILayer<T> _fusionNorm;

    /// <summary>Gets the model width.</summary>
    public int Features => _features;

    /// <summary>
    /// Creates an encoder over caller-owned layers.
    /// </summary>
    /// <param name="features">Model width.</param>
    /// <param name="kernelWidth">Causal window for the high band's local convolution.</param>
    /// <param name="lowTemporal">Projection applied to the low band after temporal mixing.</param>
    /// <param name="highTemporal">Projection applied to the high band's causal convolution.</param>
    /// <param name="spatialLow">Low band's graph-mixing projection.</param>
    /// <param name="spatialHigh">High band's graph-mixing projection (a DISTINCT layer).</param>
    /// <param name="fusion">Projection carrying the high band into the fused representation.</param>
    /// <param name="fusionNorm">
    /// Normalization applied to the fused representation. The reference has
    /// <c>nn.LayerNorm(features, elementwise_affine=False)</c> in <c>adaptiveFusion</c>; omitting it
    /// made training DIVERGE with more iterations (200-step loss worse than 50-step), because the two
    /// residual additions let activation scale grow unchecked through the fusion stage.
    /// </param>
    public StockformerDualEncoder(
        int features, int kernelWidth,
        ILayer<T> lowTemporal, ILayer<T> highTemporal,
        ILayer<T> spatialLow, ILayer<T> spatialHigh, ILayer<T> fusion, ILayer<T> fusionNorm)
    {
        if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features), features, "Width must be positive.");
        if (kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelWidth), kernelWidth, "Kernel width must be positive.");

        _features = features;
        _kernelWidth = kernelWidth;
        _lowTemporal = lowTemporal ?? throw new ArgumentNullException(nameof(lowTemporal));
        _highTemporal = highTemporal ?? throw new ArgumentNullException(nameof(highTemporal));
        _spatialLow = spatialLow ?? throw new ArgumentNullException(nameof(spatialLow));
        _spatialHigh = spatialHigh ?? throw new ArgumentNullException(nameof(spatialHigh));
        _fusion = fusion ?? throw new ArgumentNullException(nameof(fusion));
        _fusionNorm = fusionNorm ?? throw new ArgumentNullException(nameof(fusionNorm));

        if (ReferenceEquals(spatialLow, spatialHigh))
        {
            throw new ArgumentException(
                "The two bands must have SEPARATE spatial projections (ssal/ssah in the reference). " +
                "Sharing one layer would force both frequency bands to learn the same cross-asset " +
                "structure, which erases half the dual-frequency design.", nameof(spatialHigh));
        }
    }

    /// <summary>
    /// Encodes both bands and fuses them.
    /// </summary>
    /// <param name="low">Low band, <c>[assets, time, features]</c>.</param>
    /// <param name="high">High band, same shape.</param>
    /// <param name="graph">Asset graph, <c>[assets, assets]</c>.</param>
    /// <returns>The fused representation and the encoded low band.</returns>
    public (Tensor<T> Fused, Tensor<T> LowEncoded) Encode(Tensor<T> low, Tensor<T> high, Matrix<T> graph)
    {
        Validate(low, high, graph);

        int assets = low.Shape[0];
        int time = low.Shape[1];

        // Low band: causal temporal mixing, then a learned projection.
        var lowMixed = Project(CausalMix(low, assets, time, time), _lowTemporal, assets, time);

        // High band: causal local convolution (ReLU comes from the layer's activation), then projection.
        var highMixed = Project(CausalMix(high, assets, time, _kernelWidth), _highTemporal, assets, time);

        // Each band mixes across assets through the graph, then its OWN projection, added residually.
        var lowEncoded = Engine.TensorAdd(
            Project(GraphMix(lowMixed, graph, assets, time), _spatialLow, assets, time), lowMixed);
        var highEncoded = Engine.TensorAdd(
            Project(GraphMix(highMixed, graph, assets, time), _spatialHigh, assets, time), highMixed);

        // Asymmetric fusion: low is the base, high is projected in, then NORMALIZED. The reference
        // normalizes here and it is load-bearing, not cosmetic: without it the stacked residual adds
        // let activation scale drift and longer training made the objective worse instead of better.
        var fused = Project(
            Engine.TensorAdd(Project(highEncoded, _fusion, assets, time), lowEncoded),
            _fusionNorm, assets, time);
        return (fused, lowEncoded);
    }

    private void Validate(Tensor<T> low, Tensor<T> high, Matrix<T> graph)
    {
        if (low is null) throw new ArgumentNullException(nameof(low));
        if (high is null) throw new ArgumentNullException(nameof(high));
        if (graph is null) throw new ArgumentNullException(nameof(graph));

        if (low.Shape.Length != 3)
            throw new ArgumentException($"Expected [assets, time, features]; got rank {low.Shape.Length}.", nameof(low));
        for (int d = 0; d < 3; d++)
        {
            if (high.Shape[d] != low.Shape[d])
                throw new ArgumentException("The two bands must have identical shapes — they come from one split.", nameof(high));
        }
        if (low.Shape[2] != _features)
            throw new ArgumentException($"Band width {low.Shape[2]} does not match encoder width {_features}.", nameof(low));
        if (graph.Rows != low.Shape[0] || graph.Columns != low.Shape[0])
            throw new ArgumentException(
                $"Graph must be [assets, assets] = [{low.Shape[0]}, {low.Shape[0]}]; got " +
                $"[{graph.Rows}, {graph.Columns}].", nameof(graph));
    }

    /// <summary>Runs a projection over every (asset, timestep) row, via the layer so the tape records it.</summary>
    private Tensor<T> Project(Tensor<T> x, ILayer<T> layer, int assets, int time)
    {
        var flat = Engine.Reshape(x, new[] { assets * time, _features });
        var projected = layer.Forward(flat);
        return Engine.Reshape(projected, new[] { assets, time, _features });
    }

    /// <summary>
    /// Causal temporal mixing as a MATMUL against a fixed lower-triangular averaging operator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Expressed as a matrix product on purpose. The obvious implementation — loop over timesteps and
    /// write each output element — produces the same numbers and SEVERS THE GRADIENT TAPE, because
    /// writing into a fresh tensor by index is not a recorded operation. The model then reports layers
    /// and parameters but zero gradient, and training silently does nothing.
    /// </para>
    /// <para>
    /// <c>L[t, u] = 1/(t+1)</c> for <c>u &lt;= t</c> and 0 above the diagonal, so row t averages
    /// 0..t and position t can never see its future.
    /// </para>
    /// </remarks>
    private Tensor<T> CausalMix(Tensor<T> x, int assets, int time, int width)
    {
        var mixer = CausalOperator(time, width);
        return ApplyTimeOperator(x, mixer, assets, time);
    }

    /// <summary>
    /// Builds the <c>[time, time]</c> causal operator: a running mean when <paramref name="window"/>
    /// covers everything, a local window mean otherwise.
    /// </summary>
    private Tensor<T> CausalOperator(int time, int window)
    {
        var op = new Tensor<T>(new[] { time, time });
        for (int t = 0; t < time; t++)
        {
            int first = window >= time ? 0 : Math.Max(0, t - window + 1);
            int count = t - first + 1;
            var weight = Ops.FromDouble(1.0 / count);
            for (int u = first; u <= t; u++) op[(t * time) + u] = weight;
        }
        return op;
    }

    /// <summary>
    /// Applies a <c>[time, time]</c> operator across the time axis in ONE matmul.
    /// </summary>
    /// <remarks>
    /// Permuting time to the front lets a single <c>[time, time] x [time, assets*features]</c> product
    /// mix every asset and feature at once — no per-asset slicing loop, and every step
    /// (permute, reshape, matmul) is a recorded Engine op so the gradient survives.
    /// </remarks>
    private Tensor<T> ApplyTimeOperator(Tensor<T> x, Tensor<T> op, int assets, int time)
    {
        var timeFirst = Engine.Reshape(
            Engine.TensorPermute(x, new[] { 1, 0, 2 }), new[] { time, assets * _features });
        var mixed = Engine.TensorMatMul(op, timeFirst);
        return Engine.TensorPermute(
            Engine.Reshape(mixed, new[] { time, assets, _features }), new[] { 1, 0, 2 });
    }

    /// <summary>
    /// Mixes assets by matmul against the row-normalized graph, in ONE product.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The asset axis is already leading, so <c>[assets, assets] x [assets, time*features]</c> mixes
    /// every timestep simultaneously. Written as a matmul rather than an indexing loop for the same
    /// reason as the temporal operator: element-wise writes into a fresh tensor are invisible to the
    /// tape, so the graph projection would receive no gradient and the model would train without ever
    /// learning cross-asset structure.
    /// </para>
    /// <para>
    /// A zero graph entry removes the edge. An asset with no neighbours gets an identity row, so it
    /// keeps its own representation instead of collapsing to zero.
    /// </para>
    /// </remarks>
    private Tensor<T> GraphMix(Tensor<T> x, Matrix<T> graph, int assets, int time)
    {
        var normalized = new Tensor<T>(new[] { assets, assets });
        for (int a = 0; a < assets; a++)
        {
            double sum = 0.0;
            for (int n = 0; n < assets; n++)
            {
                double w = Ops.ToDouble(graph[a, n]);
                if (w > 0.0) sum += w;
            }

            if (sum <= 0.0)
            {
                normalized[(a * assets) + a] = Ops.One;
                continue;
            }

            for (int n = 0; n < assets; n++)
            {
                double w = Ops.ToDouble(graph[a, n]);
                if (w > 0.0) normalized[(a * assets) + n] = Ops.FromDouble(w / sum);
            }
        }

        var flat = Engine.Reshape(x, new[] { assets, time * _features });
        return Engine.Reshape(Engine.TensorMatMul(normalized, flat), new[] { assets, time, _features });
    }

}
