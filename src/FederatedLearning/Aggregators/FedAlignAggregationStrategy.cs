using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedAlign (Feature Alignment) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different clients with different data can learn different "languages"
/// for representing the same concepts. FedAlign adds a regularizer during local training that
/// forces each client's feature space to align with shared anchor representations, so all clients
/// speak the same "language" when their models are combined.</para>
///
/// <para>Local training objective:</para>
/// <code>L = L_task + alpha * D(f_local(anchors), f_global(anchors))</code>
/// <para>where anchors are shared reference inputs and D measures representation distance
/// using either L2 distance or CKA (Centered Kernel Alignment).</para>
///
/// <para>Protocol:</para>
/// <list type="number">
/// <item>Server generates or selects anchor inputs (small shared dataset)</item>
/// <item>Server sends global model + anchors to clients</item>
/// <item>Each client computes f_global(anchors) and f_local(anchors) during training</item>
/// <item>Alignment loss penalizes distance between the two representations</item>
/// </list>
///
/// <para>Reference: Mendieta, M., et al. (2022). "Local Learning Matters: Rethinking Data Heterogeneity
/// in Federated Learning." CVPR 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedAlignAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _alignmentWeight;
    private readonly AlignmentDistanceMetric _distanceMetric;
    private Matrix<T>? _anchorInputs;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedAlignAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="alignmentWeight">Weight of the alignment loss (alpha). Default: 1.0 per paper.</param>
    /// <param name="distanceMetric">Distance metric for feature alignment. Default: L2.</param>
    public FedAlignAggregationStrategy(
        double alignmentWeight = 1.0,
        AlignmentDistanceMetric distanceMetric = AlignmentDistanceMetric.L2)
    {
        if (alignmentWeight < 0)
        {
            throw new ArgumentException("Alignment weight must be non-negative.", nameof(alignmentWeight));
        }

        _alignmentWeight = alignmentWeight;
        _distanceMetric = distanceMetric;
    }

    /// <summary>
    /// Aggregates client models using standard weighted averaging.
    /// </summary>
    /// <remarks>
    /// <para><b>By design</b>, FedAlign uses standard FedAvg for aggregation. The alignment
    /// behavior is applied during <em>local training</em> via <see cref="ComputeAlignmentLoss"/>
    /// and <see cref="ComputeTotalLoss"/>. The aggregation step itself is unchanged from FedAvg.</para>
    /// </remarks>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Registers anchor inputs that all clients use for alignment.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Anchors are a small set of reference inputs that every client
    /// and the global model process. By comparing how each client represents these anchors
    /// versus how the global model represents them, we can measure and correct feature misalignment.</para>
    /// </remarks>
    /// <param name="anchors">Anchor input matrix [numAnchors x inputDim].</param>
    public void SetAnchors(Matrix<T> anchors)
    {
        Guard.NotNull(anchors);
        if (anchors.Rows < 2 || anchors.Columns < 1)
        {
            throw new ArgumentException("Anchor matrix must have at least 2 rows and 1 column.", nameof(anchors));
        }

        _anchorInputs = anchors.Clone();
    }

    /// <summary>
    /// Gets the registered anchor inputs, or null if none have been set.
    /// </summary>
    public Matrix<T>? Anchors => _anchorInputs?.Clone();

    /// <summary>
    /// Computes the feature alignment loss between local and global model representations on anchors.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Both the local model and the global model process the same
    /// anchor inputs to produce feature representations. This method measures how different
    /// those representations are. A large distance means the local model has "drifted" from
    /// the global model's representation space, and the loss encourages it to stay aligned.</para>
    /// </remarks>
    /// <param name="localFeatures">Local model features on anchors [numAnchors x featureDim].</param>
    /// <param name="globalFeatures">Global model features on anchors [numAnchors x featureDim].</param>
    /// <returns>The alignment loss: alpha * D(f_local, f_global).</returns>
    public T ComputeAlignmentLoss(Matrix<T> localFeatures, Matrix<T> globalFeatures)
    {
        Guard.NotNull(localFeatures);
        Guard.NotNull(globalFeatures);
        if (localFeatures.Rows != globalFeatures.Rows || localFeatures.Columns != globalFeatures.Columns)
        {
            throw new ArgumentException(
                $"Feature matrices must have the same shape. Local: [{localFeatures.Rows}x{localFeatures.Columns}], " +
                $"Global: [{globalFeatures.Rows}x{globalFeatures.Columns}].");
        }

        if (localFeatures.Rows < 2 || localFeatures.Columns < 1)
        {
            return NumOps.Zero;
        }

        if (_distanceMetric != AlignmentDistanceMetric.L2
            && _distanceMetric != AlignmentDistanceMetric.CKA
            && _distanceMetric != AlignmentDistanceMetric.MMD)
        {
            throw new ArgumentOutOfRangeException(nameof(_distanceMetric),
                $"Unknown alignment distance metric: {_distanceMetric}.");
        }

        // Half / BFloat16 features are evaluated in float: the sums of squares, Gram products and kernel
        // bandwidths below pass Half's 65504, and its 1e-8 / 1e-10 guards would round to 0.
        var local = Tensor<T>.FromMatrix(localFeatures);
        var global = Tensor<T>.FromMatrix(globalFeatures);
        T distance = AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.IsNarrowFloat<T>()
            ? NumOps.FromDouble(Distance(_distanceMetric, local.Cast<float>(), global.Cast<float>()))
            : Distance(_distanceMetric, local, global);

        return NumOps.Multiply(NumOps.FromDouble(_alignmentWeight), distance);
    }

    /// <summary>
    /// Computes the complete local training loss including task loss and alignment loss.
    /// </summary>
    /// <param name="taskLoss">The base task loss.</param>
    /// <param name="localFeatures">Local model features on anchors.</param>
    /// <param name="globalFeatures">Global model features on anchors.</param>
    /// <returns>L_total = L_task + alpha * D(f_local, f_global).</returns>
    public T ComputeTotalLoss(T taskLoss, Matrix<T> localFeatures, Matrix<T> globalFeatures)
    {
        var alignmentLoss = ComputeAlignmentLoss(localFeatures, globalFeatures);
        return NumOps.Add(taskLoss, alignmentLoss);
    }

    private static TV Distance<TV>(AlignmentDistanceMetric metric, Tensor<TV> local, Tensor<TV> global) => metric switch
    {
        AlignmentDistanceMetric.L2 => ComputeL2Distance(local, global),
        AlignmentDistanceMetric.CKA => ComputeCKADistance(local, global),
        _ => ComputeMMDDistance(local, global),
    };

    private static AiDotNet.Tensors.Interfaces.INumericOperations<TV> Ops<TV>() =>
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TV>();

    /// <summary>
    /// Mean squared L2 distance: (1/N) * sum_i ||f_local_i - f_global_i||^2.
    /// </summary>
    private static TV ComputeL2Distance<TV>(Tensor<TV> local, Tensor<TV> global)
    {
        var ops = Ops<TV>();
        var diff = Engine.TensorSubtract(local, global);
        return ops.Divide(Engine.TensorSumOfSquares(diff), ops.FromDouble(local.Shape[0]));
    }

    /// <summary>
    /// Linear CKA distance: 1 - ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F) on column-centred features.
    /// </summary>
    private static TV ComputeCKADistance<TV>(Tensor<TV> local, Tensor<TV> global)
    {
        var ops = Ops<TV>();

        // Center features (subtract column means).
        var localCentered = CenterFeatures(local);
        var globalCentered = CenterFeatures(global);

        // Linear CKA: HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
        // where HSIC with linear kernel = ||X^T Y||_F^2 / (n-1)^2. The denominator is formed as
        // sqrt(a) * sqrt(b) so the product of the two squared norms cannot overflow.
        var crossNorm = FrobeniusNormOfProduct(localCentered, globalCentered);
        var localNorm = FrobeniusNormOfProduct(localCentered, localCentered);
        var globalNorm = FrobeniusNormOfProduct(globalCentered, globalCentered);

        var denom = ops.Multiply(ops.Sqrt(localNorm), ops.Sqrt(globalNorm));
        var cka = ops.GreaterThan(denom, ops.FromDouble(1e-10))
            ? ops.Divide(crossNorm, denom)
            : ops.Zero;

        return ops.Subtract(ops.One, cka);
    }

    /// <summary>
    /// Unbiased MMD^2 U-statistic for paired samples (Gretton et al. 2012, Lemma 6 with m = n):
    /// (1 / (n(n-1))) sum_{i != j} [k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)], with an RBF
    /// kernel whose bandwidth is the median heuristic, clamped at 0.
    /// </summary>
    private static TV ComputeMMDDistance<TV>(Tensor<TV> x, Tensor<TV> y)
    {
        var ops = Ops<TV>();
        int n = x.Shape[0];

        // Compute median of pairwise distances for bandwidth selection.
        var medianDist = EstimateMedianPairwiseDistance(x, n);
        var floor = ops.FromDouble(1e-8);
        var bandwidth = ops.GreaterThan(medianDist, floor) ? medianDist : floor;
        var negGamma = ops.Negate(ops.Divide(ops.One,
            ops.Multiply(ops.FromDouble(2.0), ops.Multiply(bandwidth, bandwidth))));

        // Sums over i != j: the full kernel sum minus its diagonal (k(x_i, x_i) = 1, k(x_i, y_i) = exp(-gamma ||x_i - y_i||^2)).
        var kxx = OffDiagonalKernelSum(PairwiseSquaredDistances(x, x), negGamma);
        var kyy = OffDiagonalKernelSum(PairwiseSquaredDistances(y, y), negGamma);
        var kxy = OffDiagonalKernelSum(PairwiseSquaredDistances(x, y), negGamma);

        var normFactor = ops.FromDouble(n * (n - 1));
        var mmd = ops.Divide(
            ops.Add(ops.Subtract(kxx, ops.Multiply(ops.FromDouble(2.0), kxy)), kyy),
            normFactor);
        return ops.GreaterThan(mmd, ops.Zero) ? mmd : ops.Zero;
    }

    /// <summary>Subtracts each column's mean from an [n, d] feature tensor.</summary>
    private static Tensor<TV> CenterFeatures<TV>(Tensor<TV> features)
    {
        var means = Engine.ReduceMean(features, new[] { 0 }, true);
        return Engine.TensorSubtract(features, Engine.TensorBroadcastTo(means, new[] { features.Shape[0], features.Shape[1] }));
    }

    /// <summary>||A^T B||_F^2 for [n, d] feature tensors.</summary>
    private static TV FrobeniusNormOfProduct<TV>(Tensor<TV> a, Tensor<TV> b)
        => Engine.TensorSumOfSquares(Engine.TensorMatMul(Engine.TensorTranspose(a), b));

    /// <summary>[n, m] matrix of squared Euclidean distances between the rows of a [n, d] and b [m, d].</summary>
    private static Tensor<TV> PairwiseSquaredDistances<TV>(Tensor<TV> a, Tensor<TV> b)
    {
        int n = a.Shape[0];
        int m = b.Shape[0];
        var aSq = Engine.TensorBroadcastTo(Engine.ReduceSum(Engine.TensorSquare(a), new[] { 1 }, true), new[] { n, m });
        var bSq = Engine.TensorBroadcastTo(
            Engine.TensorTranspose(Engine.ReduceSum(Engine.TensorSquare(b), new[] { 1 }, true)), new[] { n, m });
        var cross = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(a, Engine.TensorTranspose(b)), Ops<TV>().FromDouble(2.0));
        return Engine.TensorClampMin(Engine.TensorSubtract(Engine.TensorAdd(aSq, bSq), cross), Ops<TV>().Zero);
    }

    /// <summary>Sum over i != j of exp(-gamma * d2[i, j]).</summary>
    private static TV OffDiagonalKernelSum<TV>(Tensor<TV> squaredDistances, TV negGamma)
    {
        var kernel = Engine.TensorExp(Engine.TensorMultiplyScalar(squaredDistances, negGamma));
        return Ops<TV>().Subtract(Engine.TensorSum(kernel), Engine.TensorTrace(kernel));
    }

    /// <summary>
    /// Median of a deterministic sample (every step-th pair i &lt; j in row order, at most 500) of the
    /// pairwise Euclidean distances between the rows of <paramref name="features"/>.
    /// </summary>
    private static TV EstimateMedianPairwiseDistance<TV>(Tensor<TV> features, int n)
    {
        // Use a sample of pairwise distances for efficiency.
        int maxPairs = Math.Min(n * (n - 1) / 2, 500);
        if (maxPairs == 0)
        {
            return Ops<TV>().One;
        }

        var pairIndices = new List<int>(maxPairs);
        int step = Math.Max(1, n * (n - 1) / 2 / maxPairs);
        int count = 0;

        for (int i = 0; i < n && pairIndices.Count < maxPairs; i++)
        {
            for (int j = i + 1; j < n && pairIndices.Count < maxPairs; j++)
            {
                count++;
                if (count % step == 0)
                {
                    pairIndices.Add((i * n) + j);
                }
            }
        }

        if (pairIndices.Count == 0)
        {
            return Ops<TV>().One;
        }

        var allDistances = Engine.TensorSqrt(PairwiseSquaredDistances(features, features)).Reshape(n * n);
        var sampled = Engine.TensorGather(allDistances, Vector<int>.Wrap(pairIndices.ToArray()).AsTensor(), 0);
        var sorted = Engine.TensorSort(sampled, 0, false).Values;
        return sorted[pairIndices.Count / 2];
    }

    /// <summary>Gets the feature alignment weight (alpha).</summary>
    public double AlignmentWeight => _alignmentWeight;

    /// <summary>Gets the distance metric used for alignment.</summary>
    public AlignmentDistanceMetric DistanceMetric => _distanceMetric;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedAlign(\u03b1={_alignmentWeight},{_distanceMetric})";
}

/// <summary>
/// Distance metric for feature alignment in FedAlign.
/// </summary>
public enum AlignmentDistanceMetric
{
    /// <summary>Mean squared L2 distance between representations.</summary>
    L2,

    /// <summary>Centered Kernel Alignment (CKA) — invariant to invertible linear transforms.</summary>
    CKA,

    /// <summary>Maximum Mean Discrepancy (MMD) with RBF kernel.</summary>
    MMD
}
