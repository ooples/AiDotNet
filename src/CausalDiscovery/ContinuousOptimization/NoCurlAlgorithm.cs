using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// NoCurl — DAG learning via curl-free constraints on the graph structure.
/// </summary>
/// <remarks>
/// <para>
/// NoCurl parameterizes the DAG using a variable ordering and restricted edge weights.
/// Given an ordering pi, the adjacency matrix W is constrained to have edges only from
/// earlier to later variables in the ordering — guaranteeing acyclicity by construction.
/// The algorithm alternates between optimizing the ordering (via greedy swaps) and
/// optimizing edge weights (via OLS regression restricted to the ordering).
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize with an ordering based on marginal variance</item>
/// <item>For the current ordering, compute optimal edge weights via OLS (restricted to the ordering)</item>
/// <item>Apply L1 soft-thresholding for sparsity</item>
/// <item>Try adjacent swaps to improve the L2 loss</item>
/// <item>Accept the best swap and repeat</item>
/// <item>Threshold small weights in the final W</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> NoCurl avoids the expensive matrix exponential used by NOTEARS.
/// Instead, it ensures acyclicity by finding a variable ordering (X before Y before Z)
/// and only allowing edges that follow this ordering. This is faster and simpler while
/// still finding good causal structures.
/// </para>
/// <para>
/// Reference: Yu et al. (2021), "DAGs with No Curl: An Efficient DAG Structure Learning
/// Approach", ICML.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("DAGs with No Curl: An Efficient DAG Structure Learning Approach", "https://proceedings.mlr.press/v139/yu21a.html", Year = 2021, Authors = "Yue Yu, Tian Gao, Naiyu Yin, Qiang Ji")]
public class NoCurlAlgorithm<T> : ContinuousOptimizationBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NoCurl";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes NoCurl with optional configuration.
    /// </summary>
    public NoCurlAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // NoCurl's variance ordering must be derived before z-scoring. After standardization every
        // diagonal variance is nominally one, so ordering those near-ties makes the graph depend on
        // floating-point roundoff and can change it under a uniform rescaling of the same data.
        var ordering = InitializeOrdering(data, d, n);
        var X = StandardizeData(data);

        // Compute sample covariance: S = (1/n) * X^T * X
        var S = ComputeCovarianceMatrix(X);

        // Iteratively improve ordering via adjacent swaps
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            bool improved = false;

            for (int pos = 0; pos < d - 1; pos++)
            {
                if (TrySwapAdjacent(X, S, ordering, pos, d, n))
                {
                    improved = true; // Keep the swap
                }
            }

            if (!improved) break;
        }

        // Build final W from optimal ordering with OLS weights and L1 thresholding:
        // every parent earlier in the ordering gets w = S[parent, target] / S[parent, parent].
        var position = new int[d];
        for (int idx = 0; idx < d; idx++)
            position[ordering[idx]] = idx;
        var precedes = new bool[d * d];
        for (int parent = 0; parent < d; parent++)
            for (int target = 0; target < d; target++)
                precedes[parent * d + target] = position[parent] < position[target];

        var covariance = Tensor<T>.FromMatrix(S);
        var ones = Engine.TensorOnesLike(covariance);
        var parentVariance = Engine.TensorBroadcastTo(Engine.TensorDiagonal(covariance).Reshape(new[] { d, 1 }), new[] { d, d });
        var usableParent = Engine.TensorSubtract(ones,
            Engine.TensorLessThan(parentVariance, Engine.TensorFullLike(covariance, NumOps.FromDouble(1e-10))));
        var rawWeight = Engine.TensorDivide(covariance, Engine.TensorWhere(usableParent, parentVariance, ones));
        var absWeight = Engine.TensorAbs(rawWeight);

        // L1 soft thresholding, kept when |w| > lambda1 and the shrunk weight reaches WThreshold.
        var thresholded = Engine.TensorMultiply(Engine.TensorSign(rawWeight),
            Engine.TensorSubtract(absWeight, Engine.TensorFullLike(absWeight, NumOps.FromDouble(Lambda1))));
        var keep = Engine.TensorMultiply(usableParent, Engine.TensorMultiply(
            Engine.TensorGreaterThan(absWeight, Engine.TensorFullLike(absWeight, NumOps.FromDouble(Lambda1))),
            Engine.TensorGreaterEqual(Engine.TensorAbs(thresholded), Engine.TensorFullLike(absWeight, NumOps.FromDouble(WThreshold)))));
        var zeros = Engine.TensorZerosLike(covariance);
        var W = Engine.TensorWhere(precedes, Engine.TensorWhere(keep, thresholded, zeros), zeros);

        return W.ToMatrix();
    }

    /// <summary>
    /// Swaps ordering positions <paramref name="pos"/> and <paramref name="pos"/> + 1 and keeps the
    /// swap only when it strictly lowers the ordering loss.
    /// </summary>
    /// <returns><c>true</c> when the swap was kept.</returns>
    private bool TrySwapAdjacent(Matrix<T> X, Matrix<T> S, int[] ordering, int pos, int d, int n)
    {
        // Compute loss with current ordering
        T currentLoss = ComputeOrderingLoss(X, S, ordering, d, n);

        // Try swapping positions pos and pos+1
        (ordering[pos], ordering[pos + 1]) = (ordering[pos + 1], ordering[pos]);

        T swappedLoss = ComputeOrderingLoss(X, S, ordering, d, n);

        if (NumOps.ToDouble(swappedLoss) < NumOps.ToDouble(currentLoss))
            return true;

        // Revert
        (ordering[pos], ordering[pos + 1]) = (ordering[pos + 1], ordering[pos]);
        return false;
    }

    /// <summary>
    /// Initializes ordering from the raw-data marginal variance (ascending).
    /// </summary>
    private int[] InitializeOrdering(Matrix<T> data, int d, int n)
    {
        var centered = CenterColumns(Tensor<T>.FromMatrix(data));
        var variances = Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, false);

        // Ascending variance; the stable sort breaks ties by column index.
        return Engine.TensorArgsort(variances, -1, false).ToArray();
    }

    /// <summary>
    /// Computes the total L2 loss for a given ordering.
    /// Loss = sum over targets of RSS(target | parents in ordering).
    /// </summary>
    private T ComputeOrderingLoss(Matrix<T> X, Matrix<T> S, int[] ordering, int d, int n)
    {
        // Reorder S so that row/column k is the k-th variable of the ordering.
        var order = new Tensor<int>((int[])ordering.Clone(), new[] { d });
        var ordered = Engine.TensorIndexSelect(Engine.TensorIndexSelect(Tensor<T>.FromMatrix(S), order, 0), order, 1);
        var ones = Engine.TensorOnesLike(ordered);

        // Variance reduction from parent: spt^2 / spp, for parents with spp >= 1e-10 only.
        var parentVariance = Engine.TensorBroadcastTo(Engine.TensorDiagonal(ordered).Reshape(new[] { d, 1 }), new[] { d, d });
        var usableParent = Engine.TensorSubtract(ones,
            Engine.TensorLessThan(parentVariance, Engine.TensorFullLike(ordered, NumOps.FromDouble(1e-10))));
        var reduction = Engine.TensorDivide(Engine.TensorSquare(ordered),
            Engine.TensorWhere(usableParent, parentVariance, ones));

        // Single best parent (greedy) among the earlier variables; a reduction must beat 0 to count.
        var candidate = Engine.TensorMultiply(Engine.TensorMultiply(usableParent, Engine.TensorTriu(ones, 1)),
            Engine.TensorGreaterThan(reduction, Engine.TensorZerosLike(ordered)));
        var bestReduction = Engine.ReduceMax(
            Engine.TensorWhere(candidate, reduction, Engine.TensorZerosLike(ordered)), new[] { 0 }, false);

        // Loss = sum over targets of (marginal variance - best reduction); the first has no parents.
        return Engine.TensorSum(Engine.TensorSubtract(Engine.TensorDiagonal(ordered), bestReduction));
    }
}
