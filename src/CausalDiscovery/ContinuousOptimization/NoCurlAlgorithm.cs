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
[ModelPaper("DAGs with No Curl: An Efficient DAG Structure Learning Approach", "https://proceedings.mlr.press/v139/yu21a.html", Year = 2021, Authors = "Yue Yu, Tian Gao, Naiyu Yin, Qiang Ji")]
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

        var X = StandardizeData(data);

        // Compute sample covariance: S = (1/n) * X^T * X
        var S = ComputeCovarianceMatrix(X);

        // Initialize ordering based on marginal variance (ascending)
        var ordering = InitializeOrdering(S, d);

        // Iteratively improve ordering via adjacent swaps
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            bool improved = false;

            for (int pos = 0; pos < d - 1; pos++)
            {
                // Compute loss with current ordering
                T currentLoss = ComputeOrderingLoss(X, S, ordering, d, n);

                // Try swapping positions pos and pos+1
                (ordering[pos], ordering[pos + 1]) = (ordering[pos + 1], ordering[pos]);

                T swappedLoss = ComputeOrderingLoss(X, S, ordering, d, n);

                if (NumOps.ToDouble(swappedLoss) < NumOps.ToDouble(currentLoss))
                {
                    improved = true; // Keep the swap
                }
                else
                {
                    // Revert
                    (ordering[pos], ordering[pos + 1]) = (ordering[pos + 1], ordering[pos]);
                }
            }

            if (!improved) break;
        }

        // Build final W from optimal ordering with OLS weights and L1 thresholding
        var W = new Matrix<T>(d, d);
        T lambdaT = NumOps.FromDouble(Lambda1);

        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];

            for (int predIdx = 0; predIdx < idx; predIdx++)
            {
                int parent = ordering[predIdx];

                // OLS weight: w = S[parent, target] / S[parent, parent]
                T sParent = S[parent, parent];
                if (NumOps.ToDouble(sParent) < 1e-10) continue;

                T rawWeight = NumOps.Divide(S[parent, target], sParent);
                T absWeight = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(rawWeight)));

                // L1 soft thresholding
                if (NumOps.ToDouble(absWeight) > Lambda1)
                {
                    T sign = NumOps.FromDouble(Math.Sign(NumOps.ToDouble(rawWeight)));
                    T thresholded = NumOps.Multiply(sign, NumOps.Subtract(absWeight, lambdaT));

                    if (Math.Abs(NumOps.ToDouble(thresholded)) >= WThreshold)
                        W[parent, target] = thresholded;
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Initializes ordering based on diagonal of S (marginal variance, ascending).
    /// </summary>
    private int[] InitializeOrdering(Matrix<T> S, int d)
    {
        var ordering = Enumerable.Range(0, d).ToArray();
        Array.Sort(ordering, (a, b) =>
            NumOps.ToDouble(S[a, a]).CompareTo(NumOps.ToDouble(S[b, b])));
        return ordering;
    }

    /// <summary>
    /// Computes the total L2 loss for a given ordering.
    /// Loss = sum over targets of RSS(target | parents in ordering).
    /// </summary>
    private T ComputeOrderingLoss(Matrix<T> X, Matrix<T> S, int[] ordering, int d, int n)
    {
        T totalLoss = NumOps.Zero;
        T nT = NumOps.FromDouble(n);

        for (int idx = 0; idx < d; idx++)
        {
            int target = ordering[idx];
            T targetVar = S[target, target];

            if (idx == 0)
            {
                // No parents — residual variance = marginal variance
                totalLoss = NumOps.Add(totalLoss, targetVar);
                continue;
            }

            // Single best parent (greedy): the one with highest |S[parent, target]| / S[parent, parent]
            T bestReduction = NumOps.Zero;
            for (int predIdx = 0; predIdx < idx; predIdx++)
            {
                int parent = ordering[predIdx];
                T spp = S[parent, parent];
                if (NumOps.ToDouble(spp) < 1e-10) continue;

                T spt = S[parent, target];
                // Variance reduction from parent: spt^2 / spp
                T reduction = NumOps.Divide(NumOps.Multiply(spt, spt), spp);
                if (NumOps.ToDouble(reduction) > NumOps.ToDouble(bestReduction))
                    bestReduction = reduction;
            }

            T residualVar = NumOps.Subtract(targetVar, bestReduction);
            totalLoss = NumOps.Add(totalLoss, residualVar);
        }

        return totalLoss;
    }
}
