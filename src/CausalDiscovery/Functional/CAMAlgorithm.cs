using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CAM (Causal Additive Model) — order-based causal discovery with additive nonparametric regression.
/// </summary>
/// <remarks>
/// <para>
/// CAM discovers causal structure in two stages:
/// <list type="number">
/// <item><b>Ordering:</b> Greedily selects the next variable that minimizes residual variance
/// when regressed on already-ordered variables using additive (kernel-smoothed) regression.</item>
/// <item><b>Pruning:</b> For each variable, tests whether each parent's additive contribution
/// is significant by comparing residual variance with and without that parent (likelihood ratio).</item>
/// </list>
/// </para>
/// <para>
/// The model assumes X_j = Σ_k f_k(X_pa(j)_k) + ε_j where f_k are smooth nonparametric
/// functions estimated via Nadaraya–Watson kernel regression.
/// </para>
/// <para>
/// <b>For Beginners:</b> CAM figures out the causal order of variables by finding which
/// variable is best predicted by the others using flexible (nonlinear) functions. It then
/// trims weak connections. Unlike linear methods, CAM can discover relationships where
/// the effect of one variable on another is curved or nonlinear.
/// </para>
/// <para>
/// Reference: Buhlmann et al. (2014), "CAM: Causal Additive Models, High-Dimensional
/// Order Search and Penalized Regression", Annals of Statistics.
/// </para>
/// </remarks>
internal class CAMAlgorithm<T> : FunctionalBase<T>
{
    private double _threshold = 0.1;

    /// <inheritdoc/>
    public override string Name => "CAM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CAMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true) _threshold = options.EdgeThreshold.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var standardized = StandardizeData(data);

        // Stage 1: Greedy ordering via residual variance minimization
        var causalOrder = EstimateCausalOrder(standardized, n, d);

        // Stage 2: Fit additive model in causal order and prune weak edges
        return FitAndPrune(standardized, n, d, causalOrder);
    }

    private List<int> EstimateCausalOrder(Matrix<T> data, int n, int d)
    {
        var remaining = new HashSet<int>(Enumerable.Range(0, d));
        var ordered = new List<int>();

        // First variable: least predictable by others
        int firstVar = 0;
        double minPredictability = double.MaxValue;
        foreach (int j in remaining)
        {
            double predictability = 0;
            var colJ = data.GetColumn(j);
            for (int k = 0; k < d; k++)
            {
                if (k == j) continue;
                predictability += Math.Abs(ComputeCorrelation(colJ, data.GetColumn(k)));
            }
            if (predictability < minPredictability)
            {
                minPredictability = predictability;
                firstVar = j;
            }
        }
        ordered.Add(firstVar);
        remaining.Remove(firstVar);

        while (remaining.Count > 0)
        {
            int bestVar = -1;
            double bestResidVar = double.MaxValue;

            foreach (int candidate in remaining)
            {
                double residVar = ComputeAdditiveResidualVariance(data, n, candidate, ordered);
                if (residVar < bestResidVar)
                {
                    bestResidVar = residVar;
                    bestVar = candidate;
                }
            }

            if (bestVar < 0) bestVar = remaining.First();
            ordered.Add(bestVar);
            remaining.Remove(bestVar);
        }

        return ordered;
    }

    private Matrix<T> FitAndPrune(Matrix<T> data, int n, int d, List<int> causalOrder)
    {
        var W = new Matrix<T>(d, d);

        for (int idx = 1; idx < d; idx++)
        {
            int target = causalOrder[idx];
            var predecessors = causalOrder.GetRange(0, idx);

            double fullResidVar = ComputeAdditiveResidualVariance(data, n, target, predecessors);

            foreach (int parent in predecessors)
            {
                var reducedPredecessors = predecessors.Where(p => p != parent).ToList();
                double reducedResidVar = reducedPredecessors.Count > 0
                    ? ComputeAdditiveResidualVariance(data, n, target, reducedPredecessors)
                    : NumOps.ToDouble(ComputeColumnVariance(data, target));

                double varianceReduction = reducedResidVar - fullResidVar;
                double llRatio = (fullResidVar > 1e-15 && reducedResidVar > 1e-15)
                    ? n * Math.Log(reducedResidVar / fullResidVar)
                    : 0;

                double bicPenalty = Math.Log(n);

                if (llRatio > bicPenalty && varianceReduction > _threshold * _threshold)
                {
                    W[parent, target] = NumOps.FromDouble(Math.Sqrt(Math.Max(0, varianceReduction)));
                }
            }
        }

        return W;
    }

    private double ComputeAdditiveResidualVariance(Matrix<T> data, int n, int target, List<int> parents)
    {
        if (parents.Count == 0)
            return NumOps.ToDouble(ComputeColumnVariance(data, target));

        // Backfitting algorithm for additive model
        var components = new Vector<T>[parents.Count];
        for (int k = 0; k < parents.Count; k++)
            components[k] = new Vector<T>(n);

        T nT = NumOps.FromDouble(n);
        T targetMean = NumOps.Zero;
        for (int i = 0; i < n; i++)
            targetMean = NumOps.Add(targetMean, data[i, target]);
        targetMean = NumOps.Divide(targetMean, nT);

        // Backfitting iterations
        for (int iter = 0; iter < 5; iter++)
        {
            for (int k = 0; k < parents.Count; k++)
            {
                var partialResid = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                {
                    T val = NumOps.Subtract(data[i, target], targetMean);
                    for (int j = 0; j < parents.Count; j++)
                    {
                        if (j != k) val = NumOps.Subtract(val, components[j][i]);
                    }
                    partialResid[i] = val;
                }

                components[k] = KernelSmooth(data, parents[k], partialResid);

                // Center the component
                T compMean = NumOps.Zero;
                for (int i = 0; i < n; i++) compMean = NumOps.Add(compMean, components[k][i]);
                compMean = NumOps.Divide(compMean, nT);
                for (int i = 0; i < n; i++)
                    components[k][i] = NumOps.Subtract(components[k][i], compMean);
            }
        }

        // Compute residual variance
        T residVar = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T fit = targetMean;
            for (int k = 0; k < parents.Count; k++) fit = NumOps.Add(fit, components[k][i]);
            T resid = NumOps.Subtract(data[i, target], fit);
            residVar = NumOps.Add(residVar, NumOps.Multiply(resid, resid));
        }

        return NumOps.ToDouble(NumOps.Divide(residVar, nT));
    }
}
