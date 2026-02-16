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
    private double _bandwidth = 0.5;

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
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        X = StandardizeData(X, n, d);

        // Estimate bandwidth from data (Scott's rule)
        _bandwidth = Math.Pow(n, -1.0 / 5.0);

        // Stage 1: Greedy ordering via residual variance minimization
        var causalOrder = EstimateCausalOrder(X, n, d);

        // Stage 2: Fit additive model in causal order and prune weak edges
        var W = FitAndPrune(X, n, d, causalOrder);

        return DoubleArrayToMatrix(W);
    }

    /// <summary>
    /// Greedily selects variables in causal order by choosing the next variable
    /// that is best explained by the already-ordered set via additive kernel regression.
    /// </summary>
    private List<int> EstimateCausalOrder(double[,] X, int n, int d)
    {
        var remaining = new HashSet<int>(Enumerable.Range(0, d));
        var ordered = new List<int>();

        // First variable: the one with highest marginal variance (least explained by others)
        // CAM picks the "root" as the variable with the smallest score when regressed on nothing
        // All have the same marginal variance after standardization, so pick the least predictable
        int firstVar = 0;
        double minPredictability = double.MaxValue;
        foreach (int j in remaining)
        {
            double predictability = 0;
            for (int k = 0; k < d; k++)
            {
                if (k == j) continue;
                predictability += Math.Abs(ComputeCorrelation(X, n, j, k));
            }
            if (predictability < minPredictability)
            {
                minPredictability = predictability;
                firstVar = j;
            }
        }
        ordered.Add(firstVar);
        remaining.Remove(firstVar);

        // Iteratively add the variable best explained by the current ordered set
        while (remaining.Count > 0)
        {
            int bestVar = -1;
            double bestResidVar = double.MaxValue;

            foreach (int candidate in remaining)
            {
                double residVar = ComputeAdditiveResidualVariance(X, n, candidate, ordered);
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

    /// <summary>
    /// Fits additive model for each variable on its predecessors in the causal order,
    /// then prunes edges whose contribution is below threshold.
    /// </summary>
    private double[,] FitAndPrune(double[,] X, int n, int d, List<int> causalOrder)
    {
        var W = new double[d, d];

        for (int idx = 1; idx < d; idx++)
        {
            int target = causalOrder[idx];
            var predecessors = causalOrder.GetRange(0, idx);

            // Test each predecessor's significance via variance reduction
            double fullResidVar = ComputeAdditiveResidualVariance(X, n, target, predecessors);

            foreach (int parent in predecessors)
            {
                // Compute residual variance without this parent
                var reducedPredecessors = predecessors.Where(p => p != parent).ToList();
                double reducedResidVar = reducedPredecessors.Count > 0
                    ? ComputeAdditiveResidualVariance(X, n, target, reducedPredecessors)
                    : ComputeVariance(X, n, target);

                // Variance reduction ratio as edge weight
                double varianceReduction = reducedResidVar - fullResidVar;

                // Log-likelihood ratio test (approximate): n * log(RSS_reduced / RSS_full)
                double llRatio = (fullResidVar > 1e-15 && reducedResidVar > 1e-15)
                    ? n * Math.Log(reducedResidVar / fullResidVar)
                    : 0;

                // BIC penalty for one additional nonparametric component
                double bicPenalty = Math.Log(n);

                if (llRatio > bicPenalty && varianceReduction > _threshold * _threshold)
                {
                    W[parent, target] = Math.Sqrt(Math.Max(0, varianceReduction));
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Computes residual variance of target after additive kernel regression on parents:
    /// target = Σ_k f_k(parent_k) + ε, estimated via Nadaraya–Watson kernel smoothing.
    /// </summary>
    private double ComputeAdditiveResidualVariance(double[,] X, int n, int target, List<int> parents)
    {
        if (parents.Count == 0)
            return ComputeVariance(X, n, target);

        // Backfitting algorithm for additive model
        var fitted = new double[n];
        var components = new double[parents.Count][];
        for (int k = 0; k < parents.Count; k++)
            components[k] = new double[n];

        // Initialize: partial residuals = target values
        var targetVals = new double[n];
        double targetMean = 0;
        for (int i = 0; i < n; i++)
        {
            targetVals[i] = X[i, target];
            targetMean += X[i, target];
        }
        targetMean /= n;

        // Backfitting iterations
        for (int iter = 0; iter < 5; iter++)
        {
            for (int k = 0; k < parents.Count; k++)
            {
                // Compute partial residuals: target - sum of other components
                var partialResid = new double[n];
                for (int i = 0; i < n; i++)
                {
                    partialResid[i] = targetVals[i] - targetMean;
                    for (int j = 0; j < parents.Count; j++)
                    {
                        if (j != k) partialResid[i] -= components[j][i];
                    }
                }

                // Smooth partial residuals against parent[k] using NW kernel
                components[k] = NadarayaWatsonSmooth(X, n, parents[k], partialResid);

                // Center the component
                double compMean = 0;
                for (int i = 0; i < n; i++) compMean += components[k][i];
                compMean /= n;
                for (int i = 0; i < n; i++) components[k][i] -= compMean;
            }
        }

        // Compute residual variance
        double residVar = 0;
        for (int i = 0; i < n; i++)
        {
            double fit = targetMean;
            for (int k = 0; k < parents.Count; k++) fit += components[k][i];
            double resid = targetVals[i] - fit;
            residVar += resid * resid;
        }

        return residVar / n;
    }

    /// <summary>
    /// Nadaraya–Watson kernel smoother: E[Y | X=x] = Σ K(x-xi) yi / Σ K(x-xi)
    /// using Gaussian kernel.
    /// </summary>
    private double[] NadarayaWatsonSmooth(double[,] X, int n, int predictor, double[] response)
    {
        var smoothed = new double[n];
        double h = _bandwidth;

        for (int i = 0; i < n; i++)
        {
            double xi = X[i, predictor];
            double numerator = 0;
            double denominator = 0;

            for (int j = 0; j < n; j++)
            {
                double diff = (xi - X[j, predictor]) / h;
                double kernel = Math.Exp(-0.5 * diff * diff);
                numerator += kernel * response[j];
                denominator += kernel;
            }

            smoothed[i] = denominator > 1e-15 ? numerator / denominator : 0;
        }

        return smoothed;
    }

    private static double ComputeVariance(double[,] X, int n, int col)
    {
        double mean = 0;
        for (int i = 0; i < n; i++) mean += X[i, col];
        mean /= n;
        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double d = X[i, col] - mean;
            variance += d * d;
        }
        return variance / n;
    }

    private static double ComputeCorrelation(double[,] X, int n, int i, int j)
    {
        double mi = 0, mj = 0;
        for (int k = 0; k < n; k++) { mi += X[k, i]; mj += X[k, j]; }
        mi /= n; mj /= n;
        double sij = 0, sii = 0, sjj = 0;
        for (int k = 0; k < n; k++)
        {
            double di = X[k, i] - mi, dj = X[k, j] - mj;
            sij += di * dj; sii += di * di; sjj += dj * dj;
        }
        return (sii > 1e-10 && sjj > 1e-10) ? sij / Math.Sqrt(sii * sjj) : 0;
    }
}
