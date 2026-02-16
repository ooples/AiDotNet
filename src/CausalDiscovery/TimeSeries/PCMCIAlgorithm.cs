using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// PCMCI — PC algorithm for Momentary Conditional Independence in time series.
/// </summary>
/// <remarks>
/// <para>
/// PCMCI combines a condition-selection step (based on PC's skeleton discovery) with
/// momentary conditional independence (MCI) tests. It first identifies the relevant
/// lagged parents of each variable, then tests for direct causal links conditioned
/// on those parents.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>PC₁: For each variable, identify candidate lagged parents using iterative CI tests</item>
/// <item>MCI: Test X(t-τ) → Y(t) conditioned on parents of both X and Y</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> PCMCI is designed for time series where you want to know which
/// variables' past values help predict other variables' current values. It's more reliable
/// than simple Granger causality because it conditions on the right set of variables.
/// </para>
/// <para>
/// Reference: Runge et al. (2019), "Detecting and Quantifying Causal Associations in
/// Large Nonlinear Time Series Datasets", Science Advances.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PCMCIAlgorithm<T> : TimeSeriesCausalBase<T>
{
    private double _alpha = 0.05;

    /// <inheritdoc/>
    public override string Name => "PCMCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public PCMCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        if (options?.SignificanceLevel.HasValue == true) _alpha = options.SignificanceLevel.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * d) return new Matrix<T>(d, d);

        // Step 1: PC condition selection — find candidate parents for each variable
        var candidateParents = new HashSet<(int var, int lag)>[d];
        for (int j = 0; j < d; j++)
        {
            candidateParents[j] = [];
            // Initialize with all lagged variables
            for (int i = 0; i < d; i++)
                for (int lag = 1; lag <= MaxLag; lag++)
                    candidateParents[j].Add((i, lag));

            // Iteratively remove based on conditional independence tests
            bool changed = true;
            while (changed)
            {
                changed = false;
                var toRemove = new List<(int var, int lag)>();

                foreach (var (pVar, pLag) in candidateParents[j])
                {
                    // Build conditioning set: all other candidate parents except the one being tested
                    var condSet = new HashSet<(int var, int lag)>(candidateParents[j]);
                    condSet.Remove((pVar, pLag));

                    double partCorr = ComputeLaggedPartialCorrelation(
                        data, j, pVar, pLag, condSet, effectiveN);

                    // Fisher z-transform to get p-value
                    double pValue = FisherZTestPValue(partCorr, effectiveN, condSet.Count);

                    if (pValue > _alpha) // not significant → conditionally independent → remove
                    {
                        toRemove.Add((pVar, pLag));
                        changed = true;
                    }
                }

                foreach (var item in toRemove)
                    candidateParents[j].Remove(item);
            }
        }

        // Step 2: MCI tests — test with conditioning on parents of BOTH source and target
        var W = new Matrix<T>(d, d);
        for (int j = 0; j < d; j++)
        {
            foreach (var (pVar, pLag) in candidateParents[j])
            {
                // MCI conditioning set: parents of Y (excluding current link) ∪ parents of X
                var mciCondSet = new HashSet<(int var, int lag)>(candidateParents[j]);
                mciCondSet.Remove((pVar, pLag));

                // Add parents of the source variable
                foreach (var parent in candidateParents[pVar])
                    mciCondSet.Add(parent);

                double partCorr = ComputeLaggedPartialCorrelation(
                    data, j, pVar, pLag, mciCondSet, effectiveN);

                double pValue = FisherZTestPValue(partCorr, effectiveN, mciCondSet.Count);

                if (pValue <= _alpha) // significant causal link
                {
                    // Aggregate across lags: take maximum absolute partial correlation
                    double absPartCorr = Math.Abs(partCorr);
                    double current = NumOps.ToDouble(W[pVar, j]);
                    if (absPartCorr > current)
                        W[pVar, j] = NumOps.FromDouble(absPartCorr);
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Computes partial correlation between target(t) and source(t-lag) conditioned on a set of
    /// lagged parent variables, using OLS residualization.
    /// </summary>
    private double ComputeLaggedPartialCorrelation(Matrix<T> data,
        int target, int source, int lag, HashSet<(int var, int lag)> condSet, int effectiveN)
    {
        int n = data.Rows;
        int offset = n - effectiveN;

        // Build target and source vectors
        var targetVals = new Vector<T>(effectiveN);
        var sourceVals = new Vector<T>(effectiveN);

        for (int t = 0; t < effectiveN; t++)
        {
            targetVals[t] = data[offset + t, target];
            sourceVals[t] = data[offset + t - lag, source];
        }

        if (condSet.Count == 0)
        {
            // No conditioning: compute simple Pearson correlation
            return PearsonCorrelation(sourceVals, targetVals, effectiveN);
        }

        // Build conditioning matrix (each column is a lagged conditioning variable)
        int numCond = condSet.Count;
        var condMatrix = new Matrix<T>(effectiveN, numCond);
        int col = 0;
        foreach (var (cVar, cLag) in condSet)
        {
            for (int t = 0; t < effectiveN; t++)
            {
                int tIdx = offset + t - cLag;
                condMatrix[t, col] = (tIdx >= 0 && tIdx < n) ? data[tIdx, cVar] : NumOps.Zero;
            }
            col++;
        }

        // Residualize both target and source on conditioning set via OLS
        var residTarget = OLSResiduals(condMatrix, targetVals, effectiveN, numCond);
        var residSource = OLSResiduals(condMatrix, sourceVals, effectiveN, numCond);

        return PearsonCorrelation(residSource, residTarget, effectiveN);
    }

    /// <summary>
    /// Computes OLS residuals: y - Z * (Z'Z)^{-1} Z'y using the normal equations.
    /// </summary>
    private Vector<T> OLSResiduals(Matrix<T> Z, Vector<T> y, int n, int p)
    {
        // Build normal equations: ZtZ * beta = Zty
        var ZtZ = new Matrix<T>(p, p);
        var Zty = new Vector<T>(p);

        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int t = 0; t < n; t++)
                    sum = NumOps.Add(sum, NumOps.Multiply(Z[t, i], Z[t, j]));
                ZtZ[i, j] = sum;
                ZtZ[j, i] = sum;
            }

            T sumZy = NumOps.Zero;
            for (int t = 0; t < n; t++)
                sumZy = NumOps.Add(sumZy, NumOps.Multiply(Z[t, i], y[t]));
            Zty[i] = sumZy;
        }

        // Ridge for stability
        T ridge = NumOps.FromDouble(1e-10);
        for (int i = 0; i < p; i++)
            ZtZ[i, i] = NumOps.Add(ZtZ[i, i], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(ZtZ, Zty, MatrixDecompositionType.Lu);

        // Compute residuals: y - Z * beta
        var residuals = new Vector<T>(n);
        for (int t = 0; t < n; t++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < p; j++)
                pred = NumOps.Add(pred, NumOps.Multiply(Z[t, j], beta[j]));
            residuals[t] = NumOps.Subtract(y[t], pred);
        }

        return residuals;
    }

    /// <summary>
    /// Computes Pearson correlation between two vectors.
    /// </summary>
    private double PearsonCorrelation(Vector<T> x, Vector<T> y, int n)
    {
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += NumOps.ToDouble(x[i]); my += NumOps.ToDouble(y[i]); }
        mx /= n; my /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = NumOps.ToDouble(x[i]) - mx, dy = NumOps.ToDouble(y[i]) - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    /// <summary>
    /// Computes a two-sided p-value from a partial correlation using the Fisher z-transform.
    /// z = arctanh(r) * sqrt(n - k - 3), where k is the conditioning set size.
    /// Under H0 (conditional independence), z ~ N(0,1).
    /// </summary>
    private static double FisherZTestPValue(double partialCorrelation, int n, int condSetSize)
    {
        int dof = n - condSetSize - 3;
        if (dof < 1) return 1.0; // insufficient samples

        // Fisher z-transform: z = arctanh(r)
        // Clamp to avoid infinity at ±1
        double r = Math.Max(-0.9999, Math.Min(0.9999, partialCorrelation));
        double z = 0.5 * Math.Log((1 + r) / (1 - r)); // arctanh

        // Test statistic: z * sqrt(dof)
        double testStat = Math.Abs(z) * Math.Sqrt(dof);

        // Two-sided p-value from standard normal: p = 2 * (1 - Φ(|testStat|))
        return 2.0 * NormalCdfComplement(testStat);
    }

    /// <summary>
    /// Computes 1 - Φ(x) for the standard normal distribution using the Abramowitz and Stegun approximation.
    /// Maximum error: 7.5e-8.
    /// </summary>
    private static double NormalCdfComplement(double x)
    {
        if (x < 0) return 1.0 - NormalCdfComplement(-x);

        // Abramowitz and Stegun formula 26.2.17
        const double p = 0.2316419;
        const double b1 = 0.319381530;
        const double b2 = -0.356563782;
        const double b3 = 1.781477937;
        const double b4 = -1.821255978;
        const double b5 = 1.330274429;

        double t = 1.0 / (1.0 + p * x);
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;

        double phi = Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);
        return phi * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5);
    }
}
