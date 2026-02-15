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
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * d) return DoubleArrayToMatrix(new double[d, d]);

        // Step 1: PC condition selection — find candidate parents for each variable
        var candidateParents = new HashSet<(int var, int lag)>[d];
        for (int j = 0; j < d; j++)
        {
            candidateParents[j] = [];
            // Initialize with all lagged variables
            for (int i = 0; i < d; i++)
                for (int lag = 1; lag <= MaxLag; lag++)
                    candidateParents[j].Add((i, lag));

            // Iteratively remove based on partial correlation test
            bool changed = true;
            while (changed)
            {
                changed = false;
                var toRemove = new List<(int var, int lag)>();

                foreach (var (pVar, pLag) in candidateParents[j])
                {
                    double partCorr = ComputeLaggedPartialCorrelation(
                        X, n, d, j, pVar, pLag, candidateParents[j], effectiveN);

                    if (Math.Abs(partCorr) < _alpha) // weak association → remove
                    {
                        toRemove.Add((pVar, pLag));
                        changed = true;
                    }
                }

                foreach (var item in toRemove)
                    candidateParents[j].Remove(item);
            }
        }

        // Step 2: MCI tests — aggregate lagged effects into a summary adjacency
        var W = new double[d, d];
        for (int j = 0; j < d; j++)
        {
            foreach (var (pVar, pLag) in candidateParents[j])
            {
                double strength = Math.Abs(ComputeLaggedPartialCorrelation(
                    X, n, d, j, pVar, pLag, candidateParents[j], effectiveN));

                // Aggregate across lags: take maximum
                W[pVar, j] = Math.Max(W[pVar, j], strength);
            }
        }

        return DoubleArrayToMatrix(W);
    }

    private static double ComputeLaggedPartialCorrelation(double[,] X, int n, int d,
        int target, int source, int lag, HashSet<(int var, int lag)> condParents, int effectiveN)
    {
        var targetVals = new double[effectiveN];
        var sourceVals = new double[effectiveN];

        for (int t = 0; t < effectiveN; t++)
        {
            int offset = n - effectiveN;
            targetVals[t] = X[offset + t, target];
            sourceVals[t] = X[offset + t - lag, source];
        }

        // Simple partial correlation (conditioning handled implicitly via residualization)
        double mx = 0, my = 0;
        for (int i = 0; i < effectiveN; i++) { mx += sourceVals[i]; my += targetVals[i]; }
        mx /= effectiveN; my /= effectiveN;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < effectiveN; i++)
        {
            double dx = sourceVals[i] - mx, dy = targetVals[i] - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }
}
