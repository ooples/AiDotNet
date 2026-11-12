using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Detecting and Quantifying Causal Associations in Large Nonlinear Time Series Datasets", "https://doi.org/10.1126/sciadv.aau4996", Year = 2019, Authors = "Jakob Runge, Peer Nowack, Marlene Kretschmer, Seth Flaxman, Dino Sejdinovic")]
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

        // Step 1: PC condition selection — find candidate parents for each variable.
        // Uses the PC algorithm's iterative conditioning set size expansion:
        // test with conditioning set size 0, then 1, then 2, etc.
        // This prevents over-conditioning where all edges are removed because
        // perfectly correlated variables become conditionally independent when
        // conditioned on all other variables simultaneously.
        var candidateParents = new HashSet<(int var, int lag)>[d];
        for (int j = 0; j < d; j++)
        {
            candidateParents[j] = [];
            // Initialize with all lagged variables
            for (int i = 0; i < d; i++)
                for (int lag = 1; lag <= MaxLag; lag++)
                    candidateParents[j].Add((i, lag));

            // Iterate over conditioning set sizes (PC algorithm approach)
            for (int condSize = 0; condSize < candidateParents[j].Count; condSize++)
            {
                var toRemove = new List<(int var, int lag)>();
                var currentParents = new List<(int var, int lag)>(candidateParents[j]);

                foreach (var (pVar, pLag) in currentParents)
                {
                    // Build conditioning set: pick `condSize` other parents (excluding current)
                    var otherParents = currentParents.Where(p => p != (pVar, pLag)).ToList();
                    if (otherParents.Count < condSize) continue;

                    // Use first condSize parents as conditioning set
                    var condSet = new HashSet<(int var, int lag)>(otherParents.Take(condSize));

                    double partCorr = ComputeLaggedPartialCorrelation(
                        data, j, pVar, pLag, condSet, effectiveN);

                    double pValue = FisherZTestPValue(partCorr, effectiveN, condSet.Count);

                    if (pValue > _alpha) // not significant → conditionally independent → remove
                    {
                        toRemove.Add((pVar, pLag));
                    }
                }

                foreach (var item in toRemove)
                    candidateParents[j].Remove(item);

                // Stop if no candidates left or conditioning set size exceeds remaining candidates
                if (candidateParents[j].Count <= condSize + 1)
                    break;
            }
        }

        // Step 2: MCI tests — test with conditioning on parents of BOTH source and target
        var weights = new Dictionary<int, double>();
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
                    double current = weights.TryGetValue(pVar * d + j, out double existing) ? existing : 0;
                    if (absPartCorr > current)
                        weights[pVar * d + j] = absPartCorr;
                }
            }
        }

        // Fallback: With deterministic or near-deterministic data, conditioning on
        // correlated parents makes all partial correlations zero, removing all edges.
        // In this case, use unconditional lagged cross-correlation to detect relationships.
        bool hasEdges = weights.Any(edge => edge.Key / d != edge.Key % d && Math.Abs(edge.Value) > 0);

        if (!hasEdges)
        {
            int offset = n - effectiveN;
            var x = Tensor<T>.FromMatrix(data);
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    double maxCorr = 0;
                    int bestSource = i, bestTarget = j;

                    for (int lag = 1; lag <= MaxLag; lag++)
                    {
                        // Build source(t-lag) and target(t) vectors
                        double corr = Math.Abs(SeriesCorrelation(
                            LaggedSeries(x, i, lag, offset, effectiveN), LaggedSeries(x, j, 0, offset, effectiveN)));
                        if (corr > maxCorr) { maxCorr = corr; bestSource = i; bestTarget = j; }

                        // Also check reverse direction
                        corr = Math.Abs(SeriesCorrelation(
                            LaggedSeries(x, j, lag, offset, effectiveN), LaggedSeries(x, i, 0, offset, effectiveN)));
                        if (corr > maxCorr) { maxCorr = corr; bestSource = j; bestTarget = i; }
                    }

                    if (maxCorr >= 0.1)
                        weights[bestSource * d + bestTarget] = maxCorr;
                }
            }
        }

        return CausalTensorHelpers.FromStatistics<T>(d, d, weights);
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
        var x = Tensor<T>.FromMatrix(data);

        // Build target and source vectors
        // offset = MaxLag, lag <= MaxLag, so the source rows offset + t - lag are always >= 0
        var targetVals = LaggedSeries(x, target, 0, offset, effectiveN);
        var sourceVals = LaggedSeries(x, source, lag, offset, effectiveN);

        if (condSet.Count == 0)
        {
            // No conditioning: compute simple Pearson correlation
            return SeriesCorrelation(sourceVals, targetVals);
        }

        // Build conditioning matrix (each column is a lagged conditioning variable; rows outside the data are 0)
        var condColumns = condSet.Select(c => LaggedSeries(x, c.var, c.lag, offset, effectiveN).Reshape(new[] { effectiveN, 1 }));
        var condMatrix = Engine.TensorConcatenate(condColumns.ToArray(), 1);

        // Residualize both target and source on conditioning set via OLS
        var residTarget = OlsResidualSeries(condMatrix, targetVals, false, 1e-10, MatrixDecompositionType.Lu);
        var residSource = OlsResidualSeries(condMatrix, sourceVals, false, 1e-10, MatrixDecompositionType.Lu);

        return SeriesCorrelation(residSource, residTarget);
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
