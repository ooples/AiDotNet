using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// Granger Causality — time series causal discovery via predictive improvement.
/// </summary>
/// <remarks>
/// <para>
/// Granger causality tests whether the past values of one variable X improve the prediction
/// of another variable Y beyond what Y's own past values provide. If so, X "Granger-causes" Y.
/// </para>
/// <para>
/// <b>Test procedure for each pair (i → j):</b>
/// <list type="number">
/// <item>Fit a restricted model: Y_t = f(Y_{t-1}, ..., Y_{t-L}) — autoregressive on Y only</item>
/// <item>Fit an unrestricted model: Y_t = f(Y_{t-1}, ..., Y_{t-L}, X_{t-1}, ..., X_{t-L})</item>
/// <item>Compare using F-test: F = ((RSS_r - RSS_u) / L) / (RSS_u / (n - 2L))</item>
/// <item>If F is significant, X Granger-causes Y</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine predicting tomorrow's temperature. If knowing yesterday's
/// humidity helps predict temperature better than just knowing past temperatures alone,
/// then humidity "Granger-causes" temperature. This doesn't prove true causation but
/// indicates a useful predictive relationship.
/// </para>
/// <para>
/// Reference: Granger (1969), "Investigating Causal Relations by Econometric Models
/// and Cross-spectral Methods", Econometrica.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GrangerCausalityAlgorithm<T> : TimeSeriesCausalBase<T>
{
    private readonly double _significanceLevel = 0.05;

    /// <inheritdoc/>
    public override string Name => "Granger Causality";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes Granger Causality with optional configuration.
    /// </summary>
    public GrangerCausalityAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        if (options?.SignificanceLevel.HasValue == true)
            _significanceLevel = options.SignificanceLevel.Value;
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
        if (effectiveN <= 2 * MaxLag + 1)
        {
            // Not enough data for Granger test — return empty graph
            return DoubleArrayToMatrix(new double[d, d]);
        }

        var W = new double[d, d];

        for (int target = 0; target < d; target++)
        {
            // Restricted model: AR on target only
            double rssRestricted = ComputeARModelRSS(X, n, target, MaxLag, effectiveN);

            for (int cause = 0; cause < d; cause++)
            {
                if (cause == target) continue;

                // Unrestricted model: AR on target + lags of cause
                double rssUnrestricted = ComputeGrangerRSS(X, n, target, cause, MaxLag, effectiveN);

                int dfRestricted = effectiveN - MaxLag;
                int dfUnrestricted = effectiveN - 2 * MaxLag;

                if (dfUnrestricted > 0 && rssUnrestricted > 1e-10)
                {
                    double fStat = ((rssRestricted - rssUnrestricted) / MaxLag) /
                                   (rssUnrestricted / dfUnrestricted);

                    if (fStat > 0)
                    {
                        // Use F-statistic as edge weight (normalized)
                        W[cause, target] = Math.Max(0, fStat);
                    }
                }
            }
        }

        // Normalize weights to [0, max_abs_correlation]
        double maxWeight = 0;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                maxWeight = Math.Max(maxWeight, W[i, j]);

        if (maxWeight > 1e-10)
        {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    W[i, j] /= maxWeight;
        }

        return DoubleArrayToMatrix(W);
    }

    private double ComputeARModelRSS(double[,] X, int n, int target, int lag, int effectiveN)
    {
        var design = new double[effectiveN, lag];
        var y = new double[effectiveN];

        for (int t = 0; t < effectiveN; t++)
        {
            y[t] = X[t + lag, target];
            for (int l = 0; l < lag; l++)
                design[t, l] = X[t + lag - l - 1, target];
        }

        return ComputeRSS(design, y, effectiveN, lag);
    }

    private double ComputeGrangerRSS(double[,] X, int n, int target, int cause, int lag, int effectiveN)
    {
        var design = new double[effectiveN, 2 * lag];
        var y = new double[effectiveN];

        for (int t = 0; t < effectiveN; t++)
        {
            y[t] = X[t + lag, target];
            for (int l = 0; l < lag; l++)
            {
                design[t, l] = X[t + lag - l - 1, target];
                design[t, lag + l] = X[t + lag - l - 1, cause];
            }
        }

        return ComputeRSS(design, y, effectiveN, 2 * lag);
    }
}
