using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// Base class for time series causal discovery algorithms (Granger, PCMCI, DYNOTEARS, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Time series causal discovery extends standard methods by considering temporal relationships.
/// Variable X is said to Granger-cause Y if past values of X help predict Y beyond what
/// Y's own past values can predict.
/// </para>
/// <para>
/// <b>For Beginners:</b> In time series, the order of events matters. These algorithms figure
/// out which variables help predict other variables' future values. For example, does yesterday's
/// stock price of company A help predict today's stock price of company B?
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TimeSeriesCausalBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.TimeSeries;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    /// <summary>
    /// Maximum lag order for temporal relationships.
    /// </summary>
    protected int MaxLag { get; set; } = 3;

    /// <summary>
    /// Applies time-series-specific options.
    /// </summary>
    protected void ApplyTimeSeriesOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.MaxLag.HasValue)
        {
            if (options.MaxLag.Value <= 0)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxLag must be > 0.");
            MaxLag = options.MaxLag.Value;
        }
    }

    /// <summary>
    /// Creates lagged data matrix from time series: for each time step t,
    /// includes values at lags t-1, t-2, ..., t-maxLag.
    /// </summary>
    protected (Matrix<T> LaggedX, Vector<T> Target) CreateLaggedData(
        Matrix<T> data, int targetCol, int maxLag)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (targetCol < 0 || targetCol >= d)
            throw new ArgumentOutOfRangeException(nameof(targetCol), $"targetCol must be in [0, {d - 1}].");
        if (maxLag <= 0 || maxLag >= n)
            throw new ArgumentOutOfRangeException(nameof(maxLag), $"maxLag must be in [1, {n - 1}].");

        int effectiveN = n - maxLag;
        var laggedX = new Matrix<T>(effectiveN, d * maxLag);
        var target = new Vector<T>(effectiveN);

        for (int t = 0; t < effectiveN; t++)
        {
            target[t] = data[t + maxLag, targetCol];
            for (int lag = 0; lag < maxLag; lag++)
            {
                for (int col = 0; col < d; col++)
                {
                    laggedX[t, lag * d + col] = data[t + maxLag - lag - 1, col];
                }
            }
        }

        return (laggedX, target);
    }

    /// <summary>
    /// Computes RSS (Residual Sum of Squares) for OLS regression using generic operations.
    /// </summary>
    protected double ComputeRSS(Matrix<T> X, Vector<T> y, int n, int p)
    {
        // Build normal equations: XtX * beta = Xty
        var XtX = new Matrix<T>(p, p);
        var Xty = new Vector<T>(p);
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                    sum = NumOps.Add(sum, NumOps.Multiply(X[k, i], X[k, j]));
                XtX[i, j] = sum;
            }
            T sumY = NumOps.Zero;
            for (int k = 0; k < n; k++)
                sumY = NumOps.Add(sumY, NumOps.Multiply(X[k, i], y[k]));
            Xty[i] = sumY;
        }

        // Ridge regularization
        T ridge = NumOps.FromDouble(1e-6);
        for (int i = 0; i < p; i++)
            XtX[i, i] = NumOps.Add(XtX[i, i], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX, Xty, MatrixDecompositionType.Lu);

        double rss = 0;
        for (int i = 0; i < n; i++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < p; j++)
                pred = NumOps.Add(pred, NumOps.Multiply(beta[j], X[i, j]));
            double err = NumOps.ToDouble(NumOps.Subtract(y[i], pred));
            rss += err * err;
        }

        return rss;
    }
}
