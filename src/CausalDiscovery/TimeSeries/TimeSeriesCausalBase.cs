using AiDotNet.Enums;

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
    protected int MaxLag { get; set; } = 5;

    /// <summary>
    /// Applies time-series-specific options.
    /// </summary>
    protected void ApplyTimeSeriesOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.MaxIterations.HasValue) MaxLag = options.MaxIterations.Value;
    }

    /// <summary>
    /// Creates lagged data matrix from time series: for each time step t,
    /// includes values at lags t-1, t-2, ..., t-maxLag.
    /// </summary>
    protected (double[,] LaggedX, double[] Target) CreateLaggedData(
        double[,] X, int n, int d, int targetCol, int maxLag)
    {
        int effectiveN = n - maxLag;
        var laggedX = new double[effectiveN, d * maxLag];
        var target = new double[effectiveN];

        for (int t = 0; t < effectiveN; t++)
        {
            target[t] = X[t + maxLag, targetCol];
            for (int lag = 0; lag < maxLag; lag++)
            {
                for (int col = 0; col < d; col++)
                {
                    laggedX[t, lag * d + col] = X[t + maxLag - lag - 1, col];
                }
            }
        }

        return (laggedX, target);
    }

    /// <summary>
    /// Computes RSS (Residual Sum of Squares) for OLS regression.
    /// </summary>
    protected static double ComputeRSS(double[,] X, double[] y, int n, int p)
    {
        var XtX = new double[p, p];
        var Xty = new double[p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
                for (int k = 0; k < n; k++) XtX[i, j] += X[k, i] * X[k, j];
            for (int k = 0; k < n; k++) Xty[i] += X[k, i] * y[k];
        }

        for (int i = 0; i < p; i++) XtX[i, i] += 1e-6;

        var beta = SolveSystem(XtX, Xty, p);

        double rss = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = 0;
            for (int j = 0; j < p; j++) pred += beta[j] * X[i, j];
            double err = y[i] - pred;
            rss += err * err;
        }

        return rss;
    }

    private static double[] SolveSystem(double[,] A, double[] b, int size)
    {
        var aug = new double[size, size + 1];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++) aug[i, j] = A[i, j];
            aug[i, size] = b[i];
        }

        for (int col = 0; col < size; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < size; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;
            for (int j = 0; j <= size; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);
            if (Math.Abs(aug[col, col]) < 1e-10) continue;
            for (int row = col + 1; row < size; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= size; j++) aug[row, j] -= factor * aug[col, j];
            }
        }

        var x = new double[size];
        for (int i = size - 1; i >= 0; i--)
        {
            x[i] = aug[i, size];
            for (int j = i + 1; j < size; j++) x[i] -= aug[i, j] * x[j];
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
    }

    /// <summary>
    /// Converts double array to Matrix&lt;T&gt;.
    /// </summary>
    protected Matrix<T> DoubleArrayToMatrix(double[,] data)
    {
        int rows = data.GetLength(0), cols = data.GetLength(1);
        var result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = NumOps.FromDouble(data[i, j]);
        return result;
    }
}
