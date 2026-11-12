namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for time series analysis and forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used in calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Time series analysis is a technique used to analyze data points collected over time
/// to identify patterns and predict future values. This is commonly used for forecasting trends like
/// stock prices, weather patterns, or sales data.
/// </remarks>
public static class TimeSeriesHelper<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the differences between consecutive values in a time series.
    /// </summary>
    /// <param name="y">The original time series data.</param>
    /// <param name="d">The order of differencing (how many times to apply the difference operation).</param>
    /// <returns>The differenced time series.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Differencing helps make a time series stationary by removing trends or seasonal patterns.
    /// First-order differencing (d=1) calculates the change between consecutive values.
    /// For example, if your data is [10, 13, 15, 19], first-order differencing gives [3, 2, 4],
    /// representing how much each value increased from the previous one.
    /// </remarks>
    public static Vector<T> DifferenceSeries(Vector<T> y, int d)
    {
        var engine = AiDotNetEngine.Current;
        Vector<T> result = y;
        for (int i = 0; i < d; i++)
        {
            int length = result.Length - 1;
            Vector<T> temp = new Vector<T>(length);
            if (length > 0)
            {
                // temp[j - 1] = result[j] - result[j - 1]
                temp = engine.Subtract(result.GetSubVector(1, length), result.GetSubVector(0, length));
            }
            result = temp;
        }

        return result;
    }

    /// <summary>
    /// Estimates the coefficients for an Autoregressive (AR) model.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <param name="p">The order of the AR model (number of past values to consider).</param>
    /// <param name="decompositionType">The matrix decomposition method to use for solving the system.</param>
    /// <returns>The estimated AR coefficients.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> An Autoregressive (AR) model predicts future values based on past values.
    /// The parameter 'p' determines how many past values to consider. For example, with p=2,
    /// we use the previous two values to predict the next value. The coefficients tell us how much
    /// weight to give each past value in our prediction. This is like saying "tomorrow's temperature
    /// depends 70% on today's temperature and 30% on yesterday's temperature."
    /// </remarks>
    public static Vector<T> EstimateARCoefficients(Vector<T> y, int p, MatrixDecompositionType decompositionType)
    {
        int nRows = y.Length - p;

        // Under-determined system: not enough data for reliable estimation.
        // Fall back to simple correlation-based estimation.
        if (nRows < p)
        {
            var fallback = new Vector<T>(p);
            // Use simple lag-1 autocorrelation for the first coefficient, zeros for rest
            if (nRows > 0 && y.Length > 1)
            {
                var numOps = MathHelper.GetNumericOperations<T>();
                var engine = AiDotNetEngine.Current;
                var current = y.GetSubVector(1, y.Length - 1);
                var previous = y.GetSubVector(0, y.Length - 1);
                T sum = engine.DotProduct(current, previous);
                T sumSq = engine.DotProduct(previous, previous);
                if (!numOps.Equals(sumSq, numOps.Zero))
                {
                    fallback[0] = numOps.Divide(sum, sumSq);
                }
            }
            return fallback;
        }

        Matrix<T> X = BuildLagMatrix(y, p, nRows);
        Vector<T> Y = y.GetSubVector(p, nRows);

        var coefficients = MatrixSolutionHelper.SolveLinearSystem(X, Y, decompositionType);

        // Enforce stationarity: |a1| + |a2| + ... + |ap| < 1 is sufficient
        // (though not necessary) for stationarity. See: #991
        EnforceCoefficientStability(coefficients);

        return coefficients;
    }

    /// <summary>
    /// Calculates the residuals (errors) of an Autoregressive (AR) model.
    /// </summary>
    /// <param name="y">The original time series data.</param>
    /// <param name="arCoefficients">The AR model coefficients.</param>
    /// <returns>The residuals (differences between actual and predicted values).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Residuals are the differences between what your model predicted and what actually happened.
    /// They tell you how accurate your model is. Small residuals mean your model is making good predictions.
    /// Analyzing residuals can help you improve your model or detect patterns your model missed.
    /// </remarks>
    public static Vector<T> CalculateARResiduals(Vector<T> y, Vector<T> arCoefficients)
    {
        int n = y.Length;
        int p = arCoefficients.Length;
        Vector<T> residuals = new Vector<T>(n - p);
        if (n - p == 0)
        {
            return residuals;
        }

        // residual_r = y[p + r] - sum_j a_j * y[p + r - j - 1] = (Y - X a)_r with X the lag matrix
        var engine = AiDotNetEngine.Current;
        var observed = y.GetSubVector(p, n - p);
        if (p == 0)
        {
            return observed;
        }

        var predicted = engine.MatrixVectorMultiply(BuildLagMatrix(y, p, n - p), arCoefficients);
        return engine.Subtract(observed, predicted);
    }

    /// <summary>
    /// The [rows, p] autoregressive design matrix: entry (r, j) is y[p + r - j - 1], so column j is the
    /// series shifted by j + 1 steps.
    /// </summary>
    private static Matrix<T> BuildLagMatrix(Vector<T> y, int p, int rows)
    {
        if (p == 0)
        {
            return new Matrix<T>(rows, 0);
        }

        return Matrix<T>.FromColumnVectors(Enumerable.Range(0, p).Select(j => y.GetSubVector(p - j - 1, rows)));
    }

    /// <summary>
    /// Estimates the coefficients for a Moving Average (MA) model.
    /// </summary>
    /// <param name="residuals">The residuals from a previous model (often an AR model).</param>
    /// <param name="q">The order of the MA model (number of past errors to consider).</param>
    /// <returns>The estimated MA coefficients.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A Moving Average (MA) model predicts future values based on past prediction errors.
    /// While AR models use past actual values, MA models use past mistakes in predictions.
    /// The parameter 'q' determines how many past errors to consider. This helps capture random
    /// shocks or unexpected events in your data that might affect future values.
    /// </remarks>
    public static Vector<T> EstimateMACoefficients(Vector<T> residuals, int q)
    {
        // Lag-1..q autocorrelations, each one engine dot-product reduction.
        Vector<T> maCoefficients = new Vector<T>(q);
        if (q > 0)
        {
            maCoefficients = new Vector<T>(Enumerable.Range(1, q).Select(lag => CalculateAutoCorrelation(residuals, lag)));
        }

        // Enforce invertibility: |sum| < 1 for well-defined AR(infinity) representation.
        // See: #991
        EnforceCoefficientStability(maCoefficients);

        return maCoefficients;
    }

    /// <summary>
    /// Calculates the autocorrelation function (ACF) of a time series for lags 0 through maxLag.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <param name="maxLag">The maximum lag to calculate autocorrelation for.</param>
    /// <returns>A vector of autocorrelation values for lags 0 through maxLag.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The autocorrelation function shows how similar a time series is to 
    /// time-shifted versions of itself. This helps identify patterns like trends or seasonality.
    /// For example, daily temperature data might show high autocorrelation at lag=24 hours,
    /// indicating a daily cycle in temperatures.
    /// </remarks>
    public static Vector<T> CalculateMultipleAutoCorrelation(Vector<T> y, int maxLag)
    {
        var acf = new Vector<T>(maxLag + 1);

        // Lag 0 is always 1.0 (perfect correlation with itself)
        acf[0] = _numOps.One;

        // Calculate autocorrelation for each lag using the existing method
        if (maxLag > 0)
        {
            acf = new Vector<T>(new[] { _numOps.One }.Concat(
                Enumerable.Range(1, maxLag).Select(lag => CalculateAutoCorrelation(y, lag))));
        }

        return acf;
    }

    /// <summary>
    /// Calculates the autocorrelation of a time series at a specific lag.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <param name="lag">The lag (time shift) to calculate autocorrelation for.</param>
    /// <returns>The autocorrelation value at the specified lag.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Autocorrelation measures how similar a time series is to a delayed version of itself.
    /// A high autocorrelation at lag=1 means that if today's value is high, tomorrow's value is likely to be high too.
    /// This helps identify patterns in your data. For example, temperature readings often have high autocorrelation
    /// at lag=24 hours because temperatures follow a daily cycle.
    /// </remarks>
    public static T CalculateAutoCorrelation(Vector<T> y, int lag)
    {
        T sum = _numOps.Zero;
        T sumSquared = _numOps.Zero;
        int n = y.Length;

        if (n - lag > 0)
        {
            var engine = AiDotNetEngine.Current;
            var head = y.GetSubVector(0, n - lag);
            sum = engine.DotProduct(head, y.GetSubVector(lag, n - lag));
            sumSquared = engine.DotProduct(head, head);
        }

        // Guard against division by zero (constant or zero series)
        if (!_numOps.GreaterThan(sumSquared, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(sum, sumSquared);
    }

    /// <summary>
    /// Enforces coefficient stability by sanitizing NaN/Infinity values and scaling
    /// coefficients when their absolute sum approaches or exceeds 1.
    /// </summary>
    /// <param name="coefficients">The coefficient vector to stabilize (modified in-place).</param>
    /// <param name="triggerThreshold">Absolute sum threshold that triggers scaling (default 0.99).</param>
    /// <param name="targetSum">Target absolute sum after scaling (default 0.95, providing a stability margin).</param>
    /// <remarks>
    /// <para>
    /// The condition |c1| + |c2| + ... + |cp| &lt; 1 is a sufficient (though not necessary)
    /// condition for both AR stationarity and MA invertibility. When the sum approaches 1,
    /// the model becomes unstable and predictions diverge exponentially.
    /// </para>
    /// <para>
    /// This method first replaces NaN/Infinity coefficients with zero, then scales all
    /// coefficients proportionally if their absolute sum exceeds <paramref name="triggerThreshold"/>.
    /// </para>
    /// </remarks>
    private static void EnforceCoefficientStability(
        Vector<T> coefficients,
        double triggerThreshold = 0.99,
        double targetSum = 0.95)
    {
        if (coefficients.Length == 0)
        {
            return;
        }

        // Zero the non-finite coefficients, then measure the L1 norm of what remains.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var values = Tensor<T>.FromVector(coefficients);
        var cleaned = engine.TensorWhere(engine.TensorIsFinite(values), values, new Tensor<T>(values._shape));
        double absSum = _numOps.ToDouble(engine.TensorSum(engine.TensorAbs(cleaned)));

        if (absSum >= triggerThreshold && absSum > 0)
        {
            double scaleFactor = targetSum / absSum;
            cleaned = engine.TensorMultiplyScalar(cleaned, _numOps.FromDouble(scaleFactor));
        }

        cleaned.AsSpan().CopyTo(coefficients.AsWritableSpan());
    }
}
