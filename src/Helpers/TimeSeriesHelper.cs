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
        Vector<T> result = y;
        for (int i = 0; i < d; i++)
        {
            Vector<T> temp = new Vector<T>(result.Length - 1);
            for (int j = 1; j < result.Length; j++)
            {
                temp[j - 1] = _numOps.Subtract(result[j], result[j - 1]);
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
        Matrix<T> X = new Matrix<T>(y.Length - p, p);
        Vector<T> Y = new Vector<T>(y.Length - p);

        for (int i = p; i < y.Length; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i - p, j] = y[i - j - 1];
            }
            Y[i - p] = y[i];
        }

        return MatrixSolutionHelper.SolveLinearSystem(X, Y, decompositionType);
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

        for (int i = p; i < n; i++)
        {
            T predicted = _numOps.Zero;
            for (int j = 0; j < p; j++)
            {
                predicted = _numOps.Add(predicted, _numOps.Multiply(arCoefficients[j], y[i - j - 1]));
            }
            residuals[i - p] = _numOps.Subtract(y[i], predicted);
        }

        return residuals;
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
        Vector<T> maCoefficients = new Vector<T>(q);
        for (int i = 0; i < q; i++)
        {
            maCoefficients[i] = CalculateAutoCorrelation(residuals, i + 1);
        }

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
        for (int lag = 1; lag <= maxLag; lag++)
        {
            acf[lag] = CalculateAutoCorrelation(y, lag);
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

        for (int i = 0; i < n - lag; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(y[i], y[i + lag]));
            sumSquared = _numOps.Add(sumSquared, _numOps.Multiply(y[i], y[i]));
        }

        // Guard against division by zero (constant or zero series)
        if (!_numOps.GreaterThan(sumSquared, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(sum, sumSquared);
    }
}
