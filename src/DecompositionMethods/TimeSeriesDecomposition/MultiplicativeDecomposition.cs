namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Performs multiplicative decomposition of time series data into trend, seasonal, and residual components.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Multiplicative decomposition is used when the seasonal variations in your data increase 
/// or decrease proportionally with the level of the time series. In this model, the components are multiplied 
/// together (Original = Trend × Seasonal × Residual) rather than added. This is often appropriate for economic 
/// or financial data where percentage changes are more meaningful than absolute changes.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MultiplicativeDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly MultiplicativeAlgorithmType _algorithm;
    private readonly int _seasonalPeriod;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiplicativeDecomposition{T}"/> class.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <param name="seasonalPeriod">The number of observations in one seasonal cycle (e.g., 12 for monthly data with yearly seasonality).</param>
    public MultiplicativeDecomposition(Vector<T> timeSeries, MultiplicativeAlgorithmType algorithm = MultiplicativeAlgorithmType.GeometricMovingAverage, int seasonalPeriod = 12)
        : base(timeSeries)
    {
        _algorithm = algorithm;
        _seasonalPeriod = seasonalPeriod;
        Decompose();
    }

    /// <summary>
    /// Performs the time series decomposition using the selected algorithm.
    /// </summary>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case MultiplicativeAlgorithmType.GeometricMovingAverage:
                DecomposeGeometricMovingAverage();
                break;
            case MultiplicativeAlgorithmType.MultiplicativeExponentialSmoothing:
                DecomposeMultiplicativeExponentialSmoothing();
                break;
            case MultiplicativeAlgorithmType.LogTransformedSTL:
                DecomposeLogTransformedSTL();
                break;
            default:
                throw new ArgumentException("Unsupported Multiplicative decomposition algorithm.");
        }
    }

    /// <summary>
    /// Decomposes the time series using a geometric moving average approach.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts the trend by calculating the geometric mean (multiplying values 
    /// and taking the nth root) of data points within a moving window. The geometric mean is used instead 
    /// of the arithmetic mean (simple average) because we're working with multiplicative relationships.
    /// Think of it like calculating compound growth rates rather than simple averages.
    /// </remarks>
    private void DecomposeGeometricMovingAverage()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n);
        Vector<T> seasonal = new Vector<T>(n);
        Vector<T> residual = new Vector<T>(n);

        // Calculate trend using geometric moving average
        int halfWindow = _seasonalPeriod / 2;
        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - halfWindow);
            int end = Math.Min(n - 1, i + halfWindow);
            T product = NumOps.One;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                product = NumOps.Multiply(product, TimeSeries[j]);
                count++;
            }

            trend[i] = NumOps.Power(product, NumOps.Divide(NumOps.One, NumOps.FromDouble(count)));
        }

        // Calculate seasonal component
        for (int i = 0; i < n; i++)
        {
            seasonal[i] = NumOps.Divide(TimeSeries[i], trend[i]);
        }

        // Normalize seasonal component
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            T seasonalSum = NumOps.Zero;
            int count = 0;
            for (int j = i; j < n; j += _seasonalPeriod)
            {
                seasonalSum = NumOps.Add(seasonalSum, seasonal[j]);
                count++;
            }
            T averageSeasonal = NumOps.Divide(seasonalSum, NumOps.FromDouble(count));
            for (int j = i; j < n; j += _seasonalPeriod)
            {
                seasonal[j] = NumOps.Divide(seasonal[j], averageSeasonal);
            }
        }

        // Calculate residual
        for (int i = 0; i < n; i++)
        {
            residual[i] = NumOps.Divide(TimeSeries[i], NumOps.Multiply(trend[i], seasonal[i]));
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Decomposes the time series using multiplicative exponential smoothing.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Exponential smoothing is like calculating a weighted average where recent values 
    /// have more influence than older values. The "multiplicative" part means we're working with ratios 
    /// and products rather than differences and sums. This method uses three smoothing factors (alpha, beta, gamma) 
    /// that control how quickly the model adapts to changes in level, trend, and seasonality.
    /// </remarks>
    private void DecomposeMultiplicativeExponentialSmoothing()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n);
        Vector<T> seasonal = new Vector<T>(n);
        Vector<T> residual = new Vector<T>(n);

        T alpha = NumOps.FromDouble(0.2); // Smoothing factor for level
        T beta = NumOps.FromDouble(0.1);  // Smoothing factor for trend
        T gamma = NumOps.FromDouble(0.3); // Smoothing factor for seasonal

        T level = TimeSeries[0];
        T trendComponent = NumOps.One;

        // Use a separate rotating buffer for seasonal factors
        T[] seasonalFactors = new T[_seasonalPeriod];
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            seasonalFactors[i] = NumOps.Divide(TimeSeries[i], level);
        }

        for (int t = 0; t < n; t++)
        {
            T value = TimeSeries[t];
            T lastLevel = level;
            T lastTrend = trendComponent;
            T lastSeasonal = seasonalFactors[t % _seasonalPeriod];

            // Update level
            level = NumOps.Add(
                NumOps.Multiply(alpha, NumOps.Divide(value, lastSeasonal)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, alpha), NumOps.Multiply(lastLevel, lastTrend))
            );

            // Update trend
            trendComponent = NumOps.Add(
                NumOps.Multiply(beta, NumOps.Divide(level, lastLevel)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta), lastTrend)
            );

            // Update seasonal factor
            seasonalFactors[t % _seasonalPeriod] = NumOps.Add(
                NumOps.Multiply(gamma, NumOps.Divide(value, level)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, gamma), lastSeasonal)
            );

            // Store per-timestep seasonal for correct reconstruction: data = trend * seasonal * residual
            trend[t] = NumOps.Multiply(level, trendComponent);
            seasonal[t] = seasonalFactors[t % _seasonalPeriod];
            residual[t] = NumOps.Divide(value, NumOps.Multiply(trend[t], seasonal[t]));
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Decomposes the time series using a log-transformed STL (Seasonal-Trend decomposition using LOESS) approach.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method transforms multiplicative relationships into additive ones by taking 
    /// the logarithm of the data. After this transformation, we can use STL (Seasonal-Trend decomposition 
    /// using LOESS), which is a powerful technique that works well with additive patterns. LOESS stands for 
    /// "LOcally Estimated Scatterplot Smoothing" - it's a way to find patterns in data by looking at small, 
    /// overlapping chunks. After decomposition, we transform the components back to the original scale using 
    /// the exponential function (the opposite of logarithm).
    /// </remarks>
    private void DecomposeLogTransformedSTL()
    {
        int n = TimeSeries.Length;
        Vector<T> logTimeSeries = new Vector<T>(n);

        // Log transform the time series
        for (int i = 0; i < n; i++)
        {
            logTimeSeries[i] = NumOps.Log(TimeSeries[i]);
        }

        // Perform STL decomposition on log-transformed data
        var stlOptions = new STLDecompositionOptions<T>
        {
            SeasonalPeriod = _seasonalPeriod,
            RobustIterations = 1,
            InnerLoopPasses = 2,
            SeasonalDegree = 1,
            TrendDegree = 1,
            SeasonalJump = 1,
            TrendJump = 1,
            SeasonalBandwidth = 0.75,
            TrendBandwidth = 0.75,
            LowPassBandwidth = 0.75
        };

        var stlDecomposition = new STLDecomposition<T>(stlOptions);
        stlDecomposition.Train(new Matrix<T>(logTimeSeries.Length, 1), logTimeSeries);

        Vector<T> logTrend = stlDecomposition.GetTrend();
        Vector<T> logSeasonal = stlDecomposition.GetSeasonal();
        Vector<T> logResidual = stlDecomposition.GetResidual();

        // Transform components back to original scale
        Vector<T> trend = new Vector<T>(n);
        Vector<T> seasonal = new Vector<T>(n);
        Vector<T> residual = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            trend[i] = NumOps.Exp(logTrend[i]);
            seasonal[i] = NumOps.Exp(logSeasonal[i]);
            residual[i] = NumOps.Exp(logResidual[i]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }
}
