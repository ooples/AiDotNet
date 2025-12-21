namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements additive time series decomposition, breaking a time series into trend, seasonal, and residual components.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Time series decomposition is like breaking down a complex signal (like sales data over time) 
/// into simpler parts that are easier to understand. The additive model assumes these components add up to form 
/// the original data: Original = Trend + Seasonal + Residual.
/// </para>
/// <para>
/// - Trend: The long-term progression (going up, down, or staying flat over time)
/// - Seasonal: Repeating patterns at fixed intervals (like higher sales every December)
/// - Residual: What's left over after removing trend and seasonal components (the "noise")
/// </para>
/// </remarks>
public class AdditiveDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    /// <summary>
    /// The algorithm type used for decomposition
    /// </summary>
    private readonly AdditiveDecompositionAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new instance of the AdditiveDecomposition class
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition (defaults to MovingAverage)</param>
    /// <remarks>
    /// <b>For Beginners:</b> When you create this object, it automatically decomposes your time series data
    /// using the algorithm you specify (or MovingAverage if you don't specify one).
    /// </remarks>
    public AdditiveDecomposition(Vector<T> timeSeries, AdditiveDecompositionAlgorithmType algorithm = AdditiveDecompositionAlgorithmType.MovingAverage)
        : base(timeSeries)
    {
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the decomposition based on the selected algorithm
    /// </summary>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case AdditiveDecompositionAlgorithmType.MovingAverage:
                DecomposeMovingAverage();
                break;
            case AdditiveDecompositionAlgorithmType.ExponentialSmoothing:
                DecomposeExponentialSmoothing();
                break;
            case AdditiveDecompositionAlgorithmType.STL:
                DecomposeSTL();
                break;
            default:
                throw new ArgumentException("Unsupported Additive decomposition algorithm.");
        }
    }

    /// <summary>
    /// Decomposes the time series using the Moving Average method
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Moving Average is like taking the average of a sliding window of data points.
    /// It helps smooth out short-term fluctuations and highlight longer-term trends.
    /// </remarks>
    private void DecomposeMovingAverage()
    {
        // Implementation of Moving Average decomposition
        Vector<T> trend = CalculateTrendMovingAverage();
        Vector<T> seasonal = CalculateSeasonalMovingAverage(trend);
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Decomposes the time series using the Exponential Smoothing method
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Exponential Smoothing gives more weight to recent observations and less weight
    /// to older observations. It's like having a better memory of recent events compared to distant ones.
    /// </remarks>
    private void DecomposeExponentialSmoothing()
    {
        // Implementation of Exponential Smoothing decomposition
        Vector<T> trend = CalculateTrendExponentialSmoothing();
        Vector<T> seasonal = CalculateSeasonalExponentialSmoothing(trend);
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Decomposes the time series using the Seasonal and Trend decomposition using Loess (STL) method
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> STL is a sophisticated method that can handle complex seasonal patterns that change over time.
    /// It uses a technique called "Loess" (locally estimated scatterplot smoothing) which fits simple models
    /// to small chunks of data to build a more complex and accurate overall model.
    /// </remarks>
    private void DecomposeSTL()
    {
        // Implementation of Seasonal and Trend decomposition using Loess (STL)
        (Vector<T> trend, Vector<T> seasonal) = PerformSTLDecomposition();
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Performs the STL decomposition algorithm
    /// </summary>
    /// <returns>A tuple containing the trend and seasonal components</returns>
    private (Vector<T>, Vector<T>) PerformSTLDecomposition()
    {
        int n = TimeSeries.Length;
        int seasonalPeriod = 12;
        int nInner = 2;
        int nOuter = 1;
        int trendWindow = (int)Math.Ceiling(1.5 * seasonalPeriod / (1 - 1.5 / seasonalPeriod));
        trendWindow = trendWindow % 2 == 0 ? trendWindow + 1 : trendWindow;

        Vector<T> trend = new Vector<T>(n);
        Vector<T> seasonal = new Vector<T>(n);
        Vector<T> detrended = new Vector<T>(n);

        for (int i = 0; i < nOuter; i++)
        {
            for (int j = 0; j < nInner; j++)
            {
                // Step 1: Detrending
                detrended = SubtractVectors(TimeSeries, trend);

                // Step 2: Cycle-subseries Smoothing
                seasonal = CycleSubseriesSmoothing(detrended, seasonalPeriod);

                // Step 3: Low-pass Filtering of Smoothed Cycle-subseries
                Vector<T> lowPassSeasonal = LowPassFilter(seasonal, seasonalPeriod);

                // Step 4: Detrending of Smoothed Cycle-subseries
                seasonal = SubtractVectors(seasonal, lowPassSeasonal);

                // Step 5: Deseasonalizing
                Vector<T> deseasonalized = SubtractVectors(TimeSeries, seasonal);

                // Step 6: Trend Smoothing
                trend = LoessSmoothing(deseasonalized, trendWindow);
            }
        }

        return (trend, seasonal);
    }

    /// <summary>
    /// Subtracts one vector from another element by element
    /// </summary>
    /// <param name="a">The first vector</param>
    /// <param name="b">The vector to subtract</param>
    /// <returns>A new vector containing the result of a - b</returns>
    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        int n = a.Length;
        Vector<T> result = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.Subtract(a[i], b[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs cycle-subseries smoothing on the data
    /// </summary>
    /// <param name="data">The data to smooth</param>
    /// <param name="period">The seasonal period</param>
    /// <returns>The smoothed data</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method breaks down the data into "subseries" based on the seasonal period.
    /// For example, with monthly data (period=12), it creates 12 subseries: all January values, all February values, etc.
    /// Each subseries is then smoothed separately to better capture the seasonal pattern.
    /// </remarks>
    private Vector<T> CycleSubseriesSmoothing(Vector<T> data, int period)
    {
        int n = data.Length;
        Vector<T> smoothed = new Vector<T>(n);

        for (int i = 0; i < period; i++)
        {
            List<(T x, T y)> subseries = new List<(T x, T y)>();
            for (int j = i; j < n; j += period)
            {
                subseries.Add((NumOps.FromDouble(j), data[j]));
            }

            Vector<T> smoothedSubseries = LoessSmoothing(subseries, 0.75);

            int index = 0;
            for (int j = i; j < n; j += period)
            {
                smoothed[j] = smoothedSubseries[index++];
            }
        }

        return smoothed;
    }

    /// <summary>
    /// Applies a low-pass filter to the data
    /// </summary>
    /// <param name="data">The data to filter</param>
    /// <param name="period">The period used to determine the window size</param>
    /// <returns>The filtered data</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A low-pass filter removes high-frequency variations (rapid changes) and keeps
    /// low-frequency variations (slow changes). It's like blurring a photo to remove fine details
    /// but keep the overall shapes. This helps isolate the trend component.
    /// </remarks>
    private Vector<T> LowPassFilter(Vector<T> data, int period)
    {
        int n = data.Length;
        Vector<T> filtered = new Vector<T>(n);
        int windowSize = period + 1;

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - windowSize / 2);
            int end = Math.Min(n - 1, i + windowSize / 2);
            T sum = NumOps.Zero;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                sum = NumOps.Add(sum, data[j]);
                count++;
            }

            filtered[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        return filtered;
    }

    /// <summary>
    /// Applies LOESS (Locally Estimated Scatterplot Smoothing) to a vector of data
    /// </summary>
    /// <param name="data">The data to smooth</param>
    /// <param name="windowSize">The size of the window to use for smoothing</param>
    /// <returns>A smoothed version of the input data</returns>
    /// <remarks>
    /// <b>For Beginners:</b> LOESS smoothing is like drawing a smooth curve through noisy data points.
    /// It works by looking at a small "window" of nearby points around each position and calculating
    /// a weighted average, giving more importance to closer points. This helps reveal underlying
    /// patterns by reducing noise in the data.
    /// </remarks>
    private Vector<T> LoessSmoothing(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        Vector<T> smoothed = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - windowSize / 2);
            int end = Math.Min(n - 1, i + windowSize / 2);
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = start; j <= end; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(NumOps.FromDouble(j), NumOps.FromDouble(i)));
                T weight = TriCube(NumOps.Divide(distance, NumOps.FromDouble(windowSize / 2)));
                weightedPoints.Add((distance, weight, data[j]));
            }

            smoothed[i] = WeightedLeastSquares(weightedPoints);
        }

        return smoothed;
    }

    /// <summary>
    /// Applies LOESS smoothing to a list of (x,y) data points
    /// </summary>
    /// <param name="data">The data points to smooth, as (x,y) pairs</param>
    /// <param name="span">The proportion of data to include in each local regression (0-1)</param>
    /// <returns>A smoothed version of the input data</returns>
    /// <remarks>
    /// This overload works with arbitrary x-coordinates, while the other version assumes
    /// evenly spaced data points.
    /// </remarks>
    private Vector<T> LoessSmoothing(List<(T x, T y)> data, double span)
    {
        int n = data.Count;
        Vector<T> smoothed = new Vector<T>(n);
        int windowSize = (int)(span * n);

        for (int i = 0; i < n; i++)
        {
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = 0; j < n; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(data[j].x, data[i].x));
                T weight = TriCube(NumOps.Divide(distance, NumOps.FromDouble(windowSize / 2)));
                weightedPoints.Add((distance, weight, data[j].y));
            }

            smoothed[i] = WeightedLeastSquares(weightedPoints);
        }

        return smoothed;
    }

    /// <summary>
    /// Calculates the tri-cube weight function used in LOESS smoothing
    /// </summary>
    /// <param name="x">The normalized distance value (0-1)</param>
    /// <returns>The weight value, with higher weights for smaller distances</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This function determines how much influence each nearby point has
    /// when calculating a smoothed value. Points that are closer get higher weights (more influence),
    /// while points farther away get lower weights (less influence). Points beyond a certain
    /// distance get zero weight (no influence).
    /// </remarks>
    private T TriCube(T x)
    {
        if (NumOps.GreaterThan(x, NumOps.One))
        {
            return NumOps.Zero;
        }
        T oneMinusX = NumOps.Subtract(NumOps.One, x);

        return NumOps.Multiply(NumOps.Multiply(oneMinusX, oneMinusX), oneMinusX);
    }

    /// <summary>
    /// Calculates a weighted average of data points
    /// </summary>
    /// <param name="weightedPoints">List of points with their distances, weights, and values</param>
    /// <returns>The weighted average value</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates an average where some values count more than others.
    /// Each data point is multiplied by its weight, then all these weighted values are added up
    /// and divided by the sum of all weights. This gives more importance to points with higher weights.
    /// </remarks>
    private T WeightedLeastSquares(List<(T distance, T weight, T y)> weightedPoints)
    {
        T sumWeights = NumOps.Zero;
        T sumWeightedY = NumOps.Zero;

        foreach (var (_, weight, y) in weightedPoints)
        {
            sumWeights = NumOps.Add(sumWeights, weight);
            sumWeightedY = NumOps.Add(sumWeightedY, NumOps.Multiply(weight, y));
        }

        return NumOps.Divide(sumWeightedY, sumWeights);
    }

    /// <summary>
    /// Calculates the trend component using a moving average method
    /// </summary>
    /// <returns>A vector containing the trend component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A moving average calculates the average of a "window" of data points
    /// that moves through the time series. For each position, it averages the values of nearby
    /// points (both before and after). This smooths out short-term fluctuations and highlights
    /// longer-term trends.
    /// </remarks>
    private Vector<T> CalculateTrendMovingAverage()
    {
        int _windowSize = 7;
        Vector<T> trend = new Vector<T>(TimeSeries.Length);

        for (int i = 0; i < TimeSeries.Length; i++)
        {
            int start = Math.Max(0, i - _windowSize / 2);
            int end = Math.Min(TimeSeries.Length - 1, i + _windowSize / 2);
            T sum = NumOps.Zero;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                sum = NumOps.Add(sum, TimeSeries[j]);
                count++;
            }

            trend[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        return trend;
    }

    /// <summary>
    /// Calculates the seasonal component using the moving average method
    /// </summary>
    /// <param name="trend">The trend component previously calculated</param>
    /// <returns>A vector containing the seasonal component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method finds repeating patterns in your data by:
    /// 1. Removing the trend (subtracting it from the original data)
    /// 2. Grouping what's left by season (e.g., all Januaries together)
    /// 3. Calculating the average pattern for each season
    /// 4. Using these averages as the seasonal component
    /// 
    /// For example, with monthly data, it identifies how much each month typically
    /// deviates from the trend.
    /// </remarks>
    private Vector<T> CalculateSeasonalMovingAverage(Vector<T> trend)
    {
        Vector<T> seasonal = new Vector<T>(TimeSeries.Length);
        int _seasonalPeriod = 12;

        for (int i = 0; i < TimeSeries.Length; i++)
        {
            seasonal[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        // Calculate average seasonal component for each period
        Vector<T> _averageSeasonal = new Vector<T>(_seasonalPeriod);
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            T sum = NumOps.Zero;
            int count = 0;
            for (int j = i; j < TimeSeries.Length; j += _seasonalPeriod)
            {
                sum = NumOps.Add(sum, seasonal[j]);
                count++;
            }
            _averageSeasonal[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        // Normalize seasonal component
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            seasonal[i] = _averageSeasonal[i % _seasonalPeriod];
        }

        return seasonal;
    }

    /// <summary>
    /// Calculates the trend component using exponential smoothing
    /// </summary>
    /// <returns>A vector containing the trend component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Exponential smoothing calculates a trend by giving more weight to recent
    /// observations and less weight to older ones. It's like having a better memory of recent events.
    /// 
    /// The alpha parameter (between 0 and 1) controls how quickly the influence of past observations
    /// decreases - higher values make the trend respond more quickly to recent changes.
    /// </remarks>
    private Vector<T> CalculateTrendExponentialSmoothing()
    {
        T _alpha = NumOps.FromDouble(0.2); // Smoothing factor
        Vector<T> trend = new Vector<T>(TimeSeries.Length);
        trend[0] = TimeSeries[0];

        for (int i = 1; i < TimeSeries.Length; i++)
        {
            T prevSmoothed = trend[i - 1];
            T observation = TimeSeries[i];
            trend[i] = NumOps.Add(
                NumOps.Multiply(_alpha, observation),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), prevSmoothed)
            );
        }

        return trend;
    }

    /// <summary>
    /// Calculates the seasonal component using exponential smoothing
    /// </summary>
    /// <param name="trend">The trend component previously calculated</param>
    /// <returns>A vector containing the seasonal component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method identifies repeating patterns in your data (like higher sales in December)
    /// using exponential smoothing. It works by:
    /// 
    /// 1. First calculating initial seasonal values by subtracting the trend from the original data
    /// 2. Then updating these seasonal values over time, giving more weight to recent observations
    /// 
    /// The gamma parameter (between 0 and 1) controls how quickly the seasonal pattern can change:
    /// - Higher gamma (closer to 1): Seasonal patterns adapt quickly to changes
    /// - Lower gamma (closer to 0): Seasonal patterns change slowly over time
    /// 
    /// This is useful for data where seasonal patterns might gradually evolve rather than
    /// staying exactly the same each year.
    /// </remarks>
    private Vector<T> CalculateSeasonalExponentialSmoothing(Vector<T> trend)
    {
        T _gamma = NumOps.FromDouble(0.3); // Seasonal smoothing factor
        int _seasonalPeriod = 12;
        Vector<T> seasonal = new Vector<T>(TimeSeries.Length);

        // Initialize seasonal components
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            seasonal[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        for (int i = _seasonalPeriod; i < TimeSeries.Length; i++)
        {
            int seasonIndex = i % _seasonalPeriod;
            T observation = TimeSeries[i];
            T levelTrend = trend[i];
            T prevSeasonal = seasonal[i - _seasonalPeriod];

            seasonal[i] = NumOps.Add(
                NumOps.Multiply(_gamma, NumOps.Subtract(observation, levelTrend)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, _gamma), prevSeasonal)
            );
        }

        return seasonal;
    }
}
