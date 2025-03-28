namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// STL decomposition breaks down a time series into three components: trend, seasonal, and residual.
/// It uses locally weighted regression (LOESS) to extract these components, making it robust to
/// outliers and applicable to a wide range of time series data.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// STL decomposition is like breaking down a song into its basic elements - the melody (trend),
/// the repeating chorus (seasonal pattern), and the unique variations (residuals).
/// 
/// For example, if you analyze monthly sales data:
/// - The trend component shows the long-term increase or decrease in sales
/// - The seasonal component shows regular patterns that repeat (like higher sales during holidays)
/// - The residual component shows what's left after removing trend and seasonality (like unexpected events)
/// 
/// This decomposition helps you understand what's driving your time series and can improve forecasting.
/// The model offers different algorithms (standard, robust, and fast) to handle various types of data.
/// </para>
/// </remarks>
public class STLDecomposition<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options for the STL decomposition.
    /// </summary>
    private STLDecompositionOptions<T> _stlOptions;

    /// <summary>
    /// The trend component of the time series.
    /// </summary>
    private Vector<T> _trend;

    /// <summary>
    /// The seasonal component of the time series.
    /// </summary>
    private Vector<T> _seasonal;

    /// <summary>
    /// The residual component of the time series.
    /// </summary>
    private Vector<T> _residual;

    /// <summary>
    /// Initializes a new instance of the STLDecomposition class with optional configuration options.
    /// </summary>
    /// <param name="options">The configuration options for STL decomposition. If null, default options are used.</param>
    public STLDecomposition(STLDecompositionOptions<T>? options = null) 
        : base(options ?? new STLDecompositionOptions<T>())
    {
        _stlOptions = options ?? new STLDecompositionOptions<T>();
        _trend = Vector<T>.Empty();
        _seasonal = Vector<T>.Empty();
        _residual = Vector<T>.Empty();
    }

    /// <summary>
    /// Trains the STL decomposition model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input features matrix (not used in STL decomposition).</param>
    /// <param name="y">The time series data to decompose.</param>
    /// <exception cref="ArgumentException">Thrown when the time series is too short for the specified seasonal period.</exception>
    /// <remarks>
    /// <para>
    /// The training process decomposes the time series into trend, seasonal, and residual components
    /// using one of three algorithms: standard, robust, or fast.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training this model means breaking down your time series into three parts:
    /// 
    /// 1. Trend: The long-term direction of your data (going up, down, or staying flat)
    /// 2. Seasonal: Regular patterns that repeat at fixed intervals
    /// 3. Residual: What's left after removing trend and seasonal components
    /// 
    /// The model offers three different methods to do this:
    /// - Standard: The classic STL algorithm that works well for most data
    /// - Robust: Better handles outliers (unusual data points) but takes longer
    /// - Fast: A quicker version that may be less precise but works well for large datasets
    /// 
    /// After training, you can access each component separately to better understand your data.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (y.Length < _stlOptions.SeasonalPeriod * 2)
        {
            throw new ArgumentException("Time series is too short for the specified seasonal period.");
        }

        int n = y.Length;
        _trend = new Vector<T>(n);
        _seasonal = new Vector<T>(n);
        _residual = new Vector<T>(n);

        switch (_stlOptions.AlgorithmType)
        {
            case STLAlgorithmType.Standard:
                PerformStandardSTL(y);
                break;
            case STLAlgorithmType.Robust:
                PerformRobustSTL(y);
                break;
            case STLAlgorithmType.Fast:
                PerformFastSTL(y);
                break;
            default:
                throw new ArgumentException("Invalid STL algorithm type.");
        }
    }

    /// <summary>
    /// Performs the standard STL decomposition algorithm.
    /// </summary>
    /// <param name="y">The time series data to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The standard STL algorithm works in several steps:
    /// 
    /// 1. First, it removes any existing trend estimate from the data
    /// 2. Then it extracts seasonal patterns by looking at values from the same time in different cycles
    /// 3. It smooths these seasonal patterns to make them more consistent
    /// 4. It removes the seasonal component from the original data
    /// 5. It estimates the trend from this deseasonalized data
    /// 6. Finally, it calculates residuals by subtracting both trend and seasonal components from the original data
    /// 
    /// This approach is effective for most time series with clear seasonal patterns.
    /// </para>
    /// </remarks>
    private void PerformStandardSTL(Vector<T> y)
    {
        Vector<T> detrended = y;

        // Step 1: Detrending
        detrended = SubtractVectors(y, _trend);

        // Step 2: Cycle-subseries Smoothing
        _seasonal = CycleSubseriesSmoothing(detrended, _stlOptions.SeasonalPeriod, _stlOptions.SeasonalLoessWindow);

        // Step 3: Low-pass Filtering of Smoothed Cycle-subseries
        Vector<T> lowPassSeasonal = LowPassFilter(_seasonal, _stlOptions.LowPassFilterWindowSize);

        // Step 4: Detrending of Smoothed Cycle-subseries
        _seasonal = SubtractVectors(_seasonal, lowPassSeasonal);

        // Step 5: Deseasonalizing
        Vector<T> deseasonalized = SubtractVectors(y, _seasonal);

        // Step 6: Trend Smoothing
        _trend = LoessSmoothing(deseasonalized, _stlOptions.TrendLoessWindow);

        // Step 7: Calculation of Residuals
        _residual = CalculateResiduals(y, _trend, _seasonal);
    }

    /// <summary>
    /// Performs the robust STL decomposition algorithm, which is less sensitive to outliers.
    /// </summary>
    /// <param name="y">The time series data to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The robust STL algorithm is like the standard version, but with added protection against outliers
    /// (unusual data points that could distort the results).
    /// 
    /// It works by:
    /// 1. Running the standard STL algorithm
    /// 2. Identifying potential outliers by looking at the residuals
    /// 3. Assigning lower weights to these outliers
    /// 4. Running the algorithm again with these weights
    /// 5. Repeating this process several times
    /// 
    /// This approach is particularly useful for data with anomalies or measurement errors.
    /// </para>
    /// </remarks>
    private void PerformRobustSTL(Vector<T> y)
    {
        Vector<T> detrended = y;
        Vector<T> robustnessWeights = Vector<T>.CreateDefault(y.Length, NumOps.One);

        for (int iteration = 0; iteration < _stlOptions.RobustIterations; iteration++)
        {
            PerformStandardSTL(y);

            // Apply robustness weights if not the last iteration
            if (iteration < _stlOptions.RobustIterations - 1)
            {
                robustnessWeights = CalculateRobustWeights(_residual);
                y = ApplyRobustnessWeights(y, robustnessWeights);
            }
        }
    }

    /// <summary>
    /// Performs a faster version of the STL decomposition algorithm.
    /// </summary>
    /// <param name="y">The time series data to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The fast STL algorithm is a simplified version that trades some precision for speed.
    /// It's particularly useful for very large datasets where the standard algorithm might be too slow.
    /// 
    /// It works by:
    /// 1. Estimating the trend using a simple moving average
    /// 2. Removing this trend to find seasonal patterns
    /// 3. Averaging values from the same season across different cycles
    /// 4. Smoothing these seasonal patterns
    /// 5. Removing the seasonal component and re-estimating the trend
    /// 6. Calculating residuals and normalizing the seasonal component
    /// 
    /// While not as precise as the standard algorithm, it often provides good results much faster.
    /// </para>
    /// </remarks>
    private void PerformFastSTL(Vector<T> y)
    {
        int n = y.Length;
        int period = _stlOptions.SeasonalPeriod;
        int trendWindow = _stlOptions.TrendWindowSize;
        int seasonalWindow = _stlOptions.SeasonalLoessWindow;

        // Step 1: Initial Trend Estimation (using moving average)
        _trend = MovingAverage(y, trendWindow);

        // Step 2: Detrending
        Vector<T> detrended = SubtractVectors(y, _trend);

        // Step 3: Initial Seasonal Estimation
        _seasonal = new Vector<T>(n);
        for (int i = 0; i < period; i++)
        {
            T seasonalValue = NumOps.Zero;
            int count = 0;
            for (int j = i; j < n; j += period)
            {
                seasonalValue = NumOps.Add(seasonalValue, detrended[j]);
                count++;
            }
            seasonalValue = NumOps.Divide(seasonalValue, NumOps.FromDouble(count));
            for (int j = i; j < n; j += period)
            {
                _seasonal[j] = seasonalValue;
            }
        }

        // Step 4: Seasonal Smoothing
        _seasonal = SmoothSeasonal(_seasonal, period, seasonalWindow);

        // Step 5: Seasonal Adjustment
        Vector<T> seasonallyAdjusted = SubtractVectors(y, _seasonal);

        // Step 6: Final Trend Estimation
        _trend = MovingAverage(seasonallyAdjusted, trendWindow);

        // Step 7: Calculation of Residuals
        _residual = CalculateResiduals(y, _trend, _seasonal);

        // Step 8: Normalize Seasonal Component
        NormalizeSeasonal();
    }

    /// <summary>
    /// Smooths the seasonal component by applying a moving average to each subseries.
    /// </summary>
    /// <param name="seasonal">The seasonal component to smooth.</param>
    /// <param name="period">The seasonal period.</param>
    /// <param name="window">The window size for smoothing.</param>
    /// <returns>The smoothed seasonal component.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method makes the seasonal pattern more consistent by smoothing out irregularities.
    /// It works by:
    /// 1. Grouping values from the same position in each seasonal cycle (e.g., all Januaries together)
    /// 2. Applying a moving average to each group
    /// 3. Putting these smoothed values back in their original positions
    /// 
    /// This helps identify the true seasonal pattern by reducing random variations.
    /// </para>
    /// </remarks>
    private Vector<T> SmoothSeasonal(Vector<T> seasonal, int period, int window)
    {
        Vector<T> smoothed = new Vector<T>(seasonal.Length);
        for (int i = 0; i < period; i++)
        {
            List<T> subseries = new List<T>();
            for (int j = i; j < seasonal.Length; j += period)
            {
                subseries.Add(seasonal[j]);
            }
            Vector<T> smoothedSubseries = MovingAverage(new Vector<T>(subseries), window);
            for (int j = 0; j < smoothedSubseries.Length; j++)
            {
                smoothed[i + j * period] = smoothedSubseries[j];
            }
        }

        return smoothed;
    }

    /// <summary>
    /// Normalizes the seasonal component to ensure it sums to zero, adjusting the trend accordingly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method ensures that the seasonal component represents only the seasonal variations
    /// by making it average to zero. Any overall level is moved to the trend component.
    /// 
    /// It's like making sure that seasonal adjustments don't change the overall average of the data.
    /// For example, if we say summer increases sales by 20% and winter decreases them by 20%,
    /// the average effect should be zero.
    /// </para>
    /// </remarks>
    private void NormalizeSeasonal()
    {
        T seasonalMean = _seasonal.Sum();
        seasonalMean = NumOps.Divide(seasonalMean, NumOps.FromDouble(_seasonal.Length));
        _seasonal = _seasonal.Transform(s => NumOps.Subtract(s, seasonalMean));
        _trend = _trend.Transform(t => NumOps.Add(t, seasonalMean));
    }

    /// <summary>
    /// Calculates a moving average of the input data with the specified window size.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="windowSize">The window size for the moving average.</param>
    /// <returns>The moving average of the input data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// A moving average smooths out data by averaging each point with its neighbors.
    /// For example, with a window size of 3, each point is replaced by the average of itself
    /// and the points immediately before and after it.
    /// 
    /// This method handles edge cases (points near the beginning or end of the data)
    /// by using smaller windows when necessary.
    /// 
    /// Moving averages help identify trends by reducing short-term fluctuations.
    /// </para>
    /// </remarks>
    private Vector<T> MovingAverage(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n);
        T windowSum = NumOps.Zero;
        int effectiveWindow = Math.Min(windowSize, n);

        // Initialize the first window
        for (int i = 0; i < effectiveWindow; i++)
        {
                        windowSum = NumOps.Add(windowSum, data[i]);
        }

        // Calculate moving average
        for (int i = 0; i < n; i++)
        {
            if (i < effectiveWindow / 2 || i >= n - effectiveWindow / 2)
            {
                // Edge case: use available data points
                int start = Math.Max(0, i - effectiveWindow / 2);
                int end = Math.Min(n, i + effectiveWindow / 2 + 1);
                T sum = NumOps.Zero;
                for (int j = start; j < end; j++)
                {
                    sum = NumOps.Add(sum, data[j]);
                }
                result[i] = NumOps.Divide(sum, NumOps.FromDouble(end - start));
            }
            else
            {
                // Regular case: use full window
                result[i] = NumOps.Divide(windowSum, NumOps.FromDouble(effectiveWindow));
                if (i + effectiveWindow / 2 + 1 < n)
                {
                    windowSum = NumOps.Add(windowSum, data[i + effectiveWindow / 2 + 1]);
                }
                if (i - effectiveWindow / 2 >= 0)
                {
                    windowSum = NumOps.Subtract(windowSum, data[i - effectiveWindow / 2]);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Subtracts one vector from another element-wise.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The vector to subtract.</param>
    /// <returns>The result of a - b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method simply subtracts each element of the second vector from the corresponding element
    /// of the first vector. For example, if a = [5, 10, 15] and b = [1, 2, 3], then the result
    /// would be [4, 8, 12].
    /// 
    /// In the context of STL decomposition, this is used to remove components (like trend or seasonal)
    /// from the original time series.
    /// </para>
    /// </remarks>
    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        return new Vector<T>(a.Zip(b, (x, y) => NumOps.Subtract(x, y)));
    }

    /// <summary>
    /// Performs cycle-subseries smoothing on the input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="period">The seasonal period.</param>
    /// <param name="loessWindow">The window size for LOESS smoothing.</param>
    /// <returns>The smoothed data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method extracts and smooths seasonal patterns by:
    /// 
    /// 1. Grouping values from the same position in each seasonal cycle (e.g., all Januaries together)
    /// 2. Applying LOESS smoothing to each group (which is like a sophisticated moving average)
    /// 3. Putting these smoothed values back in their original positions
    /// 
    /// This approach helps identify consistent seasonal patterns while reducing the influence of random variations.
    /// </para>
    /// </remarks>
    private Vector<T> CycleSubseriesSmoothing(Vector<T> data, int period, int loessWindow)
    {
        Vector<T> smoothed = new Vector<T>(data.Length);
        for (int i = 0; i < period; i++)
        {
            List<(T x, T y)> subseries = new List<(T x, T y)>();
            for (int j = i; j < data.Length; j += period)
            {
                subseries.Add((NumOps.FromDouble(j), data[j]));
            }
            Vector<T> smoothedSubseries = LoessSmoothing(subseries, loessWindow);
            for (int j = 0; j < smoothedSubseries.Length; j++)
            {
                smoothed[i + j * period] = smoothedSubseries[j];
            }
        }

        return smoothed;
    }

    /// <summary>
    /// Applies a low-pass filter to the input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="windowSize">The window size for filtering.</param>
    /// <returns>The filtered data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// A low-pass filter removes high-frequency variations (rapid changes) while preserving
    /// low-frequency patterns (gradual changes).
    /// 
    /// This method applies multiple moving averages in sequence, which effectively smooths out
    /// short-term fluctuations. It's like looking at your data through progressively blurrier lenses
    /// to focus only on the major patterns.
    /// 
    /// In STL decomposition, this helps ensure that the seasonal component doesn't contain any trend.
    /// </para>
    /// </remarks>
    private Vector<T> LowPassFilter(Vector<T> data, int windowSize)
    {
        return MovingAverage(MovingAverage(MovingAverage(data, windowSize), windowSize), 3);
    }

    /// <summary>
    /// Calculates the residual component by subtracting trend and seasonal components from the original data.
    /// </summary>
    /// <param name="y">The original time series data.</param>
    /// <param name="trend">The trend component.</param>
    /// <param name="seasonal">The seasonal component.</param>
    /// <returns>The residual component.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The residual component represents what's left after removing the trend and seasonal patterns.
    /// It includes random variations, noise, and any patterns not captured by the trend or seasonal components.
    /// 
    /// Calculating it is straightforward: for each point in the time series, subtract both the trend value
    /// and the seasonal value from the original value.
    /// 
    /// Analyzing residuals can help identify unusual events or additional patterns that might need modeling.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateResiduals(Vector<T> y, Vector<T> trend, Vector<T> seasonal)
    {
        return new Vector<T>(y.Zip(trend, (a, b) => NumOps.Subtract(a, b))
                 .Zip(seasonal, (a, b) => NumOps.Subtract(a, b)));
    }

    /// <summary>
    /// Calculates robustness weights based on residuals to reduce the influence of outliers.
    /// </summary>
    /// <param name="residuals">The residual component.</param>
    /// <returns>A vector of weights for each data point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method identifies potential outliers (unusual data points) and assigns them lower weights
    /// to reduce their influence on the decomposition.
    /// 
    /// It works by:
    /// 1. Calculating the absolute value of each residual
    /// 2. Finding the median (middle value) of these absolute residuals
    /// 3. Setting a threshold at 6 times this median
    /// 4. Assigning weights based on how far each residual is from zero compared to this threshold
    /// 
    /// Points with small residuals get weights close to 1 (full influence), while points with large
    /// residuals get lower weights (reduced influence). This helps the model focus on the typical
    /// patterns in the data rather than being distorted by unusual events.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateRobustWeights(Vector<T> residuals)
    {
        Vector<T> absResiduals = residuals.Transform(r => NumOps.Abs(r));
        T median = absResiduals.Median();
        T threshold = NumOps.Multiply(NumOps.FromDouble(6), median);

        return absResiduals.Transform(r => 
        {
            if (NumOps.LessThan(r, threshold))
            {
                T weight = NumOps.Subtract(NumOps.One, NumOps.Power(NumOps.Divide(r, threshold), NumOps.FromDouble(2)));
                return MathHelper.Max(weight, NumOps.FromDouble(_stlOptions.RobustWeightThreshold));
            }
            else
            {
                return NumOps.FromDouble(_stlOptions.RobustWeightThreshold);
            }
        });
    }

    /// <summary>
    /// Applies robustness weights to the input data.
    /// </summary>
    /// <param name="y">The input data.</param>
    /// <param name="weights">The weights to apply.</param>
    /// <returns>The weighted data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method applies the calculated weights to the original data, effectively reducing
    /// the influence of potential outliers. It simply multiplies each data point by its
    /// corresponding weight.
    /// 
    /// In the robust STL algorithm, this weighted data is then used for the next iteration
    /// of decomposition, resulting in estimates that are less affected by unusual data points.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyRobustnessWeights(Vector<T> y, Vector<T> weights)
    {
        return new Vector<T>(y.Zip(weights, (a, b) => NumOps.Multiply(a, b)));
    }

    /// <summary>
    /// Calculates the tri-cube weight function used in LOESS smoothing.
    /// </summary>
    /// <param name="x">The input value (typically a normalized distance).</param>
    /// <returns>The tri-cube weight.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The tri-cube function is a special weighting function used in LOESS smoothing.
    /// It assigns weights based on how far points are from the one being estimated:
    /// - Points very close get weights near 1 (high influence)
    /// - Points farther away get progressively smaller weights
    /// - Points beyond a certain distance get weights of 0 (no influence)
    /// 
    /// This creates a smooth transition from high influence to no influence as distance increases,
    /// which helps LOESS produce smooth trend and seasonal estimates.
    /// </para>
    /// </remarks>
    private T TriCube(T x)
    {
        T absX = NumOps.Abs(x);
        if (NumOps.GreaterThan(absX, NumOps.One))
        {
            return NumOps.Zero;
        }

        T oneMinusAbsX = NumOps.Subtract(NumOps.One, absX);
        T cube = NumOps.Multiply(NumOps.Multiply(oneMinusAbsX, oneMinusAbsX), oneMinusAbsX);

        return NumOps.Multiply(cube, cube);
    }

    /// <summary>
    /// Performs LOESS (locally weighted regression) smoothing on the input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="windowSize">The window size for smoothing.</param>
    /// <returns>The smoothed data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// LOESS smoothing is like a sophisticated moving average that adapts to the local structure of the data.
    /// For each point, it:
    /// 
    /// 1. Selects nearby points within a window
    /// 2. Assigns weights to these points based on their distance (closer points get higher weights)
    /// 3. Fits a weighted average to estimate the smoothed value
    /// 
    /// This approach creates a smooth curve that follows the general pattern of the data while
    /// reducing the influence of random fluctuations. It's particularly good at handling data
    /// where the underlying pattern changes gradually over time.
    /// </para>
    /// </remarks>
    private Vector<T> LoessSmoothing(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T x = NumOps.FromDouble(i);
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = 0; j < n; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(NumOps.FromDouble(j), x));
                if (NumOps.LessThanOrEquals(distance, NumOps.FromDouble(windowSize)))
                {
                    T weight = TriCube(NumOps.Divide(distance, NumOps.FromDouble(windowSize)));
                    weightedPoints.Add((distance, weight, data[j]));
                }
            }

            if (weightedPoints.Count > 0)
            {
                result[i] = WeightedLeastSquares(weightedPoints);
            }
            else
            {
                result[i] = data[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates a weighted average of points based on their weights.
    /// </summary>
    /// <param name="weightedPoints">A list of points with their distances, weights, and values.</param>
    /// <returns>The weighted average.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// A weighted average is like a regular average, but some values count more than others.
    /// Each value is multiplied by its weight, then all these products are summed and divided
    /// by the sum of the weights.
    /// 
    /// For example, if we have values [10, 20, 30] with weights [1, 2, 1]:
    /// - The weighted sum is 10×1 + 20×2 + 30×1 = 80
    /// - The sum of weights is 1 + 2 + 1 = 4
    /// - The weighted average is 80 ÷ 4 = 20
    /// 
    /// In LOESS smoothing, this gives more influence to points that are closer to the one being estimated.
    /// </para>
    /// </remarks>
    private T WeightedLeastSquares(List<(T distance, T weight, T y)> weightedPoints)
    {
        T sumWeights = NumOps.Zero;
        T sumWeightedY = NumOps.Zero;

        foreach (var point in weightedPoints)
        {
            sumWeights = NumOps.Add(sumWeights, point.weight);
            sumWeightedY = NumOps.Add(sumWeightedY, NumOps.Multiply(point.weight, point.y));
        }

        return NumOps.Divide(sumWeightedY, sumWeights);
    }

    /// <summary>
    /// Performs LOESS smoothing on a list of (x,y) points with a specified span.
    /// </summary>
    /// <param name="data">The list of (x,y) points.</param>
    /// <param name="span">The proportion of points to include in each local regression.</param>
    /// <returns>The smoothed y-values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This version of LOESS smoothing works with arbitrary x-coordinates (not just evenly spaced points).
    /// For each point, it:
    /// 
        /// 1. Calculates the distance from this point to all other points
    /// 2. Selects a proportion (span) of the closest points
    /// 3. Assigns weights based on these distances
    /// 4. Fits a weighted linear regression to estimate the smoothed value
    /// 
    /// This is more flexible than the previous method because it can handle unevenly spaced data
    /// and adapts the window size based on the data density.
    /// </para>
    /// </remarks>
    private Vector<T> LoessSmoothing(List<(T x, T y)> data, double span)
    {
        int n = data.Count;
        Vector<T> result = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T x = data[i].x;
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = 0; j < n; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(data[j].x, x));
                weightedPoints.Add((distance, NumOps.Zero, data[j].y));
            }

            weightedPoints.Sort((a, b) => 
            {
                if (NumOps.LessThan(a.distance, b.distance))
                    return -1;
                else if (NumOps.GreaterThan(a.distance, b.distance))
                    return 1;
                else
                    return 0;
            });
            int q = (int)(n * span);
            T maxDistance = weightedPoints[q - 1].distance;

            for (int j = 0; j < q; j++)
            {
                T weight = TriCube(NumOps.Divide(weightedPoints[j].distance, maxDistance));
                weightedPoints[j] = (weightedPoints[j].distance, weight, weightedPoints[j].y);
            }

            result[i] = WeightedLeastSquares(weightedPoints.Take(q).ToList(), x);
        }

        return result;
    }

    /// <summary>
    /// Performs weighted least squares regression to estimate a value at a specific x-coordinate.
    /// </summary>
    /// <param name="weightedPoints">A list of points with their distances, weights, and values.</param>
    /// <param name="x">The x-coordinate at which to estimate the value.</param>
    /// <returns>The estimated value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method fits a weighted linear regression line to nearby points and uses it to estimate
    /// the value at a specific x-coordinate. It's like drawing a "best fit" line through nearby points,
    /// giving more importance to closer points, and then reading the y-value where this line crosses
    /// the desired x-coordinate.
    /// 
    /// The process involves:
    /// 1. Calculating weighted means of x and y
    /// 2. Computing the slope of the regression line
    /// 3. Computing the intercept of the regression line
    /// 4. Using the equation y = intercept + slope * x to estimate the value
    /// 
    /// This approach allows LOESS to adapt to local trends in the data, producing a smooth curve
    /// that follows the data more closely than a simple weighted average.
    /// </para>
    /// </remarks>
    private T WeightedLeastSquares(List<(T distance, T weight, T y)> weightedPoints, T x)
    {
        T sumWeights = NumOps.Zero;
        T sumWeightedY = NumOps.Zero;
        T sumWeightedX = NumOps.Zero;
        T sumWeightedXY = NumOps.Zero;
        T sumWeightedX2 = NumOps.Zero;

        foreach (var (_, weight, y) in weightedPoints)
        {
            sumWeights = NumOps.Add(sumWeights, weight);
            sumWeightedY = NumOps.Add(sumWeightedY, NumOps.Multiply(weight, y));
            sumWeightedX = NumOps.Add(sumWeightedX, NumOps.Multiply(weight, x));
            sumWeightedXY = NumOps.Add(sumWeightedXY, NumOps.Multiply(NumOps.Multiply(weight, x), y));
            sumWeightedX2 = NumOps.Add(sumWeightedX2, NumOps.Multiply(NumOps.Multiply(weight, x), x));
        }

        T meanX = NumOps.Divide(sumWeightedX, sumWeights);
        T meanY = NumOps.Divide(sumWeightedY, sumWeights);

        T numerator = NumOps.Subtract(sumWeightedXY, NumOps.Multiply(sumWeightedX, meanY));
        T denominator = NumOps.Subtract(sumWeightedX2, NumOps.Multiply(sumWeightedX, meanX));

        T slope = NumOps.Divide(numerator, denominator);
        T intercept = NumOps.Subtract(meanY, NumOps.Multiply(slope, meanX));

        return NumOps.Add(intercept, NumOps.Multiply(slope, x));
    }

    /// <summary>
    /// Gets the trend component of the time series.
    /// </summary>
    /// <returns>The trend component.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The trend component represents the long-term direction of your time series.
    /// It shows whether your data is generally increasing, decreasing, or staying flat over time,
    /// after removing seasonal patterns and random fluctuations.
    /// 
    /// For example, in retail sales data, the trend might show steady growth over several years,
    /// even though there are seasonal peaks during holidays and random variations from month to month.
    /// </para>
    /// </remarks>
    public Vector<T> GetTrend() => _trend ?? throw new InvalidOperationException("Model has not been trained.");

    /// <summary>
    /// Gets the seasonal component of the time series.
    /// </summary>
    /// <returns>The seasonal component.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The seasonal component represents regular patterns that repeat at fixed intervals.
    /// These patterns might be related to time of day, day of week, month of year, or any other
    /// cyclical factor relevant to your data.
    /// 
    /// For example, retail sales might show a seasonal pattern with peaks in December (holiday shopping),
    /// back-to-school season, and other predictable times. The seasonal component captures these
    /// recurring patterns.
    /// </para>
    /// </remarks>
    public Vector<T> GetSeasonal() => _seasonal ?? throw new InvalidOperationException("Model has not been trained.");

    /// <summary>
    /// Gets the residual component of the time series.
    /// </summary>
    /// <returns>The residual component.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The residual component is what remains after removing both trend and seasonal patterns.
    /// It includes random variations, noise, and any patterns not captured by the other components.
    /// 
    /// Analyzing residuals can help identify:
    /// - Unusual events or outliers
    /// - Additional patterns that might need modeling
    /// - Whether the decomposition has successfully captured the main patterns in the data
    /// 
    /// Ideally, residuals should look like random noise with no obvious patterns.
    /// </para>
    /// </remarks>
    public Vector<T> GetResidual() => _residual ?? throw new InvalidOperationException("Model has not been trained.");

    /// <summary>
    /// Generates forecasts for future time periods based on the decomposed components.
    /// </summary>
    /// <param name="input">The input features matrix specifying the forecast horizon.</param>
    /// <returns>A vector of forecasted values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts future values by:
    /// 
    /// 1. Using the last trend value (assuming the trend continues at the same level)
    /// 2. Adding the appropriate seasonal pattern from the last observed season
    /// 
    /// This simple approach works well for short-term forecasts when the trend is relatively stable.
    /// For example, if we've decomposed monthly sales data and want to forecast the next few months,
    /// we'd use the most recent trend level and add the seasonal pattern from the same months in the previous year.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_trend == null || _seasonal == null)
            throw new InvalidOperationException("Model has not been trained.");

        int forecastHorizon = input.Rows;
        Vector<T> forecast = new Vector<T>(forecastHorizon);

        // Extend trend using last trend value
        T lastTrendValue = _trend[_trend.Length - 1];

        // Use last full season for seasonal component
        int seasonLength = _stlOptions.SeasonalPeriod;
        Vector<T> lastSeason = new Vector<T>(seasonLength);
        for (int i = 0; i < seasonLength; i++)
        {
            lastSeason[i] = _seasonal[_seasonal.Length - seasonLength + i];
        }

        for (int i = 0; i < forecastHorizon; i++)
        {
            forecast[i] = NumOps.Add(lastTrendValue, lastSeason[i % seasonLength]);
        }

        return forecast;
    }

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tests how well our model performs by comparing its predictions to actual values.
    /// It calculates two error metrics:
    /// 
    /// - MAE (Mean Absolute Error): The average of the absolute differences between predictions and actual values
    /// - RMSE (Root Mean Squared Error): The square root of the average squared differences, which gives more weight to larger errors
    /// 
    /// Lower values for these metrics indicate better model performance. They help you understand
    /// how accurate your forecasts are likely to be and compare different models or parameter settings.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        if (_trend == null || _seasonal == null || _residual == null)
            throw new InvalidOperationException("Model has not been trained.");

        Vector<T> yPred = Predict(xTest);
    
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Calculate Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, yPred);

        // Calculate Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, yPred);

        return metrics;
    }

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization is the process of converting the model's state into a format that can be saved to disk.
    /// This allows you to save a trained model and load it later without having to retrain it.
    /// 
    /// This method saves:
    /// - The STL configuration options (like seasonal period and window sizes)
    /// - The decomposed components (trend, seasonal, and residual)
    /// 
    /// After serializing, you can store the model and later deserialize it to make predictions
    /// or continue analysis without repeating the decomposition process.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write STL options
        writer.Write(_stlOptions.SeasonalPeriod);
        writer.Write(_stlOptions.TrendWindowSize);
        writer.Write(_stlOptions.SeasonalLoessWindow);
        writer.Write(_stlOptions.TrendLoessWindow);
        writer.Write(_stlOptions.LowPassFilterWindowSize);
        writer.Write(_stlOptions.RobustIterations);
        writer.Write(_stlOptions.RobustWeightThreshold);

        // Write decomposition components
        SerializationHelper<T>.SerializeVector(writer, _trend);
        SerializationHelper<T>.SerializeVector(writer, _seasonal);
        SerializationHelper<T>.SerializeVector(writer, _residual);
    }

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from disk.
    /// This method reads the model's parameters from a file and reconstructs the model
    /// exactly as it was when it was saved.
    /// 
    /// This allows you to:
    /// - Load a previously trained model without retraining
    /// - Make predictions with consistent results
    /// - Continue analysis from where you left off
    /// 
    /// It's like saving your work in a document and opening it later to continue editing.
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read STL options
        int seasonalPeriod = reader.ReadInt32();
        int trendWindowSize = reader.ReadInt32();
        int seasonalLoessWindow = reader.ReadInt32();
        int trendLoessWindow = reader.ReadInt32();
        int lowPassFilterWindowSize = reader.ReadInt32();
        int robustIterations = reader.ReadInt32();
        double robustWeightThreshold = reader.ReadDouble();

        // Create new STLDecompositionOptions<T> with the read values
        _stlOptions = new STLDecompositionOptions<T>
        {
            SeasonalPeriod = seasonalPeriod,
            TrendWindowSize = trendWindowSize,
            SeasonalLoessWindow = seasonalLoessWindow,
            TrendLoessWindow = trendLoessWindow,
            LowPassFilterWindowSize = lowPassFilterWindowSize,
            RobustIterations = robustIterations,
            RobustWeightThreshold = robustWeightThreshold
        };

        // Read decomposition components
        _trend = SerializationHelper<T>.DeserializeVector(reader);
        _seasonal = SerializationHelper<T>.DeserializeVector(reader);
        _residual = SerializationHelper<T>.DeserializeVector(reader);
    }
}