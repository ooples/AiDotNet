namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements the X-11 method for time series decomposition, which breaks down a time series into trend, seasonal, and irregular components.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The X-11 method is a statistical technique that helps understand patterns in data that changes over time.
/// It separates your data into three main parts:
/// - Trend: The long-term direction of your data (going up, down, or staying flat over time)
/// - Seasonal: Regular patterns that repeat at fixed intervals (like higher sales during holidays)
/// - Irregular: Random fluctuations that don't follow any pattern
/// 
/// This helps you understand what's really happening in your data by removing predictable patterns.
/// </para>
/// </remarks>
public class X11Decomposition<T> : TimeSeriesDecompositionBase<T>
{
    /// <summary>
    /// The number of observations in one complete seasonal cycle.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is how often patterns repeat in your data. For monthly data, it's 12 (for 12 months in a year).
    /// For quarterly data, it's 4. For daily data with weekly patterns, it's 7.
    /// </remarks>
    private readonly int _seasonalPeriod;

    /// <summary>
    /// The window size used for the moving average calculation of the trend component.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how many data points are used to calculate each trend value.
    /// Larger values create smoother trends by averaging over more data points.
    /// </remarks>
    private readonly int _trendCycleMovingAverageWindow;

    /// <summary>
    /// The type of X-11 algorithm to use for decomposition.
    /// </summary>
    private readonly X11AlgorithmType _algorithmType;

    /// <summary>
    /// Creates a new instance of the X11Decomposition class and performs the decomposition.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="seasonalPeriod">The number of observations in one complete seasonal cycle (default: 12 for monthly data).</param>
    /// <param name="trendCycleMovingAverageWindow">The window size for trend estimation (default: 13).</param>
    /// <param name="algorithmType">The type of X-11 algorithm to use (default: Standard).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor sets up the decomposition process with your data and some options.
    /// - timeSeries: Your data points ordered by time
    /// - seasonalPeriod: How often patterns repeat (12 for monthly data, 4 for quarterly, etc.)
    /// - trendCycleMovingAverageWindow: Controls how smooth the trend line will be
    /// - algorithmType: Different methods for breaking down your data
    /// </remarks>
    public X11Decomposition(Vector<T> timeSeries, int seasonalPeriod = 12, int trendCycleMovingAverageWindow = 13, X11AlgorithmType algorithmType = X11AlgorithmType.Standard)
        : base(timeSeries)
    {
        if (seasonalPeriod <= 0)
        {
            throw new ArgumentException("Seasonal period must be a positive integer.", nameof(seasonalPeriod));
        }
        if (trendCycleMovingAverageWindow <= 0 || trendCycleMovingAverageWindow % 2 == 0)
        {
            throw new ArgumentException("Trend-cycle moving average window must be a positive odd integer.", nameof(trendCycleMovingAverageWindow));
        }

        _seasonalPeriod = seasonalPeriod;
        _trendCycleMovingAverageWindow = trendCycleMovingAverageWindow;
        _algorithmType = algorithmType;
        Decompose();
    }

    /// <summary>
    /// Performs the time series decomposition based on the selected algorithm type.
    /// </summary>
    /// <remarks>
    /// This method selects the appropriate decomposition algorithm based on the specified algorithm type.
    /// </remarks>
    protected override void Decompose()
    {
        switch (_algorithmType)
        {
            case X11AlgorithmType.Standard:
                DecomposeStandard();
                break;
            case X11AlgorithmType.MultiplicativeAdjustment:
                DecomposeMultiplicative();
                break;
            case X11AlgorithmType.LogAdditiveAdjustment:
                DecomposeLogAdditive();
                break;
            default:
                throw new ArgumentException("Unsupported X11 algorithm type.");
        }
    }

    /// <summary>
    /// Performs standard additive X-11 decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method assumes your data components add together (trend + seasonal + irregular = original data).
    /// It's suitable when seasonal patterns have consistent amplitude regardless of the trend level.
    /// </remarks>
    private void DecomposeStandard()
    {
        // Step 1: Initial trend-cycle estimate
        Vector<T> trendCycle = CenteredMovingAverage(TimeSeries, _trendCycleMovingAverageWindow);

        // Step 2: Initial seasonal-irregular estimate
        Vector<T> seasonalIrregular = TimeSeries.Subtract(trendCycle);

        // Step 3: Initial seasonal factors
        Vector<T> seasonalFactors = EstimateSeasonalFactors(seasonalIrregular);

        // Step 4: Seasonally adjusted series
        Vector<T> seasonallyAdjusted = TimeSeries.Subtract(seasonalFactors);

        // Step 5: Final trend-cycle estimate
        trendCycle = CenteredMovingAverage(seasonallyAdjusted, _trendCycleMovingAverageWindow);

        // Step 6: Final irregular component
        Vector<T> irregular = seasonallyAdjusted.Subtract(trendCycle);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trendCycle);
        AddComponent(DecompositionComponentType.Seasonal, seasonalFactors);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    /// <summary>
    /// Performs multiplicative X-11 decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method assumes your data components multiply together (trend × seasonal × irregular = original data).
    /// It's suitable when seasonal patterns have amplitude that changes proportionally with the trend level.
    /// For example, if sales increase over time, the seasonal peaks and valleys also get larger.
    /// </remarks>
    private void DecomposeMultiplicative()
    {
        // Step 1: Initial trend-cycle estimate
        Vector<T> trendCycle = CenteredMovingAverage(TimeSeries, _trendCycleMovingAverageWindow);

        // Step 2: Initial seasonal-irregular ratios
        Vector<T> seasonalIrregular = TimeSeries.ElementwiseDivide(trendCycle);

        // Step 3: Initial seasonal factors
        Vector<T> seasonalFactors = EstimateSeasonalFactorsMultiplicative(seasonalIrregular);

        // Step 4: Seasonally adjusted series
        Vector<T> seasonallyAdjusted = TimeSeries.ElementwiseDivide(seasonalFactors);

        // Step 5: Refined trend-cycle estimate using Henderson moving average
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13); // Using Henderson 13-term moving average

        // Step 6: Refined seasonal-irregular ratios
        seasonalIrregular = TimeSeries.ElementwiseDivide(trendCycle);

        // Step 7: Final seasonal factors
        seasonalFactors = EstimateSeasonalFactorsMultiplicative(seasonalIrregular);

        // Step 8: Final seasonally adjusted series
        seasonallyAdjusted = TimeSeries.ElementwiseDivide(seasonalFactors);

        // Step 9: Final trend-cycle estimate
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13);

        // Step 10: Final irregular component
        Vector<T> irregular = seasonallyAdjusted.ElementwiseDivide(trendCycle);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trendCycle);
        AddComponent(DecompositionComponentType.Seasonal, seasonalFactors);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    /// <summary>
    /// Estimates seasonal factors for multiplicative decomposition.
    /// </summary>
    /// <param name="seasonalIrregular">The combined seasonal and irregular components.</param>
    /// <returns>The estimated seasonal factors.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method finds the repeating patterns in your data by grouping values that occur
    /// at the same point in each cycle (like all Januaries, all Februaries, etc. for monthly data).
    /// It then calculates the average pattern and ensures it averages to 1.0 (no net effect).
    /// </remarks>
    private Vector<T> EstimateSeasonalFactorsMultiplicative(Vector<T> seasonalIrregular)
    {
        int n = seasonalIrregular.Length;
        Vector<T> seasonalFactors = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T product = NumOps.One;
            int count = 0;

            for (int j = i % _seasonalPeriod; j < n; j += _seasonalPeriod)
            {
                product = NumOps.Multiply(product, seasonalIrregular[j]);
                count++;
            }

            seasonalFactors[i] = NumOps.Power(product, NumOps.Divide(NumOps.One, NumOps.FromDouble(count)));
        }

        // Normalize seasonal factors
        T totalFactor = NumOps.Zero;
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            totalFactor = NumOps.Add(totalFactor, seasonalFactors[i]);
        }
        T averageFactor = NumOps.Divide(totalFactor, NumOps.FromDouble(_seasonalPeriod));

        for (int i = 0; i < n; i++)
        {
            seasonalFactors[i] = NumOps.Divide(seasonalFactors[i], averageFactor);
        }

        return seasonalFactors;
    }

    /// <summary>
    /// Applies the Henderson moving average to smooth a time series.
    /// </summary>
    /// <param name="data">The time series data to smooth.</param>
    /// <param name="terms">The number of terms to use in the moving average (must be odd).</param>
    /// <returns>A smoothed version of the input time series.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Henderson moving average is a special type of smoothing technique that helps
    /// identify the underlying trend in your data by removing short-term fluctuations. It works by
    /// calculating a weighted average of nearby points, giving more importance to closer points.
    /// This is like looking at your data through a "smoothing lens" that helps you see the big picture.
    /// </para>
    /// </remarks>
    private Vector<T> HendersonMovingAverage(Vector<T> data, int terms)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n);
        T[] weights = CalculateHendersonWeights(terms);

        int halfTerms = terms / 2;

        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            T weightSum = NumOps.Zero;

            for (int j = -halfTerms; j <= halfTerms; j++)
            {
                int index = i + j;
                if (index >= 0 && index < n)
                {
                    T weight = weights[j + halfTerms];
                    sum = NumOps.Add(sum, NumOps.Multiply(data[index], weight));
                    weightSum = NumOps.Add(weightSum, weight);
                }
            }

            result[i] = NumOps.Divide(sum, weightSum);
        }

        return result;
    }

    /// <summary>
    /// Calculates the weights for the Henderson moving average.
    /// </summary>
    /// <param name="terms">The number of terms in the moving average (must be odd).</param>
    /// <returns>An array of weights for the Henderson moving average.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These weights determine how much importance each data point gets when calculating
    /// the trend. The formula looks complicated, but it's designed to create a smooth curve that
    /// follows the main direction of your data while ignoring random ups and downs.
    /// </para>
    /// </remarks>
    private T[] CalculateHendersonWeights(int terms)
    {
        int m = (terms - 1) / 2;
        T[] weights = new T[terms];

        for (int i = -m; i <= m; i++)
        {
            T t = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(m + 1.0));
            T tSquared = NumOps.Multiply(t, t);

            T factor1 = NumOps.Subtract(NumOps.One, tSquared);
            T factor2 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(5), tSquared));
            T factor3 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(7), tSquared));
            T factor4 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(9), tSquared));
            T factor5 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(11), tSquared));

            T numerator = NumOps.Multiply(NumOps.FromDouble(315),
                NumOps.Multiply(factor1, NumOps.Multiply(factor2, NumOps.Multiply(factor3, NumOps.Multiply(factor4, factor5)))));

            weights[i + m] = NumOps.Divide(numerator, NumOps.FromDouble(320.0));
        }

        return weights;
    }

    /// <summary>
    /// Performs log-additive decomposition of the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method handles time series where components multiply together rather than add.
    /// By taking the logarithm of the data, we convert multiplication into addition, which makes the math easier.
    /// After processing, we convert back using the exponential function.
    /// 
    /// For example, if your data shows 10% growth every month, taking the log lets us work with the constant
    /// growth rate rather than the increasing values.
    /// </para>
    /// </remarks>
    private void DecomposeLogAdditive()
    {
        // Step 1: Apply logarithmic transformation to the time series
        Vector<T> logTimeSeries = TimeSeries.Transform(x => NumOps.Log(x));

        // Step 2: Initial trend-cycle estimate
        Vector<T> trendCycle = CenteredMovingAverage(logTimeSeries, _trendCycleMovingAverageWindow);

        // Step 3: Initial seasonal-irregular estimate
        Vector<T> seasonalIrregular = logTimeSeries.Subtract(trendCycle);

        // Step 4: Initial seasonal factors
        Vector<T> seasonalFactors = EstimateSeasonalFactors(seasonalIrregular);

        // Step 5: Seasonally adjusted series
        Vector<T> seasonallyAdjusted = logTimeSeries.Subtract(seasonalFactors);

        // Step 6: Refined trend-cycle estimate using Henderson moving average
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13);

        // Step 7: Refined seasonal-irregular estimate
        seasonalIrregular = logTimeSeries.Subtract(trendCycle);

        // Step 8: Final seasonal factors
        seasonalFactors = EstimateSeasonalFactors(seasonalIrregular);

        // Step 9: Final seasonally adjusted series
        seasonallyAdjusted = logTimeSeries.Subtract(seasonalFactors);

        // Step 10: Final trend-cycle estimate
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13);

        // Step 11: Final irregular component
        Vector<T> irregular = seasonallyAdjusted.Subtract(trendCycle);

        // Step 12: Apply exponential transformation to convert back to original scale
        AddComponent(DecompositionComponentType.Trend, ApplyExp(trendCycle));
        AddComponent(DecompositionComponentType.Seasonal, ApplyExp(seasonalFactors));
        AddComponent(DecompositionComponentType.Irregular, ApplyExp(irregular));

        // Step 13: Ensure multiplicative consistency
        EnsureMultiplicativeConsistency();
    }

    /// <summary>
    /// Applies the exponential function to each element in a vector.
    /// </summary>
    /// <param name="vector">The vector to transform.</param>
    /// <returns>A new vector with the exponential function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function reverses the logarithm operation. If we took the log of our data earlier,
    /// we need to use this function to convert back to the original scale.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyExp(Vector<T> vector)
    {
        return vector.Transform(x => NumOps.Exp(x));
    }

    /// <summary>
    /// Ensures that the product of all components equals the original time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In multiplicative decomposition, the trend, seasonal, and irregular components
    /// should multiply together to give us back our original data. This method makes small adjustments
    /// to ensure this is true, distributing any differences equally among all three components.
    /// </para>
    /// </remarks>
    private void EnsureMultiplicativeConsistency()
    {
        Vector<T> trend = (Vector<T>?)GetComponent(DecompositionComponentType.Trend) ?? new Vector<T>(TimeSeries.Length);
        Vector<T> seasonal = (Vector<T>?)GetComponent(DecompositionComponentType.Seasonal) ?? new Vector<T>(TimeSeries.Length);
        Vector<T> irregular = (Vector<T>?)GetComponent(DecompositionComponentType.Irregular) ?? new Vector<T>(TimeSeries.Length);

        Vector<T> reconstructed = new Vector<T>(TimeSeries.Length);
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            reconstructed[i] = NumOps.Multiply(NumOps.Multiply(trend[i], seasonal[i]), irregular[i]);
        }

        Vector<T> adjustmentFactor = new Vector<T>(TimeSeries.Length);
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            adjustmentFactor[i] = NumOps.Divide(TimeSeries[i], reconstructed[i]);
        }

        // Distribute the adjustment factor equally among the components
        T cubicRoot = NumOps.FromDouble(1.0 / 3.0);
        Vector<T> componentAdjustment = new Vector<T>(TimeSeries.Length);
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            componentAdjustment[i] = NumOps.Power(adjustmentFactor[i], cubicRoot);
        }

        Vector<T> adjustedTrend = new Vector<T>(TimeSeries.Length);
        Vector<T> adjustedSeasonal = new Vector<T>(TimeSeries.Length);
        Vector<T> adjustedIrregular = new Vector<T>(TimeSeries.Length);

        for (int i = 0; i < TimeSeries.Length; i++)
        {
            adjustedTrend[i] = NumOps.Multiply(trend[i], componentAdjustment[i]);
            adjustedSeasonal[i] = NumOps.Multiply(seasonal[i], componentAdjustment[i]);
            adjustedIrregular[i] = NumOps.Multiply(irregular[i], componentAdjustment[i]);
        }

        AddComponent(DecompositionComponentType.Trend, adjustedTrend);
        AddComponent(DecompositionComponentType.Seasonal, adjustedSeasonal);
        AddComponent(DecompositionComponentType.Irregular, adjustedIrregular);
    }

    /// <summary>
    /// Calculates a centered moving average of the input data.
    /// </summary>
    /// <param name="data">The time series data.</param>
    /// <param name="window">The window size for the moving average (should be odd for centered).</param>
    /// <returns>A vector containing the centered moving average.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A centered moving average helps smooth out your data by replacing each point
    /// with the average of itself and nearby points. This helps reveal the underlying trend by
    /// reducing the impact of random fluctuations and seasonal patterns.
    /// 
    /// For example, with a window of 5, each point becomes the average of itself, the 2 points before it,
    /// and the 2 points after it.
    /// </para>
    /// </remarks>
    private Vector<T> CenteredMovingAverage(Vector<T> data, int window)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n);

        int halfWindow = window / 2;

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - halfWindow);
            int end = Math.Min(n - 1, i + halfWindow);
            T sum = NumOps.Zero;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                sum = NumOps.Add(sum, data[j]);
                count++;
            }

            result[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        return result;
    }

    /// <summary>
    /// Estimates seasonal factors for additive decomposition.
    /// </summary>
    /// <param name="seasonalIrregular">The combined seasonal and irregular components.</param>
    /// <returns>The estimated seasonal factors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the repeating patterns in your data by grouping values that occur
    /// at the same point in each cycle (like all Januaries, all Februaries, etc. for monthly data).
    /// 
    /// For example, if you have monthly data for 3 years, this method will:
    /// 1. Group all January values together, all February values together, etc.
    /// 2. Calculate the average for each month
    /// 3. Adjust these averages so they sum to zero (ensuring the seasonal pattern doesn't affect the overall level)
    /// 
    /// The result shows how much each season typically deviates from the trend.
    /// </para>
    /// </remarks>
    private Vector<T> EstimateSeasonalFactors(Vector<T> seasonalIrregular)
    {
        int n = seasonalIrregular.Length;
        Vector<T> seasonalFactors = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            int count = 0;

            // Group values by their position in the seasonal cycle
            for (int j = i % _seasonalPeriod; j < n; j += _seasonalPeriod)
            {
                sum = NumOps.Add(sum, seasonalIrregular[j]);
                count++;
            }

            // Calculate average for this seasonal position
            seasonalFactors[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        // Normalize seasonal factors so they sum to zero over one complete cycle
        T totalFactor = NumOps.Zero;
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            totalFactor = NumOps.Add(totalFactor, seasonalFactors[i]);
        }
        T averageFactor = NumOps.Divide(totalFactor, NumOps.FromDouble(_seasonalPeriod));

        // Subtract the average to ensure seasonal factors sum to zero
        for (int i = 0; i < n; i++)
        {
            seasonalFactors[i] = NumOps.Subtract(seasonalFactors[i], averageFactor);
        }

        return seasonalFactors;
    }
}
