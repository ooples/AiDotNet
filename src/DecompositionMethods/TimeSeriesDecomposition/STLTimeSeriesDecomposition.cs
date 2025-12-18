namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements the Seasonal-Trend decomposition using LOESS (STL) algorithm for time series analysis.
/// </summary>
/// <remarks>
/// <para>
/// STL decomposition breaks down a time series into three components:
/// trend (long-term progression), seasonal (recurring patterns), and residual (remaining noise).
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like breaking down your monthly expenses into:
/// - A trend (are you spending more or less over time?)
/// - Seasonal patterns (do you spend more during holidays?)
/// - Unexpected expenses (random costs that don't fit the patterns)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public class STLTimeSeriesDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly STLDecompositionOptions<T> _options;
    private readonly STLAlgorithmType _algorithmType;

    /// <summary>
    /// Initializes a new instance of the STL time series decomposition algorithm.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="options">Configuration options for the STL algorithm.</param>
    /// <param name="algorithmType">The type of STL algorithm to use (Standard, Robust, or Fast).</param>
    public STLTimeSeriesDecomposition(Vector<T> timeSeries, STLDecompositionOptions<T> options, STLAlgorithmType algorithmType = STLAlgorithmType.Standard)
        : base(timeSeries)
    {
        _options = options;
        _algorithmType = algorithmType;
        Decompose();
    }

    /// <summary>
    /// Performs the STL decomposition on the time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes your time series data and splits it into three parts:
    /// trend, seasonal patterns, and residuals (unexplained variations). The algorithm type
    /// determines how this splitting is done - Standard is balanced, Robust handles outliers better,
    /// and Fast prioritizes speed over precision.
    /// </para>
    /// </remarks>
    protected override void Decompose()
    {
        Matrix<T> inputMatrix;
        switch (_algorithmType)
        {
            case STLAlgorithmType.Standard:
                // Standard options are already set
                inputMatrix = new Matrix<T>(TimeSeries.Length, 1);
                break;
            case STLAlgorithmType.Robust:
                _options.RobustIterations = 2;
                inputMatrix = new Matrix<T>(TimeSeries.Length, 1);
                break;
            case STLAlgorithmType.Fast:
                _options.TrendWindowSize = Math.Max(3, TimeSeries.Length / 10);
                _options.SeasonalLoessWindow = Math.Max(3, _options.SeasonalPeriod);
                inputMatrix = CreateFastSTLInputMatrix();
                break;
            default:
                throw new ArgumentException("Invalid STL algorithm type.");
        }

        var stlDecomposition = new STLDecomposition<T>(_options);
        stlDecomposition.Train(inputMatrix, TimeSeries);

        Vector<T> trend = stlDecomposition.GetTrend();
        Vector<T> seasonal = stlDecomposition.GetSeasonal();
        Vector<T> residual = stlDecomposition.GetResidual();

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Creates an input matrix optimized for the Fast STL algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method prepares your data in a special format that helps the Fast STL
    /// algorithm work more efficiently. It creates a table with three columns: a time index (like
    /// day 1, day 2, etc.), your original values, and preprocessed values that handle issues like
    /// missing data points.
    /// </para>
    /// </remarks>
    /// <returns>A matrix with time indices, original values, and preprocessed values.</returns>
    private Matrix<T> CreateFastSTLInputMatrix()
    {
        int n = TimeSeries.Length;
        var matrix = new Matrix<T>(n, 3); // 3 columns: time index, original value, and preprocessed value

        DateTime[] dates = GetOrCreateDates();

        for (int i = 0; i < n; i++)
        {
            // Column 0: Normalized time index
            matrix[i, 0] = NumOps.FromDouble((double)i / (n - 1));

            // Column 1: Original value
            matrix[i, 1] = TimeSeries[i];

            // Column 2: Preprocessed value (e.g., handling missing values, outliers)
            matrix[i, 2] = PreprocessValue(TimeSeries[i], i, dates[i]);
        }

        return matrix;
    }

    /// <summary>
    /// Gets or creates date values for each point in the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method makes sure each data point has a date associated with it.
    /// If dates are already provided, it uses those. Otherwise, it creates dates starting from
    /// a specified date (or today) with regular intervals (like daily or hourly).
    /// </para>
    /// </remarks>
    /// <returns>An array of dates corresponding to each point in the time series.</returns>
    private DateTime[] GetOrCreateDates()
    {
        if (_options.Dates != null && _options.Dates.Length == TimeSeries.Length)
        {
            return _options.Dates;
        }

        if (TimeSeries is IEnumerable<KeyValuePair<DateTime, T>> datedSeries)
        {
            return datedSeries.Select(kvp => kvp.Key).ToArray();
        }

        // If no dates are provided, create them based on options or use default
        DateTime startDate = _options.StartDate ?? DateTime.Today.AddDays(-TimeSeries.Length + 1);
        TimeSpan interval = _options.Interval ?? TimeSpan.FromDays(1);

        return [.. Enumerable.Range(0, TimeSeries.Length).Select(i => startDate.Add(TimeSpan.FromTicks(interval.Ticks * i)))];
    }

    /// <summary>
    /// Preprocesses a value to handle missing data, outliers, and apply seasonal adjustments.
    /// </summary>
    /// <param name="value">The original value to preprocess.</param>
    /// <param name="index">The index of the value in the time series.</param>
    /// <param name="date">The date associated with the value.</param>
    /// <returns>The preprocessed value.</returns>
    private T PreprocessValue(T value, int index, DateTime date)
    {
        // Handle missing values
        if (NumOps.IsNaN(value) || NumOps.IsInfinity(value))
        {
            return ImputeMissingValue(index);
        }

        // Handle outliers
        if (IsOutlier(value, index))
        {
            return SmoothOutlier(value, index);
        }

        // Apply any seasonal adjustments if needed
        return ApplySeasonalAdjustment(value, date);
    }

    /// <summary>
    /// Fills in missing values in the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When data is missing, this method estimates what the value should be.
    /// It looks at nearby values and uses their average. If there are no valid nearby values,
    /// it uses the average of the entire series.
    /// </para>
    /// </remarks>
    /// <param name="index">The index of the missing value in the time series.</param>
    /// <returns>An estimated value to replace the missing one.</returns>
    private T ImputeMissingValue(int index)
    {
        // Simple imputation: use the average of neighboring non-missing values
        int left = index - 1;
        int right = index + 1;

        while (left >= 0 && (NumOps.IsNaN(TimeSeries[left]) || NumOps.IsInfinity(TimeSeries[left])))
        {
            left--;
        }

        while (right < TimeSeries.Length && (NumOps.IsNaN(TimeSeries[right]) || NumOps.IsInfinity(TimeSeries[right])))
        {
            right++;
        }

        if (left >= 0 && right < TimeSeries.Length)
        {
            return NumOps.Divide(NumOps.Add(TimeSeries[left], TimeSeries[right]), NumOps.FromDouble(2.0));
        }
        else if (left >= 0)
        {
            return TimeSeries[left];
        }
        else if (right < TimeSeries.Length)
        {
            return TimeSeries[right];
        }

        // If no valid neighbors found, return a default value (e.g., the mean of the entire series)
        return CalculateSeriesMean();
    }

    /// <summary>
    /// Determines if a value is an outlier using the configured detection method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An outlier is a data point that's unusually different from the rest of your data.
    /// This method checks if a value is an outlier using one of three methods:
    /// - Z-Score: Checks how many standard deviations the value is from the average
    /// - IQR: Checks if the value is far from the middle 50% of your data
    /// - Combined: Uses both methods above
    /// </para>
    /// </remarks>
    /// <param name="value">The value to check.</param>
    /// <param name="index">The index of the value in the time series.</param>
    /// <returns>True if the value is an outlier; otherwise, false.</returns>
    private bool IsOutlier(T value, int index)
    {
        return _options.OutlierDetectionMethod switch
        {
            OutlierDetectionMethod.ZScore => IsOutlierZScore(value),
            OutlierDetectionMethod.IQR => IsOutlierIQR(value),
            OutlierDetectionMethod.Combined => IsOutlierZScore(value) || IsOutlierIQR(value),
            _ => throw new ArgumentException("Invalid outlier detection method."),
        };
    }

    /// <summary>
    /// Determines if a value is an outlier using the Z-Score method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Z-Score measures how far a value is from the average in terms of standard deviations.
    /// Think of it like measuring how unusual a person's height is compared to the average height.
    /// A Z-Score of 2 means the value is 2 standard deviations away from the average.
    /// Values with high Z-Scores (typically above 2 or 3) are considered outliers.
    /// </para>
    /// </remarks>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is an outlier; otherwise, false.</returns>
    private bool IsOutlierZScore(T value)
    {
        T mean = CalculateSeriesMean();
        T stdDev = CalculateSeriesStdDev();

        // Avoid division by zero
        if (NumOps.Equals(stdDev, NumOps.Zero))
        {
            return false;
        }

        T zScore = NumOps.Divide(NumOps.Subtract(value, mean), stdDev);
        return NumOps.GreaterThan(NumOps.Abs(zScore), NumOps.FromDouble(_options.ZScoreThreshold));
    }

    /// <summary>
    /// Determines if a value is an outlier using the Interquartile Range (IQR) method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The IQR method identifies outliers by looking at the middle 50% of your data.
    /// Imagine lining up all your data points from smallest to largest:
    /// - Q1 is the value 25% of the way through your data
    /// - Q3 is the value 75% of the way through your data
    /// - IQR is the range between Q1 and Q3
    /// 
    /// Values that fall too far outside this range (typically 1.5 × IQR below Q1 or above Q3)
    /// are considered outliers.
    /// </para>
    /// </remarks>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is an outlier; otherwise, false.</returns>
    private bool IsOutlierIQR(T value)
    {
        var sortedSeries = TimeSeries.OrderBy(x => x).ToList();
        int n = sortedSeries.Count;

        T q1 = sortedSeries[n / 4];
        T q3 = sortedSeries[3 * n / 4];
        T iqr = NumOps.Subtract(q3, q1);

        T lowerBound = NumOps.Subtract(q1, NumOps.Multiply(iqr, NumOps.FromDouble(_options.IQRMultiplier)));
        T upperBound = NumOps.Add(q3, NumOps.Multiply(iqr, NumOps.FromDouble(_options.IQRMultiplier)));

        return NumOps.LessThan(value, lowerBound) || NumOps.GreaterThan(value, upperBound);
    }

    /// <summary>
    /// Calculates the mean (average) of the time series data.
    /// </summary>
    /// <returns>The mean value of the time series.</returns>
    private T CalculateSeriesMean()
    {
        T sum = TimeSeries.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        return NumOps.Divide(sum, NumOps.FromDouble(TimeSeries.Length));
    }

    /// <summary>
    /// Calculates the standard deviation of the time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Standard deviation measures how spread out the data is from the average.
    /// A low standard deviation means most values are close to the average.
    /// A high standard deviation means values are more spread out.
    /// 
    /// For example, if you measure the heights of people:
    /// - Adults might have a higher standard deviation (more variety in heights)
    /// - Children of the same age might have a lower standard deviation (more similar heights)
    /// </para>
    /// </remarks>
    /// <returns>The standard deviation of the time series.</returns>
    private T CalculateSeriesStdDev()
    {
        T mean = CalculateSeriesMean();
        T sumSquaredDiff = TimeSeries.Aggregate(NumOps.Zero, (acc, val) =>
            NumOps.Add(acc, NumOps.Multiply(NumOps.Subtract(val, mean), NumOps.Subtract(val, mean))));

        T variance = NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(TimeSeries.Length - 1));
        return NumOps.Sqrt(variance);
    }

    /// <summary>
    /// Smooths an outlier value by replacing it with the median of nearby values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When we find an unusual value (outlier), we replace it with a more reasonable one.
    /// This method looks at a small window of values around the outlier (2 values before and 2 after),
    /// and uses the median (middle value) from this window as the replacement.
    /// 
    /// For example, if we have values [10, 12, 100, 11, 9] and 100 is an outlier,
    /// we'd replace it with the median of these values, which is much closer to the others.
    /// </para>
    /// </remarks>
    /// <param name="value">The outlier value to smooth.</param>
    /// <param name="index">The position of the value in the time series.</param>
    /// <returns>A smoothed value to replace the outlier.</returns>
    private T SmoothOutlier(T value, int index)
    {
        int windowSize = 5;
        int start = Math.Max(0, index - windowSize / 2);
        int end = Math.Min(TimeSeries.Length - 1, index + windowSize / 2);

        var window = new List<T>();
        for (int i = start; i <= end; i++)
        {
            if (!NumOps.IsNaN(TimeSeries[i]) && !NumOps.IsInfinity(TimeSeries[i]))
            {
                window.Add(TimeSeries[i]);
            }
        }

        return StatisticsHelper<T>.CalculateMedian(window);
    }

    /// <summary>
    /// Applies seasonal adjustments to a value based on its date.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Some patterns in data are due to the calendar - like higher retail sales 
    /// during weekends or holidays. This method adjusts values to account for these patterns.
    /// 
    /// For example, if Mondays typically have 20% lower sales than average days, this method
    /// would apply that adjustment factor to Monday values to make them comparable to other days.
    /// </para>
    /// </remarks>
    /// <param name="value">The value to adjust.</param>
    /// <param name="date">The date associated with the value.</param>
    /// <returns>The seasonally adjusted value.</returns>
    private T ApplySeasonalAdjustment(T value, DateTime date)
    {
        T adjustedValue = value;

        if (_options.AdjustForDayOfWeek)
        {
            adjustedValue = AdjustForDayOfWeek(adjustedValue, date);
        }

        if (_options.AdjustForMonthOfYear)
        {
            adjustedValue = AdjustForMonthOfYear(adjustedValue, date);
        }

        if (_options.AdjustForHolidays)
        {
            adjustedValue = AdjustForHolidays(adjustedValue, date);
        }

        return adjustedValue;
    }

    /// <summary>
    /// Adjusts a value based on the day of the week it occurred on.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This applies a specific adjustment factor for each day of the week.
    /// For example, if weekends typically have higher values than weekdays, this method
    /// would apply those day-specific factors to make all days comparable.
    /// </para>
    /// </remarks>
    /// <param name="value">The value to adjust.</param>
    /// <param name="date">The date associated with the value.</param>
    /// <returns>The adjusted value.</returns>
    private T AdjustForDayOfWeek(T value, DateTime date)
    {
        int dayOfWeek = (int)date.DayOfWeek;
        T adjustment = _options.DayOfWeekFactors[dayOfWeek];

        return NumOps.Multiply(value, adjustment);
    }

    /// <summary>
    /// Adjusts a value based on the month of the year it occurred in.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This applies a specific adjustment factor for each month.
    /// For example, December might have higher retail sales due to holidays,
    /// so this method would apply a factor to make December sales comparable to other months.
    /// </para>
    /// </remarks>
    /// <param name="value">The value to adjust.</param>
    /// <param name="date">The date associated with the value.</param>
    /// <returns>The adjusted value.</returns>
    private T AdjustForMonthOfYear(T value, DateTime date)
    {
        int month = date.Month - 1; // 0-based index
        T adjustment = _options.MonthOfYearFactors[month];

        return NumOps.Multiply(value, adjustment);
    }

    /// <summary>
    /// Adjusts a value if it falls on a holiday.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Holidays often cause unusual patterns in data. This method checks if
    /// a value's date is a holiday, and if so, applies a special adjustment factor for that holiday.
    /// 
    /// For example, retail sales might be much higher on Black Friday, so this method would
    /// apply a factor to make Black Friday sales comparable to normal days.
    /// </para>
    /// </remarks>
    /// <param name="value">The value to adjust.</param>
    /// <param name="date">The date associated with the value.</param>
    /// <returns>The adjusted value if the date is a holiday; otherwise, the original value.</returns>
    private T AdjustForHolidays(T value, DateTime date)
    {
        if (_options.Holidays.TryGetValue(date.Date, out T? holidayFactor))
        {
            return NumOps.Multiply(value, holidayFactor);
        }

        return value;
    }
}
