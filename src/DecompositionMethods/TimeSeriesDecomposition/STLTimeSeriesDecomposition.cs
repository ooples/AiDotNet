namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class STLTimeSeriesDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly STLDecompositionOptions<T> _options;
    private readonly STLAlgorithmType _algorithmType;

    public STLTimeSeriesDecomposition(Vector<T> timeSeries, STLDecompositionOptions<T> options, STLAlgorithmType algorithmType = STLAlgorithmType.Standard)
        : base(timeSeries)
    {
        _options = options;
        _algorithmType = algorithmType;
        Decompose();
    }

    protected override void Decompose()
    {
        Matrix<T> inputMatrix;
        switch (_algorithmType)
        {
            case STLAlgorithmType.Standard:
                // Standard options are already set
                inputMatrix = new Matrix<T>(TimeSeries.Length, 1, NumOps);
                break;
            case STLAlgorithmType.Robust:
                _options.RobustIterations = 2;
                inputMatrix = new Matrix<T>(TimeSeries.Length, 1, NumOps);
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

    private Matrix<T> CreateFastSTLInputMatrix()
    {
        int n = TimeSeries.Length;
        var matrix = new Matrix<T>(n, 3, NumOps); // 3 columns: time index, original value, and preprocessed value

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

    private T CalculateSeriesMean()
    {
        T sum = TimeSeries.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        return NumOps.Divide(sum, NumOps.FromDouble(TimeSeries.Length));
    }

    private T CalculateSeriesStdDev()
    {
        T mean = CalculateSeriesMean();
        T sumSquaredDiff = TimeSeries.Aggregate(NumOps.Zero, (acc, val) => 
            NumOps.Add(acc, NumOps.Multiply(NumOps.Subtract(val, mean), NumOps.Subtract(val, mean))));
    
        T variance = NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(TimeSeries.Length - 1));
        return NumOps.Sqrt(variance);
    }

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

    private T AdjustForDayOfWeek(T value, DateTime date)
    {
        int dayOfWeek = (int)date.DayOfWeek;
        T adjustment = _options.DayOfWeekFactors[dayOfWeek];

        return NumOps.Multiply(value, adjustment);
    }

    private T AdjustForMonthOfYear(T value, DateTime date)
    {
        int month = date.Month - 1; // 0-based index
        T adjustment = _options.MonthOfYearFactors[month];

        return NumOps.Multiply(value, adjustment);
    }

    private T AdjustForHolidays(T value, DateTime date)
    {
        if (_options.Holidays.TryGetValue(date.Date, out T? holidayFactor))
        {
            return NumOps.Multiply(value, holidayFactor);
        }

        return value;
    }
}