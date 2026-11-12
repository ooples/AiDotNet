using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
[ResearchPaper("STL: A Seasonal-Trend Decomposition Procedure Based on Loess", "https://doi.org/10.6028/jres.095.015", Year = 1990, Authors = "Robert B. Cleveland, William S. Cleveland, Jean E. McRae, Irma Terpenning")]
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
        if (n == 0)
        {
            return new Matrix<T>(0, 3);
        }

        DateTime[] dates = GetOrCreateDates();

        // Column 0: Normalized time index i / (n - 1)
        var positions = Engine.TensorArange<T>(0, n, 1);
        var timeIndex = n > 1
            ? Engine.TensorDivide(positions, Engine.TensorFullLike(positions, NumOps.FromDouble(n - 1)))
            : Engine.TensorFullLike(positions, NumOps.FromDouble(double.NaN));

        // Column 1: Original value
        var values = Tensor<T>.FromVector(TimeSeries);

        // Column 2: Preprocessed value (e.g., handling missing values, outliers)
        var preprocessed = PreprocessSeries(values, dates);

        return Engine.TensorStack(new[] { timeIndex, values, preprocessed }, 1).ToMatrix();
    }

    /// <summary>
    /// Preprocesses every value: non-finite values are imputed from their nearest finite neighbours,
    /// outliers are replaced by the median of the finite values in a 5-wide window, and every other
    /// value receives the configured seasonal adjustments.
    /// </summary>
    private Tensor<T> PreprocessSeries(Tensor<T> values, DateTime[] dates)
    {
        var finite = Engine.TensorIsFinite(values);
        bool anyFinite = Engine.TensorCountNonzero(
            Engine.TensorWhere(finite, Engine.TensorOnesLike(values), Engine.TensorZerosLike(values))) > 0;

        var outliers = anyFinite ? OutlierMask(values) : Engine.TensorZerosLike(values);
        var adjusted = Engine.TensorWhere(
            Engine.TensorGreaterThan(outliers, NumOps.FromDouble(0.5)), SmoothedWindowMedians(values), ApplySeasonalAdjustments(values, dates));
        return Engine.TensorWhere(finite, adjusted, ImputedValues(values, finite));
    }

    /// <summary>1 where a value is an outlier under the configured detection method, 0 elsewhere.</summary>
    private Tensor<T> OutlierMask(Tensor<T> values)
    {
        return _options.OutlierDetectionMethod switch
        {
            OutlierDetectionMethod.ZScore => ZScoreOutlierMask(values),
            OutlierDetectionMethod.IQR => IqrOutlierMask(values),
            OutlierDetectionMethod.Combined => Engine.TensorGreaterThan(
                Engine.TensorAdd(ZScoreOutlierMask(values), IqrOutlierMask(values)), NumOps.FromDouble(0.5)),
            _ => throw new ArgumentException("Invalid outlier detection method."),
        };
    }

    private Tensor<T> ZScoreOutlierMask(Tensor<T> values)
    {
        T mean = CalculateSeriesMean();
        T stdDev = CalculateSeriesStdDev();

        // Avoid division by zero
        if (NumOps.Equals(stdDev, NumOps.Zero))
        {
            return Engine.TensorZerosLike(values);
        }

        var zScore = Engine.TensorDivide(Engine.TensorSubtractScalar(values, mean), Engine.TensorFullLike(values, stdDev));
        return Engine.TensorGreaterThan(Engine.TensorAbs(zScore), NumOps.FromDouble(_options.ZScoreThreshold));
    }

    private Tensor<T> IqrOutlierMask(Tensor<T> values)
    {
        // Quartiles of the finite values only; NaN would otherwise sort into the order statistics.
        var sortedSeries = TimeSeries
            .Where(x => !NumOps.IsNaN(x) && !NumOps.IsInfinity(x))
            .OrderBy(x => x)
            .ToList();
        int n = sortedSeries.Count;
        if (n == 0)
        {
            return Engine.TensorZerosLike(values);
        }

        T q1 = sortedSeries[n / 4];
        T q3 = sortedSeries[3 * n / 4];
        T iqr = NumOps.Subtract(q3, q1);

        T lowerBound = NumOps.Subtract(q1, NumOps.Multiply(iqr, NumOps.FromDouble(_options.IQRMultiplier)));
        T upperBound = NumOps.Add(q3, NumOps.Multiply(iqr, NumOps.FromDouble(_options.IQRMultiplier)));

        return Engine.TensorAdd(Engine.TensorLessThan(values, lowerBound), Engine.TensorGreaterThan(values, upperBound));
    }

    /// <summary>
    /// Average of the nearest finite values on the left and right of every position; a single
    /// finite side is used alone, and the series mean when neither exists.
    /// </summary>
    private Tensor<T> ImputedValues(Tensor<T> values, Tensor<AiDotNet.Tensors.Bit> finite)
    {
        int n = values.Length;
        var index = Engine.TensorArange<int>(0, n, 1);

        // Nearest finite index at or before / at or after each position (-1 / n when none).
        var left = Engine.TensorCumMax(Engine.TensorWhere(finite, index, Engine.TensorFullLike(index, -1)), 0);
        var right = Engine.TensorFlip(
            Engine.TensorCumMin(Engine.TensorFlip(Engine.TensorWhere(finite, index, Engine.TensorFullLike(index, n)), new[] { 0 }), 0),
            new[] { 0 });

        var leftValues = Engine.TensorIndexSelect(values, Engine.TensorClamp(left, 0, n - 1), 0);
        var rightValues = Engine.TensorIndexSelect(values, Engine.TensorClamp(right, 0, n - 1), 0);
        var finiteIndicator = Engine.TensorWhere(finite, Engine.TensorOnesLike(values), Engine.TensorZerosLike(values));
        var hasLeft = Engine.TensorGreaterThan(Engine.TensorCumMax(finiteIndicator, 0), NumOps.FromDouble(0.5));
        var hasRight = Engine.TensorGreaterThan(
            Engine.TensorFlip(Engine.TensorCumMax(Engine.TensorFlip(finiteIndicator, new[] { 0 }), 0), new[] { 0 }),
            NumOps.FromDouble(0.5));

        var both = Engine.TensorMultiplyScalar(Engine.TensorAdd(leftValues, rightValues), NumOps.FromDouble(0.5));
        var fallback = Engine.TensorFullLike(values, CalculateSeriesMean());
        var oneSided = Engine.TensorWhere(hasRight, rightValues, fallback);
        oneSided = Engine.TensorWhere(hasLeft, leftValues, oneSided);
        return Engine.TensorWhere(Engine.TensorMultiply(hasLeft, hasRight), both, oneSided);
    }

    /// <summary>
    /// Median of the finite values among positions [i - 2, i + 2] (clipped to the series) for every i.
    /// </summary>
    private Tensor<T> SmoothedWindowMedians(Tensor<T> values)
    {
        int windowSize = 5;
        int half = windowSize / 2;
        int n = values.Length;

        // Windows padded with non-finite entries, which sort last as +infinity.
        var padded = Engine.TensorConstantPad(values, new[] { half, half }, NumOps.FromDouble(double.NaN));
        var windows = Engine.TensorUnfold(padded, 0, windowSize, 1);
        var windowFinite = Engine.TensorIsFinite(windows);
        var ones = Engine.TensorOnesLike(windows);
        var finiteMask = Engine.TensorWhere(windowFinite, ones, Engine.TensorZerosLike(windows));
        var (sorted, _) = Engine.TensorSort(
            Engine.TensorWhere(windowFinite, windows, Engine.TensorFullLike(windows, NumOps.FromDouble(double.PositiveInfinity))), 1);
        sorted = Engine.TensorWhere(Engine.TensorIsFinite(sorted), sorted, Engine.TensorZerosLike(sorted));

        // With c finite values, the middle order statistics are at hi = floor(c / 2) and
        // lo = floor((c - 1) / 2): with d = c - 2k, hi is where d in {0, 1} and lo where d in {1, 2}.
        var count = Engine.ReduceSum(finiteMask, new[] { 1 }, true);
        var k = Engine.TensorBroadcastTo(Engine.TensorArange<T>(0, windowSize, 1).Reshape(new[] { 1, windowSize }), new[] { n, windowSize });
        var d = Engine.TensorSubtract(Engine.TensorBroadcastTo(count, new[] { n, windowSize }), Engine.TensorMultiplyScalar(k, NumOps.FromDouble(2)));
        var hiMask = Engine.TensorMultiply(Engine.TensorGreaterThan(d, NumOps.FromDouble(-0.5)), Engine.TensorLessThan(d, NumOps.FromDouble(1.5)));
        var loMask = Engine.TensorMultiply(Engine.TensorGreaterThan(d, NumOps.FromDouble(0.5)), Engine.TensorLessThan(d, NumOps.FromDouble(2.5)));
        var hi = Engine.ReduceSum(Engine.TensorMultiply(sorted, hiMask), new[] { 1 }, false);
        var lo = Engine.ReduceSum(Engine.TensorMultiply(sorted, loMask), new[] { 1 }, false);

        var flatCount = count.Reshape(new[] { n });
        var isOdd = Engine.TensorGreaterThan(
            Engine.TensorRemainder(flatCount, Engine.TensorFullLike(flatCount, NumOps.FromDouble(2))), NumOps.FromDouble(0.5));
        var even = Engine.TensorMultiplyScalar(Engine.TensorAdd(lo, hi), NumOps.FromDouble(0.5));
        return Engine.TensorWhere(isOdd, hi, even);
    }

    /// <summary>Applies the enabled day-of-week, month-of-year and holiday factors to every value.</summary>
    private Tensor<T> ApplySeasonalAdjustments(Tensor<T> values, DateTime[] dates)
    {
        var adjustedValue = values;

        if (_options.AdjustForDayOfWeek)
        {
            var days = Tensor<int>.FromVector(new Vector<int>(dates.Select(date => (int)date.DayOfWeek)));
            adjustedValue = Engine.TensorMultiply(adjustedValue, Engine.TensorIndexSelect(Tensor<T>.FromVector(_options.DayOfWeekFactors), days, 0));
        }

        if (_options.AdjustForMonthOfYear)
        {
            var months = Tensor<int>.FromVector(new Vector<int>(dates.Select(date => date.Month - 1)));
            adjustedValue = Engine.TensorMultiply(adjustedValue, Engine.TensorIndexSelect(Tensor<T>.FromVector(_options.MonthOfYearFactors), months, 0));
        }

        if (_options.AdjustForHolidays)
        {
            T noFactor = NumOps.One;
            bool[] isHoliday = dates.Select(date => _options.Holidays.ContainsKey(date.Date)).ToArray();
            var factors = Tensor<T>.FromVector(new Vector<T>(
                dates.Select(date => _options.Holidays.TryGetValue(date.Date, out T? factor) && factor is not null ? factor : noFactor)));
            adjustedValue = Engine.TensorWhere(isHoliday, Engine.TensorMultiply(adjustedValue, factors), adjustedValue);
        }

        return adjustedValue;
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
    /// Calculates the mean (average) of the time series data.
    /// </summary>
    /// <returns>The mean value of the time series.</returns>
    private T CalculateSeriesMean()
    {
        // Non-finite entries (NaN, +-Infinity) are excluded; PreprocessSeries accepts them and imputes
        // them later, and one of them would otherwise turn the mean (and every z-score) into NaN.
        var values = Tensor<T>.FromVector(TimeSeries);
        var finite = Engine.TensorIsFinite(values);
        var zeros = Engine.TensorZerosLike(values);
        T sum = Engine.TensorSum(Engine.TensorWhere(finite, values, zeros));
        T count = Engine.TensorSum(Engine.TensorWhere(finite, Engine.TensorOnesLike(values), zeros));
        return NumOps.Divide(sum, count);
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
        // Sample standard deviation over the finite entries only (see CalculateSeriesMean).
        T mean = CalculateSeriesMean();
        var values = Tensor<T>.FromVector(TimeSeries);
        var finite = Engine.TensorIsFinite(values);
        var zeros = Engine.TensorZerosLike(values);
        var deviations = Engine.TensorSubtractScalar(values, mean);
        T sumSquaredDiff = Engine.TensorSum(Engine.TensorWhere(finite, Engine.TensorMultiply(deviations, deviations), zeros));
        T count = Engine.TensorSum(Engine.TensorWhere(finite, Engine.TensorOnesLike(values), zeros));

        T variance = NumOps.Divide(sumSquaredDiff, NumOps.Subtract(count, NumOps.One));
        return NumOps.Sqrt(variance);
    }
}
