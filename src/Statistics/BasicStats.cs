namespace AiDotNet.Statistics;

/// <summary>
/// Provides a collection of basic statistical measures for a set of numeric values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// BasicStats calculates and stores a comprehensive set of descriptive statistics for a collection of values,
/// including measures of central tendency, dispersion, and distribution shape. These statistics provide
/// insights into the characteristics of the data distribution.
/// </para>
/// <para><b>For Beginners:</b> Think of BasicStats as a calculator that analyzes a set of numbers and tells you
/// their important patterns and characteristics.
/// 
/// It answers questions like:
/// - What's the typical value? (Mean, Median)
/// - How spread out are the values? (Variance, StandardDeviation)
/// - What's the range of values? (Min, Max, InterquartileRange)
/// - Is the distribution skewed or symmetric? (Skewness)
/// - Are there unusual extreme values? (Kurtosis)
/// 
/// For example, if you have test scores from a class:
/// - Mean tells you the average score
/// - StandardDeviation tells you how much scores vary from that average
/// - Skewness might reveal if more students scored above or below average
/// 
/// These statistics help you understand your data at a glance without having to examine every value.
/// </para>
/// </remarks>
[Serializable]
public class BasicStats<T> : StatisticsBase<T>
{
    #region Private Fields

    /// <summary>
    /// Number of values in the dataset.
    /// </summary>
    private int _n;
    #endregion

    #region Property Accessors

    /// <summary>
    /// Gets the arithmetic mean (average) of the values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mean is calculated by summing all values and dividing by the number of values. It represents
    /// the central tendency of the data, but can be sensitive to outliers.
    /// </para>
    /// <para><b>For Beginners:</b> The mean is the average value - add up all the numbers and divide by how many there are.
    /// 
    /// For example, for the numbers [2, 4, 6, 8, 10]:
    /// - Sum: 2 + 4 + 6 + 8 + 10 = 30
    /// - Count: 5
    /// - Mean: 30 ÷ 5 = 6
    /// 
    /// The mean gives you the "center" of your data, but can be pulled in the direction of very large or small values.
    /// </para>
    /// </remarks>
    public T Mean => GetMetric(MetricType.Mean);

    /// <summary>
    /// Gets the variance of the values, a measure of dispersion.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Variance measures how far each value in the dataset is from the mean. It is calculated as the average
    /// of the squared differences from the mean. Larger variance indicates greater dispersion in the data.
    /// </para>
    /// <para><b>For Beginners:</b> Variance measures how spread out the numbers are from the average.
    /// 
    /// To calculate variance:
    /// 1. Find how far each number is from the mean
    /// 2. Square each of those differences (to make everything positive)
    /// 3. Find the average of those squared differences
    /// 
    /// Higher variance means values are more spread out; lower variance means they're more clustered around the mean.
    /// </para>
    /// </remarks>
    public T Variance => GetMetric(MetricType.Variance);

    /// <summary>
    /// Gets the standard deviation of the values, a measure of dispersion.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Standard deviation is the square root of the variance. It measures the amount of variation or
    /// dispersion in the dataset. It is in the same units as the original data, making it more
    /// interpretable than variance.
    /// </para>
    /// <para><b>For Beginners:</b> Standard deviation is the most common way to measure how spread out your data is.
    /// 
    /// It's the square root of the variance, which puts it back in the same units as your original numbers.
    /// 
    /// For example:
    /// - Low standard deviation: Most values are close to the average
    /// - High standard deviation: Values are widely spread from the average
    /// 
    /// If you're looking at test scores with a mean of 75 and standard deviation of 5, you know most scores
    /// fall roughly between 70 and 80.
    /// </para>
    /// </remarks>
    public T StandardDeviation => GetMetric(MetricType.StandardDeviation);

    /// <summary>
    /// Gets the skewness of the distribution, a measure of asymmetry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Skewness characterizes the degree of asymmetry of a distribution around its mean. Positive skewness
    /// indicates a distribution with a longer tail to the right, while negative skewness indicates a longer
    /// tail to the left.
    /// </para>
    /// <para><b>For Beginners:</b> Skewness tells you if your data is lopsided to one side or balanced around the average.
    /// 
    /// Think of it as measuring which way your data "leans":
    /// - Positive skewness (> 0): There's a longer tail on the right side - most values are on the left with a few high outliers
    /// - Zero skewness (= 0): The data is symmetric around the mean - balanced on both sides
    /// - Negative skewness (< 0): There's a longer tail on the left side - most values are on the right with a few low outliers
    /// 
    /// For example, income distribution often has positive skewness because most people have moderate incomes,
    /// but a few extremely wealthy individuals pull the right tail out.
    /// </para>
    /// </remarks>
    public T Skewness => GetMetric(MetricType.Skewness);

    /// <summary>
    /// Gets the kurtosis of the distribution, a measure of "tailedness".
    /// </summary>
    /// <remarks>
    /// <para>
    /// Kurtosis measures the "tailedness" of the probability distribution. Higher kurtosis means more of the
    /// variance is due to infrequent extreme deviations, as opposed to frequent modestly sized deviations.
    /// </para>
    /// <para><b>For Beginners:</b> Kurtosis measures how much of your data is in the "tails" versus the "center".
    /// 
    /// Think of it as measuring the shape of your data's peaks and tails:
    /// - High kurtosis (> 3): "Heavy-tailed" - more values in the extremes, with a sharper peak
    /// - Normal kurtosis (= 3): Follows a normal distribution (bell curve)
    /// - Low kurtosis (< 3): "Light-tailed" - fewer outliers, with a flatter peak
    /// 
    /// For example, stock returns often have high kurtosis because they mostly have small changes day-to-day,
    /// but occasionally have extreme movements.
    /// </para>
    /// </remarks>
    public T Kurtosis => GetMetric(MetricType.Kurtosis);

    /// <summary>
    /// Gets the minimum value in the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The minimum is the smallest value in the dataset. It represents the lower bound of the data range.
    /// </para>
    /// <para><b>For Beginners:</b> The minimum is simply the smallest number in your data.
    /// 
    /// For the numbers [5, 12, 3, 8, 9], the minimum is 3.
    /// 
    /// It's useful to know the extreme values in your data, especially when looking for outliers
    /// or setting valid ranges.
    /// </para>
    /// </remarks>
    public T Min => GetMetric(MetricType.Min);

    /// <summary>
    /// Gets the maximum value in the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The maximum is the largest value in the dataset. It represents the upper bound of the data range.
    /// </para>
    /// <para><b>For Beginners:</b> The maximum is simply the largest number in your data.
    /// 
    /// For the numbers [5, 12, 3, 8, 9], the maximum is 12.
    /// 
    /// Together with the minimum, it tells you the full range of your data.
    /// </para>
    /// </remarks>
    public T Max => GetMetric(MetricType.Max);

    /// <summary>
    /// Gets the number of values in the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// N represents the count of values in the dataset. It is used in various statistical calculations
    /// and indicates the sample size.
    /// </para>
    /// <para><b>For Beginners:</b> N is just the count of how many numbers are in your data.
    /// 
    /// For the numbers [5, 12, 3, 8, 9], N is 5.
    /// 
    /// The sample size is important because larger samples generally give more reliable statistics.
    /// </para>
    /// </remarks>
    public int N => _n;

    /// <summary>
    /// Gets the median value of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The median is the middle value when the data is sorted in ascending order. For an even number of values,
    /// it is the average of the two middle values. The median is less sensitive to outliers than the mean.
    /// </para>
    /// <para><b>For Beginners:</b> The median is the middle value when you arrange all numbers in order.
    /// 
    /// To find the median:
    /// 1. Sort all numbers from smallest to largest
    /// 2. If there's an odd number of values, take the middle one
    /// 3. If there's an even number, take the average of the two middle values
    /// 
    /// For example:
    /// - For [3, 5, 8, 9, 12] (odd count), the median is 8
    /// - For [3, 5, 8, 9, 12, 15] (even count), the median is (8 + 9) ÷ 2 = 8.5
    /// 
    /// The median is often better than the mean for describing "typical" values when your data has outliers.
    /// </para>
    /// </remarks>
    public T Median => GetMetric(MetricType.Median);

    /// <summary>
    /// Gets the first quartile (25th percentile) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The first quartile is the value below which 25% of the observations in the dataset fall.
    /// It represents the median of the lower half of the data.
    /// </para>
    /// <para><b>For Beginners:</b> The first quartile is the value where 25% of your data falls below it.
    /// 
    /// If you divide your sorted data into four equal parts, the first quartile is the value at the boundary
    /// of the first and second parts.
    /// 
    /// For example, in a class of 20 students, the first quartile test score is the score that 5 students
    /// scored below and 15 students scored above.
    /// </para>
    /// </remarks>
    public T FirstQuartile => GetMetric(MetricType.FirstQuartile);

    /// <summary>
    /// Gets the third quartile (75th percentile) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The third quartile is the value below which 75% of the observations in the dataset fall.
    /// It represents the median of the upper half of the data.
    /// </para>
    /// <para><b>For Beginners:</b> The third quartile is the value where 75% of your data falls below it.
    /// 
    /// If you divide your sorted data into four equal parts, the third quartile is the value at the boundary
    /// of the third and fourth parts.
    /// 
    /// For example, in a class of 20 students, the third quartile test score is the score that 15 students
    /// scored below and 5 students scored above.
    /// </para>
    /// </remarks>
    public T ThirdQuartile => GetMetric(MetricType.ThirdQuartile);

    /// <summary>
    /// Gets the interquartile range (IQR) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The interquartile range is the difference between the third and first quartiles. It represents
    /// the middle 50% of the data and is a robust measure of dispersion that is less sensitive to outliers
    /// than the range or standard deviation.
    /// </para>
    /// <para><b>For Beginners:</b> The interquartile range (IQR) measures the spread of the middle 50% of your data.
    /// 
    /// It's calculated as: IQR = Third Quartile - First Quartile
    /// 
    /// The IQR is useful because:
    /// - It ignores the extreme values (potential outliers)
    /// - It gives you the range where the "typical" values fall
    /// - It's used to identify outliers (values more than 1.5 × IQR from the quartiles)
    /// 
    /// For example, if test scores have an IQR of 15 points, it means the middle 50% of students' scores
    /// span a 15-point range.
    /// </para>
    /// </remarks>
    public T InterquartileRange => GetMetric(MetricType.InterquartileRange);

    /// <summary>
    /// Gets the median absolute deviation (MAD) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The median absolute deviation is the median of the absolute deviations from the data's median.
    /// It is a robust measure of variability that is less sensitive to outliers than the standard deviation.
    /// </para>
    /// <para><b>For Beginners:</b> MAD is another way to measure spread that's less affected by outliers.
    /// 
    /// To calculate MAD:
    /// 1. Find the median of your data
    /// 2. Calculate how far each value is from the median (absolute deviation)
    /// 3. Find the median of those distances
    /// 
    /// MAD is useful when your data has outliers that might skew other measures like standard deviation.
    /// Think of it as measuring the "typical" distance from the center, ignoring extreme values.
    /// </para>
    /// </remarks>
    public T MAD => GetMetric(MetricType.MAD);

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="BasicStatsImpl{T}"/> class.
    /// </summary>
    public BasicStats(ModelType modelType) : base(modelType)
    {
        _n = 0;

        if (modelType != ModelType.None)
        {
            DetermineValidMetrics();
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BasicStatsImpl{T}"/> class with the provided values.
    /// </summary>
    /// <param name="values">The values to calculate statistics for.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new BasicStats object and calculates all the statistics
    /// based on the provided input values.
    /// </para>
    /// <para><b>For Beginners:</b> This is the easiest way to create a statistics object.
    /// Just pass in your data, and all the statistics will be calculated automatically.
    /// </para>
    /// </remarks>
    public BasicStats(Vector<T> values, ModelType modelType) : base(modelType)
    {
        _n = values.Length;

        if (modelType != ModelType.None)
        {
            DetermineValidMetrics();
        }

        CalculateStats(values);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BasicStatsImpl{T}"/> class with the provided array of values.
    /// </summary>
    /// <param name="values">The array of values to calculate statistics for.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new BasicStats object from an array of values
    /// and calculates all the statistics.
    /// </para>
    /// <para><b>For Beginners:</b> This is a convenient way to create a statistics object
    /// from a simple array of numbers.
    /// </para>
    /// </remarks>
    public BasicStats(T[] values, ModelType modelType) : base(modelType)
    {
        _n = values.Length;

        if (modelType != ModelType.None)
        {
            DetermineValidMetrics();
        }

        CalculateStats(new Vector<T>(values));
    }

    /// <summary>
    /// Creates an empty BasicStats instance with all statistics set to their default values.
    /// </summary>
    /// <returns>An empty BasicStats object.</returns>
    /// <remarks>
    /// <para>
    /// This factory method creates and returns a BasicStats object with no input values. All statistics
    /// will be initialized to their default values (typically zero). This can be useful when needing
    /// to initialize a statistics object before data is available.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an empty statistics object with no data.
    /// 
    /// Use this method when:
    /// - You need a placeholder statistics object to fill in later
    /// - You're initializing a statistics object but don't have the data yet
    /// - You need a "zero" baseline to compare against
    /// 
    /// All the statistics in this empty object will be set to zero or their equivalent default values.
    /// </para>
    /// </remarks>
    public static BasicStats<T> Empty(ModelType modelType = ModelType.None)
    {
        return new BasicStats<T>(modelType);
    }

    /// <summary>
    /// Creates a new BasicStats object from a vector of values.
    /// </summary>
    /// <param name="values">The vector of values to calculate statistics for.</param>
    /// <returns>A BasicStats object with calculated statistics.</returns>
    /// <remarks>
    /// <para>
    /// This factory method creates and returns a BasicStats object calculated from the provided vector.
    /// It's a convenient way to create statistics from data without manually constructing the inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This is the simplest way to create a statistics object from a set of numbers.
    /// 
    /// For example:
    /// <code>
    /// var numbers = new Vector<double>&lt;double&gt;([1.0, 2.0, 3.0, 4.0, 5.0]);
    /// var stats = BasicStatsImpl&lt;double&gt;.FromVector(numbers);
    /// Console.WriteLine($"Mean: {stats.Mean}");
    /// </code>
    /// </para>
    /// </remarks>
    public static BasicStats<T> FromVector(Vector<T> values, ModelType modelType)
    {
        return new BasicStats<T>(values, modelType);
    }

    /// <summary>
    /// Creates a new BasicStats object from an array of values.
    /// </summary>
    /// <param name="values">The array of values to calculate statistics for.</param>
    /// <returns>A BasicStats object with calculated statistics.</returns>
    /// <remarks>
    /// <para>
    /// This factory method creates and returns a BasicStats object calculated from the provided array.
    /// It converts the array to a vector internally before calculating statistics.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes it easy to create statistics from a simple array of numbers.
    /// 
    /// For example:
    /// <code>
    /// var numbers = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    /// var stats = BasicStatsImpl&lt;double&gt;.FromArray(numbers);
    /// Console.WriteLine($"Mean: {stats.Mean}");
    /// </code>
    /// </para>
    /// </remarks>
    public static BasicStats<T> FromArray(T[] values, ModelType modelType)
    {
        return new BasicStats<T>(values, modelType);
    }

    #endregion

    #region Core Calculation Methods

    /// <summary>
    /// Determines which metrics are valid for this statistics object.
    /// </summary>
    protected override void DetermineValidMetrics()
    {
        _validMetrics.Clear();
        var cache = MetricValidationCache.Instance;
        var modelMetrics = cache.GetValidMetrics(ModelType.None, IsBasicStatisticMetric);

        foreach (var metric in modelMetrics)
        {
            _validMetrics.Add(metric);
        }
    }

    /// <summary>
    /// Determines if a metric type is a basic statistic metric.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is a basic statistic; otherwise, false.</returns>
    private static bool IsBasicStatisticMetric(MetricType metricType)
    {
        // Define which metrics are considered basic statistics
        return metricType switch
        {
            MetricType.Mean => true,
            MetricType.Variance => true,
            MetricType.StandardDeviation => true,
            MetricType.Skewness => true,
            MetricType.Kurtosis => true,
            MetricType.Min => true,
            MetricType.Max => true,
            MetricType.N => true,
            MetricType.Median => true,
            MetricType.FirstQuartile => true,
            MetricType.ThirdQuartile => true,
            MetricType.InterquartileRange => true,
            MetricType.MAD => true,
            _ => false,
        };
    }

    /// <summary>
    /// Calculates all statistical measures from the provided values.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    /// <remarks>
    /// <para>
    /// This private method performs the actual calculation of all statistical measures. It populates
    /// the properties of the BasicStats object based on the input vector of values. It uses helper
    /// methods from the StatisticsHelper class for more complex calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This method does all the actual math to calculate the statistics.
    /// 
    /// When called with a set of numbers:
    /// - It calculates basic measures like mean, min, and max directly
    /// - It uses helper methods for more complex calculations like quartiles
    /// - It populates all the properties of the statistics object
    /// 
    /// This is a private method, meaning it's only used internally by the BasicStats class itself,
    /// not by code using this class.
    /// </para>
    /// </remarks>
    private void CalculateStats(Vector<T> values)
    {
        try
        {
            // Return early if the vector is empty
            if (_n == 0 || values.IsEmpty)
            {
                // All metrics remain at their initialized zero values
                return;
            }

            // Calculate basic statistics first
            CalculateBasicMetrics(values);

            // Calculate quartile-based statistics
            CalculateQuartileBasedMetrics(values);

            // Calculate more complex statistics
            CalculateAdvancedMetrics(values);

            // Calculate dependent metrics
            CalculateDependentMetrics();
        }
        catch (Exception ex)
        {
            // Wrap any calculation errors in a more descriptive exception
            throw new InvalidOperationException("Error calculating basic statistics.", ex);
        }
    }

    /// <summary>
    /// Calculates basic statistical metrics like mean, variance, min, and max.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    private void CalculateBasicMetrics(Vector<T> values)
    {
        // Calculate mean
        if (_validMetrics.Contains(MetricType.Mean))
        {
            _metrics[MetricType.Mean] = values.Average();
            _calculatedMetrics.Add(MetricType.Mean);
        }

        // Calculate variance
        if (_validMetrics.Contains(MetricType.Variance))
        {
            _metrics[MetricType.Variance] = values.Variance();
            _calculatedMetrics.Add(MetricType.Variance);
        }

        // Calculate min and max
        if (_validMetrics.Contains(MetricType.Min))
        {
            _metrics[MetricType.Min] = values.Min();
            _calculatedMetrics.Add(MetricType.Min);
        }

        if (_validMetrics.Contains(MetricType.Max))
        {
            _metrics[MetricType.Max] = values.Max();
            _calculatedMetrics.Add(MetricType.Max);
        }

        // Calculate median
        if (_validMetrics.Contains(MetricType.Median))
        {
            _metrics[MetricType.Median] = StatisticsHelper<T>.CalculateMedian(values);
            _calculatedMetrics.Add(MetricType.Median);
        }
    }

    /// <summary>
    /// Calculates quartile-based metrics like FirstQuartile, ThirdQuartile, and IQR.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    private void CalculateQuartileBasedMetrics(Vector<T> values)
    {
        // Calculate quartiles if either is needed
        if (_validMetrics.Contains(MetricType.FirstQuartile) ||
            _validMetrics.Contains(MetricType.ThirdQuartile) ||
            _validMetrics.Contains(MetricType.InterquartileRange))
        {
            var (q1, q3) = StatisticsHelper<T>.CalculateQuantiles(values);

            if (_validMetrics.Contains(MetricType.FirstQuartile))
            {
                _metrics[MetricType.FirstQuartile] = q1;
                _calculatedMetrics.Add(MetricType.FirstQuartile);
            }

            if (_validMetrics.Contains(MetricType.ThirdQuartile))
            {
                _metrics[MetricType.ThirdQuartile] = q3;
                _calculatedMetrics.Add(MetricType.ThirdQuartile);
            }

            // Calculate IQR if needed
            if (_validMetrics.Contains(MetricType.InterquartileRange))
            {
                _metrics[MetricType.InterquartileRange] = _numOps.Subtract(q3, q1);
                _calculatedMetrics.Add(MetricType.InterquartileRange);
            }
        }
    }

    /// <summary>
    /// Calculates more complex statistics like skewness, kurtosis, and MAD.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    private void CalculateAdvancedMetrics(Vector<T> values)
    {
        // Calculate skewness and kurtosis if either is needed
        if (_validMetrics.Contains(MetricType.Skewness) || _validMetrics.Contains(MetricType.Kurtosis))
        {
            // Need mean and standard deviation for these calculations
            var mean = GetOrCalculateMean(values);
            var stdDev = GetOrCalculateStandardDeviation(values);

            var (skewness, kurtosis) = StatisticsHelper<T>.CalculateSkewnessAndKurtosis(values, mean, stdDev, _n);

            if (_validMetrics.Contains(MetricType.Skewness))
            {
                _metrics[MetricType.Skewness] = skewness;
                _calculatedMetrics.Add(MetricType.Skewness);
            }

            if (_validMetrics.Contains(MetricType.Kurtosis))
            {
                _metrics[MetricType.Kurtosis] = kurtosis;
                _calculatedMetrics.Add(MetricType.Kurtosis);
            }
        }

        // Calculate MAD if needed
        if (_validMetrics.Contains(MetricType.MAD))
        {
            var median = GetOrCalculateMedian(values);
            _metrics[MetricType.MAD] = StatisticsHelper<T>.CalculateMeanAbsoluteDeviation(values, median);
            _calculatedMetrics.Add(MetricType.MAD);
        }
    }

    /// <summary>
    /// Calculates metrics that depend on other metrics, ensuring proper calculation order.
    /// </summary>
    private void CalculateDependentMetrics()
    {
        // Calculate standard deviation from variance if needed
        if (_validMetrics.Contains(MetricType.StandardDeviation) &&
            !_calculatedMetrics.Contains(MetricType.StandardDeviation) &&
            _calculatedMetrics.Contains(MetricType.Variance))
        {
            var variance = _metrics[MetricType.Variance];
            _metrics[MetricType.StandardDeviation] = _numOps.Sqrt(variance);
            _calculatedMetrics.Add(MetricType.StandardDeviation);
        }

        // Calculate variance from standard deviation if needed
        if (_validMetrics.Contains(MetricType.Variance) &&
            !_calculatedMetrics.Contains(MetricType.Variance) &&
            _calculatedMetrics.Contains(MetricType.StandardDeviation))
        {
            var stdDev = _metrics[MetricType.StandardDeviation];
            _metrics[MetricType.Variance] = _numOps.Multiply(stdDev, stdDev);
            _calculatedMetrics.Add(MetricType.Variance);
        }

        // Calculate range from min and max if needed
        if (_validMetrics.Contains(MetricType.Range) &&
            !_calculatedMetrics.Contains(MetricType.Range) &&
            _calculatedMetrics.Contains(MetricType.Min) &&
            _calculatedMetrics.Contains(MetricType.Max))
        {
            var min = _metrics[MetricType.Min];
            var max = _metrics[MetricType.Max];
            _metrics[MetricType.Range] = _numOps.Subtract(max, min);
            _calculatedMetrics.Add(MetricType.Range);
        }

        // Calculate interquartile range from quartiles if needed
        if (_validMetrics.Contains(MetricType.InterquartileRange) &&
            !_calculatedMetrics.Contains(MetricType.InterquartileRange) &&
            _calculatedMetrics.Contains(MetricType.FirstQuartile) &&
            _calculatedMetrics.Contains(MetricType.ThirdQuartile))
        {
            var q1 = _metrics[MetricType.FirstQuartile];
            var q3 = _metrics[MetricType.ThirdQuartile];
            _metrics[MetricType.InterquartileRange] = _numOps.Subtract(q3, q1);
            _calculatedMetrics.Add(MetricType.InterquartileRange);
        }
    }

    /// <summary>
    /// Gets the mean from the metrics dictionary or calculates it if it hasn't been calculated.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    /// <returns>The mean of the values.</returns>
    private T GetOrCalculateMean(Vector<T> values)
    {
        if (_calculatedMetrics.Contains(MetricType.Mean))
        {
            return _metrics[MetricType.Mean];
        }

        var mean = values.Average();
        _metrics[MetricType.Mean] = mean;
        _calculatedMetrics.Add(MetricType.Mean);

        return mean;
    }

    /// <summary>
    /// Gets the standard deviation from the metrics dictionary or calculates it if it hasn't been calculated.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    /// <returns>The standard deviation of the values.</returns>
    private T GetOrCalculateStandardDeviation(Vector<T> values)
    {
        if (_calculatedMetrics.Contains(MetricType.StandardDeviation))
        {
            return _metrics[MetricType.StandardDeviation];
        }

        // Calculate via variance if available
        if (_calculatedMetrics.Contains(MetricType.Variance))
        {
            var stdDev = _numOps.Sqrt(_metrics[MetricType.Variance]);
            _metrics[MetricType.StandardDeviation] = stdDev;
            _calculatedMetrics.Add(MetricType.StandardDeviation);

            return stdDev;
        }

        // Calculate both variance and standard deviation
        var variance = values.Variance();
        _metrics[MetricType.Variance] = variance;
        _calculatedMetrics.Add(MetricType.Variance);

        var standardDeviation = _numOps.Sqrt(variance);
        _metrics[MetricType.StandardDeviation] = standardDeviation;
        _calculatedMetrics.Add(MetricType.StandardDeviation);

        return standardDeviation;
    }

    /// <summary>
    /// Gets the median from the metrics dictionary or calculates it if it hasn't been calculated.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    /// <returns>The median of the values.</returns>
    private T GetOrCalculateMedian(Vector<T> values)
    {
        if (_calculatedMetrics.Contains(MetricType.Median))
        {
            return _metrics[MetricType.Median];
        }

        var median = StatisticsHelper<T>.CalculateMedian(values);
        _metrics[MetricType.Median] = median;
        _calculatedMetrics.Add(MetricType.Median);

        return median;
    }

    #endregion
}