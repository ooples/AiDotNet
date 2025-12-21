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
public class BasicStats<T>
{
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
    public T Mean { get; private set; }

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
    public T Variance { get; private set; }

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
    public T StandardDeviation { get; private set; }

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
    public T Skewness { get; private set; }

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
    public T Kurtosis { get; private set; }

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
    public T Min { get; private set; }

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
    public T Max { get; private set; }

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
    public int N { get; private set; }

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
    public T Median { get; private set; }

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
    public T FirstQuartile { get; private set; }

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
    public T ThirdQuartile { get; private set; }

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
    public T InterquartileRange { get; private set; }

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
    public T MAD { get; private set; }

    /// <summary>
    /// The numeric operations appropriate for the generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to the appropriate numeric operations implementation for the
    /// generic type T, allowing the statistical methods to perform mathematical operations
    /// regardless of whether T is float, double, or another numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that allows the code to work with different number types.
    /// 
    /// Since this class uses a generic type T (which could be float, double, etc.):
    /// - We need a way to perform math operations (+, -, *, /) on these values
    /// - _numOps provides the right methods for whatever numeric type is being used
    /// 
    /// Think of it like having different calculators for different types of numbers,
    /// and _numOps makes sure we're using the right calculator for the job.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the BasicStats class with the provided input values.
    /// </summary>
    /// <param name="inputs">The input values to calculate statistics for.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes all statistical measures based on the provided input values.
    /// It calculates the full set of statistics in a single pass through the data where possible.
    /// This constructor is marked as internal as BasicStats objects should typically be created
    /// using factory methods.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new statistics object with your data.
    /// 
    /// When you create a BasicStats object:
    /// - It takes your set of numbers as input
    /// - It calculates all the different statistics (mean, median, etc.)
    /// - It stores all those results for you to access later
    /// 
    /// This constructor is marked as "internal" which means you should use the provided
    /// factory methods like Empty() to create BasicStats objects, rather than creating
    /// them directly with this constructor.
    /// </para>
    /// </remarks>
    internal BasicStats(BasicStatsInputs<T> inputs)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize all class variables
        Mean = _numOps.Zero;
        Variance = _numOps.Zero;
        StandardDeviation = _numOps.Zero;
        Skewness = _numOps.Zero;
        Kurtosis = _numOps.Zero;
        Min = _numOps.Zero;
        Max = _numOps.Zero;
        N = 0;
        Median = _numOps.Zero;
        FirstQuartile = _numOps.Zero;
        ThirdQuartile = _numOps.Zero;
        InterquartileRange = _numOps.Zero;
        MAD = _numOps.Zero;

        CalculateStats(inputs.Values);
    }

    /// <summary>
    /// Creates an empty BasicStats object with all statistics set to their default values.
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
    public static BasicStats<T> Empty()
    {
        return new BasicStats<T>(new());
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
        N = values.Length;
        if (N == 0) return;
        Mean = values.Average();
        Variance = values.Variance();
        StandardDeviation = _numOps.Sqrt(Variance);
        (Skewness, Kurtosis) = StatisticsHelper<T>.CalculateSkewnessAndKurtosis(values, Mean, StandardDeviation, N);
        Min = values.Min();
        Max = values.Max();
        Median = StatisticsHelper<T>.CalculateMedian(values);
        (FirstQuartile, ThirdQuartile) = StatisticsHelper<T>.CalculateQuantiles(values);
        InterquartileRange = _numOps.Subtract(ThirdQuartile, FirstQuartile);
        MAD = StatisticsHelper<T>.CalculateMeanAbsoluteDeviation(values, Median);
    }

    /// <summary>
    /// Gets the value of a specific metric based on the provided MetricType.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <returns>The value of the requested metric.</returns>
    /// <remarks>
    /// <para>
    /// This method allows you to retrieve any of the calculated statistics by specifying the desired metric type.
    /// It provides a flexible way to access individual metrics without needing separate properties for each.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a vending machine for statistics.
    /// 
    /// You tell it which statistic you want (using the MetricType), and it gives you the value.
    /// For example:
    /// - If you ask for MetricType.Mean, it gives you the average
    /// - If you ask for MetricType.StandardDeviation, it gives you the standard deviation
    /// 
    /// This is useful when you want to work with different statistics in a flexible way,
    /// especially if you don't know in advance which statistic you'll need.
    /// </para>
    /// </remarks>
    public T GetMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.Mean => Mean,
            MetricType.Variance => Variance,
            MetricType.StandardDeviation => StandardDeviation,
            MetricType.Skewness => Skewness,
            MetricType.Kurtosis => Kurtosis,
            MetricType.Min => Min,
            MetricType.Max => Max,
            MetricType.N => _numOps.FromDouble(N),
            MetricType.Median => Median,
            MetricType.FirstQuartile => FirstQuartile,
            MetricType.ThirdQuartile => ThirdQuartile,
            MetricType.InterquartileRange => InterquartileRange,
            MetricType.MAD => MAD,
            _ => throw new ArgumentException($"Metric {metricType} is not available in BasicStats.", nameof(metricType)),
        };
    }

    /// <summary>
    /// Checks if a specific metric is available in this BasicStats instance.
    /// </summary>
    /// <param name="metricType">The type of metric to check for.</param>
    /// <returns>True if the metric is available, false otherwise.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method allows you to check if a particular metric is available before trying to get its value.
    /// It's useful when you're not sure if a specific metric was calculated for this set of basic stats.
    /// 
    /// For example:
    /// <code>
    /// if (stats.HasMetric(MetricType.Mean))
    /// {
    ///     var meanValue = stats.GetMetric(MetricType.Mean);
    ///     // Use meanValue...
    /// }
    /// </code>
    /// 
    /// This prevents errors that might occur if you try to access a metric that wasn't calculated.
    /// </remarks>
    public bool HasMetric(MetricType metricType)
    {
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
}
