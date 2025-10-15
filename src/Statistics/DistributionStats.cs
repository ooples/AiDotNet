namespace AiDotNet.Statistics;

/// <summary>
/// Provides comprehensive statistical measures for analyzing probability distributions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// DistributionStats calculates and stores various statistical measures that help characterize 
/// the shape, center, spread, and other properties of a data distribution. It includes metrics
/// for central tendency, dispersion, and distribution shape.
/// </para>
/// <para><b>For Beginners:</b> Think of DistributionStats as a detailed analyzer for sets of numbers.
/// 
/// It examines your data and tells you important information about how the values are distributed:
/// - Where is the center? (mean, median, mode)
/// - How spread out are the values? (variance, standard deviation)
/// - Is the distribution symmetric or skewed? (skewness)
/// - Are there unusual extreme values? (kurtosis)
/// - What kind of distribution best fits your data? (distribution type)
/// 
/// This helps you understand the underlying patterns in your data and make informed decisions
/// based on its statistical properties.
/// </para>
/// </remarks>
[Serializable]
public class DistributionStats<T> : StatisticsBase<T>
{
    /// <summary>
    /// The observed values from which statistics are calculated.
    /// </summary>
    public Vector<T> Values { get; private set; }

    /// <summary>
    /// Gets the number of values in the dataset.
    /// </summary>
    public int Count => Values is null ? 0 : Values.Length;

    /// <summary>
    /// Gets the arithmetic mean (average) of the values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The mean is the sum of all values divided by the number of values.
    /// It represents the central value of a distribution.
    /// </para>
    /// </remarks>
    public T Mean => GetMetric(MetricType.Mean);

    /// <summary>
    /// Gets the median value of the dataset.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The median is the middle value when all values are arranged in order.
    /// It's less affected by outliers than the mean.
    /// </para>
    /// </remarks>
    public T Median => GetMetric(MetricType.Median);

    /// <summary>
    /// Gets the mode (most frequently occurring value) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The mode is the value that appears most often in your data.
    /// A distribution can have multiple modes or no mode.
    /// </para>
    /// </remarks>
    public T Mode => GetMetric(MetricType.Mode);

    /// <summary>
    /// Gets the variance of the values, a measure of dispersion.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Variance measures how far each value is from the mean.
    /// Higher variance indicates more spread-out data.
    /// </para>
    /// </remarks>
    public T Variance => GetMetric(MetricType.Variance);

    /// <summary>
    /// Gets the standard deviation of the values, a measure of dispersion.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard deviation is the square root of variance.
    /// It measures spread in the same units as your data.
    /// </para>
    /// </remarks>
    public T StandardDeviation => GetMetric(MetricType.StandardDeviation);

    /// <summary>
    /// Gets the skewness of the distribution, a measure of asymmetry.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Skewness tells you if your distribution is lopsided:
    /// - Positive skewness: Long tail to the right
    /// - Zero skewness: Symmetric
    /// - Negative skewness: Long tail to the left
    /// </para>
    /// </remarks>
    public T Skewness => GetMetric(MetricType.Skewness);

    /// <summary>
    /// Gets the kurtosis of the distribution, a measure of "tailedness".
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Kurtosis measures how much of your data is in the "tails":
    /// - High kurtosis: More outliers, sharper central peak
    /// - Low kurtosis: Fewer outliers, flatter central peak
    /// </para>
    /// </remarks>
    public T Kurtosis => GetMetric(MetricType.Kurtosis);

    /// <summary>
    /// Gets the minimum value in the dataset.
    /// </summary>
    public T Min => GetMetric(MetricType.Min);

    /// <summary>
    /// Gets the maximum value in the dataset.
    /// </summary>
    public T Max => GetMetric(MetricType.Max);

    /// <summary>
    /// Gets the range of the dataset (Max - Min).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Range is the difference between the highest and lowest values.
    /// It gives a simple measure of the spread of your data.
    /// </para>
    /// </remarks>
    public T Range => GetMetric(MetricType.Range);

    /// <summary>
    /// Gets the interquartile range (IQR) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> IQR is the range of the middle 50% of your data.
    /// It's useful for identifying outliers and understanding the spread of typical values.
    /// </para>
    /// </remarks>
    public T InterquartileRange => GetMetric(MetricType.InterquartileRange);

    /// <summary>
    /// Gets the first quartile (25th percentile) of the dataset.
    /// </summary>
    public T FirstQuartile => GetMetric(MetricType.FirstQuartile);

    /// <summary>
    /// Gets the third quartile (75th percentile) of the dataset.
    /// </summary>
    public T ThirdQuartile => GetMetric(MetricType.ThirdQuartile);

    /// <summary>
    /// Gets the coefficient of variation, a standardized measure of dispersion.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Coefficient of variation (CV) is standard deviation divided by the mean.
    /// It allows comparing variation between distributions with different means.
    /// </para>
    /// </remarks>
    public T CoefficientOfVariation => GetMetric(MetricType.CoefficientOfVariation);

    /// <summary>
    /// Gets the entropy of the distribution, a measure of uncertainty.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Entropy measures the uncertainty or randomness in a distribution.
    /// Higher entropy means less predictable values.
    /// </para>
    /// </remarks>
    public T Entropy => GetMetric(MetricType.Entropy);

    /// <summary>
    /// Gets the best-fitting distribution type for the data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you which standard probability distribution
    /// (like Normal, Exponential, etc.) most closely matches your data.
    /// </para>
    /// </remarks>
    public DistributionType BestFitDistribution => (DistributionType)Convert.ToInt32(_numOps.ToInt32(GetMetric(MetricType.BestDistributionType)));

    /// <summary>
    /// Gets the parameters for the best-fitting distribution.
    /// </summary>
    public Dictionary<string, T> DistributionParameters { get; private set; }

    /// <summary>
    /// Gets the p-value from the distribution goodness-of-fit test.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The p-value tells you how well your data fits the best distribution.
    /// Higher values (closer to 1) indicate a better fit.
    /// </para>
    /// </remarks>
    public T GoodnessOfFitPValue => GetMetric(MetricType.GoodnessOfFitPValue);

    /// <summary>
    /// Gets a collection of quantiles (values that divide the data into equal portions).
    /// </summary>
    public List<(T Quantile, T Value)> Quantiles { get; private set; }

    #region Constructors

    /// <summary>
    /// Creates a new DistributionStats instance and calculates statistics for the provided values.
    /// </summary>
    /// <param name="values">The values to analyze.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor takes your data and calculates all the statistics
    /// that help you understand its distribution properties.
    /// </para>
    /// </remarks>
    public DistributionStats(Vector<T> values, ModelType modelType) : base(modelType)
    {
        Values = values ?? Vector<T>.Empty();
        DistributionParameters = [];
        Quantiles = [];

        // Calculate all valid metrics
        if (!Values.IsEmpty)
        {
            CalculateDistributionStats();
        }
    }

    /// <summary>
    /// Creates an empty DistributionStats instance with all statistics set to their default values.
    /// </summary>
    /// <returns>An empty DistributionStats object.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a DistributionStats object with no data.
    /// All statistics will be set to zero or their default values.
    /// </para>
    /// </remarks>
    public static DistributionStats<T> Empty()
    {
        return new DistributionStats<T>(Vector<T>.Empty(), ModelType.None);
    }

    #endregion

    #region Core Calculation Methods

    /// <summary>
    /// Determines which metrics are valid for this statistics provider.
    /// </summary>
    protected override void DetermineValidMetrics()
    {
        // Define which metrics are valid for distribution statistics
        var validMetrics = new MetricType[]
        {
            MetricType.Mean,
            MetricType.Median,
            MetricType.Mode,
            MetricType.Variance,
            MetricType.StandardDeviation,
            MetricType.Skewness,
            MetricType.Kurtosis,
            MetricType.Min,
            MetricType.Max,
            MetricType.Range,
            MetricType.FirstQuartile,
            MetricType.ThirdQuartile,
            MetricType.InterquartileRange,
            MetricType.CoefficientOfVariation,
            MetricType.Entropy,
            MetricType.BestDistributionType,
            MetricType.GoodnessOfFitPValue
        };

        foreach (var metricType in validMetrics)
        {
            _validMetrics.Add(metricType);
        }
    }

    /// <summary>
    /// Calculates all valid distribution statistics.
    /// </summary>
    private void CalculateDistributionStats()
    {
        // Calculate basic statistics
        CalculateBasicStats();

        // Calculate distribution-specific statistics
        CalculateDistributionFit();

        // Calculate quantiles
        CalculateQuantiles();
    }

    /// <summary>
    /// Calculates basic statistical measures.
    /// </summary>
    private void CalculateBasicStats()
    {
        // Calculate mean and central tendency metrics
        if (_validMetrics.Contains(MetricType.Mean))
        {
            _metrics[MetricType.Mean] = StatisticsHelper<T>.CalculateMean(Values);
            _calculatedMetrics.Add(MetricType.Mean);
        }

        if (_validMetrics.Contains(MetricType.Median))
        {
            _metrics[MetricType.Median] = StatisticsHelper<T>.CalculateMedian(Values);
            _calculatedMetrics.Add(MetricType.Median);
        }

        if (_validMetrics.Contains(MetricType.Mode))
        {
            _metrics[MetricType.Mode] = StatisticsHelper<T>.CalculateMode(Values);
            _calculatedMetrics.Add(MetricType.Mode);
        }

        // Calculate dispersion metrics
        if (_validMetrics.Contains(MetricType.Variance))
        {
            _metrics[MetricType.Variance] = StatisticsHelper<T>.CalculateVariance(Values);
            _calculatedMetrics.Add(MetricType.Variance);
        }

        if (_validMetrics.Contains(MetricType.StandardDeviation) && _calculatedMetrics.Contains(MetricType.Variance))
        {
            _metrics[MetricType.StandardDeviation] = _numOps.Sqrt(_metrics[MetricType.Variance]);
            _calculatedMetrics.Add(MetricType.StandardDeviation);
        }

        // Calculate range metrics
        if (_validMetrics.Contains(MetricType.Min))
        {
            _metrics[MetricType.Min] = Values.Min();
            _calculatedMetrics.Add(MetricType.Min);
        }

        if (_validMetrics.Contains(MetricType.Max))
        {
            _metrics[MetricType.Max] = Values.Max();
            _calculatedMetrics.Add(MetricType.Max);
        }

        if (_validMetrics.Contains(MetricType.Range) &&
            _calculatedMetrics.Contains(MetricType.Min) &&
            _calculatedMetrics.Contains(MetricType.Max))
        {
            _metrics[MetricType.Range] = _numOps.Subtract(_metrics[MetricType.Max], _metrics[MetricType.Min]);
            _calculatedMetrics.Add(MetricType.Range);
        }

        // Calculate quartiles and IQR
        if (_validMetrics.Contains(MetricType.FirstQuartile) ||
            _validMetrics.Contains(MetricType.ThirdQuartile) ||
            _validMetrics.Contains(MetricType.InterquartileRange))
        {
            var (q1, q3) = StatisticsHelper<T>.CalculateQuantiles(Values);

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

            if (_validMetrics.Contains(MetricType.InterquartileRange))
            {
                _metrics[MetricType.InterquartileRange] = _numOps.Subtract(q3, q1);
                _calculatedMetrics.Add(MetricType.InterquartileRange);
            }
        }

        // Calculate shape metrics
        if (_validMetrics.Contains(MetricType.Skewness) || _validMetrics.Contains(MetricType.Kurtosis))
        {
            // If we need both metrics, calculate them together efficiently
            if (_validMetrics.Contains(MetricType.Skewness) && _validMetrics.Contains(MetricType.Kurtosis))
            {
                var mean = _calculatedMetrics.Contains(MetricType.Mean) ?
                    _metrics[MetricType.Mean] :
                    StatisticsHelper<T>.CalculateMean(Values);

                var stdDev = _calculatedMetrics.Contains(MetricType.StandardDeviation) ?
                    _metrics[MetricType.StandardDeviation] :
                    _numOps.Sqrt(StatisticsHelper<T>.CalculateVariance(Values));

                (var skewness, var kurtosis) = StatisticsHelper<T>.CalculateSkewnessAndKurtosis(Values, mean, stdDev, Values.Length);

                _metrics[MetricType.Skewness] = skewness;
                _calculatedMetrics.Add(MetricType.Skewness);

                _metrics[MetricType.Kurtosis] = kurtosis;
                _calculatedMetrics.Add(MetricType.Kurtosis);
            }
            // Otherwise calculate them individually
            else
            {
                if (_validMetrics.Contains(MetricType.Skewness))
                {
                    _metrics[MetricType.Skewness] = StatisticsHelper<T>.CalculateSkewness(Values);
                    _calculatedMetrics.Add(MetricType.Skewness);
                }

                if (_validMetrics.Contains(MetricType.Kurtosis))
                {
                    _metrics[MetricType.Kurtosis] = StatisticsHelper<T>.CalculateKurtosis(Values);
                    _calculatedMetrics.Add(MetricType.Kurtosis);
                }
            }
        }

        // Calculate coefficient of variation
        if (_validMetrics.Contains(MetricType.CoefficientOfVariation) &&
            _calculatedMetrics.Contains(MetricType.StandardDeviation) &&
            _calculatedMetrics.Contains(MetricType.Mean))
        {
            var mean = _metrics[MetricType.Mean];
            if (!_numOps.Equals(mean, _numOps.Zero) && !_numOps.IsNaN(mean))
            {
                var stdDev = _metrics[MetricType.StandardDeviation];
                _metrics[MetricType.CoefficientOfVariation] = _numOps.Divide(stdDev, _numOps.Abs(mean));
                _calculatedMetrics.Add(MetricType.CoefficientOfVariation);
            }
        }

        // Calculate entropy
        if (_validMetrics.Contains(MetricType.Entropy))
        {
            _metrics[MetricType.Entropy] = StatisticsHelper<T>.CalculateEntropy(Values);
            _calculatedMetrics.Add(MetricType.Entropy);
        }
    }

    /// <summary>
    /// Calculates the best-fitting distribution for the data.
    /// </summary>
    private void CalculateDistributionFit()
    {
        if (_validMetrics.Contains(MetricType.BestDistributionType) ||
            _validMetrics.Contains(MetricType.GoodnessOfFitPValue))
        {
            var fitResult = StatisticsHelper<T>.DetermineBestFitDistribution(Values);

            if (_validMetrics.Contains(MetricType.BestDistributionType))
            {
                _metrics[MetricType.BestDistributionType] = _numOps.FromDouble((double)fitResult.DistributionType);
                _calculatedMetrics.Add(MetricType.BestDistributionType);
            }

            if (_validMetrics.Contains(MetricType.GoodnessOfFitPValue))
            {
                _metrics[MetricType.GoodnessOfFitPValue] = fitResult.PValue;
                _calculatedMetrics.Add(MetricType.GoodnessOfFitPValue);
            }

            // Store distribution parameters
            DistributionParameters = fitResult.Parameters;
        }
    }

    /// <summary>
    /// Calculates quantiles of the distribution.
    /// </summary>
    private void CalculateQuantiles()
    {
        // Default quantiles at 10% intervals
        var percentiles = new[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

        // Get the quantile values as a dictionary
        var quantileDict = StatisticsHelper<T>.CalculateQuantiles(Values, [.. percentiles.Select(p => _numOps.FromDouble(p))]);

        Quantiles = [];

        foreach (var kvp in quantileDict)
        {
            if (double.TryParse(kvp.Key, out double percentileValue))
            {
                Quantiles.Add((_numOps.FromDouble(percentileValue), kvp.Value));
            }
        }
    }

    #endregion

    #region Public API Methods

    /// <summary>
    /// Gets the value at a specific percentile in the distribution.
    /// </summary>
    /// <param name="percentile">The percentile to find (between 0 and 1).</param>
    /// <returns>The value at the specified percentile.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method finds the value below which a certain percentage of your data falls.
    /// For example, the 0.25 percentile is the value below which 25% of your data falls.
    /// </para>
    /// </remarks>
    public T GetPercentile(T percentile)
    {
        if (_numOps.LessThan(percentile, _numOps.Zero) || _numOps.GreaterThan(percentile, _numOps.One))
        {
            throw new ArgumentOutOfRangeException(nameof(percentile), "Percentile must be between 0 and 1");
        }

        return StatisticsHelper<T>.CalculatePercentile(Values, percentile);
    }

    /// <summary>
    /// Creates a probability density function (PDF) for the distribution.
    /// </summary>
    /// <param name="points">The number of points to generate.</param>
    /// <returns>A list of (x, probability density) pairs representing the PDF.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The PDF shows how likely different values are in your distribution.
    /// Higher points on the curve represent values that are more likely to occur.
    /// </para>
    /// </remarks>
    public List<(T X, T Density)> CreateProbabilityDensityFunction(int points = 100)
    {
        if (Values.IsEmpty)
        {
            return [];
        }

        return StatisticsHelper<T>.CreateProbabilityDensityFunction(
            Values,
            _numOps.Subtract(_metrics[MetricType.Min], _numOps.Multiply(_metrics[MetricType.StandardDeviation], _numOps.FromDouble(3))),
            _numOps.Add(_metrics[MetricType.Max], _numOps.Multiply(_metrics[MetricType.StandardDeviation], _numOps.FromDouble(3))),
            points,
            BestFitDistribution,
            DistributionParameters);
    }

    /// <summary>
    /// Creates a cumulative distribution function (CDF) for the distribution.
    /// </summary>
    /// <param name="points">The number of points to generate.</param>
    /// <returns>A list of (x, cumulative probability) pairs representing the CDF.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The CDF shows the probability of a value being less than or equal to each point.
    /// It ranges from 0 to 1 and increases from left to right.
    /// </para>
    /// </remarks>
    public List<(T X, T CumulativeProbability)> CreateCumulativeDistributionFunction(int points = 100)
    {
        if (Values.IsEmpty)
        {
            return new List<(T, T)>();
        }

        return StatisticsHelper<T>.CreateCumulativeDistributionFunction(
            Values,
            _numOps.Subtract(_metrics[MetricType.Min], _numOps.Multiply(_metrics[MetricType.StandardDeviation], _numOps.FromDouble(3))),
            _numOps.Add(_metrics[MetricType.Max], _numOps.Multiply(_metrics[MetricType.StandardDeviation], _numOps.FromDouble(3))),
            points,
            BestFitDistribution,
            DistributionParameters);
    }

    /// <summary>
    /// Calculates the probability that a value falls within a specified range.
    /// </summary>
    /// <param name="lowerBound">The lower bound of the range.</param>
    /// <param name="upperBound">The upper bound of the range.</param>
    /// <returns>The probability (between 0 and 1) that a value falls within the specified range.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how likely it is for a value from your distribution
    /// to fall between the lower and upper bounds you specify.
    /// </para>
    /// </remarks>
    public T CalculateProbabilityInRange(T lowerBound, T upperBound)
    {
        if (Values.IsEmpty)
        {
            return _numOps.Zero;
        }

        return StatisticsHelper<T>.CalculateProbabilityInRange(
            lowerBound,
            upperBound,
            BestFitDistribution,
            DistributionParameters);
    }

    /// <summary>
    /// Performs a normality test to determine if the data follows a normal distribution.
    /// </summary>
    /// <returns>A tuple containing the test statistic and p-value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tests whether your data follows a bell-shaped (normal) distribution.
    /// A high p-value (typically > 0.05) suggests your data could be normally distributed.
    /// </para>
    /// </remarks>
    public (T TestStatistic, T PValue) TestNormality()
    {
        if (Values.IsEmpty)
        {
            return (_numOps.Zero, _numOps.Zero);
        }

        return StatisticsHelper<T>.TestNormality(Values);
    }

    #endregion
}