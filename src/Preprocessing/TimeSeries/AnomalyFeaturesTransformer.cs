using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Computes rolling anomaly detection features for time series data.
/// </summary>
/// <remarks>
/// <para>
/// This transformer calculates features that help identify unusual patterns,
/// including Z-scores, IQR-based outlier detection, CUSUM control charts, and isolation scores.
/// </para>
/// <para><b>For Beginners:</b> Anomaly detection helps find unusual or unexpected values in your data.
///
/// This is useful for:
/// - Fraud detection (unusual transactions)
/// - Equipment failure prediction (abnormal sensor readings)
/// - Quality control (out-of-specification products)
/// - Financial monitoring (unusual market movements)
///
/// The transformer creates several types of anomaly features:
/// - Z-score: How many standard deviations from normal
/// - IQR outliers: Values outside the typical range
/// - CUSUM: Detects gradual shifts in the data
/// - Isolation score: Machine learning-based anomaly scoring
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class AnomalyFeaturesTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The enabled anomaly features.
    /// </summary>
    private readonly AnomalyFeatures _enabledFeatures;

    /// <summary>
    /// Z-score threshold for anomaly flagging.
    /// </summary>
    private readonly double _zScoreThreshold;

    /// <summary>
    /// IQR multiplier for outlier detection.
    /// </summary>
    private readonly double _iqrMultiplier;

    /// <summary>
    /// CUSUM sensitivity parameter (k).
    /// </summary>
    private readonly double _cusumK;

    /// <summary>
    /// CUSUM decision threshold (h).
    /// </summary>
    private readonly double _cusumH;

    /// <summary>
    /// Cached operation names.
    /// </summary>
    private readonly string[] _operationNames;

    /// <summary>
    /// Random number generator for isolation forest.
    /// </summary>
    private readonly Random _random;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new anomaly features transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public AnomalyFeaturesTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _enabledFeatures = Options.EnabledAnomalyFeatures;
        _zScoreThreshold = Options.ZScoreThreshold;
        _iqrMultiplier = Options.IqrMultiplier;
        _cusumK = Options.CusumK;
        _cusumH = Options.CusumH;
        _random = new Random(42); // Fixed seed for reproducibility

        _operationNames = BuildOperationNames();
    }

    #endregion

    #region Properties

    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    #endregion

    #region Core Implementation

    /// <inheritdoc />
    protected override void FitCore(Tensor<T> data)
    {
        // Anomaly features don't need to learn parameters
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();

        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        int maxWindow = GetMaxWindowSize();
        int firstValidIndex = -1;

        for (int outT = 0; outT < outputTimeSteps; outT++)
        {
            int t = outT + startIndex;
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                if (isEdge && Options.EdgeHandling == EdgeHandling.NaN)
                {
                    int measuresCount = CountEnabledFeatures() * inputFeatures;
                    for (int i = 0; i < measuresCount; i++)
                    {
                        output[outT, outputIdx++] = GetNaN();
                    }
                    continue;
                }

                if (firstValidIndex < 0 && !IsEdgeRegion(t, maxWindow))
                {
                    firstValidIndex = outT;
                }

                int effectiveWindow = ShouldComputePartialWindows() && isEdge
                    ? GetEffectiveWindowSize(t, windowSize)
                    : windowSize;

                for (int f = 0; f < inputFeatures; f++)
                {
                    var window = ExtractWindow(data, t, f, effectiveWindow);
                    double currentValue = window[^1];
                    ComputeAnomalyFeatures(window, currentValue, output, outT, ref outputIdx);
                }
            }
        }

        if (Options.EdgeHandling == EdgeHandling.ForwardFill && firstValidIndex > 0)
        {
            ApplyForwardFill(output, firstValidIndex);
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();
        int maxWindow = GetMaxWindowSize();

        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });
        int featuresPerWindow = CountEnabledFeatures();

        int firstValidIndex = -1;
        object lockObj = new object();

        Parallel.For(0, outputTimeSteps, outT =>
        {
            int t = outT + startIndex;
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                if (isEdge && Options.EdgeHandling == EdgeHandling.NaN)
                {
                    int measuresCount = featuresPerWindow * inputFeatures;
                    for (int i = 0; i < measuresCount; i++)
                    {
                        output[outT, outputIdx++] = GetNaN();
                    }
                    continue;
                }

                if (!IsEdgeRegion(t, maxWindow))
                {
                    lock (lockObj)
                    {
                        if (firstValidIndex < 0 || outT < firstValidIndex)
                            firstValidIndex = outT;
                    }
                }

                int effectiveWindow = ShouldComputePartialWindows() && isEdge
                    ? GetEffectiveWindowSize(t, windowSize)
                    : windowSize;

                for (int f = 0; f < inputFeatures; f++)
                {
                    var window = ExtractWindow(data, t, f, effectiveWindow);
                    double currentValue = window[^1];
                    int localIdx = outputIdx;
                    ComputeAnomalyFeaturesThreadSafe(window, currentValue, output, outT, localIdx);
                    outputIdx += featuresPerWindow;
                }
            }
        });

        if (Options.EdgeHandling == EdgeHandling.ForwardFill && firstValidIndex > 0)
        {
            ApplyForwardFill(output, firstValidIndex);
        }

        return output;
    }

    #endregion

    #region Feature Naming

    /// <inheritdoc />
    protected override string[] GenerateFeatureNames()
    {
        var names = new List<string>();
        var inputNames = GetInputFeatureNames();
        var sep = GetSeparator();
        var ops = GetOperationNames();

        foreach (int windowSize in WindowSizes)
        {
            foreach (string inputName in inputNames)
            {
                foreach (string op in ops)
                {
                    names.Add($"{inputName}{sep}{op}{sep}{windowSize}");
                }
            }
        }

        return [.. names];
    }

    /// <inheritdoc />
    protected override string[] GetOperationNames()
    {
        return _operationNames;
    }

    private string[] BuildOperationNames()
    {
        var ops = new List<string>();

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScore))
            ops.Add("z_score");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScoreFlag))
            ops.Add("z_score_flag");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ModifiedZScore))
            ops.Add("mod_z_score");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierScore))
            ops.Add("iqr_score");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierFlag))
            ops.Add("iqr_flag");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumStatistic))
            ops.Add("cusum");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumFlag))
            ops.Add("cusum_flag");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IsolationScore))
            ops.Add("isolation_score");

        if (_enabledFeatures.HasFlag(AnomalyFeatures.PercentileRank))
            ops.Add("percentile_rank");

        return [.. ops];
    }

    private int CountEnabledFeatures()
    {
        return _operationNames.Length;
    }

    #endregion

    #region Window Extraction

    private double[] ExtractWindow(Tensor<T> data, int endTime, int feature, int windowSize)
    {
        int actualSize = Math.Min(windowSize, endTime + 1);
        actualSize = Math.Max(1, actualSize);

        int startTime = endTime - actualSize + 1;
        var window = new double[actualSize];

        for (int i = 0; i < actualSize; i++)
        {
            int t = startTime + i;
            window[i] = t < 0 ? double.NaN : NumOps.ToDouble(GetValue(data, t, feature));
        }

        return window;
    }

    #endregion

    #region Anomaly Calculations

    private void ComputeAnomalyFeatures(double[] window, double currentValue, Tensor<T> output, int outT, ref int outputIdx)
    {
        var validValues = window.Where(v => !double.IsNaN(v)).ToArray();

        if (validValues.Length < 2)
        {
            int numFeatures = CountEnabledFeatures();
            for (int i = 0; i < numFeatures; i++)
            {
                output[outT, outputIdx++] = GetNaN();
            }
            return;
        }

        double mean = validValues.Average();
        double std = ComputeStdDev(validValues);
        double median = ComputeMedian(validValues);
        double mad = ComputeMAD(validValues, median);

        // Create a sorted copy for percentile calculations
        // Keep original validValues in temporal order for CUSUM
        var sortedValues = (double[])validValues.Clone();
        Array.Sort(sortedValues);
        double q1 = ComputePercentile(sortedValues, 0.25);
        double q3 = ComputePercentile(sortedValues, 0.75);
        double iqr = q3 - q1;

        // Z-score
        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScore))
        {
            double zScore = std > 1e-10 ? (currentValue - mean) / std : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(zScore);
        }

        // Z-score flag
        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScoreFlag))
        {
            double zScore = std > 1e-10 ? (currentValue - mean) / std : 0;
            double flag = Math.Abs(zScore) > _zScoreThreshold ? 1.0 : 0.0;
            output[outT, outputIdx++] = NumOps.FromDouble(flag);
        }

        // Modified Z-score (using MAD)
        if (_enabledFeatures.HasFlag(AnomalyFeatures.ModifiedZScore))
        {
            // Modified Z-score = 0.6745 * (x - median) / MAD
            double modZScore = mad > 1e-10 ? 0.6745 * (currentValue - median) / mad : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(modZScore);
        }

        // IQR outlier score
        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierScore))
        {
            double lowerBound = q1 - _iqrMultiplier * iqr;
            double upperBound = q3 + _iqrMultiplier * iqr;
            double iqrScore = 0;

            if (currentValue < lowerBound)
                iqrScore = (lowerBound - currentValue) / (iqr > 1e-10 ? iqr : 1);
            else if (currentValue > upperBound)
                iqrScore = (currentValue - upperBound) / (iqr > 1e-10 ? iqr : 1);

            output[outT, outputIdx++] = NumOps.FromDouble(iqrScore);
        }

        // IQR outlier flag
        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierFlag))
        {
            double lowerBound = q1 - _iqrMultiplier * iqr;
            double upperBound = q3 + _iqrMultiplier * iqr;
            double flag = (currentValue < lowerBound || currentValue > upperBound) ? 1.0 : 0.0;
            output[outT, outputIdx++] = NumOps.FromDouble(flag);
        }

        // CUSUM statistic
        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumStatistic))
        {
            double cusum = ComputeCusum(validValues, mean, std);
            output[outT, outputIdx++] = NumOps.FromDouble(cusum);
        }

        // CUSUM flag
        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumFlag))
        {
            double cusum = ComputeCusum(validValues, mean, std);
            double threshold = _cusumH * (std > 1e-10 ? std : 1);
            double flag = cusum > threshold ? 1.0 : 0.0;
            output[outT, outputIdx++] = NumOps.FromDouble(flag);
        }

        // Isolation score
        if (_enabledFeatures.HasFlag(AnomalyFeatures.IsolationScore))
        {
            double isolationScore = ComputeIsolationScore(currentValue, validValues);
            output[outT, outputIdx++] = NumOps.FromDouble(isolationScore);
        }

        // Percentile rank
        if (_enabledFeatures.HasFlag(AnomalyFeatures.PercentileRank))
        {
            int countBelow = validValues.Count(v => v < currentValue);
            double percentileRank = (double)countBelow / validValues.Length;
            output[outT, outputIdx++] = NumOps.FromDouble(percentileRank);
        }
    }

    private void ComputeAnomalyFeaturesThreadSafe(double[] window, double currentValue, Tensor<T> output, int outT, int startIdx)
    {
        int outputIdx = startIdx;

        var validValues = window.Where(v => !double.IsNaN(v)).ToArray();

        if (validValues.Length < 2)
        {
            int numFeatures = CountEnabledFeatures();
            for (int i = 0; i < numFeatures; i++)
            {
                output[outT, outputIdx++] = GetNaN();
            }
            return;
        }

        double mean = validValues.Average();
        double std = ComputeStdDev(validValues);
        double median = ComputeMedian(validValues);
        double mad = ComputeMAD(validValues, median);

        var sortedValues = (double[])validValues.Clone();
        Array.Sort(sortedValues);
        double q1 = ComputePercentile(sortedValues, 0.25);
        double q3 = ComputePercentile(sortedValues, 0.75);
        double iqr = q3 - q1;

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScore))
        {
            double zScore = std > 1e-10 ? (currentValue - mean) / std : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(zScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScoreFlag))
        {
            double zScore = std > 1e-10 ? (currentValue - mean) / std : 0;
            double flag = Math.Abs(zScore) > _zScoreThreshold ? 1.0 : 0.0;
            output[outT, outputIdx++] = NumOps.FromDouble(flag);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ModifiedZScore))
        {
            double modZScore = mad > 1e-10 ? 0.6745 * (currentValue - median) / mad : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(modZScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierScore))
        {
            double lowerBound = q1 - _iqrMultiplier * iqr;
            double upperBound = q3 + _iqrMultiplier * iqr;
            double iqrScore = 0;

            if (currentValue < lowerBound)
                iqrScore = (lowerBound - currentValue) / (iqr > 1e-10 ? iqr : 1);
            else if (currentValue > upperBound)
                iqrScore = (currentValue - upperBound) / (iqr > 1e-10 ? iqr : 1);

            output[outT, outputIdx++] = NumOps.FromDouble(iqrScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierFlag))
        {
            double lowerBound = q1 - _iqrMultiplier * iqr;
            double upperBound = q3 + _iqrMultiplier * iqr;
            double flag = (currentValue < lowerBound || currentValue > upperBound) ? 1.0 : 0.0;
            output[outT, outputIdx++] = NumOps.FromDouble(flag);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumStatistic))
        {
            double cusum = ComputeCusum(validValues, mean, std);
            output[outT, outputIdx++] = NumOps.FromDouble(cusum);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumFlag))
        {
            double cusum = ComputeCusum(validValues, mean, std);
            double threshold = _cusumH * (std > 1e-10 ? std : 1);
            double flag = cusum > threshold ? 1.0 : 0.0;
            output[outT, outputIdx++] = NumOps.FromDouble(flag);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IsolationScore))
        {
            double isolationScore = ComputeIsolationScore(currentValue, validValues);
            output[outT, outputIdx++] = NumOps.FromDouble(isolationScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.PercentileRank))
        {
            int countBelow = validValues.Count(v => v < currentValue);
            double percentileRank = (double)countBelow / validValues.Length;
            output[outT, outputIdx++] = NumOps.FromDouble(percentileRank);
        }
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Computes anomaly features incrementally from the circular buffer.
    /// </summary>
    protected override T[] ComputeIncrementalFeatures(IncrementalState<T> state, T[] newDataPoint)
    {
        var features = new T[OutputFeatureCount];
        int featureIdx = 0;

        foreach (int windowSize in WindowSizes)
        {
            for (int f = 0; f < InputFeatureCount; f++)
            {
                // Extract window data from the circular buffer
                var windowData = ExtractIncrementalWindow(state, f, windowSize);
                double currentValue = windowData[^1];

                // Compute anomaly features
                ComputeAnomalyFeaturesIncremental(windowData, currentValue, features, ref featureIdx);
            }
        }

        return features;
    }

    /// <summary>
    /// Computes anomaly features for a window of data incrementally.
    /// </summary>
    private void ComputeAnomalyFeaturesIncremental(double[] window, double currentValue, T[] features, ref int featureIdx)
    {
        var validValues = window.Where(v => !double.IsNaN(v)).ToArray();

        if (validValues.Length < 2)
        {
            int numFeatures = CountEnabledFeatures();
            for (int i = 0; i < numFeatures; i++)
            {
                features[featureIdx++] = GetNaN();
            }
            return;
        }

        double mean = validValues.Average();
        double std = ComputeStdDev(validValues);
        double median = ComputeMedian(validValues);
        double mad = ComputeMAD(validValues, median);

        var sortedValues = (double[])validValues.Clone();
        Array.Sort(sortedValues);
        double q1 = ComputePercentile(sortedValues, 0.25);
        double q3 = ComputePercentile(sortedValues, 0.75);
        double iqr = q3 - q1;

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScore))
        {
            double zScore = std > 1e-10 ? (currentValue - mean) / std : 0;
            features[featureIdx++] = NumOps.FromDouble(zScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ZScoreFlag))
        {
            double zScore = std > 1e-10 ? (currentValue - mean) / std : 0;
            double flag = Math.Abs(zScore) > _zScoreThreshold ? 1.0 : 0.0;
            features[featureIdx++] = NumOps.FromDouble(flag);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.ModifiedZScore))
        {
            double modZScore = mad > 1e-10 ? 0.6745 * (currentValue - median) / mad : 0;
            features[featureIdx++] = NumOps.FromDouble(modZScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierScore))
        {
            double lowerBound = q1 - _iqrMultiplier * iqr;
            double upperBound = q3 + _iqrMultiplier * iqr;
            double iqrScore = 0;

            if (currentValue < lowerBound)
                iqrScore = (lowerBound - currentValue) / (iqr > 1e-10 ? iqr : 1);
            else if (currentValue > upperBound)
                iqrScore = (currentValue - upperBound) / (iqr > 1e-10 ? iqr : 1);

            features[featureIdx++] = NumOps.FromDouble(iqrScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IqrOutlierFlag))
        {
            double lowerBound = q1 - _iqrMultiplier * iqr;
            double upperBound = q3 + _iqrMultiplier * iqr;
            double flag = (currentValue < lowerBound || currentValue > upperBound) ? 1.0 : 0.0;
            features[featureIdx++] = NumOps.FromDouble(flag);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumStatistic))
        {
            double cusum = ComputeCusum(validValues, mean, std);
            features[featureIdx++] = NumOps.FromDouble(cusum);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.CusumFlag))
        {
            double cusum = ComputeCusum(validValues, mean, std);
            double threshold = _cusumH * (std > 1e-10 ? std : 1);
            double flag = cusum > threshold ? 1.0 : 0.0;
            features[featureIdx++] = NumOps.FromDouble(flag);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.IsolationScore))
        {
            double isolationScore = ComputeIsolationScore(currentValue, validValues);
            features[featureIdx++] = NumOps.FromDouble(isolationScore);
        }

        if (_enabledFeatures.HasFlag(AnomalyFeatures.PercentileRank))
        {
            int countBelow = validValues.Count(v => v < currentValue);
            double percentileRank = (double)countBelow / validValues.Length;
            features[featureIdx++] = NumOps.FromDouble(percentileRank);
        }
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Exports transformer-specific parameters for serialization.
    /// </summary>
    protected override Dictionary<string, object> ExportParameters()
    {
        return new Dictionary<string, object>
        {
            ["EnabledFeatures"] = (int)_enabledFeatures,
            ["ZScoreThreshold"] = _zScoreThreshold,
            ["IqrMultiplier"] = _iqrMultiplier,
            ["CusumK"] = _cusumK,
            ["CusumH"] = _cusumH,
            ["OperationNames"] = _operationNames
        };
    }

    /// <summary>
    /// Imports transformer-specific parameters for validation.
    /// </summary>
    protected override void ImportParameters(Dictionary<string, object> parameters)
    {
        if (parameters.TryGetValue("EnabledFeatures", out var featuresObj))
        {
            int savedFeatures = Convert.ToInt32(featuresObj);
            if (savedFeatures != (int)_enabledFeatures)
            {
                throw new ArgumentException(
                    $"Saved EnabledFeatures ({savedFeatures}) does not match current configuration ({(int)_enabledFeatures}).");
            }
        }
    }

    #endregion

    #region Statistical Helpers

    private static double ComputeStdDev(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = values.Average();
        double sumSq = values.Select(x => (x - mean) * (x - mean)).Sum();
        return Math.Sqrt(sumSq / (values.Length - 1));
    }

    private static double ComputeMedian(double[] values)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 0
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
            : sorted[n / 2];
    }

    private static double ComputeMAD(double[] values, double median)
    {
        var deviations = values.Select(v => Math.Abs(v - median)).ToArray();
        return ComputeMedian(deviations);
    }

    private static double ComputePercentile(double[] sortedValues, double percentile)
    {
        if (sortedValues.Length == 0) return double.NaN;
        if (sortedValues.Length == 1) return sortedValues[0];

        double index = percentile * (sortedValues.Length - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper)
            return sortedValues[lower];

        return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * (index - lower);
    }

    /// <summary>
    /// Computes two-sided CUSUM statistic.
    /// </summary>
    private double ComputeCusum(double[] values, double mean, double std)
    {
        if (std < 1e-10) return 0;

        double cusumHigh = 0;
        double cusumLow = 0;

        foreach (double value in values)
        {
            double z = (value - mean) / std;
            cusumHigh = Math.Max(0, cusumHigh + z - _cusumK);
            cusumLow = Math.Max(0, cusumLow - z - _cusumK);
        }

        return Math.Max(cusumHigh, cusumLow);
    }

    /// <summary>
    /// Computes a simplified isolation score based on how easily a point can be isolated.
    /// </summary>
    /// <remarks>
    /// This is a simplified version that approximates isolation forest behavior
    /// using the statistical properties of the window rather than building actual trees.
    /// </remarks>
    private double ComputeIsolationScore(double currentValue, double[] values)
    {
        if (values.Length < 2) return 0.5;

        // Calculate isolation score based on deviation from distribution center
        // and how many values would need to be eliminated to isolate this point
        double min = values.Min();
        double max = values.Max();

        if (Math.Abs(max - min) < 1e-10) return 0.5; // No variation

        // Normalize value to [0, 1] range
        double normalized = (currentValue - min) / (max - min);

        // Score based on distance from center (0.5)
        // Points near edges get higher isolation scores
        double distanceFromCenter = Math.Abs(normalized - 0.5) * 2;

        // Count how many values are on the same side as current
        int sameDirection = 0;
        foreach (double v in values)
        {
            if ((currentValue > values.Average()) == (v > values.Average()))
                sameDirection++;
        }

        double proportionOnSameSide = (double)sameDirection / values.Length;

        // Combine factors: isolated points are far from center and have few neighbors
        double isolationScore = 0.5 + 0.5 * distanceFromCenter * (1 - proportionOnSameSide);

        return Math.Min(1.0, Math.Max(0.0, isolationScore));
    }

    #endregion
}
