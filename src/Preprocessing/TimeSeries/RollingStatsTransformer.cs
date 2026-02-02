using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Computes rolling statistics over time series data for feature engineering.
/// </summary>
/// <remarks>
/// <para>
/// This transformer calculates various statistical measures over rolling windows,
/// creating features that capture local patterns in time series data.
/// </para>
/// <para><b>For Beginners:</b> Rolling statistics are like taking a moving snapshot of your data.
///
/// For example, with a 7-day rolling mean:
/// - Day 7: Average of days 1-7
/// - Day 8: Average of days 2-8
/// - Day 9: Average of days 3-9
///
/// This helps capture trends and patterns at different time scales. Common uses include:
/// - Smoothing noisy data (rolling mean)
/// - Detecting volatility changes (rolling std deviation)
/// - Finding extreme values (rolling min/max)
/// - Understanding distribution changes (rolling skewness/kurtosis)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RollingStatsTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// Cached operation names for feature naming (readonly for thread safety).
    /// </summary>
    private readonly string[] _operationNames;

    /// <summary>
    /// The enabled statistics from options.
    /// </summary>
    private readonly RollingStatistics _enabledStats;

    /// <summary>
    /// Custom percentiles to compute.
    /// </summary>
    private readonly double[] _percentiles;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new rolling statistics transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public RollingStatsTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _enabledStats = Options.EnabledStatistics;
        _percentiles = Options.CustomPercentiles;
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
        // Rolling stats don't need to learn parameters from data
        // All computation happens in Transform
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        // Determine output dimensions based on edge handling
        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();

        // Handle Truncate mode with empty output
        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        // Create output tensor
        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        // Track first valid index for forward fill
        int firstValidIndex = -1;
        int maxWindow = GetMaxWindowSize();

        // For each output time step
        for (int outT = 0; outT < outputTimeSteps; outT++)
        {
            int t = outT + startIndex; // Map to input time step
            int outputIdx = 0;

            // For each window size
            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                // Handle edge cases based on EdgeHandling mode
                if (isEdge)
                {
                    switch (Options.EdgeHandling)
                    {
                        case EdgeHandling.NaN:
                            // Fill with NaN for incomplete windows
                            int statsCount = CountEnabledStats() * inputFeatures;
                            for (int i = 0; i < statsCount; i++)
                            {
                                output[outT, outputIdx++] = GetNaN();
                            }
                            continue;

                        case EdgeHandling.Partial:
                        case EdgeHandling.ForwardFill:
                            // Use partial window - continue to compute with available data
                            break;

                        case EdgeHandling.Truncate:
                            // Should not reach here - startIndex skips edge region
                            break;
                    }
                }

                // Track first fully valid index for forward fill
                if (firstValidIndex < 0 && !IsEdgeRegion(t, maxWindow))
                {
                    firstValidIndex = outT;
                }

                // For each input feature
                for (int f = 0; f < inputFeatures; f++)
                {
                    // Extract window data (may be partial)
                    int effectiveWindow = ShouldComputePartialWindows() && isEdge
                        ? GetEffectiveWindowSize(t, windowSize)
                        : windowSize;
                    var windowData = ExtractWindowWithSize(data, t, f, windowSize, effectiveWindow);

                    // Compute enabled statistics
                    ComputeStatistics(windowData, output, outT, ref outputIdx);
                }
            }
        }

        // Apply forward fill if needed
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

        // Determine output dimensions based on edge handling
        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();
        int maxWindow = GetMaxWindowSize();

        // Handle Truncate mode with empty output
        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        // Create output tensor
        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        // Track first valid index for forward fill (thread-safe)
        int firstValidIndex = -1;
        object lockObj = new object();

        // Process time steps in parallel
        Parallel.For(0, outputTimeSteps, outT =>
        {
            int t = outT + startIndex; // Map to input time step
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                // Handle edge cases based on EdgeHandling mode
                if (isEdge)
                {
                    switch (Options.EdgeHandling)
                    {
                        case EdgeHandling.NaN:
                            int statsCount = CountEnabledStats() * inputFeatures;
                            for (int i = 0; i < statsCount; i++)
                            {
                                output[outT, outputIdx++] = GetNaN();
                            }
                            continue;

                        case EdgeHandling.Partial:
                        case EdgeHandling.ForwardFill:
                            // Use partial window
                            break;

                        case EdgeHandling.Truncate:
                            break;
                    }
                }

                // Track first valid index for forward fill (thread-safe)
                if (!IsEdgeRegion(t, maxWindow))
                {
                    lock (lockObj)
                    {
                        if (firstValidIndex < 0 || outT < firstValidIndex)
                            firstValidIndex = outT;
                    }
                }

                for (int f = 0; f < inputFeatures; f++)
                {
                    int effectiveWindow = ShouldComputePartialWindows() && isEdge
                        ? GetEffectiveWindowSize(t, windowSize)
                        : windowSize;
                    var windowData = ExtractWindowWithSize(data, t, f, windowSize, effectiveWindow);
                    int localIdx = outputIdx;
                    ComputeStatisticsToArray(windowData, output, outT, localIdx);
                    outputIdx += CountEnabledStats();
                }
            }
        });

        // Apply forward fill if needed (sequential post-processing)
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
                    names.Add($"{inputName}{sep}rolling{sep}{op}{sep}{windowSize}");
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

    /// <summary>
    /// Builds the list of operation names based on enabled statistics.
    /// </summary>
    private string[] BuildOperationNames()
    {
        var ops = new List<string>();

        if (_enabledStats.HasFlag(RollingStatistics.Mean)) ops.Add("mean");
        if (_enabledStats.HasFlag(RollingStatistics.Median)) ops.Add("median");
        if (_enabledStats.HasFlag(RollingStatistics.StandardDeviation)) ops.Add("std");
        if (_enabledStats.HasFlag(RollingStatistics.Variance)) ops.Add("var");
        if (_enabledStats.HasFlag(RollingStatistics.Min)) ops.Add("min");
        if (_enabledStats.HasFlag(RollingStatistics.Max)) ops.Add("max");
        if (_enabledStats.HasFlag(RollingStatistics.Sum)) ops.Add("sum");
        if (_enabledStats.HasFlag(RollingStatistics.Count)) ops.Add("count");
        if (_enabledStats.HasFlag(RollingStatistics.Range)) ops.Add("range");
        if (_enabledStats.HasFlag(RollingStatistics.Skewness)) ops.Add("skew");
        if (_enabledStats.HasFlag(RollingStatistics.Kurtosis)) ops.Add("kurt");
        if (_enabledStats.HasFlag(RollingStatistics.IQR)) ops.Add("iqr");
        if (_enabledStats.HasFlag(RollingStatistics.MAD)) ops.Add("mad");
        if (_enabledStats.HasFlag(RollingStatistics.FirstQuartile)) ops.Add("q1");
        if (_enabledStats.HasFlag(RollingStatistics.ThirdQuartile)) ops.Add("q3");

        // Add custom percentiles
        foreach (double p in _percentiles)
        {
            ops.Add($"p{(int)(p * 100)}");
        }

        return [.. ops];
    }

    /// <summary>
    /// Counts the number of enabled statistics.
    /// </summary>
    private int CountEnabledStats()
    {
        return _operationNames.Length;
    }

    #endregion

    #region Window Extraction

    /// <summary>
    /// Extracts data for a rolling window ending at the specified time step.
    /// </summary>
    /// <param name="data">The source data.</param>
    /// <param name="endTime">The end time step (inclusive).</param>
    /// <param name="feature">The feature index.</param>
    /// <param name="windowSize">The window size.</param>
    /// <returns>Array of values in the window, or null values for missing data.</returns>
    private double[] ExtractWindow(Tensor<T> data, int endTime, int feature, int windowSize)
    {
        return ExtractWindowWithSize(data, endTime, feature, windowSize, windowSize);
    }

    /// <summary>
    /// Extracts data for a rolling window with support for partial windows.
    /// </summary>
    /// <param name="data">The source data.</param>
    /// <param name="endTime">The end time step (inclusive).</param>
    /// <param name="feature">The feature index.</param>
    /// <param name="requestedSize">The requested window size.</param>
    /// <param name="effectiveSize">The actual size to extract (may be smaller for partial windows).</param>
    /// <returns>Array of values in the window.</returns>
    private double[] ExtractWindowWithSize(Tensor<T> data, int endTime, int feature, int requestedSize, int effectiveSize)
    {
        // For partial windows, we only extract available data
        int actualSize = Math.Min(effectiveSize, endTime + 1);
        actualSize = Math.Max(1, actualSize); // At least 1 element

        int startTime = endTime - actualSize + 1;
        var window = new double[actualSize];

        for (int i = 0; i < actualSize; i++)
        {
            int t = startTime + i;
            window[i] = t < 0
                ? double.NaN // Edge case: before data starts
                : NumOps.ToDouble(GetValue(data, t, feature));
        }

        return window;
    }

    #endregion

    #region Statistics Computation

    /// <summary>
    /// Computes all enabled statistics for a window and writes to output.
    /// </summary>
    private void ComputeStatistics(double[] windowData, Tensor<T> output, int timeStep, ref int outputIdx)
    {
        // Filter out NaN values for statistics
        var validData = windowData.Where(x => !double.IsNaN(x)).ToArray();

        bool hasValidData = validData.Length > 0;
        bool hasEnoughData = validData.Length >= 2;

        // Compute statistics in order matching _operationNames
        if (_enabledStats.HasFlag(RollingStatistics.Mean))
        {
            double val = hasValidData ? validData.Average() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Median))
        {
            double val = hasValidData ? ComputeMedian(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.StandardDeviation))
        {
            double val = hasEnoughData ? ComputeStdDev(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Variance))
        {
            double val = hasEnoughData ? ComputeVariance(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Min))
        {
            double val = hasValidData ? validData.Min() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Max))
        {
            double val = hasValidData ? validData.Max() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Sum))
        {
            double val = hasValidData ? validData.Sum() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Count))
        {
            double val = validData.Length;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Range))
        {
            double val = hasValidData ? validData.Max() - validData.Min() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Skewness))
        {
            double val = validData.Length >= 3 ? ComputeSkewness(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Kurtosis))
        {
            double val = validData.Length >= 4 ? ComputeKurtosis(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.IQR))
        {
            double val = hasValidData ? ComputeIQR(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.MAD))
        {
            double val = hasValidData ? ComputeMAD(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.FirstQuartile))
        {
            double val = hasValidData ? ComputePercentile(validData, 0.25) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.ThirdQuartile))
        {
            double val = hasValidData ? ComputePercentile(validData, 0.75) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        // Custom percentiles
        foreach (double p in _percentiles)
        {
            double val = hasValidData ? ComputePercentile(validData, p) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }
    }

    /// <summary>
    /// Thread-safe version that computes stats starting at a specific output index.
    /// </summary>
    private void ComputeStatisticsToArray(double[] windowData, Tensor<T> output, int timeStep, int startIdx)
    {
        int outputIdx = startIdx;
        var validData = windowData.Where(x => !double.IsNaN(x)).ToArray();

        bool hasValidData = validData.Length > 0;
        bool hasEnoughData = validData.Length >= 2;

        if (_enabledStats.HasFlag(RollingStatistics.Mean))
        {
            double val = hasValidData ? validData.Average() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Median))
        {
            double val = hasValidData ? ComputeMedian(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.StandardDeviation))
        {
            double val = hasEnoughData ? ComputeStdDev(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Variance))
        {
            double val = hasEnoughData ? ComputeVariance(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Min))
        {
            double val = hasValidData ? validData.Min() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Max))
        {
            double val = hasValidData ? validData.Max() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Sum))
        {
            double val = hasValidData ? validData.Sum() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Count))
        {
            double val = validData.Length;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Range))
        {
            double val = hasValidData ? validData.Max() - validData.Min() : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Skewness))
        {
            double val = validData.Length >= 3 ? ComputeSkewness(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Kurtosis))
        {
            double val = validData.Length >= 4 ? ComputeKurtosis(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.IQR))
        {
            double val = hasValidData ? ComputeIQR(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.MAD))
        {
            double val = hasValidData ? ComputeMAD(validData) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.FirstQuartile))
        {
            double val = hasValidData ? ComputePercentile(validData, 0.25) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.ThirdQuartile))
        {
            double val = hasValidData ? ComputePercentile(validData, 0.75) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }

        foreach (double p in _percentiles)
        {
            double val = hasValidData ? ComputePercentile(validData, p) : double.NaN;
            output[timeStep, outputIdx++] = NumOps.FromDouble(val);
        }
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Computes rolling statistics features incrementally from the circular buffer.
    /// </summary>
    /// <param name="state">The current incremental state containing the rolling buffer.</param>
    /// <param name="newDataPoint">The new data point that was just added.</param>
    /// <returns>Array of computed feature values matching batch computation.</returns>
    protected override T[] ComputeIncrementalFeatures(IncrementalState<T> state, T[] newDataPoint)
    {
        var features = new T[OutputFeatureCount];
        int featureIdx = 0;

        // For each window size
        foreach (int windowSize in WindowSizes)
        {
            // For each input feature
            for (int f = 0; f < InputFeatureCount; f++)
            {
                // Extract window data from the circular buffer
                var windowData = ExtractIncrementalWindow(state, f, windowSize);

                // Compute all enabled statistics and add to features array
                ComputeStatisticsIncremental(windowData, features, ref featureIdx);
            }
        }

        return features;
    }

    /// <summary>
    /// Computes all enabled statistics for a window and writes to the features array.
    /// This is the incremental version of ComputeStatistics.
    /// </summary>
    /// <param name="windowData">The window of data to compute statistics for.</param>
    /// <param name="features">The output features array.</param>
    /// <param name="featureIdx">The current index in the features array (will be incremented).</param>
    private void ComputeStatisticsIncremental(double[] windowData, T[] features, ref int featureIdx)
    {
        // Filter out NaN values for statistics
        var validData = windowData.Where(x => !double.IsNaN(x)).ToArray();

        bool hasValidData = validData.Length > 0;
        bool hasEnoughData = validData.Length >= 2;

        // Compute statistics in order matching _operationNames
        if (_enabledStats.HasFlag(RollingStatistics.Mean))
        {
            double val = hasValidData ? validData.Average() : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Median))
        {
            double val = hasValidData ? ComputeMedian(validData) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.StandardDeviation))
        {
            double val = hasEnoughData ? ComputeStdDev(validData) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Variance))
        {
            double val = hasEnoughData ? ComputeVariance(validData) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Min))
        {
            double val = hasValidData ? validData.Min() : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Max))
        {
            double val = hasValidData ? validData.Max() : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Sum))
        {
            double val = hasValidData ? validData.Sum() : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Count))
        {
            double val = validData.Length;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Range))
        {
            double val = hasValidData ? validData.Max() - validData.Min() : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Skewness))
        {
            double val = validData.Length >= 3 ? ComputeSkewness(validData) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.Kurtosis))
        {
            double val = validData.Length >= 4 ? ComputeKurtosis(validData) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.IQR))
        {
            double val = hasValidData ? ComputeIQR(validData) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.MAD))
        {
            double val = hasValidData ? ComputeMAD(validData) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.FirstQuartile))
        {
            double val = hasValidData ? ComputePercentile(validData, 0.25) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledStats.HasFlag(RollingStatistics.ThirdQuartile))
        {
            double val = hasValidData ? ComputePercentile(validData, 0.75) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        // Custom percentiles
        foreach (double p in _percentiles)
        {
            double val = hasValidData ? ComputePercentile(validData, p) : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Exports transformer-specific parameters for serialization.
    /// </summary>
    /// <returns>Dictionary containing enabled statistics and percentiles.</returns>
    protected override Dictionary<string, object> ExportParameters()
    {
        return new Dictionary<string, object>
        {
            ["EnabledStatistics"] = (int)_enabledStats,
            ["Percentiles"] = _percentiles,
            ["OperationNames"] = _operationNames
        };
    }

    /// <summary>
    /// Imports transformer-specific parameters. Note: For RollingStatsTransformer,
    /// the core parameters (_enabledStats, _percentiles, _operationNames) are readonly
    /// fields set during construction. Full state restoration requires creating a new
    /// instance with the appropriate options. This method validates compatibility.
    /// </summary>
    /// <param name="parameters">The parameters from serialized state.</param>
    protected override void ImportParameters(Dictionary<string, object> parameters)
    {
        if (parameters.TryGetValue("EnabledStatistics", out var statsObj))
        {
            int savedStats = Convert.ToInt32(statsObj);
            if (savedStats != (int)_enabledStats)
            {
                throw new ArgumentException(
                    $"Saved EnabledStatistics ({savedStats}) does not match current configuration ({(int)_enabledStats}). " +
                    "Create a new transformer with matching options.");
            }
        }

        if (parameters.TryGetValue("Percentiles", out var percentilesObj))
        {
            var savedPercentiles = percentilesObj switch
            {
                double[] arr => arr,
                IEnumerable<object> enumerable => enumerable.Select(x => Convert.ToDouble(x)).ToArray(),
                _ => []
            };

            if (!savedPercentiles.SequenceEqual(_percentiles))
            {
                throw new ArgumentException(
                    "Saved Percentiles do not match current configuration. " +
                    "Create a new transformer with matching options.");
            }
        }
    }

    #endregion

    #region Statistical Functions

    /// <summary>
    /// Computes the median of the values.
    /// </summary>
    private static double ComputeMedian(double[] values)
    {
        var sorted = values.OrderBy(x => x).ToArray();
        int n = sorted.Length;

        return n % 2 == 1
            ? sorted[n / 2]
            : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    /// <summary>
    /// Computes the sample variance.
    /// </summary>
    private static double ComputeVariance(double[] values)
    {
        double mean = values.Average();
        double sumSq = values.Select(x => (x - mean) * (x - mean)).Sum();
        return sumSq / (values.Length - 1); // Sample variance (Bessel's correction)
    }

    /// <summary>
    /// Computes the sample standard deviation.
    /// </summary>
    private static double ComputeStdDev(double[] values)
    {
        return Math.Sqrt(ComputeVariance(values));
    }

    /// <summary>
    /// Computes the skewness (asymmetry measure).
    /// </summary>
    private static double ComputeSkewness(double[] values)
    {
        int n = values.Length;
        double mean = values.Average();
        double std = ComputeStdDev(values);

        if (std < 1e-10) return 0;

        double sumCubed = values.Select(x => Math.Pow((x - mean) / std, 3)).Sum();

        // Fisher's skewness with sample adjustment
        // Cast to double early to avoid integer overflow
        return (n / ((n - 1.0) * (n - 2.0))) * sumCubed;
    }

    /// <summary>
    /// Computes the excess kurtosis (tail heaviness measure).
    /// </summary>
    private static double ComputeKurtosis(double[] values)
    {
        int n = values.Length;
        double mean = values.Average();
        double std = ComputeStdDev(values);

        if (std < 1e-10) return 0;

        double sumFourth = values.Select(x => Math.Pow((x - mean) / std, 4)).Sum();

        // Fisher's excess kurtosis with sample adjustment
        // Use double literals throughout to avoid integer overflow in multiplications
        double k = (double)n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * sumFourth;
        double adjustment = 3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));

        return k - adjustment;
    }

    /// <summary>
    /// Computes a percentile value.
    /// </summary>
    private static double ComputePercentile(double[] values, double percentile)
    {
        var sorted = values.OrderBy(x => x).ToArray();
        int n = sorted.Length;

        if (n == 1) return sorted[0];

        double rank = percentile * (n - 1);
        int lower = (int)Math.Floor(rank);
        int upper = (int)Math.Ceiling(rank);

        double weight = rank - lower;
        return lower == upper
            ? sorted[lower]
            : sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    /// <summary>
    /// Computes the interquartile range (Q3 - Q1).
    /// </summary>
    private static double ComputeIQR(double[] values)
    {
        return ComputePercentile(values, 0.75) - ComputePercentile(values, 0.25);
    }

    /// <summary>
    /// Computes the median absolute deviation.
    /// </summary>
    private static double ComputeMAD(double[] values)
    {
        double median = ComputeMedian(values);
        var absDeviations = values.Select(x => Math.Abs(x - median)).ToArray();
        return ComputeMedian(absDeviations);
    }

    #endregion
}
