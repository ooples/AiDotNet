namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Time Series Isolation Forest anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Time Series Isolation Forest extends the classic Isolation Forest algorithm to handle
/// temporal data by incorporating lag features, rolling statistics, and seasonal decomposition.
/// </para>
/// <para><b>For Beginners:</b> Isolation Forest works by randomly isolating observations.
/// Anomalies are easier to isolate because they are "few and different" - they end up
/// in shorter branches of the isolation trees.
///
/// For time series, we enhance this by considering:
/// - **Lag Features**: How the current value relates to recent past values
/// - **Rolling Statistics**: Moving averages, standard deviations
/// - **Seasonal Patterns**: Accounting for regular patterns like daily/weekly cycles
/// - **Trend**: Long-term direction of the data
///
/// This makes it effective for detecting:
/// - Sudden spikes or drops (point anomalies)
/// - Values that are unusual given the context (contextual anomalies)
/// - Unusual patterns over time (collective anomalies)
/// </para>
/// </remarks>
public class TimeSeriesIsolationForestOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public TimeSeriesIsolationForestOptions() { }

    /// <summary>
    /// Creates a copy of the specified options.
    /// </summary>
    public TimeSeriesIsolationForestOptions(TimeSeriesIsolationForestOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        NumTrees = other.NumTrees;
        SampleSize = other.SampleSize;
        MaxDepth = other.MaxDepth;
        ContaminationRate = other.ContaminationRate;
        LagFeatures = other.LagFeatures;
        RollingWindowSize = other.RollingWindowSize;
        UseSeasonalDecomposition = other.UseSeasonalDecomposition;
        SeasonalPeriod = other.SeasonalPeriod;
        UseTrendFeatures = other.UseTrendFeatures;
    }

    /// <summary>
    /// Gets or sets the number of isolation trees in the forest.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More trees generally give more stable anomaly scores,
    /// but take longer to build. 100-200 trees is usually sufficient.
    /// </para>
    /// </remarks>
    public int NumTrees { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of samples to use when building each tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A smaller sample size (e.g., 256) actually works better
    /// because anomalies are easier to isolate in smaller samples. This also makes the
    /// algorithm faster. If null, defaults to min(256, n_samples).
    /// </para>
    /// </remarks>
    public int? SampleSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the maximum depth of each isolation tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Limiting depth speeds up the algorithm without much
    /// accuracy loss. A depth of ceil(log2(sample_size)) is typically sufficient
    /// since anomalies are isolated early. If null, uses the theoretical limit.
    /// </para>
    /// </remarks>
    public int? MaxDepth { get; set; } = null;

    /// <summary>
    /// Gets or sets the expected proportion of anomalies in the dataset.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is used to set the threshold for what counts
    /// as an anomaly. A value of 0.1 means you expect about 10% of points to be anomalies.
    /// If unsure, start with 0.05-0.1 and adjust based on results.
    /// </para>
    /// </remarks>
    public double ContaminationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of lag features to include.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lag features capture how the current value relates
    /// to past values. For hourly data, lag=24 captures the daily pattern.
    /// For daily data, lag=7 captures the weekly pattern.
    /// </para>
    /// </remarks>
    public int LagFeatures { get; set; } = 10;

    /// <summary>
    /// Gets or sets the window size for rolling statistics (mean, std, min, max).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rolling statistics help detect if a value is
    /// unusual compared to its recent neighbors. A window of 10-30 is typical.
    /// </para>
    /// </remarks>
    public int RollingWindowSize { get; set; } = 20;

    /// <summary>
    /// Gets or sets whether to decompose the series into trend and seasonal components.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Seasonal decomposition helps detect anomalies
    /// that deviate from expected seasonal patterns. Enable this for data with
    /// clear daily, weekly, or yearly patterns.
    /// </para>
    /// </remarks>
    public bool UseSeasonalDecomposition { get; set; } = false;

    /// <summary>
    /// Gets or sets the seasonal period for decomposition.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of time steps in one seasonal cycle.
    /// For hourly data with daily patterns, use 24.
    /// For daily data with weekly patterns, use 7.
    /// </para>
    /// </remarks>
    public new int SeasonalPeriod { get; set; } = 24;

    /// <summary>
    /// Gets or sets whether to include trend-based features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Trend features help detect anomalies that deviate
    /// from the long-term direction of the data.
    /// </para>
    /// </remarks>
    public bool UseTrendFeatures { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; } = 42;
}
