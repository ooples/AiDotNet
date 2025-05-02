namespace AiDotNet.Statistics;

/// <summary>
/// Base interface for all statistics providers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IStatisticsProvider<T>
{
    /// <summary>
    /// Gets the value of a specific metric.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <returns>The value of the requested metric.</returns>
    T GetMetric(MetricType metricType);

    /// <summary>
    /// Checks if a specific metric is valid for this statistics provider.
    /// </summary>
    /// <param name="metricType">The type of metric to check.</param>
    /// <returns>True if the metric is valid; otherwise, false.</returns>
    bool IsValidMetric(MetricType metricType);

    /// <summary>
    /// Checks if a specific metric has been calculated.
    /// </summary>
    /// <param name="metricType">The type of metric to check.</param>
    /// <returns>True if the metric has been calculated; otherwise, false.</returns>
    bool IsCalculatedMetric(MetricType metricType);

    /// <summary>
    /// Gets all metric types that are valid for this statistics provider.
    /// </summary>
    /// <returns>An array of valid metric types.</returns>
    MetricType[] GetValidMetricTypes();

    /// <summary>
    /// Gets all metric types that have been calculated.
    /// </summary>
    /// <returns>An array of calculated metric types.</returns>
    MetricType[] GetCalculatedMetricTypes();

    /// <summary>
    /// Gets a dictionary of all calculated metrics.
    /// </summary>
    /// <returns>A dictionary mapping metric types to their values.</returns>
    Dictionary<MetricType, T> GetAllCalculatedMetrics();
}

/// <summary>
/// Interface for model-based statistics providers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IModelStatistics<T> : IStatisticsProvider<T>
{
    /// <summary>
    /// Gets the type of model being evaluated.
    /// </summary>
    ModelType ModelType { get; }

    /// <summary>
    /// Gets the number of features or parameters in the model.
    /// </summary>
    int FeatureCount { get; }
}

/// <summary>
/// Interface for statistics providers that support prediction intervals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IIntervalProvider<T>
{
    /// <summary>
    /// Gets an interval by type.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <returns>The interval as a tuple of (Lower, Upper) bounds.</returns>
    (T Lower, T Upper) GetInterval(IntervalType intervalType);

    /// <summary>
    /// Tries to get a specific interval.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <param name="interval">The interval as a tuple of (Lower, Upper) bounds if successful.</param>
    /// <returns>True if the interval was successfully retrieved; otherwise, false.</returns>
    bool TryGetInterval(IntervalType intervalType, out (T Lower, T Upper) interval);

    /// <summary>
    /// Checks if a specific interval is valid for this provider.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval is valid; otherwise, false.</returns>
    bool IsValidInterval(IntervalType intervalType);

    /// <summary>
    /// Checks if a specific interval has been calculated.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval has been calculated; otherwise, false.</returns>
    bool IsCalculatedInterval(IntervalType intervalType);

    /// <summary>
    /// Gets all interval types that are valid for this provider.
    /// </summary>
    /// <returns>An array of valid interval types.</returns>
    IntervalType[] GetValidIntervalTypes();

    /// <summary>
    /// Gets all interval types that have been calculated.
    /// </summary>
    /// <returns>An array of calculated interval types.</returns>
    IntervalType[] GetCalculatedIntervalTypes();
}

/// <summary>
/// Interface for prediction-based statistics that combine model statistics and intervals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IPredictionStatistics<T> : IModelStatistics<T>, IIntervalProvider<T>
{
    /// <summary>
    /// Gets the type of prediction (regression, classification, etc.).
    /// </summary>
    PredictionType PredictionType { get; }
}