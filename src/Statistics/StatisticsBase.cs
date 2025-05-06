namespace AiDotNet.Statistics;

/// <summary>
/// Base class for all statistics providers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[Serializable]
public abstract class StatisticsBase<T> : IStatisticsProvider<T>
{
    /// <summary>
    /// Provides mathematical operations for the generic type T.
    /// </summary>
    protected readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Dictionary to store calculated metrics.
    /// </summary>
    protected readonly Dictionary<MetricType, T> _metrics = [];

    /// <summary>
    /// Set of metrics that have been calculated.
    /// </summary>
    protected readonly HashSet<MetricType> _calculatedMetrics = [];

    /// <summary>
    /// Set of metrics valid for this statistics provider.
    /// </summary>
    protected readonly HashSet<MetricType> _validMetrics = [];

    /// <summary>
    /// The type of model being evaluated.
    /// </summary>
    protected readonly ModelType ModelType;

    /// <summary>
    /// Initializes a new instance of the <see cref="StatisticsBase{T}"/> class.
    /// </summary>
    protected StatisticsBase(ModelType modelType)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        ModelType = modelType;

        // Determine which metrics are valid for this statistics provider
        DetermineValidMetrics();
    }

    /// <summary>
    /// Determines which metrics are valid for this statistics provider.
    /// </summary>
    protected abstract void DetermineValidMetrics();

    /// <summary>
    /// Gets the value of a specific metric.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <returns>The value of the requested metric.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the requested metric is not valid.</exception>
    public virtual T GetMetric(MetricType metricType)
    {
        return _metrics.TryGetValue(metricType, out var value) ? value : _numOps.Zero;
    }

    /// <summary>
    /// Checks if a specific metric is valid for this statistics provider.
    /// </summary>
    /// <param name="metricType">The type of metric to check.</param>
    /// <returns>True if the metric is valid; otherwise, false.</returns>
    public virtual bool IsValidMetric(MetricType metricType)
    {
        return _validMetrics.Contains(metricType);
    }

    /// <summary>
    /// Checks if a specific metric has been calculated.
    /// </summary>
    /// <param name="metricType">The type of metric to check.</param>
    /// <returns>True if the metric has been calculated; otherwise, false.</returns>
    public virtual bool IsCalculatedMetric(MetricType metricType)
    {
        return _calculatedMetrics.Contains(metricType);
    }

    /// <summary>
    /// Gets all metric types that are valid for this statistics provider.
    /// </summary>
    /// <returns>An array of valid metric types.</returns>
    public virtual MetricType[] GetValidMetricTypes()
    {
        return [.. _validMetrics];
    }

    /// <summary>
    /// Gets all metric types that have been calculated.
    /// </summary>
    /// <returns>An array of calculated metric types.</returns>
    public virtual MetricType[] GetCalculatedMetricTypes()
    {
        return [.. _calculatedMetrics];
    }

    /// <summary>
    /// Gets a dictionary of all calculated metrics.
    /// </summary>
    /// <returns>A dictionary mapping metric types to their values.</returns>
    public virtual Dictionary<MetricType, T> GetAllCalculatedMetrics()
    {
        return new Dictionary<MetricType, T>(_metrics);
    }
}