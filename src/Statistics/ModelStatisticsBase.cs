namespace AiDotNet.Statistics;

/// <summary>
/// Base class for model-based statistics providers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[Serializable]
public abstract class ModelStatisticsBase<T> : StatisticsBase<T>, IModelStatistics<T>
{
    /// <summary>
    /// Gets the number of features or parameters in the model.
    /// </summary>
    public int FeatureCount { get; }

    public new ModelType ModelType { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelStatisticsBase{T}"/> class.
    /// </summary>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="featureCount">The number of features or parameters in the model.</param>
    protected ModelStatisticsBase(ModelType modelType, int featureCount)
        : base(modelType)
    {
        ModelType = modelType;
        FeatureCount = featureCount;
    }

    /// <summary>
    /// Determines which metrics are valid for this model statistics provider.
    /// </summary>
    protected override void DetermineValidMetrics()
    {
        _validMetrics.Clear();
        var cache = MetricValidationCache.Instance;
        var modelMetrics = cache.GetValidMetrics(ModelType, IsProviderStatisticMetric);

        foreach (var metric in modelMetrics)
        {
            _validMetrics.Add(metric);
        }
    }

    /// <summary>
    /// Determines if a metric type is provided by this specific statistics provider.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is provided by this statistics class; otherwise, false.</returns>
    protected abstract bool IsProviderStatisticMetric(MetricType metricType);
}