namespace AiDotNet.Caching;

/// <summary>
/// Provides a thread-safe singleton cache for valid metrics per model type.
/// </summary>
public sealed class MetricValidationCache
{
    private static readonly Lazy<MetricValidationCache> _instance =
        new(() => new MetricValidationCache());

    private readonly ConcurrentDictionary<ModelType, HashSet<MetricType>> _validMetricsCache = default!;

    /// <summary>
    /// Gets the singleton instance of the MetricValidationCache.
    /// </summary>
    public static MetricValidationCache Instance => _instance.Value;

    private MetricValidationCache()
    {
        _validMetricsCache = new ConcurrentDictionary<ModelType, HashSet<MetricType>>();
        PreloadAllValidMetrics();
    }

    /// <summary>
    /// Preloads all valid metrics for all model types.
    /// </summary>
    private void PreloadAllValidMetrics()
    {
        var allModelTypes = Enum.GetValues(typeof(ModelType)).Cast<ModelType>();
        var allMetricTypes = Enum.GetValues(typeof(MetricType)).Cast<MetricType>();

        foreach (var modelType in allModelTypes)
        {
            var validMetrics = new HashSet<MetricType>();

            foreach (var metricType in allMetricTypes)
            {
                if (ModelTypeHelper.IsValidMetric(modelType, metricType))
                {
                    validMetrics.Add(metricType);
                }
            }

            _validMetricsCache[modelType] = validMetrics;
        }
    }

    /// <summary>
    /// Gets the valid metrics for a specific model type.
    /// </summary>
    /// <param name="modelType">The model type to get valid metrics for.</param>
    /// <returns>A HashSet of valid metric types.</returns>
    public HashSet<MetricType> GetValidMetrics(ModelType modelType)
    {
        if (_validMetricsCache.TryGetValue(modelType, out var metrics))
        {
            // Return a copy to prevent modification of the cached set
            return [];
        }

        // Fallback: compute on demand if not found (shouldn't happen normally)
        return ComputeValidMetrics(modelType);
    }

    /// <summary>
    /// Gets the valid metrics for a specific model type and statistics provider.
    /// </summary>
    /// <param name="modelType">The model type.</param>
    /// <param name="providerFilter">A function that filters metrics valid for the specific provider.</param>
    /// <returns>A HashSet of valid metric types.</returns>
    public HashSet<MetricType> GetValidMetrics(ModelType modelType, Func<MetricType, bool> providerFilter)
    {
        if (_validMetricsCache.TryGetValue(modelType, out var metrics))
        {
            // Apply the provider-specific filter
            return [.. metrics.Where(providerFilter)];
        }

        // Fallback: compute on demand if not found
        return ComputeValidMetrics(modelType, providerFilter);
    }

    /// <summary>
    /// Checks if a specific metric is valid for a model type.
    /// </summary>
    /// <param name="modelType">The model type.</param>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is valid for the model type; otherwise, false.</returns>
    public bool IsValidMetric(ModelType modelType, MetricType metricType)
    {
        if (_validMetricsCache.TryGetValue(modelType, out var metrics))
        {
            return metrics.Contains(metricType);
        }

        // Fallback to direct check
        return ModelTypeHelper.IsValidMetric(modelType, metricType);
    }

    /// <summary>
    /// Computes valid metrics for a model type on demand.
    /// </summary>
    private static HashSet<MetricType> ComputeValidMetrics(ModelType modelType, Func<MetricType, bool>? providerFilter = null)
    {
        var validMetrics = new HashSet<MetricType>();
        var allMetricTypes = Enum.GetValues(typeof(MetricType)).Cast<MetricType>();

        foreach (var metricType in allMetricTypes)
        {
            if (ModelTypeHelper.IsValidMetric(modelType, metricType) &&
                (providerFilter == null || providerFilter(metricType)))
            {
                validMetrics.Add(metricType);
            }
        }

        return validMetrics;
    }

    /// <summary>
    /// Refreshes the cache for all model types. Useful if ModelTypeHelper logic changes.
    /// </summary>
    public void RefreshCache()
    {
        _validMetricsCache.Clear();
        PreloadAllValidMetrics();
    }

    /// <summary>
    /// Refreshes the cache for a specific model type.
    /// </summary>
    public void RefreshCache(ModelType modelType)
    {
        var validMetrics = ComputeValidMetrics(modelType);
        _validMetricsCache[modelType] = validMetrics;
    }
}