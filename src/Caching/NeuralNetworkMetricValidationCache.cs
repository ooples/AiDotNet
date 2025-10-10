using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Caching;

/// <summary>
/// Provides a cache of valid metrics for neural network task types.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class remembers which evaluation metrics are appropriate for different
/// types of neural network tasks. It stores this information in memory so it doesn't have to be 
/// recalculated each time, making your code faster and more efficient.
/// </para>
/// </remarks>
public sealed class NeuralNetworkMetricValidationCache
{
    private static readonly Lazy<NeuralNetworkMetricValidationCache> _instance =
        new(() => new NeuralNetworkMetricValidationCache());

    private readonly ConcurrentDictionary<NeuralNetworkTaskType, HashSet<MetricType>> _validMetricsCache = default!;

    /// <summary>
    /// Gets the singleton instance of the NeuralNetworkMetricValidationCache.
    /// </summary>
    public static NeuralNetworkMetricValidationCache Instance => _instance.Value;

    private NeuralNetworkMetricValidationCache()
    {
        _validMetricsCache = new ConcurrentDictionary<NeuralNetworkTaskType, HashSet<MetricType>>();
        PreloadAllValidMetrics();
    }

    /// <summary>
    /// Preloads all valid metrics for all neural network task types.
    /// </summary>
    private void PreloadAllValidMetrics()
    {
        var allTaskTypes = Enum.GetValues(typeof(NeuralNetworkTaskType)).Cast<NeuralNetworkTaskType>();

        foreach (var taskType in allTaskTypes)
        {
            var validMetrics = NeuralNetworkMetricHelper.GetValidMetricTypes(taskType);
            _validMetricsCache[taskType] = validMetrics;
        }
    }

    /// <summary>
    /// Gets the valid metrics for a specific neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type to get valid metrics for.</param>
    /// <returns>A HashSet of valid metric types.</returns>
    public HashSet<MetricType> GetValidMetrics(NeuralNetworkTaskType taskType)
    {
        if (_validMetricsCache.TryGetValue(taskType, out var metrics))
        {
            // Return a copy to prevent modification of the cached set
            return new HashSet<MetricType>(metrics);
        }

        // Fallback: compute on demand if not found (shouldn't happen normally)
        return ComputeValidMetrics(taskType);
    }

    /// <summary>
    /// Gets the valid metrics for a specific neural network task type and statistics provider.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <param name="providerFilter">A function that filters metrics valid for the specific provider.</param>
    /// <returns>A HashSet of valid metric types.</returns>
    public HashSet<MetricType> GetValidMetrics(NeuralNetworkTaskType taskType, Func<MetricType, bool> providerFilter)
    {
        if (_validMetricsCache.TryGetValue(taskType, out var metrics))
        {
            // Apply the provider-specific filter
            return new HashSet<MetricType>(metrics.Where(providerFilter));
        }

        // Fallback: compute on demand if not found
        return ComputeValidMetrics(taskType, providerFilter);
    }

    /// <summary>
    /// Gets the primary metrics for a specific neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>An array of the most important metrics for evaluating the specified task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method gives you a smaller, focused list of the most important metrics 
    /// for evaluating your neural network. While many metrics may be valid, these are the ones you should 
    /// pay the most attention to when assessing how well your model is performing.
    /// </para>
    /// </remarks>
    public MetricType[] GetPrimaryMetrics(NeuralNetworkTaskType taskType)
    {
        return NeuralNetworkMetricHelper.GetPrimaryMetrics(taskType);
    }

    /// <summary>
    /// Checks if a specific metric is valid for a neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is valid for the task type; otherwise, false.</returns>
    public bool IsValidMetric(NeuralNetworkTaskType taskType, MetricType metricType)
    {
        if (_validMetricsCache.TryGetValue(taskType, out var metrics))
        {
            return metrics.Contains(metricType);
        }

        // Fallback to direct check
        return NeuralNetworkMetricHelper.IsValidMetricForTask(metricType, taskType);
    }

    /// <summary>
    /// Computes valid metrics for a neural network task type on demand.
    /// </summary>
    private static HashSet<MetricType> ComputeValidMetrics(NeuralNetworkTaskType taskType, Func<MetricType, bool>? providerFilter = null)
    {
        var validMetrics = NeuralNetworkMetricHelper.GetValidMetricTypes(taskType);

        if (providerFilter != null)
        {
            return [.. validMetrics.Where(providerFilter)];
        }

        return validMetrics;
    }

    /// <summary>
    /// Refreshes the cache for all neural network task types.
    /// Useful if NeuralNetworkMetricHelper logic changes.
    /// </summary>
    public void RefreshCache()
    {
        _validMetricsCache.Clear();
        PreloadAllValidMetrics();
    }

    /// <summary>
    /// Refreshes the cache for a specific neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type to refresh in the cache.</param>
    public void RefreshCache(NeuralNetworkTaskType taskType)
    {
        var validMetrics = ComputeValidMetrics(taskType);
        _validMetricsCache[taskType] = validMetrics;
    }

    /// <summary>
    /// Gets the metric groups appropriate for a neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>An array of metric groups appropriate for the specified task.</returns>
    public static MetricGroups[] GetMetricGroups(NeuralNetworkTaskType taskType)
    {
        return NeuralNetworkMetricHelper.GetMetricGroupsForTask(taskType);
    }

    /// <summary>
    /// Gets a user-friendly name for a neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>A descriptive name for the task type.</returns>
    public static string GetTaskName(NeuralNetworkTaskType taskType)
    {
        return NeuralNetworkMetricHelper.GetTaskName(taskType);
    }
}