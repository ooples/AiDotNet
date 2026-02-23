using System.Collections.Concurrent;

namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Collects telemetry data for deployed models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> TelemetryCollector provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class TelemetryCollector
{
    private readonly bool _enabled;
    private readonly ConcurrentBag<TelemetryEvent> _events;
    private readonly ConcurrentDictionary<string, ModelMetrics> _metrics;

    public TelemetryCollector(bool enabled = true)
    {
        _enabled = enabled;
        _events = new ConcurrentBag<TelemetryEvent>();
        _metrics = new ConcurrentDictionary<string, ModelMetrics>();
    }

    /// <summary>
    /// Records a telemetry event.
    /// </summary>
    public void RecordEvent(string eventName, Dictionary<string, object> properties)
    {
        if (!_enabled) return;

        var telemetryEvent = new TelemetryEvent
        {
            Name = eventName,
            Timestamp = DateTime.UtcNow,
            Properties = properties
        };

        _events.Add(telemetryEvent);
    }

    /// <summary>
    /// Records an inference execution.
    /// </summary>
    public void RecordInference(string modelName, string version, long latencyMs, bool fromCache)
    {
        if (!_enabled) return;

        var key = GetMetricsKey(modelName, version);
        var metrics = _metrics.GetOrAdd(key, _ => new ModelMetrics
        {
            ModelName = modelName,
            Version = version
        });

        lock (metrics)
        {
            Interlocked.Increment(ref metrics.TotalInferences);
            if (fromCache)
                Interlocked.Increment(ref metrics.CacheHits);
            Interlocked.Add(ref metrics.TotalLatencyMs, latencyMs);

            // Update min/max atomically
            long currentMin;
            do
            {
                currentMin = Interlocked.Read(ref metrics.MinLatencyMs);
                if (latencyMs >= currentMin) break;
            } while (Interlocked.CompareExchange(ref metrics.MinLatencyMs, latencyMs, currentMin) != currentMin);

            long currentMax;
            do
            {
                currentMax = Interlocked.Read(ref metrics.MaxLatencyMs);
                if (latencyMs <= currentMax) break;
            } while (Interlocked.CompareExchange(ref metrics.MaxLatencyMs, latencyMs, currentMax) != currentMax);

            metrics.LastInferenceTime = DateTime.UtcNow;
        }

        RecordEvent("Inference", new Dictionary<string, object>
        {
            ["ModelName"] = modelName,
            ["Version"] = version,
            ["LatencyMs"] = latencyMs,
            ["FromCache"] = fromCache
        });
    }

    /// <summary>
    /// Records an error.
    /// </summary>
    public void RecordError(string modelName, string version, Exception exception)
    {
        if (!_enabled) return;

        var key = GetMetricsKey(modelName, version);
        var metrics = _metrics.GetOrAdd(key, _ => new ModelMetrics
        {
            ModelName = modelName,
            Version = version
        });

        lock (metrics)
        {
            Interlocked.Increment(ref metrics.TotalErrors);
            metrics.LastError = exception.Message;
            metrics.LastErrorTime = DateTime.UtcNow;
        }

        RecordEvent("Error", new Dictionary<string, object>
        {
            ["ModelName"] = modelName,
            ["Version"] = version,
            ["ErrorMessage"] = exception.Message,
            ["ErrorType"] = exception.GetType().Name,
            ["StackTrace"] = exception.StackTrace ?? string.Empty
        });
    }

    /// <summary>
    /// Gets statistics for a model.
    /// Note: This method reads from metrics that may be concurrently updated. Results provide a snapshot view.
    /// </summary>
    public ModelStatistics GetStatistics(string modelName, string? version = null)
    {
        // Take a snapshot to avoid repeated enumeration during concurrent modifications
        var relevantMetrics = (version == null
            ? _metrics.Values.Where(m => m.ModelName == modelName)
            : _metrics.Values.Where(m => m.ModelName == modelName && m.Version == version)).ToList();

        var totalInferences = relevantMetrics.Sum(m => Interlocked.Read(ref m.TotalInferences));
        var totalErrors = relevantMetrics.Sum(m => Interlocked.Read(ref m.TotalErrors));
        var totalLatency = relevantMetrics.Sum(m => Interlocked.Read(ref m.TotalLatencyMs));
        var cacheHits = relevantMetrics.Sum(m => Interlocked.Read(ref m.CacheHits));

        // Total requests includes both successful inferences and errors
        var totalRequests = totalInferences + totalErrors;

        return new ModelStatistics
        {
            ModelName = modelName,
            Version = version,
            TotalInferences = totalInferences,
            TotalErrors = totalErrors,
            ErrorRate = totalRequests > 0 ? (double)totalErrors / totalRequests : 0.0,
            AverageLatencyMs = totalInferences > 0 ? (double)totalLatency / totalInferences : 0.0,
            MinLatencyMs = relevantMetrics.Any() ? relevantMetrics.Min(m => Interlocked.Read(ref m.MinLatencyMs)) : 0,
            MaxLatencyMs = relevantMetrics.Any() ? relevantMetrics.Max(m => Interlocked.Read(ref m.MaxLatencyMs)) : 0,
            CacheHitRate = totalInferences > 0 ? (double)cacheHits / totalInferences : 0.0,
            LastInferenceTime = relevantMetrics.Any()
                ? relevantMetrics.Max(m => m.LastInferenceTime)
                : DateTime.MinValue
        };
    }

    /// <summary>
    /// Gets the most recent recorded events ordered by timestamp descending.
    /// </summary>
    /// <param name="limit">Maximum number of events to return (default: 100)</param>
    /// <returns>List of telemetry events ordered from most recent to oldest</returns>
    public List<TelemetryEvent> GetEvents(int limit = 100)
    {
        // Order FIRST by timestamp descending, THEN take limit
        // ConcurrentBag doesn't guarantee ordering, so Take() before OrderBy() would return arbitrary items
        return _events.OrderByDescending(e => e.Timestamp).Take(limit).ToList();
    }

    /// <summary>
    /// Clears all telemetry data.
    /// Note: While ConcurrentBag.Clear() is thread-safe in .NET 5+, it may lose events added concurrently during clear.
    /// </summary>
    public void Clear()
    {
        // .NET Framework 462 doesn't have ConcurrentBag.Clear()
        while (_events.TryTake(out _)) { }
        _metrics.Clear();
    }

    private string GetMetricsKey(string modelName, string version) => $"{modelName}:{version}";
}

/// <summary>
/// Represents a telemetry event.
/// </summary>
public class TelemetryEvent
{
    public string Name { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public Dictionary<string, object> Properties { get; set; } = new();
}

/// <summary>
/// Internal metrics for a model version.
/// Thread-safe: Numeric fields are accessed via Interlocked operations. DateTime fields are protected by lock in RecordInference/RecordError.
/// </summary>
internal class ModelMetrics
{
    public string ModelName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public long TotalInferences;
    public long TotalErrors;
    public long TotalLatencyMs;
    public long MinLatencyMs = long.MaxValue;
    public long MaxLatencyMs;
    public long CacheHits;
    public DateTime LastInferenceTime { get; set; }
    public string? LastError { get; set; }
    public DateTime? LastErrorTime { get; set; }
}

/// <summary>
/// Statistics for a model.
/// </summary>
public class ModelStatistics
{
    public string ModelName { get; set; } = string.Empty;
    public string? Version { get; set; }
    public long TotalInferences { get; set; }
    public long TotalErrors { get; set; }
    public double ErrorRate { get; set; }
    public double AverageLatencyMs { get; set; }
    public long MinLatencyMs { get; set; }
    public long MaxLatencyMs { get; set; }
    public double CacheHitRate { get; set; }
    public DateTime LastInferenceTime { get; set; }
}
