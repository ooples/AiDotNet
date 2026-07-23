using AiDotNet.Enums;

namespace AiDotNet.Benchmarking.Models;

/// <summary>
/// Represents a single metric value in a benchmark report.
/// </summary>
public sealed class BenchmarkMetricValue
{
    /// <summary>
    /// Gets the metric identifier.
    /// </summary>
    public BenchmarkMetric Metric { get; internal set; }

    /// <summary>
    /// Gets the metric value.
    /// </summary>
    public double Value { get; internal set; }
}

