using AiDotNet.Enums;

namespace AiDotNet.Benchmarking.Models;

/// <summary>
/// Represents the outcome and metrics for a single benchmark suite run.
/// </summary>
public sealed class BenchmarkSuiteReport
{
    /// <summary>
    /// Gets the benchmark suite identifier.
    /// </summary>
    public BenchmarkSuite Suite { get; internal set; }

    /// <summary>
    /// Gets the suite kind/category.
    /// </summary>
    public BenchmarkSuiteKind Kind { get; internal set; }

    /// <summary>
    /// Gets the suite display name.
    /// </summary>
    public string Name { get; internal set; } = string.Empty;

    /// <summary>
    /// Gets the UTC time when this suite started.
    /// </summary>
    public DateTimeOffset StartedUtc { get; internal set; }

    /// <summary>
    /// Gets the UTC time when this suite ended.
    /// </summary>
    public DateTimeOffset EndedUtc { get; internal set; }

    /// <summary>
    /// Gets the duration for this suite.
    /// </summary>
    public TimeSpan Duration => EndedUtc - StartedUtc;

    /// <summary>
    /// Gets the execution status.
    /// </summary>
    public BenchmarkExecutionStatus Status { get; internal set; }

    /// <summary>
    /// Gets an optional failure reason when <see cref="Status"/> is <see cref="BenchmarkExecutionStatus.Failed"/>.
    /// </summary>
    public string? FailureReason { get; internal set; }

    /// <summary>
    /// Gets the standardized metrics for this suite.
    /// </summary>
    public IReadOnlyList<BenchmarkMetricValue> Metrics { get; internal set; } = Array.Empty<BenchmarkMetricValue>();

    /// <summary>
    /// Gets optional category-level accuracy breakdowns (when available and requested).
    /// </summary>
    public IReadOnlyList<BenchmarkCategoryResult>? CategoryAccuracies { get; internal set; }

    /// <summary>
    /// Gets optional dataset selection details for dataset-backed suites.
    /// </summary>
    public BenchmarkDataSelectionSummary? DataSelection { get; internal set; }
}
