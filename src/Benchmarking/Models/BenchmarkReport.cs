using AiDotNet.Enums;

namespace AiDotNet.Benchmarking.Models;

/// <summary>
/// Represents a structured report from running one or more benchmark suites.
/// </summary>
public sealed class BenchmarkReport
{
    /// <summary>
    /// Gets the UTC time when the benchmark run started.
    /// </summary>
    public DateTimeOffset StartedUtc { get; internal set; }

    /// <summary>
    /// Gets the UTC time when the benchmark run ended.
    /// </summary>
    public DateTimeOffset EndedUtc { get; internal set; }

    /// <summary>
    /// Gets the total run duration.
    /// </summary>
    public TimeSpan TotalDuration => EndedUtc - StartedUtc;

    /// <summary>
    /// Gets the per-suite reports for this run.
    /// </summary>
    public IReadOnlyList<BenchmarkSuiteReport> Suites { get; internal set; } = Array.Empty<BenchmarkSuiteReport>();

    /// <summary>
    /// Gets the overall status for the run.
    /// </summary>
    public BenchmarkExecutionStatus OverallStatus
    {
        get
        {
            if (Suites.Count == 0)
            {
                return BenchmarkExecutionStatus.Skipped;
            }

            if (Suites.Any(x => x.Status == BenchmarkExecutionStatus.Failed))
            {
                return BenchmarkExecutionStatus.Failed;
            }

            if (Suites.All(x => x.Status == BenchmarkExecutionStatus.Skipped))
            {
                return BenchmarkExecutionStatus.Skipped;
            }

            return BenchmarkExecutionStatus.Succeeded;
        }
    }
}

