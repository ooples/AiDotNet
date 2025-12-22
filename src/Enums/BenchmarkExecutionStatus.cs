namespace AiDotNet.Enums;

/// <summary>
/// Represents the execution outcome for a benchmark suite run.
/// </summary>
public enum BenchmarkExecutionStatus
{
    /// <summary>
    /// The benchmark suite completed successfully.
    /// </summary>
    Succeeded,

    /// <summary>
    /// The benchmark suite failed to complete.
    /// </summary>
    Failed,

    /// <summary>
    /// The benchmark suite was skipped due to configuration or environment constraints.
    /// </summary>
    Skipped
}

