namespace AiDotNet.Enums;

/// <summary>
/// Controls how the benchmark runner should behave when one or more suites fail.
/// </summary>
public enum BenchmarkFailurePolicy
{
    /// <summary>
    /// Stop at the first failure and throw immediately.
    /// </summary>
    FailFast,

    /// <summary>
    /// Run all requested suites, then throw an aggregate exception if any failed.
    /// </summary>
    ContinueAndThrowAggregate,

    /// <summary>
    /// Run all requested suites and return a report even if some fail.
    /// </summary>
    ContinueAndAttachReport
}

