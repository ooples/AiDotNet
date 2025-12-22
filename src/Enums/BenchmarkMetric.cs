namespace AiDotNet.Enums;

/// <summary>
/// Defines standardized metrics used in benchmark reports.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Benchmarks produce scores and measurements. This enum provides a type-safe,
/// standardized vocabulary for common metrics so we avoid "stringly-typed" metric keys.
/// </para>
/// </remarks>
public enum BenchmarkMetric
{
    /// <summary>
    /// Proportion of correct answers (0.0 to 1.0).
    /// </summary>
    Accuracy,

    /// <summary>
    /// Average confidence value (0.0 to 1.0) when available.
    /// </summary>
    AverageConfidence,

    /// <summary>
    /// Total number of items evaluated.
    /// </summary>
    TotalEvaluated,

    /// <summary>
    /// Number of correct items.
    /// </summary>
    CorrectCount,

    /// <summary>
    /// Total duration in milliseconds.
    /// </summary>
    TotalDurationMilliseconds,

    /// <summary>
    /// Average time per item in milliseconds.
    /// </summary>
    AverageTimePerItemMilliseconds,

    /// <summary>
    /// Mean squared error (regression-style error metric).
    /// </summary>
    MeanSquaredError,

    /// <summary>
    /// Root mean squared error (regression-style error metric).
    /// </summary>
    RootMeanSquaredError
}
