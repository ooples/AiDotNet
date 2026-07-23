namespace AiDotNet.Benchmarking.Models;

/// <summary>
/// Represents a category-level metric result for benchmarks that support category breakdowns.
/// </summary>
public sealed class BenchmarkCategoryResult
{
    /// <summary>
    /// Gets the category name.
    /// </summary>
    public string Category { get; internal set; } = string.Empty;

    /// <summary>
    /// Gets the category accuracy as a value between 0.0 and 1.0.
    /// </summary>
    public double Accuracy { get; internal set; }
}

