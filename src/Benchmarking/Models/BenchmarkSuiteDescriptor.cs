using AiDotNet.Enums;

namespace AiDotNet.Benchmarking.Models;

/// <summary>
/// Describes a benchmark suite and how it is categorized within AiDotNet.
/// </summary>
public sealed class BenchmarkSuiteDescriptor
{
    /// <summary>
    /// Gets the benchmark suite identifier.
    /// </summary>
    public BenchmarkSuite Suite { get; internal set; }

    /// <summary>
    /// Gets the benchmark suite kind/category.
    /// </summary>
    public BenchmarkSuiteKind Kind { get; internal set; }

    /// <summary>
    /// Gets a stable display name for this suite.
    /// </summary>
    public string DisplayName { get; internal set; } = string.Empty;

    /// <summary>
    /// Gets a short description of what this suite measures.
    /// </summary>
    public string Description { get; internal set; } = string.Empty;
}

