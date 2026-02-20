namespace AiDotNet.Safety.Benchmarking;

/// <summary>
/// Comprehensive report from running all safety benchmarks.
/// </summary>
public class SafetyBenchmarkReport
{
    /// <summary>Individual benchmark results by benchmark name.</summary>
    public IReadOnlyDictionary<string, SafetyBenchmarkResult> BenchmarkResults { get; init; }
        = new Dictionary<string, SafetyBenchmarkResult>();

    /// <summary>Overall aggregate score across all benchmarks (0.0-1.0).</summary>
    public double AggregateScore { get; init; }

    /// <summary>Total test cases across all benchmarks.</summary>
    public int TotalTestCases { get; init; }

    /// <summary>Overall precision across all benchmarks.</summary>
    public double OverallPrecision { get; init; }

    /// <summary>Overall recall across all benchmarks.</summary>
    public double OverallRecall { get; init; }

    /// <summary>Overall F1 score across all benchmarks.</summary>
    public double OverallF1 { get; init; }

    /// <summary>Recommendations based on benchmark results.</summary>
    public IReadOnlyList<string> Recommendations { get; init; } = Array.Empty<string>();
}
