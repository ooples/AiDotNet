namespace AiDotNet.Enums;

/// <summary>
/// Categorizes benchmark suites by their evaluation style and infrastructure requirements.
/// </summary>
public enum BenchmarkSuiteKind
{
    /// <summary>
    /// Reasoning-style benchmarks (prompt/agent driven evaluation).
    /// </summary>
    Reasoning,

    /// <summary>
    /// Dataset suites (supervised/unsupervised tasks backed by data loaders).
    /// </summary>
    DatasetSuite,

    /// <summary>
    /// System-level benchmarks (latency/throughput/communication and other runtime metrics).
    /// </summary>
    System
}

