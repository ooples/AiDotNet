namespace AiDotNet.Benchmarking.Models;

/// <summary>
/// Summarizes how data was selected/sampled for a dataset-backed benchmark suite run.
/// </summary>
public sealed class BenchmarkDataSelectionSummary
{
    /// <summary>
    /// Gets the number of clients/users included.
    /// </summary>
    public int ClientsUsed { get; internal set; }

    /// <summary>
    /// Gets the number of aggregated training samples included.
    /// </summary>
    public int TrainSamplesUsed { get; internal set; }

    /// <summary>
    /// Gets the number of aggregated test samples included.
    /// </summary>
    public int TestSamplesUsed { get; internal set; }

    /// <summary>
    /// Gets the feature count for the aggregated dataset.
    /// </summary>
    public int FeatureCount { get; internal set; }

    /// <summary>
    /// Gets whether CI mode was enabled for this run.
    /// </summary>
    public bool CiMode { get; internal set; }

    /// <summary>
    /// Gets the seed used for deterministic sampling when applicable.
    /// </summary>
    public int Seed { get; internal set; }

    /// <summary>
    /// Gets the maximum samples per user applied (0 means not applied).
    /// </summary>
    public int MaxSamplesPerUser { get; internal set; }
}

