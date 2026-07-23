using AiDotNet.FederatedLearning.Benchmarks.Leaf;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running LEAF-backed federated benchmark suites.
/// </summary>
/// <remarks>
/// <para>
/// This options class supplies the dataset context required to run the LEAF suite (train/test JSON split files)
/// while keeping the user-facing facade surface minimal.
/// </para>
/// <para><b>For Beginners:</b> LEAF datasets are stored as JSON files where each "user" corresponds to one federated client.
/// You provide the file paths, and AiDotNet loads and evaluates the model against the suite.
/// </para>
/// </remarks>
public sealed class LeafFederatedBenchmarkOptions
{
    /// <summary>
    /// Gets or sets the path to the LEAF train split JSON file.
    /// </summary>
    /// <remarks>
    /// This is required for suite execution because it defines the per-client partitioning.
    /// </remarks>
    public string? TrainFilePath { get; set; }

    /// <summary>
    /// Gets or sets the optional path to the LEAF test split JSON file.
    /// </summary>
    /// <remarks>
    /// If null, the suite will evaluate on the train split (CI/local-only convenience).
    /// </remarks>
    public string? TestFilePath { get; set; }

    /// <summary>
    /// Gets or sets load options controlling how many users/clients are loaded.
    /// </summary>
    public LeafFederatedDatasetLoadOptions LoadOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the maximum number of samples to use per user/client (null uses all available).
    /// </summary>
    /// <remarks>
    /// This is primarily used to keep benchmark runs fast and deterministic in CI mode.
    /// </remarks>
    public int? MaxSamplesPerUser { get; set; }
}

