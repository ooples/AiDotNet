using AiDotNet.FederatedLearning.Benchmarks.Leaf;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running the Shakespeare LEAF federated benchmark suite.
/// </summary>
/// <remarks>
/// <para>
/// Shakespeare is a federated next-character prediction benchmark. LEAF stores the dataset as JSON where each
/// "user" corresponds to a federated client and each sample is a fixed-length text window with the next character
/// as the label.
/// </para>
/// <para><b>For Beginners:</b> You provide the train/test JSON files, and AiDotNet loads the per-user partitions,
/// tokenizes the character sequences, and evaluates your model.
/// </para>
/// </remarks>
public sealed class ShakespeareFederatedBenchmarkOptions
{
    /// <summary>
    /// Gets or sets the path to the Shakespeare train split JSON file.
    /// </summary>
    public string? TrainFilePath { get; set; }

    /// <summary>
    /// Gets or sets the optional path to the Shakespeare test split JSON file.
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
    public int? MaxSamplesPerUser { get; set; }

    /// <summary>
    /// Gets or sets the fixed character window length used as model input.
    /// </summary>
    /// <remarks>
    /// LEAF uses 80 by default, but this can be smaller for CI or experimentation.
    /// </remarks>
    public int? SequenceLength { get; set; }
}

