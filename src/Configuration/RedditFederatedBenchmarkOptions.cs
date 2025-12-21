using AiDotNet.FederatedLearning.Benchmarks.Leaf;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running the Reddit federated benchmark suite.
/// </summary>
/// <remarks>
/// <para>
/// Reddit is a large-scale federated text benchmark. This suite uses a token-sequence formulation (next-token
/// prediction) and evaluates models without exposing model internals.
/// </para>
/// <para><b>For Beginners:</b> You provide the train/test JSON files, and AiDotNet loads the per-user partitions,
/// builds a vocabulary (with safe defaults), and evaluates your model on a standardized next-token task.
/// </para>
/// </remarks>
public sealed class RedditFederatedBenchmarkOptions
{
    /// <summary>
    /// Gets or sets the path to the Reddit train split JSON file.
    /// </summary>
    public string? TrainFilePath { get; set; }

    /// <summary>
    /// Gets or sets the optional path to the Reddit test split JSON file.
    /// </summary>
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
    /// Gets or sets the fixed token sequence length used as model input.
    /// </summary>
    /// <remarks>
    /// If null, AiDotNet infers the length from the dataset and validates consistency.
    /// </remarks>
    public int? SequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the maximum vocabulary size used for token-to-ID mapping.
    /// </summary>
    /// <remarks>
    /// If null, AiDotNet uses a sensible default (smaller in CI mode).
    /// </remarks>
    public int? MaxVocabularySize { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of sequences used to build the default vocabulary.
    /// </summary>
    /// <remarks>
    /// If null, AiDotNet uses a sensible default (smaller in CI mode).
    /// </remarks>
    public int? VocabularyTrainingSampleCount { get; set; }
}

