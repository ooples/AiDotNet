using AiDotNet.FederatedLearning.Benchmarks.Leaf;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running the Sent140 LEAF federated benchmark suite.
/// </summary>
/// <remarks>
/// <para>
/// Sent140 is a federated sentiment classification benchmark derived from tweets. LEAF stores the dataset as JSON
/// where each "user" corresponds to a federated client.
/// </para>
/// <para><b>For Beginners:</b> You provide the train/test JSON files, and AiDotNet loads the per-user partitions,
/// tokenizes the tweet text, and evaluates your model.
/// </para>
/// </remarks>
public sealed class Sent140FederatedBenchmarkOptions
{
    /// <summary>
    /// Gets or sets the path to the Sent140 train split JSON file.
    /// </summary>
    public string? TrainFilePath { get; set; }

    /// <summary>
    /// Gets or sets the optional path to the Sent140 test split JSON file.
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
    /// Gets or sets the maximum token sequence length for each tweet after tokenization.
    /// </summary>
    /// <remarks>
    /// If null, AiDotNet uses an industry-standard default (and smaller defaults in CI mode).
    /// </remarks>
    public int? MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the desired WordPiece vocabulary size when the model result does not already provide a tokenizer.
    /// </summary>
    /// <remarks>
    /// If null, AiDotNet uses an industry-standard default (and smaller defaults in CI mode).
    /// </remarks>
    public int? TokenizerVocabularySize { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of texts used to train a default WordPiece tokenizer when needed.
    /// </summary>
    /// <remarks>
    /// If null, AiDotNet uses a sensible default (smaller in CI mode).
    /// </remarks>
    public int? TokenizerTrainingSampleCount { get; set; }
}

