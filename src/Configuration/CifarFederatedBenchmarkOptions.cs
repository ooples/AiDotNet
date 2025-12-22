using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running CIFAR-based federated benchmark suites.
/// </summary>
/// <remarks>
/// <para>
/// CIFAR datasets are distributed as binary batch files. AiDotNet loads the dataset from the provided directory,
/// applies an industry-standard synthetic federated partitioning strategy (for example, Dirichlet label skew),
/// and evaluates the model with a structured report.
/// </para>
/// <para><b>For Beginners:</b> This suite tests image classification using the well-known CIFAR datasets.
/// You point AiDotNet to the extracted CIFAR folder, and it handles the rest.
/// </para>
/// </remarks>
public sealed class CifarFederatedBenchmarkOptions
{
    /// <summary>
    /// Gets or sets the dataset directory path containing CIFAR binary files.
    /// </summary>
    public string? DataDirectoryPath { get; set; }

    /// <summary>
    /// Gets or sets the number of federated clients to simulate (null uses defaults).
    /// </summary>
    public int? ClientCount { get; set; }

    /// <summary>
    /// Gets or sets the dataset-to-client partitioning strategy.
    /// </summary>
    public FederatedPartitioningStrategy PartitioningStrategy { get; set; } = FederatedPartitioningStrategy.DirichletLabel;

    /// <summary>
    /// Gets or sets how many label shards should be assigned to each client when using <see cref="FederatedPartitioningStrategy.ShardByLabel"/>.
    /// </summary>
    public int? ShardsPerClient { get; set; }

    /// <summary>
    /// Gets or sets the Dirichlet concentration parameter used when <see cref="PartitioningStrategy"/> is DirichletLabel.
    /// </summary>
    public double DirichletAlpha { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets an optional maximum number of training samples to load (null loads all available).
    /// </summary>
    public int? MaxTrainSamples { get; set; }

    /// <summary>
    /// Gets or sets an optional maximum number of test samples to load (null loads all available).
    /// </summary>
    public int? MaxTestSamples { get; set; }

    /// <summary>
    /// Gets or sets whether pixel values should be normalized to the range [0,1].
    /// </summary>
    public bool NormalizePixels { get; set; } = true;
}
