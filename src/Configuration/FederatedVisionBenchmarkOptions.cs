namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for federated vision benchmark suites.
/// </summary>
/// <remarks>
/// <para>
/// This groups dataset-specific options for vision benchmarks (for example, FEMNIST and CIFAR) under a single
/// facade-facing configuration object.
/// </para>
/// <para><b>For Beginners:</b> Vision benchmarks test models on image-like data. You select a suite (enum)
/// and provide the minimal dataset configuration here.
/// </para>
/// </remarks>
public sealed class FederatedVisionBenchmarkOptions
{
    /// <summary>
    /// Gets or sets FEMNIST options (LEAF JSON split files).
    /// </summary>
    public LeafFederatedBenchmarkOptions? Femnist { get; set; }

    /// <summary>
    /// Gets or sets CIFAR-10 options (CIFAR binary files with synthetic federated partitioning).
    /// </summary>
    public CifarFederatedBenchmarkOptions? Cifar10 { get; set; }

    /// <summary>
    /// Gets or sets CIFAR-100 options (CIFAR-100 binary files with synthetic federated partitioning).
    /// </summary>
    public CifarFederatedBenchmarkOptions? Cifar100 { get; set; }
}

