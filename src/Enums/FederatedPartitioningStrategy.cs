namespace AiDotNet.Enums;

/// <summary>
/// Defines how a centralized dataset should be partitioned into per-client datasets for federated simulations.
/// </summary>
public enum FederatedPartitioningStrategy
{
    /// <summary>
    /// IID partitioning (uniform random assignment of samples to clients).
    /// </summary>
    IID,

    /// <summary>
    /// Dirichlet label distribution partitioning (common FL benchmark approach).
    /// </summary>
    DirichletLabel,

    /// <summary>
    /// Shard-by-label partitioning (sort by label then assign label shards to clients).
    /// </summary>
    ShardByLabel
}

