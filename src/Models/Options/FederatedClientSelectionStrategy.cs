namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how clients are selected to participate in a federated training round.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In large deployments, not every client participates in every round.
/// Client selection strategies control which clients are chosen each round.
/// </remarks>
public enum FederatedClientSelectionStrategy
{
    /// <summary>
    /// Uniform random sampling.
    /// </summary>
    UniformRandom = 0,

    /// <summary>
    /// Weighted random sampling (typically proportional to client sample counts).
    /// </summary>
    WeightedRandom = 1,

    /// <summary>
    /// Stratified sampling based on group keys.
    /// </summary>
    Stratified = 2,

    /// <summary>
    /// Prefer clients with higher reported availability.
    /// </summary>
    AvailabilityAware = 3,

    /// <summary>
    /// Prefer clients with better observed performance, with exploration.
    /// </summary>
    PerformanceAware = 4,

    /// <summary>
    /// Cluster clients (e.g., by embeddings) and sample across clusters.
    /// </summary>
    Clustered = 5
}

