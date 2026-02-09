namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for client selection in federated learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Client selection controls which devices/organizations participate in each round.
/// In real deployments, many clients may be offline or slow, so selecting a subset per round is common.
/// </remarks>
public class ClientSelectionOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the selection strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This selects the "rule" used to choose clients each round.
    /// </remarks>
    public FederatedClientSelectionStrategy Strategy { get; set; } = FederatedClientSelectionStrategy.UniformRandom;

    /// <summary>
    /// Gets or sets an optional mapping from client ID to a group key for stratified sampling.
    /// </summary>
    public Dictionary<int, string>? ClientGroupKeys { get; set; } = null;

    /// <summary>
    /// Gets or sets an optional mapping from client ID to an availability probability (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Availability-aware selection prefers clients that are more likely to be online.
    /// </remarks>
    public Dictionary<int, double>? ClientAvailabilityProbabilities { get; set; } = null;

    /// <summary>
    /// Gets or sets the exploration probability for performance-aware sampling (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A value of 0.1 means "10% of the time, pick random clients to explore;
    /// 90% of the time, pick the best-known clients."
    /// </remarks>
    public double ExplorationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of clusters to use for cluster-based sampling.
    /// </summary>
    public int ClusterCount { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of k-means iterations for cluster-based sampling.
    /// </summary>
    public int KMeansIterations { get; set; } = 5;

    /// <summary>
    /// Gets or sets the minimum availability probability required for a client to be considered available.
    /// </summary>
    public double AvailabilityThreshold { get; set; } = 0.0;
}

