using AiDotNet.Models.Options;

namespace AiDotNet.Clustering.Ensemble;

/// <summary>
/// Configuration options for Consensus (Ensemble) Clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Consensus clustering combines multiple clustering solutions to produce
/// a more robust final clustering. It works by aggregating partitions from
/// different algorithms or the same algorithm with different parameters.
/// </para>
/// <para><b>For Beginners:</b> Ensemble clustering is like taking a vote.
///
/// The idea:
/// 1. Run multiple clustering algorithms (or same algorithm multiple times)
/// 2. Each gives a different answer
/// 3. Combine the answers to get a more reliable result
///
/// Why it works:
/// - Different algorithms have different strengths
/// - Random initialization can give different results
/// - Combining reduces the impact of any single bad result
///
/// Common approaches:
/// - Co-association matrix: How often are points clustered together?
/// - Voting: Which cluster assignment is most popular?
/// </para>
/// </remarks>
public class ConsensusClusteringOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of base clusterings to generate.
    /// </summary>
    /// <value>Number of base clusterings. Default is 10.</value>
    public int NumBaseClusterings { get; set; } = 10;

    /// <summary>
    /// Gets or sets the consensus method.
    /// </summary>
    /// <value>Consensus method. Default is CoAssociation.</value>
    public ConsensusMethod Method { get; set; } = ConsensusMethod.CoAssociation;

    /// <summary>
    /// Gets or sets the final clustering algorithm.
    /// </summary>
    /// <value>Final algorithm. Default is Hierarchical.</value>
    public FinalClusteringAlgorithm FinalAlgorithm { get; set; } = FinalClusteringAlgorithm.Hierarchical;

    /// <summary>
    /// Gets or sets the target number of clusters.
    /// </summary>
    /// <value>Target clusters, or null for automatic.</value>
    public int? NumClusters { get; set; }

}

/// <summary>
/// Methods for combining multiple clusterings.
/// </summary>
public enum ConsensusMethod
{
    /// <summary>
    /// Co-association matrix: Cluster based on pair frequency.
    /// </summary>
    CoAssociation,

    /// <summary>
    /// Evidence accumulation: Similar to co-association with normalization.
    /// </summary>
    EvidenceAccumulation,

    /// <summary>
    /// Voting: Each clustering votes for assignments.
    /// </summary>
    Voting
}

/// <summary>
/// Algorithm for final clustering of consensus matrix.
/// </summary>
public enum FinalClusteringAlgorithm
{
    /// <summary>
    /// Use hierarchical clustering on co-association matrix.
    /// </summary>
    Hierarchical,

    /// <summary>
    /// Use spectral clustering on co-association matrix.
    /// </summary>
    Spectral
}
