using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

/// <summary>
/// Contains the results of Leiden community detection, including hierarchical partitions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Leiden algorithm produces a hierarchy of communities:
/// - Level 0: Finest partition — each node belongs to a small community
/// - Level 1: Coarser — small communities merged into bigger ones
/// - Level N: Coarsest — entire graph in a few large communities
///
/// The Communities dictionary gives the final (finest) partition:
/// mapping each node ID to its community ID.
/// </para>
/// </remarks>
public class LeidenResult
{
    /// <summary>
    /// Hierarchical partitions: level → (nodeId → communityId).
    /// Level 0 is the finest partition, higher levels are coarser.
    /// </summary>
    public List<Dictionary<string, int>> HierarchicalPartitions { get; set; } = [];

    /// <summary>
    /// Final community assignments: nodeId → communityId (finest level).
    /// </summary>
    public Dictionary<string, int> Communities { get; set; } = [];

    /// <summary>
    /// Modularity score at each level of the hierarchy.
    /// </summary>
    public List<double> ModularityScores { get; set; } = [];
}
