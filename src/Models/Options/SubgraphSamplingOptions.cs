namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for subgraph neighborhood sampling during federated GNN training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> GNNs learn by aggregating features from neighboring nodes (message passing).
/// In a large graph, using all neighbors at every hop is too expensive. Neighborhood sampling limits
/// how many neighbors are used at each hop, trading off accuracy for speed.</para>
///
/// <para><b>Example:</b> With HopCount=2 and MaxNeighborsPerHop=10, for each node we sample up to 10
/// neighbors at hop 1, and for each of those, sample up to 10 neighbors at hop 2. This gives a
/// 2-hop subgraph of up to ~110 nodes instead of potentially thousands.</para>
/// </remarks>
public class SubgraphSamplingOptions
{
    /// <summary>
    /// Gets or sets the number of hops for neighborhood sampling. Default is 2.
    /// </summary>
    public int HopCount { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum number of neighbors to sample per hop. Default is 10.
    /// </summary>
    public int MaxNeighborsPerHop { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use importance sampling (degree-proportional) instead of uniform.
    /// Default is false (uniform sampling).
    /// </summary>
    public bool UseImportanceSampling { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to include self-loops in the sampled subgraph. Default is true.
    /// </summary>
    public bool IncludeSelfLoops { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum degree a node must have to be included in sampling. Default is 0.
    /// </summary>
    public int MinNodeDegree { get; set; } = 0;
}
