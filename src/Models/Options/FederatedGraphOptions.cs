namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for federated graph learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Graph FL extends standard federated learning to handle graph-structured
/// data (social networks, molecular graphs, knowledge graphs). Each client holds a subgraph, and
/// the server coordinates GNN training across all subgraphs. These options control the graph-specific
/// aspects of training.</para>
///
/// <para><b>Key decisions:</b></para>
/// <list type="bullet">
/// <item><description><b>Mode:</b> What graph task? (node classification, link prediction, graph classification)</description></item>
/// <item><description><b>Partitioning:</b> How was the graph split? (METIS, community, random, pre-assigned)</description></item>
/// <item><description><b>Pseudo-nodes:</b> How to handle missing cross-client neighbors?</description></item>
/// <item><description><b>Cross-client edges:</b> How to discover edges between clients' subgraphs?</description></item>
/// </list>
/// </remarks>
public class FederatedGraphOptions
{
    /// <summary>
    /// Gets or sets the graph FL mode (subgraph, node, link, or graph classification). Default is SubgraphLevel.
    /// </summary>
    public GraphFLMode Mode { get; set; } = GraphFLMode.SubgraphLevel;

    /// <summary>
    /// Gets or sets the graph partition strategy. Default is Preassigned.
    /// </summary>
    public GraphPartitionStrategy PartitionStrategy { get; set; } = GraphPartitionStrategy.Preassigned;

    /// <summary>
    /// Gets or sets the pseudo-node strategy for handling missing cross-client neighbors.
    /// Default is FeatureAverage.
    /// </summary>
    public PseudoNodeStrategy PseudoNodeStrategy { get; set; } = PseudoNodeStrategy.FeatureAverage;

    /// <summary>
    /// Gets or sets subgraph neighborhood sampling options.
    /// </summary>
    public SubgraphSamplingOptions Sampling { get; set; } = new SubgraphSamplingOptions();

    /// <summary>
    /// Gets or sets cross-client edge handling options.
    /// </summary>
    public CrossClientEdgeOptions CrossClientEdges { get; set; } = new CrossClientEdgeOptions();

    /// <summary>
    /// Gets or sets the dimensionality of node feature vectors. Default is 64.
    /// </summary>
    public int NodeFeatureDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden layer dimension for GNN layers. Default is 128.
    /// </summary>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of GNN message-passing layers. Default is 2.
    /// </summary>
    public int NumGnnLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to use prototype-based learning instead of full model sharing.
    /// Default is false.
    /// </summary>
    public bool UsePrototypeLearning { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of prototypes per class for prototype-based learning. Default is 5.
    /// </summary>
    public int PrototypesPerClass { get; set; } = 5;

    /// <summary>
    /// Gets or sets the neighborhood privacy epsilon for LDP on topology queries. Default is 2.0.
    /// </summary>
    public double NeighborhoodPrivacyEpsilon { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the number of target partitions when using graph partitioning. Default matches client count.
    /// </summary>
    public int? NumberOfPartitions { get; set; } = null;
}
