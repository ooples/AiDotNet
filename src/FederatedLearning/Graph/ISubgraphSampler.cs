using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Samples neighborhoods from a client's local subgraph for mini-batch GNN training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> GNNs learn by aggregating features from neighboring nodes (message passing).
/// For large subgraphs, using all neighbors is too expensive. A sampler selects a fixed-size neighborhood
/// at each hop, creating small "computation trees" for efficient training.</para>
///
/// <para><b>Example:</b> For a target node, with 2 hops and max 10 neighbors per hop:</para>
/// <list type="number">
/// <item><description>Sample up to 10 direct neighbors (hop 1).</description></item>
/// <item><description>For each hop-1 neighbor, sample up to 10 of their neighbors (hop 2).</description></item>
/// <item><description>Result: a 2-hop subgraph of ~110 nodes for message passing.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISubgraphSampler<T>
{
    /// <summary>
    /// Samples a k-hop neighborhood around the specified target nodes.
    /// </summary>
    /// <param name="adjacency">Full adjacency matrix of the local subgraph.</param>
    /// <param name="targetNodes">Indices of the target nodes to build neighborhoods for.</param>
    /// <param name="nodeFeatures">Feature matrix [numNodes, featureDim].</param>
    /// <returns>Sampled subgraph: adjacency, features, and mapping from sampled to original indices.</returns>
    SampledSubgraph<T> Sample(Tensor<T> adjacency, int[] targetNodes, Tensor<T> nodeFeatures);
}

/// <summary>
/// Represents a sampled subgraph from neighborhood sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SampledSubgraph<T>
{
    /// <summary>
    /// Gets or sets the adjacency matrix of the sampled subgraph.
    /// </summary>
    public Tensor<T> Adjacency { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>
    /// Gets or sets the node features of the sampled subgraph.
    /// </summary>
    public Tensor<T> NodeFeatures { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>
    /// Gets or sets the mapping from sampled node indices to original node indices.
    /// </summary>
    public int[] NodeMapping { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the indices of the target nodes within the sampled subgraph.
    /// </summary>
    public int[] TargetNodeIndices { get; set; } = Array.Empty<int>();
}
