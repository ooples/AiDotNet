using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Graph-aware model aggregation strategy for federated GNN training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard FedAvg treats all model parameters equally during aggregation.
/// But GNNs have distinct parameter types — message-passing weights, attention heads, node embeddings,
/// readout layers — that benefit from different aggregation treatment.</para>
///
/// <para><b>Example:</b> Message-passing layers should be weighted by subgraph size (number of edges),
/// while readout layers should be weighted by number of labeled nodes. A graph-aware strategy handles
/// this distinction.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IGraphAggregationStrategy<T>
{
    /// <summary>
    /// Aggregates GNN model parameters from multiple clients with graph-aware weighting.
    /// </summary>
    /// <param name="clientModels">Client ID to model parameters mapping.</param>
    /// <param name="clientGraphStats">Graph statistics per client (node count, edge count, etc.).</param>
    /// <returns>Aggregated global model parameters.</returns>
    Tensor<T> Aggregate(
        Dictionary<int, Tensor<T>> clientModels,
        Dictionary<int, ClientGraphStats> clientGraphStats);

    /// <summary>
    /// Gets the name of this graph aggregation strategy.
    /// </summary>
    string StrategyName { get; }
}

/// <summary>
/// Statistics about a client's local subgraph, used for graph-aware aggregation weighting.
/// </summary>
public class ClientGraphStats
{
    /// <summary>
    /// Gets or sets the number of nodes in the client's subgraph.
    /// </summary>
    public int NodeCount { get; set; }

    /// <summary>
    /// Gets or sets the number of edges in the client's subgraph.
    /// </summary>
    public int EdgeCount { get; set; }

    /// <summary>
    /// Gets or sets the number of labeled nodes (for supervised tasks).
    /// </summary>
    public int LabeledNodeCount { get; set; }

    /// <summary>
    /// Gets or sets the number of cross-client border nodes.
    /// </summary>
    public int BorderNodeCount { get; set; }

    /// <summary>
    /// Gets or sets the average node degree in the subgraph.
    /// </summary>
    public double AverageDegree { get; set; }
}
