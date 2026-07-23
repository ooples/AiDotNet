using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Orchestrates federated learning across clients holding subgraphs of a larger graph.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard FL assumes clients have independent datasets. Graph FL is different
/// because clients' data is interconnected — edges may cross client boundaries. This trainer handles
/// the unique challenges of graph FL:</para>
/// <list type="bullet">
/// <item><description>Distributing a GNN model to subgraph-holding clients.</description></item>
/// <item><description>Managing cross-client edge discovery (using PSI).</description></item>
/// <item><description>Handling missing neighbors via pseudo-node strategies.</description></item>
/// <item><description>Aggregating GNN parameters with graph-aware strategies.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IFederatedGraphTrainer<T>
{
    /// <summary>
    /// Registers a client's subgraph with the trainer.
    /// </summary>
    /// <param name="clientId">Unique client identifier.</param>
    /// <param name="adjacency">Adjacency matrix (sparse or dense) for the client's subgraph.</param>
    /// <param name="nodeFeatures">Node feature matrix [numNodes, featureDim].</param>
    /// <param name="nodeLabels">Node label vector (for node classification tasks). May be null for unsupervised.</param>
    void RegisterSubgraph(int clientId, Tensor<T> adjacency, Tensor<T> nodeFeatures, Tensor<T>? nodeLabels);

    /// <summary>
    /// Executes one round of federated graph learning.
    /// </summary>
    /// <param name="roundNumber">Current round number.</param>
    /// <returns>Aggregated global model parameters after this round.</returns>
    Tensor<T> ExecuteRound(int roundNumber);

    /// <summary>
    /// Gets the current global GNN model parameters.
    /// </summary>
    Tensor<T> GetGlobalModel();

    /// <summary>
    /// Gets the number of registered clients.
    /// </summary>
    int ClientCount { get; }

    /// <summary>
    /// Initiates cross-client edge discovery between all client pairs.
    /// </summary>
    void DiscoverCrossClientEdges();
}
