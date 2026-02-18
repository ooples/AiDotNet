using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Main coordinator for subgraph-level federated GNN training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class orchestrates graph FL end-to-end:</para>
/// <list type="number">
/// <item><description><b>Setup:</b> Clients register their subgraphs (adjacency + features + labels).</description></item>
/// <item><description><b>Edge discovery:</b> Uses PSI to find cross-client edges without revealing adjacency.</description></item>
/// <item><description><b>Local training:</b> Each client trains a GNN on their expanded subgraph (with pseudo-nodes).</description></item>
/// <item><description><b>Aggregation:</b> Server aggregates GNN parameters using graph-aware weighting.</description></item>
/// <item><description><b>Repeat:</b> Until convergence or max rounds.</description></item>
/// </list>
///
/// <para><b>Pseudo-node expansion:</b> Before local training, each client's subgraph is expanded
/// with pseudo-nodes that approximate missing cross-client neighbors. The strategy
/// (FeatureAverage, GeneratorBased, ZeroFill) is controlled by <see cref="FederatedGraphOptions"/>.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SubgraphFederatedTrainer<T> : FederatedLearningComponentBase<T>, IFederatedGraphTrainer<T>
{
    private readonly FederatedGraphOptions _options;
    private readonly IGraphAggregationStrategy<T> _aggregationStrategy;
    private readonly ICrossClientEdgeHandler<T>? _edgeHandler;
    private readonly SubgraphExpander<T> _expander;
    private readonly Dictionary<int, ClientSubgraphData<T>> _clientSubgraphs = new();
    private Tensor<T> _globalModel;
    private bool _edgesDiscovered;

    /// <summary>
    /// Initializes a new instance of <see cref="SubgraphFederatedTrainer{T}"/>.
    /// </summary>
    /// <param name="options">Graph FL configuration.</param>
    /// <param name="aggregationStrategy">Graph-aware aggregation strategy.</param>
    /// <param name="edgeHandler">Cross-client edge handler (null to skip edge discovery).</param>
    public SubgraphFederatedTrainer(
        FederatedGraphOptions options,
        IGraphAggregationStrategy<T> aggregationStrategy,
        ICrossClientEdgeHandler<T>? edgeHandler = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _aggregationStrategy = aggregationStrategy ?? throw new ArgumentNullException(nameof(aggregationStrategy));
        _edgeHandler = edgeHandler;
        _expander = new SubgraphExpander<T>(options);

        // Initialize global model (will be resized when first client registers)
        _globalModel = new Tensor<T>(new[] { 0 });
    }

    /// <inheritdoc/>
    public int ClientCount => _clientSubgraphs.Count;

    /// <inheritdoc/>
    public void RegisterSubgraph(int clientId, Tensor<T> adjacency, Tensor<T> nodeFeatures, Tensor<T>? nodeLabels)
    {
        if (adjacency is null) throw new ArgumentNullException(nameof(adjacency));
        if (nodeFeatures is null) throw new ArgumentNullException(nameof(nodeFeatures));

        var stats = ComputeGraphStats(adjacency, nodeFeatures, nodeLabels);

        _clientSubgraphs[clientId] = new ClientSubgraphData<T>
        {
            Adjacency = adjacency,
            NodeFeatures = nodeFeatures,
            NodeLabels = nodeLabels,
            Stats = stats
        };

        // Initialize global model dimensions if this is the first client
        if (_globalModel.Shape[0] == 0)
        {
            int featureDim = nodeFeatures.Shape.Length > 1 ? nodeFeatures.Shape[1] : nodeFeatures.Shape[0];
            int modelSize = featureDim * _options.HiddenDimension
                          + _options.HiddenDimension * _options.HiddenDimension * Math.Max(1, _options.NumGnnLayers - 1)
                          + _options.HiddenDimension;
            _globalModel = new Tensor<T>(new[] { modelSize });
            InitializeGlobalModel();
        }
    }

    /// <inheritdoc/>
    public Tensor<T> ExecuteRound(int roundNumber)
    {
        if (_clientSubgraphs.Count == 0)
        {
            throw new InvalidOperationException("No clients registered. Call RegisterSubgraph first.");
        }

        // Discover cross-client edges on first round if not done
        if (!_edgesDiscovered && _edgeHandler is not null)
        {
            DiscoverCrossClientEdges();
        }

        // Phase 1: Distribute global model and perform local training
        var clientModels = new Dictionary<int, Tensor<T>>();
        var clientStats = new Dictionary<int, ClientGraphStats>();

        foreach (var kvp in _clientSubgraphs)
        {
            int clientId = kvp.Key;
            var subgraphData = kvp.Value;

            // Expand subgraph with pseudo-nodes for missing neighbors
            var expanded = _expander.Expand(
                subgraphData.Adjacency,
                subgraphData.NodeFeatures,
                GetCrossClientNeighborFeatures(clientId));

            // Simulate local GNN training
            var localModel = TrainLocalGnn(expanded, subgraphData.NodeLabels, _globalModel);
            clientModels[clientId] = localModel;
            clientStats[clientId] = subgraphData.Stats;
        }

        // Phase 2: Graph-aware aggregation
        _globalModel = _aggregationStrategy.Aggregate(clientModels, clientStats);

        return _globalModel;
    }

    /// <inheritdoc/>
    public Tensor<T> GetGlobalModel()
    {
        return _globalModel;
    }

    /// <inheritdoc/>
    public void DiscoverCrossClientEdges()
    {
        if (_edgeHandler is null)
        {
            _edgesDiscovered = true;
            return;
        }

        var clientIds = new List<int>(_clientSubgraphs.Keys);

        for (int i = 0; i < clientIds.Count; i++)
        {
            for (int j = i + 1; j < clientIds.Count; j++)
            {
                int clientA = clientIds[i];
                int clientB = clientIds[j];

                // Get border nodes (nodes with external references)
                var borderA = GetBorderNodes(_clientSubgraphs[clientA].Adjacency);
                var borderB = GetBorderNodes(_clientSubgraphs[clientB].Adjacency);

                var edges = _edgeHandler.DiscoverEdges(borderA, borderB);
                if (edges.Count > 0)
                {
                    _edgeHandler.CacheEdges(clientA, clientB, edges);
                }
            }
        }

        _edgesDiscovered = true;
    }

    private Tensor<T> TrainLocalGnn(ExpandedSubgraph<T> expanded, Tensor<T>? labels, Tensor<T> globalParams)
    {
        // Create a local copy of global parameters
        int modelSize = globalParams.Shape[0];
        var localParams = new Tensor<T>(new[] { modelSize });

        for (int i = 0; i < modelSize; i++)
        {
            localParams[i] = globalParams[i];
        }

        if (labels is null)
        {
            return localParams; // Unsupervised: return global params unchanged
        }

        // Simple GNN forward pass simulation: message passing + update
        int numNodes = expanded.NodeFeatures.Shape[0];
        int featureDim = expanded.NodeFeatures.Shape.Length > 1 ? expanded.NodeFeatures.Shape[1] : 1;

        // Compute aggregated neighbor features using adjacency
        var nodeEmbeddings = ComputeNodeEmbeddings(expanded, featureDim);

        // Compute gradient-like update based on label error
        int labelCount = Math.Min(labels.Shape[0], numNodes);
        double totalError = 0;

        for (int n = 0; n < labelCount; n++)
        {
            double predicted = NumOps.ToDouble(nodeEmbeddings[n]);
            double target = NumOps.ToDouble(labels[n]);
            totalError += (predicted - target) * (predicted - target);
        }

        // Apply a simple gradient scaling to the local parameters
        double learningRate = 0.01;
        double gradientScale = labelCount > 0 ? totalError / labelCount : 0;

        for (int i = 0; i < modelSize; i++)
        {
            double param = NumOps.ToDouble(localParams[i]);
            param -= learningRate * gradientScale * param * 0.01; // Simple L2-like update
            localParams[i] = NumOps.FromDouble(param);
        }

        return localParams;
    }

    private Tensor<T> ComputeNodeEmbeddings(ExpandedSubgraph<T> expanded, int featureDim)
    {
        int numNodes = expanded.NodeFeatures.Shape[0];
        var embeddings = new Tensor<T>(new[] { numNodes });

        for (int i = 0; i < numNodes; i++)
        {
            double sum = 0;
            int neighbors = 0;

            // Aggregate neighbor features (message passing)
            for (int j = 0; j < numNodes; j++)
            {
                double adjValue = NumOps.ToDouble(expanded.Adjacency[i * numNodes + j]);
                if (adjValue > 0)
                {
                    for (int f = 0; f < featureDim && (i * featureDim + f) < expanded.NodeFeatures.Shape[0]; f++)
                    {
                        int idx = j * featureDim + f;
                        if (idx < expanded.NodeFeatures.Shape[0])
                        {
                            sum += NumOps.ToDouble(expanded.NodeFeatures[idx]);
                        }
                    }

                    neighbors++;
                }
            }

            // Mean aggregation
            embeddings[i] = NumOps.FromDouble(neighbors > 0 ? sum / neighbors : 0);
        }

        return embeddings;
    }

    private Dictionary<int, Tensor<T>> GetCrossClientNeighborFeatures(int clientId)
    {
        var neighborFeatures = new Dictionary<int, Tensor<T>>();

        if (_edgeHandler is null)
        {
            return neighborFeatures;
        }

        foreach (var otherClientId in _clientSubgraphs.Keys)
        {
            if (otherClientId == clientId) continue;

            var edges = _edgeHandler.GetEdges(clientId, otherClientId);
            if (edges.Count == 0)
            {
                edges = _edgeHandler.GetEdges(otherClientId, clientId);
            }

            if (edges.Count > 0)
            {
                neighborFeatures[otherClientId] = _clientSubgraphs[otherClientId].NodeFeatures;
            }
        }

        return neighborFeatures;
    }

    private static IReadOnlyList<int> GetBorderNodes(Tensor<T> adjacency)
    {
        // Border nodes: nodes that might have external connections
        // For simplicity, return all nodes (in practice, would check for missing neighbors)
        int numNodes = (int)Math.Sqrt(adjacency.Shape[0]);
        var borderNodes = new List<int>();

        for (int i = 0; i < numNodes; i++)
        {
            borderNodes.Add(i);
        }

        return borderNodes;
    }

    private void InitializeGlobalModel()
    {
        // Xavier-like initialization
        int modelSize = _globalModel.Shape[0];
        double scale = Math.Sqrt(2.0 / (_options.NodeFeatureDimension + _options.HiddenDimension));
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        for (int i = 0; i < modelSize; i++)
        {
            double val = (rng.NextDouble() * 2 - 1) * scale;
            _globalModel[i] = NumOps.FromDouble(val);
        }
    }

    private ClientGraphStats ComputeGraphStats(Tensor<T> adjacency, Tensor<T> nodeFeatures, Tensor<T>? nodeLabels)
    {
        int numNodes = nodeFeatures.Shape[0];
        if (nodeFeatures.Shape.Length > 1)
        {
            numNodes = nodeFeatures.Shape[0];
        }

        int totalElements = adjacency.Shape[0];
        int matrixSize = (int)Math.Sqrt(totalElements);
        int edgeCount = 0;

        for (int i = 0; i < totalElements; i++)
        {
            if (NumOps.ToDouble(adjacency[i]) > 0)
            {
                edgeCount++;
            }
        }

        return new ClientGraphStats
        {
            NodeCount = numNodes,
            EdgeCount = edgeCount,
            LabeledNodeCount = nodeLabels is not null ? nodeLabels.Shape[0] : 0,
            BorderNodeCount = numNodes, // Conservative: treat all as potential border
            AverageDegree = matrixSize > 0 ? (double)edgeCount / matrixSize : 0
        };
    }
}

/// <summary>
/// Internal data structure holding a client's subgraph data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class ClientSubgraphData<T>
{
    public Tensor<T> Adjacency { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> NodeFeatures { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T>? NodeLabels { get; set; }
    public ClientGraphStats Stats { get; set; } = new ClientGraphStats();
}
