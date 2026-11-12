using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
        var localParams = Engine.TensorMultiplyScalar(globalParams.Reshape(new[] { globalParams.Length }), NumOps.One);

        if (labels is null)
        {
            return localParams; // Unsupervised: return global params unchanged
        }

        // Simple GNN forward pass simulation: message passing + update
        // Compute aggregated neighbor features using adjacency
        var nodeEmbeddings = ComputeNodeEmbeddings(expanded);
        int numNodes = nodeEmbeddings.Length;

        // Compute gradient-like update based on label error
        int labelCount = Math.Min(labels.Shape[0], numNodes);
        double gradientScale = 0;
        if (labelCount > 0)
        {
            var errors = Engine.TensorSubtract(
                Engine.TensorSlice(nodeEmbeddings, new[] { 0 }, new[] { labelCount }),
                Engine.TensorSlice(labels.Reshape(new[] { labels.Length }), new[] { 0 }, new[] { labelCount }));
            gradientScale = SumOfSquares(errors) / labelCount;
        }

        // Apply a simple gradient scaling to the local parameters: p -= lr * scale * p * 0.01
        double learningRate = 0.01;
        return Engine.TensorMultiplyScalar(localParams, NumOps.FromDouble(1.0 - learningRate * gradientScale * 0.01));
    }

    /// <summary>
    /// Mean over each node's neighbors of the neighbor's summed features. The node count comes from the
    /// adjacency (N x N, flattened) and the feature width from the flattened features (N * F).
    /// </summary>
    private Tensor<T> ComputeNodeEmbeddings(ExpandedSubgraph<T> expanded)
    {
        int numNodes = (int)Math.Sqrt(expanded.Adjacency.Length);
        if (numNodes == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        int featureDim = expanded.NodeFeatures.Length / numNodes;

        // Aggregate neighbor features (message passing)
        var adjacency = Engine.TensorSlice(expanded.Adjacency.Reshape(new[] { expanded.Adjacency.Length }), new[] { 0 }, new[] { numNodes * numNodes })
            .Reshape(new[] { numNodes, numNodes });
        var neighborMask = Engine.TensorGreaterThan(adjacency, NumOps.Zero);
        var featureSums = featureDim == 0
            ? new Tensor<T>(new[] { numNodes, 1 })
            : Engine.ReduceSum(
                Engine.TensorSlice(expanded.NodeFeatures.Reshape(new[] { expanded.NodeFeatures.Length }), new[] { 0 }, new[] { numNodes * featureDim })
                    .Reshape(new[] { numNodes, featureDim }),
                new[] { 1 }, true);
        var sums = Engine.TensorMatMul(neighborMask, featureSums).Reshape(new[] { numNodes });
        var counts = Engine.ReduceSum(neighborMask, new[] { 1 }, false);

        // Mean aggregation (nodes without neighbors embed to 0)
        var hasNeighbors = Engine.TensorGreaterThan(counts, NumOps.Zero);
        var safeCounts = Engine.TensorWhere(hasNeighbors, counts, Engine.TensorOnesLike(counts));
        return Engine.TensorMultiply(Engine.TensorDivide(sums, safeCounts), hasNeighbors);
    }

    /// <summary>
    /// Σ v² computed as max|v|² · Σ (v / max|v|)² so the squares cannot overflow narrow T.
    /// </summary>
    private double SumOfSquares(Tensor<T> values)
    {
        T maxAbs = Engine.TensorMaxValue(Engine.TensorAbs(values));
        if (!NumOps.GreaterThan(maxAbs, NumOps.Zero))
        {
            return 0;
        }

        double scaled = NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(Engine.TensorDivideScalar(values, maxAbs))));
        double max = NumOps.ToDouble(maxAbs);
        return max * max * scaled;
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

        var values = new double[modelSize];
        for (int i = 0; i < modelSize; i++)
        {
            values[i] = (rng.NextDouble() * 2 - 1) * scale;
        }

        _globalModel = new Tensor<double>(values, new[] { modelSize }).Cast<T>();
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
        int edgeCount = totalElements == 0
            ? 0
            : Engine.TensorCountNonzero(Engine.TensorGreaterThan(
                Engine.TensorSlice(adjacency.Reshape(new[] { adjacency.Length }), new[] { 0 }, new[] { totalElements }), NumOps.Zero));

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
