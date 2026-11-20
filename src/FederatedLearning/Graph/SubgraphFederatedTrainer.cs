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
            // Flattened features are [nodes * featureDim]; the node count comes from the N x N adjacency.
            int adjacencyNodes = (int)Math.Sqrt(adjacency.Length);
            int featureDim = nodeFeatures.Shape.Length > 1
                ? nodeFeatures.Shape[1]
                : adjacencyNodes > 0 ? nodeFeatures.Length / adjacencyNodes : nodeFeatures.Length;
            int modelSize = featureDim * _options.HiddenDimension
                          + _options.HiddenDimension * _options.HiddenDimension * HiddenLayerCount
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

            // Local GNN training on the expanded subgraph
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

    /// <summary>Hidden-to-hidden GNN layers in the parameter layout (at least one).</summary>
    private int HiddenLayerCount => Math.Max(1, _options.NumGnnLayers - 1);

    /// <summary>
    /// Local training of the client GCN on its expanded subgraph: <see cref="FederatedGraphOptions.LocalEpochs"/>
    /// full-batch gradient-descent steps on the mean squared error of the labeled nodes.
    /// </summary>
    /// <remarks>
    /// <para>The parameters are laid out W_1 [F x H], then HiddenLayerCount matrices W_l [H x H], then the node
    /// regression head w [H]. With Â = D⁻¹(A + I) (mean over a node and its neighbors) the forward pass is
    /// H_0 = X, H_l = ReLU(Â H_{l-1} W_l), ŷ = H_L w (Kipf &amp; Welling 2017, with the row-normalized propagation
    /// of GraphSAGE-mean). The loss is (1 / m) Σ_{i &lt; m} (ŷ_i − y_i)² over the first m = min(labels, nodes)
    /// nodes, the original (labeled) nodes; pseudo-nodes only pass messages.</para>
    /// <para>Backward: g_ŷ = 2 (ŷ − y) / m on labeled rows, g_w = H_Lᵀ g_ŷ, g_{H_L} = g_ŷ wᵀ, and per layer
    /// g_Z = g_H ⊙ [Z &gt; 0], g_{W_l} = (Â H_{l-1})ᵀ g_Z, g_{H_{l-1}} = Âᵀ g_Z W_lᵀ.</para>
    /// </remarks>
    internal Tensor<T> TrainLocalGnn(ExpandedSubgraph<T> expanded, Tensor<T>? labels, Tensor<T> globalParams)
    {
        if (_options.LocalEpochs < 0)
            throw new InvalidOperationException("FederatedGraphOptions.LocalEpochs must be non-negative.");
        if (!(_options.LocalLearningRate > 0) || double.IsInfinity(_options.LocalLearningRate))
            throw new InvalidOperationException("FederatedGraphOptions.LocalLearningRate must be positive and finite.");

        var localParams = Engine.TensorMultiplyScalar(globalParams.Reshape(new[] { globalParams.Length }), NumOps.One);
        if (labels is null)
        {
            return localParams; // Unsupervised: no labeled loss to descend.
        }

        int numNodes = (int)Math.Sqrt(expanded.Adjacency.Length);
        int labelCount = Math.Min(labels.Length, numNodes);
        if (labelCount == 0 || _options.LocalEpochs == 0)
        {
            return localParams;
        }

        int hidden = _options.HiddenDimension;
        int headAndHidden = hidden * hidden * HiddenLayerCount + hidden;
        int featureDim = (localParams.Length - headAndHidden) / hidden;
        if (featureDim <= 0 || featureDim * hidden + headAndHidden != localParams.Length)
        {
            throw new ArgumentException(
                $"The model has {localParams.Length} parameters, which is not the GCN layout for hidden size {hidden}.",
                nameof(globalParams));
        }

        if (expanded.NodeFeatures.Length != numNodes * featureDim)
        {
            throw new ArgumentException(
                $"The expanded subgraph has {expanded.NodeFeatures.Length} feature values; the model expects " +
                $"{numNodes} nodes x {featureDim} features.", nameof(expanded));
        }

        var adjacency = Engine.TensorSlice(expanded.Adjacency.Reshape(new[] { expanded.Adjacency.Length }), new[] { 0 }, new[] { numNodes * numNodes })
            .Reshape(new[] { numNodes, numNodes });
        var features = expanded.NodeFeatures.Reshape(new[] { numNodes, featureDim });
        var targets = Engine.TensorSlice(labels.Reshape(new[] { labels.Length }), new[] { 0 }, new[] { labelCount })
            .Reshape(new[] { labelCount, 1 });

        // Half and BFloat16 train in float: a node degree past 2048 or a squared error past 65504 is lost in Half.
        return FederatedTensorMath<T>.IsNarrow
            ? GcnGradientDescent(adjacency.Cast<float>(), features.Cast<float>(), targets.Cast<float>(), localParams.Cast<float>(),
                featureDim, hidden, HiddenLayerCount, _options.LocalLearningRate, _options.LocalEpochs).Cast<T>()
            : GcnGradientDescent(adjacency, features, targets, localParams,
                featureDim, hidden, HiddenLayerCount, _options.LocalLearningRate, _options.LocalEpochs);
    }

    private static Tensor<TV> GcnGradientDescent<TV>(
        Tensor<TV> adjacency, Tensor<TV> features, Tensor<TV> targets, Tensor<TV> parameters,
        int featureDim, int hidden, int hiddenLayers, double learningRate, int epochs)
    {
        var ops = MathHelper.GetNumericOperations<TV>();
        int numNodes = adjacency.Shape[0];
        int labelCount = targets.Shape[0];

        // Â = D⁻¹(A + I): 1 on every edge and on the diagonal, each row divided by its count.
        var identity = new double[numNodes * numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            identity[i * numNodes + i] = 1.0;
        }

        var linked = Engine.TensorGreaterThan(
            Engine.TensorAdd(Engine.TensorGreaterThan(adjacency, ops.Zero).Cast<TV>(), new Tensor<double>(identity, new[] { numNodes, numNodes }).Cast<TV>()),
            ops.Zero).Cast<TV>();
        var degrees = Engine.TensorMultiplyScalar(Engine.ReduceMean(linked, new[] { 1 }, true), ops.FromDouble(numNodes));
        var propagation = Engine.TensorDivide(linked, Engine.TensorBroadcastTo(degrees, new[] { numNodes, numNodes }));
        var propagationT = Engine.TensorTranspose(propagation);

        // Weight shapes in layout order: W_1, the hidden W_l, then the head.
        int layers = 1 + hiddenLayers;
        var shapes = new int[layers + 1][];
        shapes[0] = new[] { featureDim, hidden };
        for (int l = 1; l < layers; l++)
        {
            shapes[l] = new[] { hidden, hidden };
        }

        shapes[layers] = new[] { hidden, 1 };

        var weights = new Tensor<TV>[layers + 1];
        int offset = 0;
        for (int l = 0; l <= layers; l++)
        {
            int size = shapes[l][0] * shapes[l][1];
            weights[l] = Engine.TensorSlice(parameters, new[] { offset }, new[] { size }).Reshape(shapes[l]);
            offset += size;
        }

        var step = ops.FromDouble(learningRate);
        var lossScale = ops.FromDouble(2.0 / labelCount);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Forward: P_l = Â H_{l-1}, Z_l = P_l W_l, H_l = ReLU(Z_l).
            var propagated = new Tensor<TV>[layers];
            var active = new Tensor<TV>[layers];
            var h = features;
            for (int l = 0; l < layers; l++)
            {
                propagated[l] = Engine.TensorMatMul(propagation, h);
                var z = Engine.TensorMatMul(propagated[l], weights[l]);
                active[l] = Engine.TensorGreaterThan(z, ops.Zero).Cast<TV>();
                h = Engine.TensorMultiply(z, active[l]);
            }

            var predictions = Engine.TensorMatMul(h, weights[layers]); // [N, 1]

            // g_ŷ = 2 (ŷ − y) / m on the labeled rows, 0 on the rest.
            var labeledError = Engine.TensorMultiplyScalar(Engine.TensorSubtract(
                Engine.TensorSlice(predictions, new[] { 0, 0 }, new[] { labelCount, 1 }), targets), lossScale);
            var gradPrediction = labelCount == numNodes
                ? labeledError
                : Engine.TensorConcatenate(new[] { labeledError, new Tensor<TV>(new[] { numNodes - labelCount, 1 }) }, 0);

            var gradients = new Tensor<TV>[layers + 1];
            gradients[layers] = Engine.TensorMatMul(Engine.TensorTranspose(h), gradPrediction);
            var gradH = Engine.TensorMatMul(gradPrediction, Engine.TensorTranspose(weights[layers]));
            for (int l = layers - 1; l >= 0; l--)
            {
                var gradZ = Engine.TensorMultiply(gradH, active[l]);
                gradients[l] = Engine.TensorMatMul(Engine.TensorTranspose(propagated[l]), gradZ);
                if (l > 0)
                {
                    gradH = Engine.TensorMatMul(propagationT, Engine.TensorMatMul(gradZ, Engine.TensorTranspose(weights[l])));
                }
            }

            for (int l = 0; l <= layers; l++)
            {
                weights[l] = Engine.TensorSubtract(weights[l], Engine.TensorMultiplyScalar(gradients[l], step));
            }
        }

        var flat = new Tensor<TV>[layers + 1];
        for (int l = 0; l <= layers; l++)
        {
            flat[l] = weights[l].Reshape(new[] { weights[l].Length });
        }

        return Engine.TensorConcatenate(flat, 0);
    }

    /// <summary>
    /// Mean over each node's neighbors of the neighbor's summed features. The node count comes from the
    /// adjacency (N x N, flattened) and the feature width from the flattened features (N * F).
    /// </summary>
    internal Tensor<T> ComputeNodeEmbeddings(ExpandedSubgraph<T> expanded)
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
        if (featureDim == 0)
        {
            return new Tensor<T>(new[] { numNodes });
        }

        var features = Engine.TensorSlice(expanded.NodeFeatures.Reshape(new[] { expanded.NodeFeatures.Length }), new[] { 0 }, new[] { numNodes * featureDim })
            .Reshape(new[] { numNodes, featureDim });

        // Neighbor counts are formed in int: a 0/1 sum in Half stops increasing at 2048.
        var neighborCounts = FederatedTensorMath<T>.CountNonzeroPerRow(neighborMask);

        // Half and BFloat16 sum and divide in float: two neighbor features of 40000 sum past 65504 although
        // their mean is in range.
        return FederatedTensorMath<T>.IsNarrow
            ? MeanNeighborFeatures(neighborMask.Cast<float>(), features.Cast<float>(), neighborCounts).Cast<T>()
            : MeanNeighborFeatures(neighborMask, features, neighborCounts);
    }

    /// <summary>Per-node mean over its neighbors of their summed features; nodes without neighbors embed to 0.</summary>
    private static Tensor<TV> MeanNeighborFeatures<TV>(Tensor<TV> neighborMask, Tensor<TV> features, int[] neighborCounts)
    {
        int numNodes = neighborCounts.Length;
        var sums = Engine.TensorMatMul(neighborMask, Engine.ReduceSum(features, new[] { 1 }, true)).Reshape(new[] { numNodes });
        var counts = new Tensor<int>(neighborCounts, new[] { numNodes }).Cast<TV>();
        var hasNeighbors = Engine.TensorGreaterThan(counts, AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TV>().Zero);
        var safeCounts = Engine.TensorWhere(hasNeighbors, counts, Engine.TensorOnesLike(counts));
        return Engine.TensorMultiply(Engine.TensorDivide(sums, safeCounts), hasNeighbors);
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
