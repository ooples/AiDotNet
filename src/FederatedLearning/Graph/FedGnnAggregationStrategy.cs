using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// GNN-aware federated aggregation that weights contributions by subgraph topology characteristics.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard FedAvg weights clients by dataset size (number of samples).
/// For graph FL, "dataset size" is ambiguous — is it nodes, edges, or labeled nodes?
/// This strategy uses a composite weight based on graph topology:</para>
///
/// <list type="bullet">
/// <item><description><b>Edge-weighted:</b> More edges = more message-passing information = higher weight for
/// GNN layers. This prevents sparse subgraphs from diluting aggregate quality.</description></item>
/// <item><description><b>Label-weighted:</b> More labeled nodes = more supervised signal = higher weight for
/// classification/readout layers.</description></item>
/// <item><description><b>Degree-adjusted:</b> Higher average degree means the subgraph is more structurally
/// representative of the global graph.</description></item>
/// </list>
///
/// <para><b>Formula:</b> weight_i = alpha * E_i/E_total + beta * L_i/L_total + gamma * deg_i/deg_avg,
/// where E = edges, L = labeled nodes, deg = average degree, and alpha+beta+gamma = 1.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FedGnnAggregationStrategy<T> : FederatedLearningComponentBase<T>, IGraphAggregationStrategy<T>
{
    private readonly double _edgeWeight;
    private readonly double _labelWeight;
    private readonly double _degreeWeight;

    /// <inheritdoc/>
    public string StrategyName => "FedGNN";

    /// <summary>
    /// Initializes a new instance of <see cref="FedGnnAggregationStrategy{T}"/>.
    /// </summary>
    /// <param name="edgeWeight">Weight for edge-count component. Default 0.4.</param>
    /// <param name="labelWeight">Weight for labeled-node component. Default 0.4.</param>
    /// <param name="degreeWeight">Weight for average-degree component. Default 0.2.</param>
    public FedGnnAggregationStrategy(
        double edgeWeight = 0.4,
        double labelWeight = 0.4,
        double degreeWeight = 0.2)
    {
        double total = edgeWeight + labelWeight + degreeWeight;
        if (total <= 0)
        {
            throw new ArgumentException("At least one weight component must be positive.");
        }

        // Normalize to sum to 1
        _edgeWeight = edgeWeight / total;
        _labelWeight = labelWeight / total;
        _degreeWeight = degreeWeight / total;
    }

    /// <inheritdoc/>
    public Tensor<T> Aggregate(
        Dictionary<int, Tensor<T>> clientModels,
        Dictionary<int, ClientGraphStats> clientGraphStats)
    {
        if (clientModels is null || clientModels.Count == 0)
        {
            throw new ArgumentException("No client models to aggregate.", nameof(clientModels));
        }

        // Compute composite weights
        var weights = ComputeCompositeWeights(clientModels.Keys, clientGraphStats);

        // Determine model size from first client
        int modelSize = 0;
        Tensor<T>? firstModel = null;
        foreach (var model in clientModels.Values)
        {
            firstModel = model;
            modelSize = model.Shape[0];
            break;
        }

        if (firstModel is null || modelSize == 0)
        {
            throw new InvalidOperationException("Client models are empty.");
        }

        // Weighted average
        var aggregated = new Tensor<T>(new[] { modelSize });
        double totalWeight = 0;

        foreach (var kvp in clientModels)
        {
            int clientId = kvp.Key;
            double weight = weights.ContainsKey(clientId) ? weights[clientId] : 1.0;
            totalWeight += weight;

            var model = kvp.Value;
            for (int i = 0; i < modelSize; i++)
            {
                double current = NumOps.ToDouble(aggregated[i]);
                double clientVal = NumOps.ToDouble(model[i]);
                aggregated[i] = NumOps.FromDouble(current + clientVal * weight);
            }
        }

        // Normalize
        if (totalWeight > 0)
        {
            for (int i = 0; i < modelSize; i++)
            {
                double val = NumOps.ToDouble(aggregated[i]);
                aggregated[i] = NumOps.FromDouble(val / totalWeight);
            }
        }

        return aggregated;
    }

    private Dictionary<int, double> ComputeCompositeWeights(
        IEnumerable<int> clientIds,
        Dictionary<int, ClientGraphStats> stats)
    {
        var weights = new Dictionary<int, double>();

        // Compute totals for normalization
        double totalEdges = 0, totalLabeled = 0, totalDegree = 0;
        int clientCount = 0;

        foreach (int clientId in clientIds)
        {
            if (stats.ContainsKey(clientId))
            {
                var s = stats[clientId];
                totalEdges += s.EdgeCount;
                totalLabeled += s.LabeledNodeCount;
                totalDegree += s.AverageDegree;
            }

            clientCount++;
        }

        // Avoid division by zero
        if (totalEdges <= 0) totalEdges = clientCount;
        if (totalLabeled <= 0) totalLabeled = clientCount;
        if (totalDegree <= 0) totalDegree = clientCount;

        foreach (int clientId in clientIds)
        {
            if (stats.ContainsKey(clientId))
            {
                var s = stats[clientId];
                double edgeComponent = s.EdgeCount / totalEdges;
                double labelComponent = s.LabeledNodeCount > 0 ? s.LabeledNodeCount / totalLabeled : 1.0 / clientCount;
                double degreeComponent = s.AverageDegree / totalDegree;

                weights[clientId] = _edgeWeight * edgeComponent
                                  + _labelWeight * labelComponent
                                  + _degreeWeight * degreeComponent;
            }
            else
            {
                weights[clientId] = 1.0 / clientCount;
            }
        }

        return weights;
    }
}
