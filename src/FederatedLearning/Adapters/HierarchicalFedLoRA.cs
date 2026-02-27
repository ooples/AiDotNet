namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements HierFedLoRA — Hierarchical LoRA aggregation for edge-cloud federated topologies.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In a hierarchical FL system, devices first aggregate within
/// their local "edge" group (e.g., same hospital, same region), then edges aggregate with
/// the central cloud server. HierFedLoRA applies different LoRA ranks at different levels:
/// edge clients may use very low ranks (e.g., rank 4) for fast local communication, while the
/// edge-to-cloud aggregation uses a higher rank to preserve more information.</para>
///
/// <para>Topology:</para>
/// <code>
/// Cloud (rank=16)
///  +-- Edge 1 (rank=4)
///  |    +-- Client 1a
///  |    +-- Client 1b
///  +-- Edge 2 (rank=4)
///       +-- Client 2a
///       +-- Client 2b
/// </code>
///
/// <para>Rank promotion: When edge-level adapters (rank 4) are sent to the cloud, they are
/// promoted to the higher cloud rank (rank 16) by padding with zeros, then the cloud aggregates
/// in the higher-rank space and distributes back to edges.</para>
///
/// <para>Reference: Hierarchical LoRA Aggregation for Cross-Silo Federated Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class HierarchicalFedLoRA<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _localRank;
    private readonly int _globalRank;
    private readonly int _modelDim;
    private readonly int _numAdaptedLayers;
    private readonly int _layerDim;

    /// <inheritdoc/>
    public int AdapterParameterCount { get; }

    /// <inheritdoc/>
    public double CompressionRatio { get; }

    /// <summary>
    /// Creates a new HierFedLoRA strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="localRank">LoRA rank for edge-level aggregation. Default: 4.</param>
    /// <param name="globalRank">LoRA rank for cloud-level aggregation. Default: 16.</param>
    /// <param name="numAdaptedLayers">Number of adapted layers. Default: 4.</param>
    /// <param name="layerDim">Dimension of each adapted layer. Default: 768.</param>
    public HierarchicalFedLoRA(
        int modelDim,
        int localRank = 4,
        int globalRank = 16,
        int numAdaptedLayers = 4,
        int layerDim = 768)
    {
        if (modelDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim), "Model dimension must be positive.");
        }

        if (localRank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(localRank), "Local rank must be positive.");
        }

        if (globalRank <= 0 || globalRank < localRank)
        {
            throw new ArgumentOutOfRangeException(nameof(globalRank), "Global rank must be positive and >= local rank.");
        }

        _localRank = localRank;
        _globalRank = globalRank;
        _modelDim = modelDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerDim = layerDim;

        // At client level we use localRank for communication.
        int paramsPerLayer = 2 * _layerDim * _localRank;
        AdapterParameterCount = _numAdaptedLayers * paramsPerLayer;
        CompressionRatio = _modelDim > 0 ? (double)AdapterParameterCount / _modelDim : 0;
    }

    /// <inheritdoc/>
    public Vector<T> ExtractAdapterParameters(Vector<T> fullModelParameters)
    {
        int totalParams = fullModelParameters.Length;
        int adapterCount = Math.Min(AdapterParameterCount, totalParams);
        int start = totalParams - adapterCount;

        var adapterParams = new T[adapterCount];
        for (int i = 0; i < adapterCount; i++)
        {
            adapterParams[i] = fullModelParameters[start + i];
        }

        return new Vector<T>(adapterParams);
    }

    /// <inheritdoc/>
    public Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters)
    {
        int totalParams = fullModelParameters.Length;
        int adapterCount = aggregatedAdapters.Length;
        int start = totalParams - adapterCount;

        var merged = new T[totalParams];
        for (int i = 0; i < start; i++)
        {
            merged[i] = fullModelParameters[i];
        }

        for (int i = 0; i < adapterCount; i++)
        {
            merged[start + i] = aggregatedAdapters[i];
        }

        return new Vector<T>(merged);
    }

    /// <inheritdoc/>
    /// <remarks>Default aggregation uses edge-level weighted averaging (localRank).</remarks>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        return AggregateEdge(clientAdapters, clientWeights);
    }

    /// <summary>
    /// Performs edge-level aggregation of client adapters using weighted averaging at local rank.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> At the edge level (e.g., within one hospital), clients send
    /// their low-rank LoRA updates and the edge server averages them. This is fast because
    /// the updates are small (low rank) and communication stays within the local network.</para>
    /// </remarks>
    /// <param name="clientAdapters">Client adapter vectors at local rank.</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <returns>Aggregated edge adapter at local rank.</returns>
    public Vector<T> AggregateEdge(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        int adapterLen = clientAdapters.Values.First().Length;
        var aggregated = new T[adapterLen];
        double totalWeight = 0;

        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            var wT = NumOps.FromDouble(w);
            for (int i = 0; i < adapterLen; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(adapters[i], wT));
            }
        }

        var invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < adapterLen; i++)
        {
            aggregated[i] = NumOps.Multiply(aggregated[i], invTotal);
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>
    /// Promotes a local-rank adapter to the global (cloud) rank by zero-padding the
    /// B and A matrices. This preserves the existing low-rank representation while
    /// providing room for the cloud aggregation to capture additional structure.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Think of this like upgrading from a small box to a bigger box.
    /// The original content (low-rank adapters) stays the same, but there's now extra room.
    /// When the cloud combines adapters from multiple edges, this extra room lets it capture
    /// information that wouldn't fit in the smaller rank.</para>
    ///
    /// <para>For each layer, B goes from [dim x localRank] to [dim x globalRank],
    /// and A goes from [localRank x dim] to [globalRank x dim]. New elements are zero.</para>
    /// </remarks>
    /// <param name="localAdapter">Adapter parameters at local rank.</param>
    /// <returns>Adapter parameters promoted to global rank (zero-padded).</returns>
    public Vector<T> PromoteToGlobalRank(Vector<T> localAdapter)
    {
        int localParamsPerLayer = 2 * _layerDim * _localRank;
        int globalParamsPerLayer = 2 * _layerDim * _globalRank;
        int globalTotalParams = _numAdaptedLayers * globalParamsPerLayer;

        var promoted = new T[globalTotalParams];
        // Initialize all to zero (already default for value types via NumOps).

        for (int layer = 0; layer < _numAdaptedLayers; layer++)
        {
            int localOffset = layer * localParamsPerLayer;
            int globalOffset = layer * globalParamsPerLayer;

            int localBSize = _layerDim * _localRank;
            int globalBSize = _layerDim * _globalRank;

            // Promote B matrix: [dim x localRank] -> [dim x globalRank].
            // Copy each row, zero-padding extra rank columns.
            for (int row = 0; row < _layerDim; row++)
            {
                for (int r = 0; r < _localRank; r++)
                {
                    if (localOffset + row * _localRank + r < localAdapter.Length)
                    {
                        promoted[globalOffset + row * _globalRank + r] = localAdapter[localOffset + row * _localRank + r];
                    }
                }
                // Remaining columns [localRank..globalRank) stay zero.
            }

            // Promote A matrix: [localRank x dim] -> [globalRank x dim].
            // Copy first localRank rows, remaining rows stay zero.
            for (int r = 0; r < _localRank; r++)
            {
                for (int col = 0; col < _layerDim; col++)
                {
                    int localIdx = localOffset + localBSize + r * _layerDim + col;
                    int globalIdx = globalOffset + globalBSize + r * _layerDim + col;
                    if (localIdx < localAdapter.Length)
                    {
                        promoted[globalIdx] = localAdapter[localIdx];
                    }
                }
            }
        }

        return new Vector<T>(promoted);
    }

    /// <summary>
    /// Demotes a global-rank adapter back to local rank by truncating the B and A matrices.
    /// </summary>
    /// <param name="globalAdapter">Adapter parameters at global rank.</param>
    /// <returns>Adapter parameters truncated to local rank.</returns>
    public Vector<T> DemoteToLocalRank(Vector<T> globalAdapter)
    {
        int localParamsPerLayer = 2 * _layerDim * _localRank;
        int globalParamsPerLayer = 2 * _layerDim * _globalRank;
        int localTotalParams = _numAdaptedLayers * localParamsPerLayer;

        var demoted = new T[localTotalParams];

        for (int layer = 0; layer < _numAdaptedLayers; layer++)
        {
            int localOffset = layer * localParamsPerLayer;
            int globalOffset = layer * globalParamsPerLayer;

            int localBSize = _layerDim * _localRank;
            int globalBSize = _layerDim * _globalRank;

            // Truncate B matrix: [dim x globalRank] -> [dim x localRank].
            for (int row = 0; row < _layerDim; row++)
            {
                for (int r = 0; r < _localRank; r++)
                {
                    demoted[localOffset + row * _localRank + r] = globalAdapter[globalOffset + row * _globalRank + r];
                }
            }

            // Truncate A matrix: [globalRank x dim] -> [localRank x dim].
            for (int r = 0; r < _localRank; r++)
            {
                for (int col = 0; col < _layerDim; col++)
                {
                    int localIdx = localOffset + localBSize + r * _layerDim + col;
                    int globalIdx = globalOffset + globalBSize + r * _layerDim + col;
                    demoted[localIdx] = globalAdapter[globalIdx];
                }
            }
        }

        return new Vector<T>(demoted);
    }

    /// <summary>
    /// Performs cloud-level aggregation of edge adapters. Edge adapters are first promoted
    /// to global rank, aggregated in the higher-rank space, then the result is optionally
    /// demoted back to local rank for distribution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When edge servers send their aggregated adapters to the cloud,
    /// they first get "promoted" to a higher rank. The cloud then combines all the edge results
    /// in this higher-rank space, which can capture cross-edge patterns that wouldn't fit in the
    /// lower rank. The result is sent back to edges after "demotion" back to local rank.</para>
    /// </remarks>
    /// <param name="edgeAdapters">Dictionary of edge ID to edge-aggregated adapter vectors (at local rank).</param>
    /// <param name="edgeWeights">Optional per-edge weights (typically proportional to number of clients).</param>
    /// <param name="demoteResult">If true, demote the result back to local rank. Default: true.</param>
    /// <returns>Cloud-aggregated adapter at the appropriate rank.</returns>
    public Vector<T> AggregateCloud(
        Dictionary<int, Vector<T>> edgeAdapters,
        Dictionary<int, double>? edgeWeights,
        bool demoteResult = true)
    {
        if (edgeAdapters.Count == 0)
        {
            throw new ArgumentException("No edge adapters provided.", nameof(edgeAdapters));
        }

        // Step 1: Promote all edge adapters to global rank.
        var promotedAdapters = new Dictionary<int, Vector<T>>(edgeAdapters.Count);
        foreach (var (edgeId, adapter) in edgeAdapters)
        {
            promotedAdapters[edgeId] = PromoteToGlobalRank(adapter);
        }

        // Step 2: Weighted average in the global-rank space.
        int globalLen = promotedAdapters.Values.First().Length;
        var aggregated = new T[globalLen];
        double totalWeight = 0;

        foreach (var (edgeId, adapter) in promotedAdapters)
        {
            double w = edgeWeights?.GetValueOrDefault(edgeId, 1.0) ?? 1.0;
            totalWeight += w;

            var wT = NumOps.FromDouble(w);
            for (int i = 0; i < globalLen; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(adapter[i], wT));
            }
        }

        var invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < globalLen; i++)
        {
            aggregated[i] = NumOps.Multiply(aggregated[i], invTotal);
        }

        var globalResult = new Vector<T>(aggregated);

        // Step 3: Optionally demote back to local rank for distribution.
        return demoteResult ? DemoteToLocalRank(globalResult) : globalResult;
    }

    /// <summary>
    /// Full hierarchical aggregation: first aggregate within each edge group,
    /// then aggregate across edges at the cloud level.
    /// </summary>
    /// <param name="edgeGroups">Dictionary of edge ID to (client ID to adapter) mapping.</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <param name="edgeWeights">Optional per-edge weights.</param>
    /// <returns>Cloud-aggregated adapter at local rank.</returns>
    public Vector<T> AggregateHierarchical(
        Dictionary<int, Dictionary<int, Vector<T>>> edgeGroups,
        Dictionary<int, double>? clientWeights = null,
        Dictionary<int, double>? edgeWeights = null)
    {
        if (edgeGroups.Count == 0)
        {
            throw new ArgumentException("No edge groups provided.", nameof(edgeGroups));
        }

        // Step 1: Edge-level aggregation (within each group).
        var edgeAdapters = new Dictionary<int, Vector<T>>(edgeGroups.Count);
        foreach (var (edgeId, clients) in edgeGroups)
        {
            edgeAdapters[edgeId] = AggregateEdge(clients, clientWeights);
        }

        // Step 2: Cloud-level aggregation (across edges, with rank promotion).
        return AggregateCloud(edgeAdapters, edgeWeights);
    }

    /// <summary>Gets the local (edge) LoRA rank.</summary>
    public int LocalRank => _localRank;

    /// <summary>Gets the global (cloud) LoRA rank.</summary>
    public int GlobalRank => _globalRank;

    /// <summary>Gets the number of adapted layers.</summary>
    public int NumAdaptedLayers => _numAdaptedLayers;

    /// <summary>Gets the layer dimension.</summary>
    public int LayerDim => _layerDim;
}
