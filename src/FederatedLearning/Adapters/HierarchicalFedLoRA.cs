using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        if (numAdaptedLayers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numAdaptedLayers), "Number of adapted layers must be positive.");
        }

        if (layerDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(layerDim), "Layer dimension must be positive.");
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

        var adapterParams = new Vector<T>(adapterCount);
        fullModelParameters.AsSpan().Slice(start, adapterCount).CopyTo(adapterParams.AsWritableSpan());
        return adapterParams;
    }

    /// <inheritdoc/>
    public Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters)
    {
        Guard.NotNull(fullModelParameters);
        Guard.NotNull(aggregatedAdapters);
        int totalParams = fullModelParameters.Length;
        int adapterCount = aggregatedAdapters.Length;

        if (adapterCount > totalParams)
        {
            throw new ArgumentException(
                $"Adapter length ({adapterCount}) exceeds full model length ({totalParams}).",
                nameof(aggregatedAdapters));
        }

        int start = totalParams - adapterCount;

        var merged = fullModelParameters.Clone();
        aggregatedAdapters.AsSpan().CopyTo(merged.AsWritableSpan().Slice(start, adapterCount));
        return merged;
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
        Guard.NotNull(clientAdapters);
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        int adapterLen = clientAdapters.Values.First().Length;

        // Validate all clients have matching adapter lengths.
        foreach (var (clientId, adapter) in clientAdapters)
        {
            if (adapter.Length != adapterLen)
            {
                throw new ArgumentException(
                    $"Client {clientId} adapter length {adapter.Length} != expected {adapterLen}.",
                    nameof(clientAdapters));
            }
        }

        var aggregated = new Vector<T>(adapterLen);
        double totalWeight = 0;

        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;
            aggregated = Engine.Add(aggregated, Engine.Multiply(adapters, NumOps.FromDouble(w)));
        }

        if (totalWeight <= 0)
        {
            throw new InvalidOperationException("Total client weight must be positive.");
        }

        return Engine.Multiply(aggregated, NumOps.FromDouble(1.0 / totalWeight));
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
        Guard.NotNull(localAdapter);
        int localParamsPerLayer = 2 * _layerDim * _localRank;
        int expectedLocalLen = _numAdaptedLayers * localParamsPerLayer;

        if (localAdapter.Length < expectedLocalLen)
        {
            throw new ArgumentException(
                $"Local adapter length ({localAdapter.Length}) is shorter than expected ({expectedLocalLen}).",
                nameof(localAdapter));
        }

        var (b, a) = SplitFactors(localAdapter, _localRank);
        int extra = _globalRank - _localRank;
        if (extra > 0)
        {
            // B gains zero rank columns, A gains zero rank rows.
            b = Engine.TensorConcatenate(new[] { b, new Tensor<T>(new[] { _numAdaptedLayers, _layerDim, extra }) }, 2);
            a = Engine.TensorConcatenate(new[] { a, new Tensor<T>(new[] { _numAdaptedLayers, extra, _layerDim }) }, 1);
        }

        return JoinFactors(b, a, _globalRank);
    }

    /// <summary>
    /// Demotes a global-rank adapter back to local rank by truncating the B and A matrices.
    /// </summary>
    /// <param name="globalAdapter">Adapter parameters at global rank.</param>
    /// <returns>Adapter parameters truncated to local rank.</returns>
    public Vector<T> DemoteToLocalRank(Vector<T> globalAdapter)
    {
        Guard.NotNull(globalAdapter);
        int globalParamsPerLayer = 2 * _layerDim * _globalRank;
        int expectedGlobalLen = _numAdaptedLayers * globalParamsPerLayer;

        if (globalAdapter.Length < expectedGlobalLen)
        {
            throw new ArgumentException(
                $"Global adapter length ({globalAdapter.Length}) is shorter than expected ({expectedGlobalLen}).",
                nameof(globalAdapter));
        }

        var (b, a) = SplitFactors(globalAdapter, _globalRank);
        var localB = Engine.TensorSlice(b, new[] { 0, 0, 0 }, new[] { _numAdaptedLayers, _layerDim, _localRank });
        var localA = Engine.TensorSlice(a, new[] { 0, 0, 0 }, new[] { _numAdaptedLayers, _localRank, _layerDim });
        return JoinFactors(localB, localA, _localRank);
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
    /// <para>The higher-rank space is used through the edges' full updates: per layer the cloud forms
    /// dW = sum_e p_e B_e A_e (rank up to edges * localRank) and re-decomposes it by SVD into the best
    /// globalRank factors B = U_r sqrt(S_r), A = sqrt(S_r) V_r^T. Averaging zero-padded factors instead
    /// would only reproduce the local-rank average avg(B) avg(A). Demotion keeps the leading localRank
    /// singular directions, which is the best local-rank approximation.</para>
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

        double totalWeight = 0;
        foreach (var edgeId in promotedAdapters.Keys)
        {
            totalWeight += edgeWeights?.GetValueOrDefault(edgeId, 1.0) ?? 1.0;
        }

        if (totalWeight <= 0)
        {
            throw new InvalidOperationException("Total edge weight must be positive.");
        }

        // Step 2: Weighted average of the full updates in the global-rank space, batched over layers.
        var deltaW = new Tensor<T>(new[] { _numAdaptedLayers, _layerDim, _layerDim });
        foreach (var (edgeId, adapter) in promotedAdapters)
        {
            double p = (edgeWeights?.GetValueOrDefault(edgeId, 1.0) ?? 1.0) / totalWeight;
            var (b, a) = SplitFactors(adapter, _globalRank);
            deltaW = Engine.TensorAdd(deltaW, Engine.TensorMultiplyScalar(Engine.TensorMatMul(b, a), NumOps.FromDouble(p)));
        }

        // Step 3: Re-decompose each layer at global rank.
        var layerB = new Tensor<T>[_numAdaptedLayers];
        var layerA = new Tensor<T>[_numAdaptedLayers];
        int kept = Math.Min(_globalRank, _layerDim);
        for (int layer = 0; layer < _numAdaptedLayers; layer++)
        {
            var layerDelta = Engine.TensorSlice(deltaW, new[] { layer, 0, 0 }, new[] { 1, _layerDim, _layerDim })
                .Reshape(new[] { _layerDim, _layerDim });
            var (u, s, vh) = LinalgDispatch<T>.Svd(layerDelta, fullMatrices: false);
            var sqrtSigma = Engine.TensorSqrt(Engine.TensorSlice(s, new[] { 0 }, new[] { kept }));
            var b = Engine.TensorMultiply(
                Engine.TensorSlice(u, new[] { 0, 0 }, new[] { _layerDim, kept }), sqrtSigma.Reshape(new[] { 1, kept }));
            var a = Engine.TensorMultiply(
                Engine.TensorSlice(vh, new[] { 0, 0 }, new[] { kept, _layerDim }), sqrtSigma.Reshape(new[] { kept, 1 }));
            if (kept < _globalRank)
            {
                b = Engine.TensorConcatenate(new[] { b, new Tensor<T>(new[] { _layerDim, _globalRank - kept }) }, 1);
                a = Engine.TensorConcatenate(new[] { a, new Tensor<T>(new[] { _globalRank - kept, _layerDim }) }, 0);
            }

            layerB[layer] = b.Reshape(new[] { 1, _layerDim, _globalRank });
            layerA[layer] = a.Reshape(new[] { 1, _globalRank, _layerDim });
        }

        var globalResult = JoinFactors(
            Engine.TensorConcatenate(layerB, 0), Engine.TensorConcatenate(layerA, 0), _globalRank);

        // Step 4: Optionally demote back to local rank for distribution.
        return demoteResult ? DemoteToLocalRank(globalResult) : globalResult;
    }

    /// <summary>
    /// Splits the leading layers * 2 * dim * rank adapter entries into B [layers, dim, rank] and
    /// A [layers, rank, dim].
    /// </summary>
    private (Tensor<T> B, Tensor<T> A) SplitFactors(Vector<T> adapter, int rank)
    {
        int factorSize = _layerDim * rank;
        var layers = new Tensor<T>(
            new[] { _numAdaptedLayers, 2 * factorSize }, adapter.GetSubVector(0, _numAdaptedLayers * 2 * factorSize));
        var b = Engine.TensorSlice(layers, new[] { 0, 0 }, new[] { _numAdaptedLayers, factorSize })
            .Reshape(new[] { _numAdaptedLayers, _layerDim, rank });
        var a = Engine.TensorSlice(layers, new[] { 0, factorSize }, new[] { _numAdaptedLayers, factorSize })
            .Reshape(new[] { _numAdaptedLayers, rank, _layerDim });
        return (b, a);
    }

    /// <summary>Flattens B [layers, dim, rank] and A [layers, rank, dim] back to the per-layer [B, A] layout.</summary>
    private Vector<T> JoinFactors(Tensor<T> b, Tensor<T> a, int rank)
    {
        int factorSize = _layerDim * rank;
        var joined = Engine.TensorConcatenate(
            new[] { b.Reshape(new[] { _numAdaptedLayers, factorSize }), a.Reshape(new[] { _numAdaptedLayers, factorSize }) }, 1);
        return joined.ToVector();
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
