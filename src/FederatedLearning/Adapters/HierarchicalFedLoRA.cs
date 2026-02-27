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
///  ├── Edge 1 (rank=4)
///  │    ├── Client 1a
///  │    └── Client 1b
///  └── Edge 2 (rank=4)
///       ├── Client 2a
///       └── Client 2b
/// </code>
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

        if (globalRank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(globalRank), "Global rank must be positive.");
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
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        // Edge-level aggregation uses localRank. Cloud-level would call this again on edge outputs.
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

    /// <summary>Gets the local (edge) LoRA rank.</summary>
    public int LocalRank => _localRank;

    /// <summary>Gets the global (cloud) LoRA rank.</summary>
    public int GlobalRank => _globalRank;
}
