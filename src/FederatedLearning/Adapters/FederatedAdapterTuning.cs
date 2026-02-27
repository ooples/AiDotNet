namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements FedAdapter — federated bottleneck adapter tuning where small adapter modules
/// are inserted into each transformer block and only these are communicated.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of modifying a large model's weights directly, adapters
/// insert small "bottleneck" layers (down-project → activation → up-project) after the
/// attention and feed-forward layers in each transformer block. Only these tiny bottleneck
/// layers are trained and shared in federated learning, keeping the base model frozen.</para>
///
/// <para>Architecture per adapted layer:</para>
/// <code>
/// x → DownProject(d → bottleneck) → ReLU → UpProject(bottleneck → d) → + x (residual)
/// Params per layer = 2 * d * bottleneck + bottleneck (bias)
/// </code>
///
/// <para>Reference: Cai, X., et al. (2023). "FedAdapter: Efficient Federated Learning via
/// Bottleneck Adapters." NeurIPS Workshop 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FederatedAdapterTuning<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _modelDim;
    private readonly int _bottleneckDim;
    private readonly int _numAdaptedLayers;
    private readonly int _layerDim;

    /// <inheritdoc/>
    public int AdapterParameterCount { get; }

    /// <inheritdoc/>
    public double CompressionRatio { get; }

    /// <summary>
    /// Creates a new FedAdapter strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="bottleneckDim">Bottleneck hidden dimension. Default: 64.</param>
    /// <param name="numAdaptedLayers">Number of transformer layers with adapters. Default: 12.</param>
    /// <param name="layerDim">Hidden dimension of each transformer layer. Default: 768.</param>
    public FederatedAdapterTuning(
        int modelDim,
        int bottleneckDim = 64,
        int numAdaptedLayers = 12,
        int layerDim = 768)
    {
        if (modelDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim), "Model dimension must be positive.");
        }

        if (bottleneckDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bottleneckDim), "Bottleneck dimension must be positive.");
        }

        _modelDim = modelDim;
        _bottleneckDim = bottleneckDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerDim = layerDim;

        // Down projection + up projection + bias per layer
        int paramsPerLayer = 2 * _layerDim * _bottleneckDim + _bottleneckDim;
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

    /// <summary>Gets the bottleneck hidden dimension.</summary>
    public int BottleneckDimension => _bottleneckDim;
}
