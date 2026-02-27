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

        if (numAdaptedLayers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numAdaptedLayers), "Number of adapted layers must be positive.");
        }

        if (layerDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(layerDim), "Layer dimension must be positive.");
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

        if (totalWeight <= 0)
        {
            throw new InvalidOperationException("Total client weight must be positive.");
        }

        var invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < adapterLen; i++)
        {
            aggregated[i] = NumOps.Multiply(aggregated[i], invTotal);
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>
    /// Applies the adapter forward pass with residual connection: output = x + scale * UpProject(ReLU(DownProject(x))).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The adapter is a bottleneck layer that compresses the input down
    /// to a small dimension (DownProject), applies an activation (ReLU), and expands back up
    /// (UpProject). The result is added to the original input (residual connection) so the adapter
    /// only learns the "delta" — what to change. The scale factor controls how much influence
    /// the adapter has; a smaller scale means more conservative updates.</para>
    /// </remarks>
    /// <param name="input">Input activations of shape [batchSize x layerDim].</param>
    /// <param name="adapterParams">Adapter parameters for one layer: [DownProject, UpProject, Bias].</param>
    /// <param name="residualScale">Scale factor for the adapter output before adding to residual. Default: 1.0.</param>
    /// <returns>Output activations with adapter applied.</returns>
    public T[] ApplyAdapterForward(T[] input, T[] adapterParams, double residualScale = 1.0)
    {
        Guard.NotNull(input);
        Guard.NotNull(adapterParams);
        int downSize = _layerDim * _bottleneckDim;
        int upSize = _layerDim * _bottleneckDim;

        if (adapterParams.Length < downSize + upSize + _bottleneckDim)
        {
            throw new ArgumentException(
                $"Adapter params too short. Expected at least {downSize + upSize + _bottleneckDim}, got {adapterParams.Length}.",
                nameof(adapterParams));
        }

        // DownProject: [layerDim x bottleneck]
        var hidden = new T[_bottleneckDim];
        for (int b = 0; b < _bottleneckDim; b++)
        {
            hidden[b] = NumOps.Zero;
            for (int d = 0; d < _layerDim && d < input.Length; d++)
            {
                hidden[b] = NumOps.Add(hidden[b], NumOps.Multiply(input[d], adapterParams[d * _bottleneckDim + b]));
            }

            // Add bias.
            hidden[b] = NumOps.Add(hidden[b], adapterParams[downSize + upSize + b]);
        }

        // ReLU activation.
        for (int b = 0; b < _bottleneckDim; b++)
        {
            if (NumOps.ToDouble(hidden[b]) < 0)
            {
                hidden[b] = NumOps.Zero;
            }
        }

        // UpProject: [bottleneck x layerDim]
        var output = new T[input.Length];

        // Preserve residual for dimensions beyond _layerDim (adapter doesn't touch them).
        for (int d = _layerDim; d < input.Length; d++)
        {
            output[d] = input[d];
        }

        var scaleT = NumOps.FromDouble(residualScale);
        for (int d = 0; d < input.Length && d < _layerDim; d++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < _bottleneckDim; b++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(hidden[b], adapterParams[downSize + b * _layerDim + d]));
            }

            // Residual connection: output = x + scale * adapter(x)
            output[d] = NumOps.Add(input[d], NumOps.Multiply(sum, scaleT));
        }

        return output;
    }

    /// <summary>
    /// Extracts adapter parameters for a specific layer and position (attention or FFN).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each transformer layer typically has TWO adapter insertions:
    /// one after the self-attention sub-layer and one after the feed-forward sub-layer.
    /// This method extracts the parameters for one specific adapter position.</para>
    /// </remarks>
    /// <param name="allAdapterParams">All adapter parameters (from ExtractAdapterParameters).</param>
    /// <param name="layerIndex">Which transformer layer (0-based).</param>
    /// <param name="position">Which position in the layer: 0 = after attention, 1 = after FFN.</param>
    /// <returns>Adapter parameters for the specified position.</returns>
    public T[] GetLayerAdapterParams(Vector<T> allAdapterParams, int layerIndex, int position)
    {
        Guard.NotNull(allAdapterParams);

        if (layerIndex < 0 || layerIndex >= _numAdaptedLayers)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index must be in [0, {_numAdaptedLayers}), got {layerIndex}.");
        }

        if (position < 0 || position > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(position),
                "Position must be 0 (attention) or 1 (FFN).");
        }

        int paramsPerAdapter = 2 * _layerDim * _bottleneckDim + _bottleneckDim;
        int adaptersPerLayer = 2; // attention + FFN
        int offset = (layerIndex * adaptersPerLayer + position) * paramsPerAdapter;

        if (offset + paramsPerAdapter > allAdapterParams.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer {layerIndex} position {position} exceeds adapter parameter bounds.");
        }

        var result = new T[paramsPerAdapter];
        for (int i = 0; i < paramsPerAdapter; i++)
        {
            result[i] = allAdapterParams[offset + i];
        }

        return result;
    }

    /// <summary>Gets the bottleneck hidden dimension.</summary>
    public int BottleneckDimension => _bottleneckDim;

    /// <summary>Gets the number of adapted transformer layers.</summary>
    public int NumAdaptedLayers => _numAdaptedLayers;

    /// <summary>Gets the transformer hidden dimension.</summary>
    public int LayerDimension => _layerDim;
}
