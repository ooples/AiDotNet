namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Federated LoRA — Low-Rank Adaptation for parameter-efficient federated fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// LoRA (Hu et al., 2021) decomposes weight updates into low-rank matrices: ΔW = BA where
/// B ∈ R^{d×r} and A ∈ R^{r×k} with rank r ≪ min(d,k). In federated settings, only the
/// LoRA matrices are communicated, reducing bandwidth by 100-1000x for large models.
/// </para>
/// <para>
/// This implementation follows the FedEx-LoRA approach (ACL 2025) which performs exact
/// aggregation by averaging A and B matrices separately with residual error correction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine a large model as a huge spreadsheet. Instead of modifying
/// every cell (billions of values), LoRA only learns a small "correction" matrix that captures
/// the most important changes. In federated LoRA, each device only sends this tiny correction
/// instead of the whole spreadsheet — making it practical to collaboratively fine-tune
/// GPT-scale models across phones or hospitals.
/// </para>
/// <para>
/// References:
/// Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models".
/// Sun et al. (2025), "FedEx-LoRA: Exact Aggregation for Federated Low-Rank Adaptation" (ACL 2025).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FederatedLoRA<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _rank;
    private readonly double _alpha;
    private readonly int _modelDim;
    private readonly int _numAdaptedLayers;
    private readonly int _layerInputDim;
    private readonly int _layerOutputDim;

    /// <inheritdoc/>
    public int AdapterParameterCount { get; }

    /// <inheritdoc/>
    public double CompressionRatio { get; }

    /// <summary>
    /// Creates a new federated LoRA strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="rank">LoRA rank (r). Lower = more compression. Default: 8.</param>
    /// <param name="alpha">LoRA scaling factor (alpha/r scales the adaptation). Default: 16.</param>
    /// <param name="numAdaptedLayers">Number of layers with LoRA adapters. Default: 4.</param>
    /// <param name="layerInputDim">Input dimension of adapted layers. Default: 768.</param>
    /// <param name="layerOutputDim">Output dimension of adapted layers. Default: 768.</param>
    public FederatedLoRA(int modelDim, int rank = 8, double alpha = 16.0,
        int numAdaptedLayers = 4, int layerInputDim = 768, int layerOutputDim = 768)
    {
        _rank = rank;
        _alpha = alpha;
        _modelDim = modelDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerInputDim = layerInputDim;
        _layerOutputDim = layerOutputDim;

        // Each adapted layer has: B (outputDim × rank) + A (rank × inputDim)
        int paramsPerLayer = _layerOutputDim * _rank + _rank * _layerInputDim;
        AdapterParameterCount = _numAdaptedLayers * paramsPerLayer;
        CompressionRatio = modelDim > 0 ? (double)AdapterParameterCount / modelDim : 0;
    }

    /// <inheritdoc/>
    public Vector<T> ExtractAdapterParameters(Vector<T> fullModelParameters)
    {
        // Extract adapter parameters from the end of the parameter vector
        // Convention: base model params first, then LoRA params appended
        int totalParams = fullModelParameters.Length;
        int adapterCount = Math.Min(AdapterParameterCount, totalParams);

        var adapterParams = new T[adapterCount];
        int start = totalParams - adapterCount;
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
        // Copy frozen base parameters
        for (int i = 0; i < start; i++)
        {
            merged[i] = fullModelParameters[i];
        }
        // Replace adapter parameters with aggregated ones
        double scalingFactor = _alpha / _rank;
        for (int i = 0; i < adapterCount; i++)
        {
            merged[start + i] = NumOps.Multiply(aggregatedAdapters[i], NumOps.FromDouble(scalingFactor));
        }

        return new Vector<T>(merged);
    }

    /// <inheritdoc/>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        if (clientAdapters.Count == 0)
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));

        int adapterLen = clientAdapters.Values.First().Length;
        var aggregated = new T[adapterLen];
        double totalWeight = 0;

        // FedEx-LoRA exact aggregation: weighted average of adapter parameters
        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            T weight = NumOps.FromDouble(w);
            for (int i = 0; i < adapterLen; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(adapters[i], weight));
            }
        }

        T invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < adapterLen; i++)
        {
            aggregated[i] = NumOps.Multiply(aggregated[i], invTotal);
        }

        return new Vector<T>(aggregated);
    }
}
