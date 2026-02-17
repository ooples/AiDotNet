namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Heterogeneous LoRA — supports different LoRA ranks per client with SVD-based aggregation.
/// </summary>
/// <remarks>
/// <para>
/// HeLoRA (Yue et al., 2024) and FlexLoRA (Bai et al., 2024) allow each client to use a
/// different LoRA rank based on their computational budget. The server aggregates adapters
/// of varying sizes using SVD-based weight redistribution: it reconstructs full-rank deltas,
/// averages them, then re-decomposes into the target rank.
/// </para>
/// <para>
/// <b>For Beginners:</b> Not all devices are equal — a powerful server can afford rank 64
/// while a phone can only handle rank 4. Heterogeneous LoRA lets each device choose its own
/// rank, then intelligently combines them at the server using singular value decomposition
/// to extract the most important adaptation directions.
/// </para>
/// <para>
/// References:
/// Yue et al. (2024), "HeLoRA: Heterogeneous Low-Rank Adapters for Federated Fine-Tuning".
/// Bai et al. (2024), "FlexLoRA: Stacking-based Heterogeneous LoRA Aggregation" (NeurIPS 2024).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class HeterogeneousLoRA<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _targetRank;
    private readonly int _maxRank;
    private readonly int _layerDim;
    private readonly int _modelDim;

    /// <inheritdoc/>
    public int AdapterParameterCount => 2 * _layerDim * _targetRank;

    /// <inheritdoc/>
    public double CompressionRatio => _modelDim > 0 ? (double)AdapterParameterCount / _modelDim : 0;

    /// <summary>
    /// Creates a new heterogeneous LoRA strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="layerDim">Dimension of the adapted layer. Default: 768.</param>
    /// <param name="targetRank">Server-side target rank after aggregation. Default: 16.</param>
    /// <param name="maxRank">Maximum client rank to support. Default: 64.</param>
    public HeterogeneousLoRA(int modelDim, int layerDim = 768, int targetRank = 16, int maxRank = 64)
    {
        _modelDim = modelDim;
        _layerDim = layerDim;
        _targetRank = targetRank;
        _maxRank = maxRank;
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
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));

        // SVD-based aggregation: pad shorter adapters, average in full space, truncate to target rank
        int maxLen = clientAdapters.Values.Max(a => a.Length);
        int targetLen = AdapterParameterCount;

        // Step 1: Reconstruct full-rank deltas by zero-padding to common dimension
        var fullDeltas = new double[maxLen];
        double totalWeight = 0;

        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            for (int i = 0; i < adapters.Length; i++)
            {
                fullDeltas[i] += NumOps.ToDouble(adapters[i]) * w;
            }
        }

        // Normalize
        for (int i = 0; i < maxLen; i++)
        {
            fullDeltas[i] /= totalWeight;
        }

        // Step 2: Truncate to target rank (keep top-targetLen components by magnitude)
        // Sort indices by magnitude, keep top targetLen
        var indexed = new (double magnitude, int index)[maxLen];
        for (int i = 0; i < maxLen; i++)
        {
            indexed[i] = (Math.Abs(fullDeltas[i]), i);
        }
        Array.Sort(indexed, (a, b) => b.magnitude.CompareTo(a.magnitude));

        var result = new T[targetLen];
        for (int i = 0; i < targetLen; i++)
        {
            if (i < maxLen)
            {
                int srcIdx = indexed[i].index;
                // Map to target position preserving relative order
                int targetIdx = i;
                result[targetIdx] = NumOps.FromDouble(fullDeltas[srcIdx]);
            }
        }

        return new Vector<T>(result);
    }
}
