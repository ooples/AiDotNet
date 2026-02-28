namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements SLoRA — Sparse LoRA for communication-efficient federated fine-tuning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Even with LoRA, sending all adapter parameters can be expensive
/// when many layers are adapted. SLoRA adds a sparsity step: after local training, each client
/// identifies which adapter elements changed most and only sends those (top-k by magnitude).
/// The server aggregates the sparse updates and broadcasts the result. This can reduce
/// communication by another 2-10x on top of LoRA's compression.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// 1. Client trains LoRA adapters locally
/// 2. Compute delta = new_adapters - old_adapters
/// 3. Keep only top-k% of delta by magnitude (sparse mask)
/// 4. Send sparse delta + mask to server
/// 5. Server aggregates sparse deltas, applies to global adapters
/// </code>
///
/// <para>Reference: Sparse LoRA for Federated Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SparseLoRA<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _rank;
    private readonly double _alpha;
    private readonly double _sparsityRatio;
    private readonly int _modelDim;
    private readonly int _numAdaptedLayers;
    private readonly int _layerInputDim;
    private readonly int _layerOutputDim;

    /// <inheritdoc/>
    public int AdapterParameterCount { get; }

    /// <inheritdoc/>
    public double CompressionRatio { get; }

    /// <summary>
    /// Creates a new SLoRA strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="rank">LoRA rank. Default: 8.</param>
    /// <param name="alpha">LoRA alpha. Default: 16.</param>
    /// <param name="sparsityRatio">Fraction of adapter elements to keep (top-k). Default: 0.5.</param>
    /// <param name="numAdaptedLayers">Number of adapted layers. Default: 4.</param>
    /// <param name="layerInputDim">Input dimension. Default: 768.</param>
    /// <param name="layerOutputDim">Output dimension. Default: 768.</param>
    public SparseLoRA(
        int modelDim,
        int rank = 8,
        double alpha = 16.0,
        double sparsityRatio = 0.5,
        int numAdaptedLayers = 4,
        int layerInputDim = 768,
        int layerOutputDim = 768)
    {
        if (modelDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim), "Model dimension must be positive.");
        }

        if (rank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), "Rank must be positive.");
        }

        if (sparsityRatio <= 0 || sparsityRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sparsityRatio), "Sparsity ratio must be in (0, 1].");
        }

        if (alpha <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be positive.");
        }

        if (numAdaptedLayers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numAdaptedLayers), "Number of adapted layers must be positive.");
        }

        if (layerInputDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(layerInputDim), "Layer input dimension must be positive.");
        }

        if (layerOutputDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(layerOutputDim), "Layer output dimension must be positive.");
        }

        _rank = rank;
        _alpha = alpha;
        _sparsityRatio = sparsityRatio;
        _modelDim = modelDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerInputDim = layerInputDim;
        _layerOutputDim = layerOutputDim;

        int paramsPerLayer = _layerOutputDim * _rank + _rank * _layerInputDim;
        AdapterParameterCount = _numAdaptedLayers * paramsPerLayer;
        CompressionRatio = _modelDim > 0 ? (double)AdapterParameterCount * _sparsityRatio / _modelDim : 0;
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

        double scale = _alpha / _rank;
        var scaleT = NumOps.FromDouble(scale);
        for (int i = 0; i < adapterCount; i++)
        {
            merged[start + i] = NumOps.Multiply(aggregatedAdapters[i], scaleT);
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
        foreach (var (clientId, adapters) in clientAdapters)
        {
            if (adapters.Length != adapterLen)
            {
                throw new ArgumentException(
                    $"Client {clientId} adapter length {adapters.Length} differs from expected {adapterLen}.");
            }
        }
        var aggregated = new T[adapterLen];
        var counts = new double[adapterLen]; // Track how many clients contributed to each element.
        double totalWeight = 0;

        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            var wT = NumOps.FromDouble(w);
            for (int i = 0; i < adapterLen; i++)
            {
                double val = NumOps.ToDouble(adapters[i]);
                if (val != 0.0) // Sparse: only non-zero elements were sent.
                {
                    aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(adapters[i], wT));
                    counts[i] += w;
                }
            }
        }

        // Normalize by the weight of contributing clients per element.
        for (int i = 0; i < adapterLen; i++)
        {
            if (counts[i] > 0)
            {
                aggregated[i] = NumOps.Multiply(aggregated[i], NumOps.FromDouble(1.0 / counts[i]));
            }
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>
    /// Applies top-k sparsification to an adapter update delta, keeping only the largest
    /// elements by magnitude. This is the core communication reduction step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After local training, the adapter parameters have changed.
    /// But not all changes are equally important. This method keeps only the most significant
    /// changes (the ones with the largest absolute value) and zeros out the rest. This can
    /// reduce the amount of data sent by 2-10x with minimal loss of accuracy.</para>
    /// </remarks>
    /// <param name="adapterBefore">Adapter parameters before local training.</param>
    /// <param name="adapterAfter">Adapter parameters after local training.</param>
    /// <returns>A sparse update containing only the top-k elements by magnitude.</returns>
    public SparseUpdate<T> SparsifyDelta(Vector<T> adapterBefore, Vector<T> adapterAfter)
    {
        Guard.NotNull(adapterBefore);
        Guard.NotNull(adapterAfter);
        if (adapterBefore.Length != adapterAfter.Length)
        {
            throw new ArgumentException(
                $"Before ({adapterBefore.Length}) and after ({adapterAfter.Length}) adapter lengths must match.");
        }

        int len = adapterAfter.Length;
        int k = Math.Max(1, (int)(len * _sparsityRatio));

        // Compute the delta.
        var delta = new double[len];
        for (int i = 0; i < len; i++)
        {
            delta[i] = NumOps.ToDouble(adapterAfter[i]) - NumOps.ToDouble(adapterBefore[i]);
        }

        // Find the top-k threshold by magnitude using partial sort.
        var magnitudes = new double[len];
        for (int i = 0; i < len; i++)
        {
            magnitudes[i] = Math.Abs(delta[i]);
        }

        // Find the k-th largest magnitude.
        var sorted = (double[])magnitudes.Clone();
        Array.Sort(sorted);
        double threshold = sorted[len - k]; // Elements >= this threshold are in top-k.

        // Build sparse representation.
        var indices = new List<int>(k);
        var values = new List<T>(k);

        for (int i = 0; i < len; i++)
        {
            if (magnitudes[i] >= threshold && indices.Count < k)
            {
                indices.Add(i);
                values.Add(NumOps.FromDouble(delta[i]));
            }
        }

        return new SparseUpdate<T>(indices.ToArray(), values.ToArray(), len);
    }

    /// <summary>
    /// Converts a sparse update back to a dense vector (filling non-sparse positions with zero).
    /// </summary>
    /// <param name="sparse">The sparse update.</param>
    /// <returns>Dense vector representation.</returns>
    public static Vector<T> ToDense(SparseUpdate<T> sparse)
    {
        var dense = new T[sparse.TotalLength];
        for (int i = 0; i < sparse.Indices.Length; i++)
        {
            dense[sparse.Indices[i]] = sparse.Values[i];
        }

        return new Vector<T>(dense);
    }

    /// <summary>
    /// Applies a sparse update to existing adapter parameters: new = old + sparse_delta.
    /// </summary>
    /// <param name="adapter">Current adapter parameters.</param>
    /// <param name="sparseUpdate">Sparse update delta.</param>
    /// <returns>Updated adapter parameters.</returns>
    public static Vector<T> ApplySparseUpdate(Vector<T> adapter, SparseUpdate<T> sparseUpdate)
    {
        var result = new T[adapter.Length];
        for (int i = 0; i < adapter.Length; i++)
        {
            result[i] = adapter[i];
        }

        for (int i = 0; i < sparseUpdate.Indices.Length; i++)
        {
            int idx = sparseUpdate.Indices[i];
            result[idx] = NumOps.Add(result[idx], sparseUpdate.Values[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Aggregates multiple sparse updates from clients. Only non-zero positions are considered,
    /// and the aggregation averages by the number of clients that contributed to each position.
    /// </summary>
    /// <param name="sparseUpdates">Dictionary of client ID to sparse updates.</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <returns>Dense aggregated delta vector.</returns>
    public Vector<T> AggregateSparseUpdates(
        Dictionary<int, SparseUpdate<T>> sparseUpdates,
        Dictionary<int, double>? clientWeights = null)
    {
        if (sparseUpdates.Count == 0)
        {
            throw new ArgumentException("No sparse updates provided.", nameof(sparseUpdates));
        }

        int len = sparseUpdates.Values.First().TotalLength;
        var aggregated = new double[len];
        var weightSums = new double[len];

        foreach (var (clientId, sparse) in sparseUpdates)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;

            for (int i = 0; i < sparse.Indices.Length; i++)
            {
                int idx = sparse.Indices[i];
                aggregated[idx] += w * NumOps.ToDouble(sparse.Values[i]);
                weightSums[idx] += w;
            }
        }

        var result = new T[len];
        for (int i = 0; i < len; i++)
        {
            if (weightSums[i] > 0)
            {
                result[i] = NumOps.FromDouble(aggregated[i] / weightSums[i]);
            }
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes the actual communication cost of a sparse update (number of non-zero elements).
    /// </summary>
    /// <param name="sparse">The sparse update.</param>
    /// <returns>Number of values transmitted (each needing an index + value pair).</returns>
    public static int ComputeCommunicationCost(SparseUpdate<T> sparse)
    {
        return sparse.Indices.Length;
    }

    /// <summary>Gets the sparsity ratio (fraction of elements communicated).</summary>
    public double SparsityRatio => _sparsityRatio;

    /// <summary>Gets the LoRA rank.</summary>
    public int Rank => _rank;
}

/// <summary>
/// Represents a sparse update: only a subset of indices have non-zero values.
/// This is the communication-efficient representation used by SparseLoRA.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class SparseUpdate<T>
{
    /// <summary>Creates a new sparse update.</summary>
    /// <param name="indices">Indices of non-zero elements.</param>
    /// <param name="values">Values at those indices.</param>
    /// <param name="totalLength">Total vector length (including zero positions).</param>
    public SparseUpdate(int[] indices, T[] values, int totalLength)
    {
        Guard.NotNull(indices);
        Guard.NotNull(values);
        if (indices.Length != values.Length)
        {
            throw new ArgumentException("Indices and values must have the same length.");
        }

        if (totalLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalLength), "Total length must be non-negative.");
        }

        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= totalLength)
            {
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {indices[i]} at position {i} is out of bounds [0, {totalLength}).");
            }
        }

        Indices = indices;
        Values = values;
        TotalLength = totalLength;
    }

    /// <summary>Indices of non-zero elements.</summary>
    public int[] Indices { get; }

    /// <summary>Values at the non-zero indices.</summary>
    public T[] Values { get; }

    /// <summary>Total length of the full dense vector.</summary>
    public int TotalLength { get; }

    /// <summary>Number of non-zero elements.</summary>
    public int NnzCount => Indices.Length;

    /// <summary>Effective sparsity: fraction of elements that are non-zero.</summary>
    public double Density => TotalLength > 0 ? (double)NnzCount / TotalLength : 0;
}
