using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        var adapterParams = new Vector<T>(adapterCount);
        fullModelParameters.AsSpan().Slice(start, adapterCount).CopyTo(adapterParams.AsWritableSpan());
        return adapterParams;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The stored LoRA factors are written back unscaled. LoRA (Hu et al. 2021) applies alpha / rank to the
    /// product B*A in the forward pass; scaling both stored factors here would apply it squared and compound
    /// it every round.
    /// </remarks>
    public Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters)
    {
        int totalParams = fullModelParameters.Length;
        int adapterCount = aggregatedAdapters.Length;
        int start = totalParams - adapterCount;

        var merged = fullModelParameters.Clone();
        aggregatedAdapters.AsSpan().CopyTo(merged.AsWritableSpan().Slice(start, adapterCount));
        return merged;
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

        // Each element is averaged over the clients that sent it (non-zero). Weights are normalized by
        // their total first so the per-element weight sums stay within [0, 1] for narrow T.
        double totalWeight = 0;
        foreach (var clientId in clientAdapters.Keys)
        {
            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        double weightScale = totalWeight > 0 ? 1.0 / totalWeight : 1.0;
        var aggregated = new Tensor<T>(new[] { adapterLen });
        var counts = new Tensor<T>(new[] { adapterLen }); // Weight of the clients that sent each element.

        foreach (var (clientId, adapters) in clientAdapters)
        {
            var wT = NumOps.FromDouble((clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0) * weightScale);
            var sent = Engine.TensorNotEquals(adapters.AsTensor(), NumOps.Zero); // Sparse: only non-zero elements were sent.
            aggregated = Engine.TensorAdd(aggregated, Engine.TensorMultiplyScalar(adapters.AsTensor(), wT));
            counts = Engine.TensorAdd(counts, Engine.TensorMultiplyScalar(sent, wT));
        }

        // Normalize by the weight of contributing clients per element.
        var normalized = Engine.TensorDivide(aggregated, counts);
        return Engine.TensorWhere(Engine.TensorGreaterThan(counts, NumOps.Zero), normalized, aggregated).ToVector();
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
    /// <para>Every element strictly larger than the k-th largest magnitude is kept; ties at the k-th
    /// magnitude fill the remaining slots in index order. Indices are returned in ascending order.</para>
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
        if (len == 0)
        {
            return new SparseUpdate<T>(Array.Empty<int>(), Array.Empty<T>(), 0);
        }

        int k = Math.Max(1, (int)(len * _sparsityRatio));

        // Compute the delta and its magnitudes.
        var delta = Engine.Subtract(adapterAfter, adapterBefore).AsTensor();
        var magnitudes = Engine.TensorAbs(delta);

        // The k-th largest magnitude is the threshold.
        var (sortedMagnitudes, _) = Engine.TensorSort(magnitudes, 0, false);
        T threshold = sortedMagnitudes[len - k];

        // Keep everything above the threshold, then the first ties in index order (counted in int).
        var above = Engine.TensorGreaterThan(magnitudes, threshold).Cast<int>();
        int tieSlots = k - Engine.TensorCountNonzero(above);
        var ties = Engine.TensorEquals(magnitudes, threshold).Cast<int>();
        var tieRank = Engine.TensorCumSum(ties, 0);
        var keptTies = Engine.TensorMultiply(ties, Engine.TensorLessThan(tieRank, tieSlots + 1));
        var selected = Engine.TensorAdd(above, keptTies);

        var nonzero = Engine.TensorNonzero(selected);
        var indexTensor = nonzero.Reshape(new[] { nonzero.Length });
        var values = Engine.TensorGather(delta, indexTensor, 0);
        return new SparseUpdate<T>(indexTensor.ToArray(), values.ToArray(), len);
    }

    /// <summary>
    /// Converts a sparse update back to a dense vector (filling non-sparse positions with zero).
    /// </summary>
    /// <param name="sparse">The sparse update.</param>
    /// <returns>Dense vector representation.</returns>
    public static Vector<T> ToDense(SparseUpdate<T> sparse)
    {
        var dense = new Tensor<T>(new[] { sparse.TotalLength });
        if (sparse.Indices.Length == 0)
        {
            return dense.ToVector();
        }

        var indices = new Tensor<int>(sparse.Indices, new[] { sparse.Indices.Length });
        var values = new Tensor<T>(sparse.Values, new[] { sparse.Values.Length });
        return Engine.TensorIndexCopy(dense, 0, indices, values).ToVector();
    }

    /// <summary>
    /// Applies a sparse update to existing adapter parameters: new = old + sparse_delta.
    /// </summary>
    /// <param name="adapter">Current adapter parameters.</param>
    /// <param name="sparseUpdate">Sparse update delta.</param>
    /// <returns>Updated adapter parameters.</returns>
    public static Vector<T> ApplySparseUpdate(Vector<T> adapter, SparseUpdate<T> sparseUpdate)
    {
        if (sparseUpdate.Indices.Length == 0)
        {
            return adapter.Clone();
        }

        var indices = new Tensor<int>(sparseUpdate.Indices, new[] { sparseUpdate.Indices.Length });
        var values = new Tensor<T>(sparseUpdate.Values, new[] { sparseUpdate.Values.Length });
        return Engine.TensorIndexAdd(adapter.AsTensor().Clone(), 0, indices, values).ToVector();
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

        // Weights are normalized by their total first so the per-position weight sums stay within [0, 1].
        double totalWeight = 0;
        foreach (var clientId in sparseUpdates.Keys)
        {
            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        double weightScale = totalWeight > 0 ? 1.0 / totalWeight : 1.0;
        var aggregated = new Tensor<T>(new[] { len });
        var weightSums = new Tensor<T>(new[] { len });

        foreach (var (clientId, sparse) in sparseUpdates)
        {
            int nnz = sparse.Indices.Length;
            if (nnz == 0)
            {
                continue;
            }

            var wT = NumOps.FromDouble((clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0) * weightScale);
            var indices = new Tensor<int>(sparse.Indices, new[] { nnz });
            var values = new Tensor<T>(sparse.Values, new[] { nnz });
            aggregated = Engine.TensorIndexAdd(aggregated, 0, indices, Engine.TensorMultiplyScalar(values, wT));
            weightSums = Engine.TensorIndexAdd(weightSums, 0, indices, Engine.TensorFullLike(values, wT));
        }

        var normalized = Engine.TensorDivide(aggregated, weightSums);
        return Engine.TensorWhere(
            Engine.TensorGreaterThan(weightSums, NumOps.Zero), normalized, new Tensor<T>(new[] { len })).ToVector();
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
