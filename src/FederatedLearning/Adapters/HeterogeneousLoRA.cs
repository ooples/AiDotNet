using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        var adapterParams = new Vector<T>(adapterCount);
        fullModelParameters.AsSpan().Slice(start, adapterCount).CopyTo(adapterParams.AsWritableSpan());
        return adapterParams;
    }

    /// <inheritdoc/>
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
    /// <remarks>
    /// Returns the global adapters at the server's target rank; see <see cref="AggregateAdaptersForRank"/>.
    /// </remarks>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        return AggregateAdaptersForRank(clientAdapters, clientWeights, _targetRank);
    }

    /// <summary>
    /// Aggregates heterogeneous-rank LoRA adapters and re-decomposes the result at the requested rank.
    /// </summary>
    /// <remarks>
    /// <para>FlexLoRA (Bai et al. 2024): every client k sends B_k [layerDim x r_k] followed by
    /// A_k [r_k x layerDim], so its rank is r_k = length / (2 * layerDim). The server forms the full-size
    /// update dW = sum_k p_k B_k A_k, takes its SVD dW = U S V^T and hands each client the rank-r_k
    /// truncation B = U_r S_r, A = V_r^T (the best rank-r approximation of dW).</para>
    /// <para><b>For Beginners:</b> Call this with each client's own rank to give it a global adapter it
    /// can load directly.</para>
    /// </remarks>
    /// <param name="clientAdapters">Adapter parameters from each client, possibly with different ranks.</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <param name="rank">Rank of the returned adapters.</param>
    /// <returns>Adapters [B (layerDim x rank), A (rank x layerDim)] of length 2 * layerDim * rank.</returns>
    public Vector<T> AggregateAdaptersForRank(
        Dictionary<int, Vector<T>> clientAdapters,
        Dictionary<int, double>? clientWeights,
        int rank)
    {
        Guard.NotNull(clientAdapters);
        if (clientAdapters.Count == 0)
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        if (rank <= 0)
            throw new ArgumentOutOfRangeException(nameof(rank), "Rank must be positive.");

        int blockSize = 2 * _layerDim;
        double totalWeight = 0;
        foreach (var (clientId, adapters) in clientAdapters)
        {
            if (adapters.Length == 0 || adapters.Length % blockSize != 0)
            {
                throw new ArgumentException(
                    $"Client {clientId} adapter length {adapters.Length} is not a positive multiple of 2 * layerDim ({blockSize}).",
                    nameof(clientAdapters));
            }

            int clientRank = adapters.Length / blockSize;
            if (clientRank > _maxRank)
            {
                throw new ArgumentException(
                    $"Client {clientId} rank {clientRank} exceeds the maximum supported rank {_maxRank}.",
                    nameof(clientAdapters));
            }

            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        if (totalWeight <= 0)
        {
            throw new ArgumentException("Total client weight must be positive.", nameof(clientWeights));
        }

        // Step 1: reconstruct and average the full-size updates dW = sum_k p_k B_k A_k.
        var deltaW = new Tensor<T>(new[] { _layerDim, _layerDim });
        foreach (var (clientId, adapters) in clientAdapters)
        {
            int clientRank = adapters.Length / blockSize;
            int factorSize = _layerDim * clientRank;
            var b = new Tensor<T>(new[] { _layerDim, clientRank }, adapters.GetSubVector(0, factorSize));
            var a = new Tensor<T>(new[] { clientRank, _layerDim }, adapters.GetSubVector(factorSize, factorSize));
            double p = (clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0) / totalWeight;
            deltaW = Engine.TensorAdd(deltaW, Engine.TensorMultiplyScalar(Engine.TensorMatMul(b, a), NumOps.FromDouble(p)));
        }

        // Step 2: SVD and rank-r truncation, B = U_r S_r, A = V_r^T.
        var (u, s, vh) = LinalgDispatch<T>.Svd(deltaW, fullMatrices: false);
        int kept = Math.Min(rank, s.Length);
        var sigma = Engine.TensorSlice(s, new[] { 0 }, new[] { kept }).Reshape(new[] { 1, kept });
        var newB = Engine.TensorMultiply(Engine.TensorSlice(u, new[] { 0, 0 }, new[] { _layerDim, kept }), sigma);
        var newA = Engine.TensorSlice(vh, new[] { 0, 0 }, new[] { kept, _layerDim });

        int outFactorSize = _layerDim * rank;
        var result = new Vector<T>(2 * outFactorSize);
        var resultSpan = result.AsWritableSpan();
        if (kept == rank)
        {
            newB.AsSpan().CopyTo(resultSpan.Slice(0, outFactorSize));
        }
        else
        {
            // Ranks beyond min(layerDim, layerDim) are zero columns of B.
            Engine.TensorConcatenate(new[] { newB, new Tensor<T>(new[] { _layerDim, rank - kept }) }, 1)
                .AsSpan().CopyTo(resultSpan.Slice(0, outFactorSize));
        }

        newA.AsSpan().CopyTo(resultSpan.Slice(outFactorSize, kept * _layerDim));
        return result;
    }
}
