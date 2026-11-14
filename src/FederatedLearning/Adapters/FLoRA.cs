using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements FLoRA — Federated Low-Rank Adaptation with stacked lossless aggregation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard federated LoRA averages the A and B matrices from
/// different clients, which introduces approximation error. FLoRA instead <em>stacks</em>
/// the local LoRA updates — each client's (B_k, A_k) pair is concatenated vertically/horizontally,
/// preserving all information. The server then uses an SVD-based compression to bring the stacked
/// result back to the target rank. This gives lossless aggregation without the information loss
/// of simple averaging.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// 1. Stack: B_stacked = [B_1; B_2; ...; B_K], A_stacked = [A_1; A_2; ...; A_K]
/// 2. Compute ΔW = B_stacked * A_stacked (full rank update)
/// 3. SVD: ΔW = U Σ V^T, truncate to rank r
/// 4. Return B_new = U[:, :r] * sqrt(Σ[:r]), A_new = sqrt(Σ[:r]) * V[:, :r]^T
/// </code>
///
/// <para>Reference: Wang, Y., et al. (2024). "FLoRA: Federated Fine-Tuning Large Language
/// Models with Heterogeneous Low-Rank Adaptations." arXiv:2405.14739.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FLoRA<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
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
    /// Creates a new FLoRA strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="rank">Target LoRA rank after aggregation. Default: 8.</param>
    /// <param name="alpha">LoRA scaling factor. Default: 16.</param>
    /// <param name="numAdaptedLayers">Number of adapted layers. Default: 4.</param>
    /// <param name="layerInputDim">Input dimension of adapted layers. Default: 768.</param>
    /// <param name="layerOutputDim">Output dimension of adapted layers. Default: 768.</param>
    public FLoRA(
        int modelDim,
        int rank = 8,
        double alpha = 16.0,
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
        _modelDim = modelDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerInputDim = layerInputDim;
        _layerOutputDim = layerOutputDim;

        int paramsPerLayer = _layerOutputDim * _rank + _rank * _layerInputDim;
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
    /// <remarks>
    /// The stored LoRA factors are written back unscaled. LoRA (Hu et al. 2021) applies alpha / rank to the
    /// product B*A in the forward pass (see <see cref="Alpha"/> and <see cref="Rank"/>); scaling both stored
    /// factors here would apply it squared and compound it every round.
    /// </remarks>
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
        int paramsPerLayer = (_layerOutputDim * _rank) + (_rank * _layerInputDim);
        int bSize = _layerOutputDim * _rank; // B matrix: [out_dim x rank]
        int aSize = _rank * _layerInputDim;  // A matrix: [rank x in_dim]

        double totalWeight = 0;
        foreach (var (clientId, _) in clientAdapters)
        {
            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        if (totalWeight <= 0)
        {
            throw new InvalidOperationException("Total client weight must be positive.");
        }

        var aggregated = new Vector<T>(adapterLen);

        // Process each adapted layer independently.
        for (int layer = 0; layer < _numAdaptedLayers; layer++)
        {
            int layerOffset = layer * paramsPerLayer;

            if (layerOffset + paramsPerLayer > adapterLen)
            {
                break;
            }

            // Step 1: Reconstruct weighted sum of ΔW = sum(w_k * B_k * A_k).
            // ΔW is [out_dim x in_dim]; B_k is [out_dim x rank] and A_k is [rank x in_dim], row-major.
            var deltaW = new Tensor<T>(new[] { _layerOutputDim, _layerInputDim });

            foreach (var (clientId, adapters) in clientAdapters)
            {
                double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
                double normalizedW = w / totalWeight;

                var bK = new Tensor<T>(new[] { _layerOutputDim, _rank }, adapters.GetSubVector(layerOffset, bSize));
                var aK = new Tensor<T>(new[] { _rank, _layerInputDim }, adapters.GetSubVector(layerOffset + bSize, aSize));
                deltaW = Engine.TensorAdd(
                    deltaW,
                    Engine.TensorMultiplyScalar(Engine.TensorMatMul(bK, aK), NumOps.FromDouble(normalizedW)));
            }

            // Step 2: Truncated SVD of ΔW re-decomposes it into rank-r factors.
            var (newB, newA) = TruncatedSVD(deltaW, _layerOutputDim, _layerInputDim, _rank);

            // Step 3: Write B_new and A_new back to the aggregated adapter vector.
            var aggregatedSpan = aggregated.AsWritableSpan();
            newB.AsSpan().CopyTo(aggregatedSpan.Slice(layerOffset, bSize));
            newA.AsSpan().CopyTo(aggregatedSpan.Slice(layerOffset + bSize, aSize));
        }

        return aggregated;
    }

    /// <summary>
    /// Truncated SVD. Decomposes M [rows x cols] into B [rows x rank] and A [rank x cols] such that
    /// M ≈ B * A is the best rank-r approximation (Eckart-Young).
    /// Singular values are split evenly: B absorbs sqrt(sigma), A absorbs sqrt(sigma).
    /// Components beyond min(rows, cols) are zero.
    /// </summary>
    private static (Tensor<T> B, Tensor<T> A) TruncatedSVD(Tensor<T> matrix, int rows, int cols, int rank)
    {
        var (u, s, vh) = LinalgDispatch<T>.Svd(matrix, fullMatrices: false);
        int kept = Math.Min(rank, s.Length);

        var sqrtSigma = Engine.TensorSqrt(Engine.TensorSlice(s, new[] { 0 }, new[] { kept }));
        var b = Engine.TensorMultiply(
            Engine.TensorSlice(u, new[] { 0, 0 }, new[] { rows, kept }), sqrtSigma.Reshape(new[] { 1, kept }));
        var a = Engine.TensorMultiply(
            Engine.TensorSlice(vh, new[] { 0, 0 }, new[] { kept, cols }), sqrtSigma.Reshape(new[] { kept, 1 }));

        if (kept < rank)
        {
            b = Engine.TensorConcatenate(new[] { b, new Tensor<T>(new[] { rows, rank - kept }) }, 1);
            a = Engine.TensorConcatenate(new[] { a, new Tensor<T>(new[] { rank - kept, cols }) }, 0);
        }

        return (b, a);
    }

    /// <summary>Gets the target LoRA rank.</summary>
    public int Rank => _rank;

    /// <summary>Gets the LoRA alpha scaling factor.</summary>
    public double Alpha => _alpha;
}
