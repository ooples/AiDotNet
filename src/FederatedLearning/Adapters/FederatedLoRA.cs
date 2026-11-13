using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        // Copy frozen base parameters, then replace adapter parameters with aggregated ones.
        var merged = fullModelParameters.Clone();
        aggregatedAdapters.AsSpan().CopyTo(merged.AsWritableSpan().Slice(start, adapterCount));
        return merged;
    }

    /// <inheritdoc/>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        if (clientAdapters.Count == 0)
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));

        int adapterLen = clientAdapters.Values.First().Length;
        var aggregated = new Vector<T>(adapterLen);
        double totalWeight = 0;

        // FedEx-LoRA: A and B are each the weighted average of the clients' factors.
        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;
            aggregated = Engine.Add(aggregated, Engine.Multiply(adapters, NumOps.FromDouble(w)));
        }

        return Engine.Multiply(aggregated, NumOps.FromDouble(1.0 / totalWeight));
    }

    /// <summary>
    /// Computes the FedEx-LoRA residual that makes the aggregation exact.
    /// </summary>
    /// <remarks>
    /// <para>Averaging B and A separately gives avg(B) * avg(A), which differs from the ideal update
    /// avg(B_k * A_k). FedEx-LoRA (Sun et al. 2025) folds the difference into the frozen base weights:</para>
    /// <code>
    /// residual_l = sum_k p_k * B_k,l * A_k,l - (sum_k p_k * B_k,l) * (sum_k p_k * A_k,l)
    /// W0_l += (alpha / rank) * residual_l
    /// </code>
    /// <para><b>For Beginners:</b> The averaged adapters lose a little information. This returns, per
    /// adapted layer, the exact correction to add to that layer's frozen weight matrix (after scaling by
    /// alpha / rank) so the global model equals the average of the clients' fine-tuned models.</para>
    /// </remarks>
    /// <param name="clientAdapters">Adapter parameters from each client (layout per layer: B then A).</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <returns>Residuals of shape [numAdaptedLayers, layerOutputDim, layerInputDim].</returns>
    public Tensor<T> ComputeExactAggregationResidual(
        Dictionary<int, Vector<T>> clientAdapters,
        Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientAdapters);
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        int bSize = _layerOutputDim * _rank;
        int aSize = _rank * _layerInputDim;
        int paramsPerLayer = bSize + aSize;
        int required = _numAdaptedLayers * paramsPerLayer;
        foreach (var (clientId, adapters) in clientAdapters)
        {
            if (adapters.Length < required)
            {
                throw new ArgumentException(
                    $"Client {clientId} adapter length {adapters.Length} is shorter than the LoRA layout ({required}).",
                    nameof(clientAdapters));
            }
        }

        double totalWeight = 0;
        foreach (var clientId in clientAdapters.Keys)
        {
            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        if (totalWeight <= 0)
        {
            throw new ArgumentException("Total client weight must be positive.", nameof(clientWeights));
        }

        // Batched over layers: B is [layers, out, rank] and A is [layers, rank, in].
        var bShape = new[] { _numAdaptedLayers, _layerOutputDim, _rank };
        var aShape = new[] { _numAdaptedLayers, _rank, _layerInputDim };
        var products = new List<Tensor<T>>(clientAdapters.Count);
        var factorsB = new List<Tensor<T>>(clientAdapters.Count);
        var factorsA = new List<Tensor<T>>(clientAdapters.Count);
        var weights = new List<double>(clientAdapters.Count);
        foreach (var (clientId, adapters) in clientAdapters)
        {
            var layers = new Tensor<T>(new[] { _numAdaptedLayers, paramsPerLayer }, adapters.GetSubVector(0, required));
            var b = Engine.TensorSlice(layers, new[] { 0, 0 }, new[] { _numAdaptedLayers, bSize }).Reshape(bShape);
            var a = Engine.TensorSlice(layers, new[] { 0, bSize }, new[] { _numAdaptedLayers, aSize }).Reshape(aShape);
            products.Add(Engine.TensorMatMul(b, a));
            factorsB.Add(b);
            factorsA.Add(a);
            weights.Add(clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0);
        }

        var meanProduct = Infrastructure.FederatedTensorMath<T>.WeightedSum(products, weights, totalWeight);
        var meanB = Infrastructure.FederatedTensorMath<T>.WeightedSum(factorsB, weights, totalWeight);
        var meanA = Infrastructure.FederatedTensorMath<T>.WeightedSum(factorsA, weights, totalWeight);
        return Engine.TensorSubtract(meanProduct, Engine.TensorMatMul(meanB, meanA));
    }
}
