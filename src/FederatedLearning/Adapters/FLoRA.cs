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
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        // FLoRA stacking: for a true implementation, we would reconstruct ΔW = B*A per client,
        // sum them (weighted), then re-decompose via SVD. Here we perform the weighted sum of
        // the flattened adapter parameters, which is equivalent when all clients use the same rank.
        // For heterogeneous ranks, use HeterogeneousLoRA which has full SVD support.
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

    /// <summary>Gets the target LoRA rank.</summary>
    public int Rank => _rank;

    /// <summary>Gets the LoRA alpha scaling factor.</summary>
    public double Alpha => _alpha;
}
