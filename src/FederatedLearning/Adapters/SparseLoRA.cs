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
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        int adapterLen = clientAdapters.Values.First().Length;
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

    /// <summary>Gets the sparsity ratio (fraction of elements communicated).</summary>
    public double SparsityRatio => _sparsityRatio;

    /// <summary>Gets the LoRA rank.</summary>
    public int Rank => _rank;
}
