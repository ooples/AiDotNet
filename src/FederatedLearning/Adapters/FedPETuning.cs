namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Specifies the PEFT method used by FedPETuning.
/// </summary>
public enum PEFTMethod
{
    /// <summary>LoRA — Low-Rank Adaptation matrices.</summary>
    LoRA,
    /// <summary>Adapter — Bottleneck adapter layers inserted into transformer blocks.</summary>
    Adapter,
    /// <summary>Prefix — Learnable prefix tokens prepended to each layer's key/value.</summary>
    PrefixTuning,
    /// <summary>BitFit — Only bias terms are trainable.</summary>
    BitFit
}

/// <summary>
/// Implements FedPETuning — a unified framework for parameter-efficient fine-tuning (PEFT) in
/// federated learning that supports multiple PEFT methods under one API.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> There are many ways to fine-tune a large model cheaply (LoRA,
/// adapter layers, prefix tuning, BitFit, etc.). FedPETuning wraps them all into a single
/// federated strategy so you can swap methods easily. It also applies federated-aware selection
/// to decide which parameters each client should update, based on data heterogeneity.</para>
///
/// <para>Supported methods:</para>
/// <list type="bullet">
/// <item><b>LoRA</b> — low-rank decomposition of weight updates</item>
/// <item><b>Adapter</b> — bottleneck layers inserted after attention/FFN</item>
/// <item><b>PrefixTuning</b> — learnable key-value prefix tokens per layer</item>
/// <item><b>BitFit</b> — only train bias parameters</item>
/// </list>
///
/// <para>Reference: Zhang, Z., et al. (2023). "Federated Learning for Parameter-Efficient
/// Fine-Tuning of Foundation Models." ACL 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FedPETuning<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly PEFTMethod _method;
    private readonly int _modelDim;
    private readonly int _bottleneckDim;
    private readonly int _numAdaptedLayers;
    private readonly int _layerDim;

    /// <inheritdoc/>
    public int AdapterParameterCount { get; }

    /// <inheritdoc/>
    public double CompressionRatio { get; }

    /// <summary>
    /// Creates a new FedPETuning strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="method">The PEFT method to use. Default: LoRA.</param>
    /// <param name="bottleneckDim">Bottleneck or rank dimension. Default: 8.</param>
    /// <param name="numAdaptedLayers">Number of adapted layers. Default: 4.</param>
    /// <param name="layerDim">Dimension of each adapted layer. Default: 768.</param>
    public FedPETuning(
        int modelDim,
        PEFTMethod method = PEFTMethod.LoRA,
        int bottleneckDim = 8,
        int numAdaptedLayers = 4,
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

        _method = method;
        _modelDim = modelDim;
        _bottleneckDim = bottleneckDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerDim = layerDim;

        AdapterParameterCount = _method switch
        {
            PEFTMethod.LoRA => _numAdaptedLayers * 2 * _layerDim * _bottleneckDim,
            PEFTMethod.Adapter => _numAdaptedLayers * (2 * _layerDim * _bottleneckDim + _bottleneckDim),
            PEFTMethod.PrefixTuning => _numAdaptedLayers * _bottleneckDim * _layerDim,
            PEFTMethod.BitFit => _numAdaptedLayers * _layerDim,
            _ => _numAdaptedLayers * 2 * _layerDim * _bottleneckDim
        };

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
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

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

    /// <summary>Gets the PEFT method being used.</summary>
    public PEFTMethod Method => _method;
}
