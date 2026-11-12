using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        if (numAdaptedLayers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numAdaptedLayers), "Number of adapted layers must be positive.");
        }

        if (layerDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(layerDim), "Layer dimension must be positive.");
        }

        _method = method;
        _modelDim = modelDim;
        _bottleneckDim = bottleneckDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerDim = layerDim;

        // Prefix tuning uses 2x parameters (separate K and V prefix matrices).
        // Use checked arithmetic to detect integer overflow for large dimensions.
        try
        {
            AdapterParameterCount = _method switch
            {
                PEFTMethod.LoRA => checked(_numAdaptedLayers * 2 * _layerDim * _bottleneckDim),
                PEFTMethod.Adapter => checked(_numAdaptedLayers * (2 * _layerDim * _bottleneckDim + _bottleneckDim)),
                PEFTMethod.PrefixTuning => checked(_numAdaptedLayers * 2 * _bottleneckDim * _layerDim),
                PEFTMethod.BitFit => checked(_numAdaptedLayers * _layerDim),
                _ => checked(_numAdaptedLayers * 2 * _layerDim * _bottleneckDim)
            };
        }
        catch (OverflowException)
        {
            throw new ArgumentException(
                $"Adapter parameter count overflows int32 for numAdaptedLayers={numAdaptedLayers}, layerDim={layerDim}, bottleneckDim={bottleneckDim}.");
        }

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
        var aggregated = new Vector<T>(adapterLen);
        double totalWeight = 0;

        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;
            aggregated = Engine.Add(aggregated, Engine.Multiply(adapters, NumOps.FromDouble(w)));
        }

        return Engine.Multiply(aggregated, NumOps.FromDouble(1.0 / totalWeight));
    }

    /// <summary>
    /// Aggregates adapters for the configured PEFT method.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> FedPETuning (Zhang et al., ACL 2023) aggregates the trainable PEFT
    /// parameters of every method (LoRA B and A, adapters, prefixes, BitFit biases) with FedAvg, so every
    /// method uses the sample-weighted average of all adapter parameters.</para>
    /// </remarks>
    /// <param name="clientAdapters">Adapter parameters from each client.</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <returns>Method-aware aggregated adapters.</returns>
    public Vector<T> AggregateAdaptersMethodAware(
        Dictionary<int, Vector<T>> clientAdapters,
        Dictionary<int, double>? clientWeights)
    {
        if (_method == PEFTMethod.LoRA)
        {
            return AggregateLoRAAdapters(clientAdapters, clientWeights);
        }

        // For Adapter, PrefixTuning, BitFit: standard weighted average is appropriate.
        return AggregateAdapters(clientAdapters, clientWeights);
    }

    private Vector<T> AggregateLoRAAdapters(
        Dictionary<int, Vector<T>> clientAdapters,
        Dictionary<int, double>? clientWeights)
    {
        // LoRA adapter layout per layer: [B (layerDim x rank), A (rank x layerDim)].
        // Validate adapter lengths are uniform.
        int adapterLenLora = clientAdapters.Values.First().Length;
        foreach (var (clientId, adapters) in clientAdapters)
        {
            if (adapters.Length != adapterLenLora)
            {
                throw new ArgumentException(
                    $"Client {clientId} LoRA adapter length {adapters.Length} differs from expected {adapterLenLora}.");
            }
        }

        // FedPETuning aggregates the LoRA factors with FedAvg: the weighted average of every B and A entry
        // of every layer. Entries past the per-layer layout are averaged too rather than dropped.
        var aggregated = new Vector<T>(adapterLenLora);
        double totalWeight = 0;

        foreach (var (clientId, adapters) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;
            aggregated = Engine.Add(aggregated, Engine.Multiply(adapters, NumOps.FromDouble(w)));
        }

        double invTotal = totalWeight > 0 ? 1.0 / totalWeight : 0;
        return Engine.Multiply(aggregated, NumOps.FromDouble(invTotal));
    }

    /// <summary>
    /// Selects which adapter parameters each client should update based on data heterogeneity.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Not all adapter parameters are equally important for every client.
    /// Clients with more heterogeneous data benefit from updating different subsets of parameters.
    /// This method computes a binary mask indicating which parameters each client should train,
    /// based on the gradient variance across its local data.</para>
    ///
    /// <para>Reference: Zhang et al., ACL 2023 — federated-aware parameter selection.</para>
    /// </remarks>
    /// <param name="clientGradientNorms">Per-parameter gradient norm estimates from each client.</param>
    /// <param name="selectionRatio">Fraction of parameters to select (top by norm). Default: 0.8.</param>
    /// <returns>Per-client binary selection masks (true = update this parameter).</returns>
    public Dictionary<int, bool[]> SelectParametersByHeterogeneity(
        Dictionary<int, double[]> clientGradientNorms,
        double selectionRatio = 0.8)
    {
        if (selectionRatio <= 0 || selectionRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(selectionRatio), "Selection ratio must be in (0, 1].");
        }

        var masks = new Dictionary<int, bool[]>();

        foreach (var (clientId, norms) in clientGradientNorms)
        {
            int k = Math.Max(1, (int)(norms.Length * selectionRatio));
            var mask = new bool[norms.Length];

            // Find the k-th largest norm as threshold.
            var sorted = (double[])norms.Clone();
            Array.Sort(sorted);
            double threshold = sorted[norms.Length - k];

            int selected = 0;
            for (int i = 0; i < norms.Length && selected < k; i++)
            {
                if (norms[i] >= threshold)
                {
                    mask[i] = true;
                    selected++;
                }
            }

            masks[clientId] = mask;
        }

        return masks;
    }

    /// <summary>
    /// Applies a selection mask to adapter parameters, zeroing out unselected positions.
    /// </summary>
    /// <param name="adapters">Full adapter parameters.</param>
    /// <param name="mask">Boolean mask from SelectParametersByHeterogeneity.</param>
    /// <returns>Masked adapter parameters.</returns>
    public Vector<T> ApplySelectionMask(Vector<T> adapters, bool[] mask)
    {
        Guard.NotNull(adapters);
        Guard.NotNull(mask);
        // Positions past the end of the mask are unselected.
        var keep = new bool[adapters.Length];
        Array.Copy(mask, keep, Math.Min(mask.Length, adapters.Length));
        return Engine.TensorWhere(keep, adapters.AsTensor(), new Tensor<T>(new[] { adapters.Length })).ToVector();
    }

    /// <summary>Gets the PEFT method being used.</summary>
    public PEFTMethod Method => _method;

    /// <summary>Gets the bottleneck/rank dimension.</summary>
    public int BottleneckDim => _bottleneckDim;
}
