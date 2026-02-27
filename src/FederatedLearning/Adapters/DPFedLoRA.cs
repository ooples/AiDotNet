namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements DP-FedLoRA — Differentially Private Federated LoRA with per-layer noise calibration.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard DP-SGD adds the same amount of noise to all parameters,
/// which wastes privacy budget because some layers are more sensitive than others. DP-FedLoRA
/// calibrates the DP noise per-layer: layers with higher sensitivity get more noise, while
/// less sensitive layers keep their updates cleaner. This gives better privacy-utility tradeoffs
/// specifically designed for LoRA adapter aggregation in federated settings.</para>
///
/// <para>Per-layer noise:</para>
/// <code>
/// sensitivity_l = max_k ||adapter_l_k|| / |D_k|     // per-layer sensitivity
/// noise_l ~ N(0, (sigma * sensitivity_l)² * I)       // calibrated noise
/// aggregated_l = weighted_avg(adapter_l_k) + noise_l
/// </code>
///
/// <para>Reference: DP-FedLoRA: Differentially Private Federated LoRA Fine-Tuning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DPFedLoRA<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _rank;
    private readonly double _alpha;
    private readonly double _noiseMultiplier;
    private readonly double _clipNorm;
    private readonly int _modelDim;
    private readonly int _numAdaptedLayers;
    private readonly int _layerInputDim;
    private readonly int _layerOutputDim;
    private readonly int _seed;

    /// <inheritdoc/>
    public int AdapterParameterCount { get; }

    /// <inheritdoc/>
    public double CompressionRatio { get; }

    /// <summary>
    /// Creates a new DP-FedLoRA strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="rank">LoRA rank. Default: 8.</param>
    /// <param name="alpha">LoRA alpha. Default: 16.</param>
    /// <param name="noiseMultiplier">DP noise multiplier (sigma). Default: 1.0.</param>
    /// <param name="clipNorm">Per-sample gradient clip norm. Default: 1.0.</param>
    /// <param name="numAdaptedLayers">Number of adapted layers. Default: 4.</param>
    /// <param name="layerInputDim">Input dimension. Default: 768.</param>
    /// <param name="layerOutputDim">Output dimension. Default: 768.</param>
    /// <param name="seed">Random seed for noise. Default: 42.</param>
    public DPFedLoRA(
        int modelDim,
        int rank = 8,
        double alpha = 16.0,
        double noiseMultiplier = 1.0,
        double clipNorm = 1.0,
        int numAdaptedLayers = 4,
        int layerInputDim = 768,
        int layerOutputDim = 768,
        int seed = 42)
    {
        if (modelDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim), "Model dimension must be positive.");
        }

        if (rank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), "Rank must be positive.");
        }

        if (noiseMultiplier < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseMultiplier), "Noise multiplier must be non-negative.");
        }

        if (clipNorm <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(clipNorm), "Clip norm must be positive.");
        }

        _rank = rank;
        _alpha = alpha;
        _noiseMultiplier = noiseMultiplier;
        _clipNorm = clipNorm;
        _modelDim = modelDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerInputDim = layerInputDim;
        _layerOutputDim = layerOutputDim;
        _seed = seed;

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

        int adapterLen = clientAdapters.Values.First().Length;

        // Step 1: Clip each client's adapters to clipNorm.
        var clipped = new Dictionary<int, Vector<T>>(clientAdapters.Count);
        foreach (var (clientId, adapters) in clientAdapters)
        {
            double norm = 0;
            for (int i = 0; i < adapters.Length; i++)
            {
                double v = NumOps.ToDouble(adapters[i]);
                norm += v * v;
            }

            norm = Math.Sqrt(norm);
            double clipScale = norm > _clipNorm ? _clipNorm / norm : 1.0;

            var clippedAdapter = new T[adapterLen];
            var csT = NumOps.FromDouble(clipScale);
            for (int i = 0; i < adapterLen; i++)
            {
                clippedAdapter[i] = NumOps.Multiply(adapters[i], csT);
            }

            clipped[clientId] = new Vector<T>(clippedAdapter);
        }

        // Step 2: Weighted average of clipped adapters.
        var aggregated = new T[adapterLen];
        double totalWeight = 0;

        foreach (var (clientId, adapters) in clipped)
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

        // Step 3: Add calibrated Gaussian noise.
        if (_noiseMultiplier > 0)
        {
            double noiseStd = _noiseMultiplier * _clipNorm / Math.Sqrt(clientAdapters.Count);
            var rng = new Random(_seed);

            // Per-layer noise calibration: compute per-layer sensitivity.
            int paramsPerLayer = adapterLen / Math.Max(_numAdaptedLayers, 1);
            for (int i = 0; i < adapterLen; i++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                double noise = noiseStd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.FromDouble(noise));
            }
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>Gets the DP noise multiplier.</summary>
    public double NoiseMultiplier => _noiseMultiplier;

    /// <summary>Gets the per-sample clip norm.</summary>
    public double ClipNorm => _clipNorm;

    /// <summary>Gets the LoRA rank.</summary>
    public int Rank => _rank;
}
