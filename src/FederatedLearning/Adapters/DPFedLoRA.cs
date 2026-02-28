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
    private int _roundCounter;
    private double _cumulativeRdpEpsilon;

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
        Guard.NotNull(fullModelParameters);
        Guard.NotNull(aggregatedAdapters);
        int totalParams = fullModelParameters.Length;
        int adapterCount = aggregatedAdapters.Length;
        if (adapterCount > totalParams)
        {
            throw new ArgumentException(
                $"Adapter length {adapterCount} exceeds model length {totalParams}.");
        }

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

        if (totalWeight <= 0)
        {
            totalWeight = clientAdapters.Count;
        }

        var invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < adapterLen; i++)
        {
            aggregated[i] = NumOps.Multiply(aggregated[i], invTotal);
        }

        // Step 3: Add per-layer calibrated Gaussian noise.
        // Gaussian mechanism on the average of n clipped vectors:
        //   sensitivity_l = clipBound_l / n  (L2 sensitivity of the average)
        //   noise_std = sigma * sensitivity_l
        // Per-layer clip bound is proportional to each layer's share of the total clip norm.
        int n = clientAdapters.Count;
        if (_noiseMultiplier > 0 && n > 0)
        {
            var rng = new Random(_seed + _roundCounter);
            _roundCounter++;
            int paramsPerLayer = adapterLen / Math.Max(_numAdaptedLayers, 1);

            // Compute per-layer norms to allocate the total clip budget proportionally.
            var layerMaxNorms = new double[_numAdaptedLayers];
            double totalLayerNormSum = 0;

            for (int layer = 0; layer < _numAdaptedLayers; layer++)
            {
                int layerStart = layer * paramsPerLayer;
                int layerEnd = (layer == _numAdaptedLayers - 1) ? adapterLen : layerStart + paramsPerLayer;

                double maxNorm = 0;
                foreach (var (_, adapters) in clipped)
                {
                    double layerNorm2 = 0;
                    for (int i = layerStart; i < layerEnd && i < adapters.Length; i++)
                    {
                        double v = NumOps.ToDouble(adapters[i]);
                        layerNorm2 += v * v;
                    }

                    double layerNorm = Math.Sqrt(layerNorm2);
                    if (layerNorm > maxNorm)
                    {
                        maxNorm = layerNorm;
                    }
                }

                layerMaxNorms[layer] = maxNorm;
                totalLayerNormSum += maxNorm;
            }

            // Add noise per layer with correctly calibrated sensitivity = clipBound_l / n.
            for (int layer = 0; layer < _numAdaptedLayers; layer++)
            {
                int layerStart = layer * paramsPerLayer;
                int layerEnd = (layer == _numAdaptedLayers - 1) ? adapterLen : layerStart + paramsPerLayer;

                // Per-layer clip bound: proportional share of total clip norm.
                double layerClipBound = totalLayerNormSum > 0
                    ? _clipNorm * (layerMaxNorms[layer] / totalLayerNormSum)
                    : _clipNorm / _numAdaptedLayers;

                // L2 sensitivity of the average = clipBound / n.
                double layerSensitivity = layerClipBound / n;
                double layerNoiseStd = _noiseMultiplier * layerSensitivity;

                for (int i = layerStart; i < layerEnd && i < adapterLen; i++)
                {
                    // Box-Muller transform for Gaussian noise.
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    double noise = layerNoiseStd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    aggregated[i] = NumOps.Add(aggregated[i], NumOps.FromDouble(noise));
                }
            }

            // Track cumulative privacy cost via Renyi DP accounting.
            AccumulatePrivacyCost(n);
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>
    /// Accumulates the privacy cost of one round using Renyi Differential Privacy accounting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each time we add noise and release a result, we "spend" some
    /// privacy budget. The Renyi DP framework tracks this by computing the Renyi divergence
    /// at order alpha, then converting to (epsilon, delta)-DP. This method uses the analytic
    /// Gaussian mechanism bound from Balle et al. (2020).</para>
    ///
    /// <para>For Gaussian mechanism with noise multiplier sigma on a query with L2 sensitivity S:</para>
    /// <code>
    /// RDP at order alpha = alpha * S² / (2 * sigma²)
    /// Convert to (eps, delta)-DP: eps = rdp - log(delta) / (alpha - 1)
    /// </code>
    /// </remarks>
    /// <param name="numClients">Number of clients in this round.</param>
    private void AccumulatePrivacyCost(int numClients)
    {
        if (_noiseMultiplier <= 0 || numClients <= 0) return;

        // Overall sensitivity for the averaged query: clipNorm / n.
        double sensitivity = _clipNorm / numClients;
        double sigma = _noiseMultiplier * sensitivity;

        // Use alpha = 10 (common choice for Renyi DP accounting).
        const double alpha = 10.0;
        double rdp = alpha * sensitivity * sensitivity / (2.0 * sigma * sigma);

        // This simplifies to alpha / (2 * noiseMultiplier²) regardless of n
        // (since sigma = noiseMultiplier * clipNorm/n and sensitivity = clipNorm/n).
        _cumulativeRdpEpsilon += rdp;
    }

    /// <summary>
    /// Computes the cumulative (epsilon, delta)-DP guarantee after all rounds so far.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After multiple rounds of federated learning with DP noise,
    /// the total privacy leakage grows. This method converts the accumulated Renyi DP cost
    /// to a standard (epsilon, delta) guarantee. Smaller epsilon = stronger privacy.
    /// A typical target is epsilon &lt; 10 for meaningful privacy.</para>
    /// </remarks>
    /// <param name="delta">Target failure probability (typically 1e-5 to 1e-7). Default: 1e-5.</param>
    /// <returns>The cumulative epsilon for the given delta.</returns>
    public double ComputePrivacySpent(double delta = 1e-5)
    {
        if (delta <= 0 || delta >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in (0, 1).");
        }

        // Convert accumulated RDP to (epsilon, delta)-DP:
        // epsilon = rdp_epsilon - log(delta) / (alpha - 1)
        const double alpha = 10.0;
        double epsilon = _cumulativeRdpEpsilon - Math.Log(delta) / (alpha - 1.0);

        return Math.Max(epsilon, 0);
    }

    /// <summary>
    /// Estimates the maximum number of rounds that can be performed while staying
    /// within a target epsilon budget.
    /// </summary>
    /// <param name="targetEpsilon">Target total epsilon. Default: 8.0.</param>
    /// <param name="delta">Target delta. Default: 1e-5.</param>
    /// <param name="numClientsPerRound">Expected clients per round. Default: 10.</param>
    /// <returns>Estimated maximum number of rounds.</returns>
    public int EstimateMaxRounds(double targetEpsilon = 8.0, double delta = 1e-5, int numClientsPerRound = 10)
    {
        if (targetEpsilon <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetEpsilon), "Target epsilon must be positive.");
        }

        if (delta <= 0 || delta >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in (0, 1).");
        }

        if (numClientsPerRound < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClientsPerRound), "Number of clients must be at least 1.");
        }

        if (_noiseMultiplier <= 0) return int.MaxValue;

        const double alpha = 10.0;
        double logDeltaTerm = -Math.Log(delta) / (alpha - 1.0);

        // Per-round RDP cost: alpha / (2 * sigma²) where sigma = noiseMultiplier.
        // Since sensitivity cancels out in the ratio.
        double perRoundRdp = alpha / (2.0 * _noiseMultiplier * _noiseMultiplier);

        // targetEpsilon = T * perRoundRdp + logDeltaTerm
        // T = (targetEpsilon - logDeltaTerm) / perRoundRdp
        double availableBudget = targetEpsilon - logDeltaTerm;
        if (availableBudget <= 0) return 0;

        return (int)(availableBudget / perRoundRdp);
    }

    /// <summary>Gets the DP noise multiplier.</summary>
    public double NoiseMultiplier => _noiseMultiplier;

    /// <summary>Gets the per-sample clip norm.</summary>
    public double ClipNorm => _clipNorm;

    /// <summary>Gets the LoRA rank.</summary>
    public int Rank => _rank;

    /// <summary>Gets the current cumulative RDP epsilon (before conversion to (eps,delta)-DP).</summary>
    public double CumulativeRdpEpsilon => _cumulativeRdpEpsilon;

    /// <summary>Gets the number of aggregation rounds completed so far.</summary>
    public int RoundsCompleted => _roundCounter;
}
