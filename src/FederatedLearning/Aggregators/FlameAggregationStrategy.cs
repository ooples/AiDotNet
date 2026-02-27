namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FLAME (Filtering via cosine similarity + Adaptive clipping + Noise) for
/// Byzantine-robust federated learning with backdoor resistance.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Backdoor attacks in federated learning try to implant hidden
/// triggers in the global model. FLAME defends against this with a three-step approach:
/// (1) filter out client updates whose direction (cosine similarity) is too different from
/// the majority, (2) clip surviving updates to a common norm to prevent magnitude-based
/// attacks, and (3) add calibrated noise to the aggregated result to erase any residual
/// backdoor signal.</para>
///
/// <para>Pipeline:</para>
/// <list type="number">
/// <item>Compute pairwise cosine similarity, remove clients below threshold (HDBSCAN-inspired)</item>
/// <item>Clip remaining updates to the median norm (adaptive clipping)</item>
/// <item>Average the clipped updates</item>
/// <item>Add Gaussian noise with std = <c>NoiseMultiplier</c> * <c>clipNorm</c></item>
/// </list>
///
/// <para>Reference: Nguyen, T. D., et al. (2022). "FLAME: Taming Backdoors in Federated
/// Learning." USENIX Security 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FlameAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _cosineThreshold;
    private readonly double _noiseMultiplier;
    private readonly int _seed;

    /// <summary>
    /// Initializes a new instance of the <see cref="FlameAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="cosineThreshold">Minimum cosine similarity to the centroid for inclusion.
    /// Default: 0.5. Clients below this are filtered out.</param>
    /// <param name="noiseMultiplier">Gaussian noise multiplier relative to clip norm. Default: 0.001.</param>
    /// <param name="seed">Random seed for noise generation. Default: 42.</param>
    public FlameAggregationStrategy(double cosineThreshold = 0.5, double noiseMultiplier = 0.001, int seed = 42)
    {
        if (cosineThreshold < -1.0 || cosineThreshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(cosineThreshold), "Cosine threshold must be in [-1, 1].");
        }

        if (noiseMultiplier < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseMultiplier), "Noise multiplier must be non-negative.");
        }

        _cosineThreshold = cosineThreshold;
        _noiseMultiplier = noiseMultiplier;
        _seed = seed;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        if (clientModels.Count == 1)
        {
            return clientModels.First().Value;
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);
        var clientIds = clientModels.Keys.ToList();
        int n = clientIds.Count;

        // Flatten to double vectors and compute norms.
        var flatVectors = new double[n][];
        var norms = new double[n];

        for (int c = 0; c < n; c++)
        {
            flatVectors[c] = new double[totalParams];
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    double v = NumOps.ToDouble(cp[i]);
                    flatVectors[c][offset] = v;
                    norms[c] += v * v;
                    offset++;
                }
            }

            norms[c] = Math.Sqrt(norms[c]);
        }

        // Step 1: Compute centroid and filter by cosine similarity.
        var centroid = new double[totalParams];
        for (int c = 0; c < n; c++)
        {
            for (int i = 0; i < totalParams; i++)
            {
                centroid[i] += flatVectors[c][i] / n;
            }
        }

        double centroidNorm = 0;
        for (int i = 0; i < totalParams; i++)
        {
            centroidNorm += centroid[i] * centroid[i];
        }

        centroidNorm = Math.Sqrt(centroidNorm);

        var trusted = new List<int>();
        for (int c = 0; c < n; c++)
        {
            if (norms[c] <= 0 || centroidNorm <= 0)
            {
                continue;
            }

            double dot = 0;
            for (int i = 0; i < totalParams; i++)
            {
                dot += flatVectors[c][i] * centroid[i];
            }

            double cosSim = dot / (norms[c] * centroidNorm);
            if (cosSim >= _cosineThreshold)
            {
                trusted.Add(c);
            }
        }

        // If all clients were filtered, fall back to all clients.
        if (trusted.Count == 0)
        {
            trusted = Enumerable.Range(0, n).ToList();
        }

        // Step 2: Adaptive clipping — clip to median norm of trusted clients.
        var trustedNorms = trusted.Select(c => norms[c]).OrderBy(x => x).ToArray();
        double medianNorm = trustedNorms.Length % 2 == 1
            ? trustedNorms[trustedNorms.Length / 2]
            : (trustedNorms[trustedNorms.Length / 2 - 1] + trustedNorms[trustedNorms.Length / 2]) / 2.0;

        double clipNorm = Math.Max(medianNorm, 1e-10);

        // Step 3: Average the clipped updates.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        double weight = 1.0 / trusted.Count;
        foreach (int c in trusted)
        {
            double scale = norms[c] > clipNorm ? clipNorm / norms[c] : 1.0;
            double combinedScale = scale * weight;
            var csT = NumOps.FromDouble(combinedScale);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], csT));
                }
            }
        }

        // Step 4: Add calibrated Gaussian noise.
        if (_noiseMultiplier > 0)
        {
            double noiseStd = _noiseMultiplier * clipNorm;
            var rng = new Random(_seed);

            foreach (var layerName in layerNames)
            {
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    // Box-Muller transform for Gaussian noise.
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    double noise = noiseStd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    rp[i] = NumOps.Add(rp[i], NumOps.FromDouble(noise));
                }
            }
        }

        return result;
    }

    /// <summary>Gets the cosine similarity threshold for filtering.</summary>
    public double CosineThreshold => _cosineThreshold;

    /// <summary>Gets the noise multiplier for DP noise injection.</summary>
    public double NoiseMultiplier => _noiseMultiplier;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FLAME(τ={_cosineThreshold},σ={_noiseMultiplier})";
}
