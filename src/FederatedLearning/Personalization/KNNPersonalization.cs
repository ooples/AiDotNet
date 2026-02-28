namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements kNN-Per — kNN-based personalization at inference time with zero extra training cost.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> kNN-Per is the simplest way to personalize a federated model:
/// after FL training, each client builds a small cache of (feature, label) pairs from their
/// local data using the global model's feature extractor. At inference time, the global model's
/// prediction is combined with a kNN lookup in this local cache. If the test input is similar
/// to local training examples, the kNN component dominates; if it's novel, the global model
/// dominates. This adds zero extra training cost — just a one-time cache construction.</para>
///
/// <para>Prediction:</para>
/// <code>
/// p_final = lambda * p_kNN(features, cache) + (1 - lambda) * p_global(features)
/// </code>
/// <para>where lambda is tuned by cross-validation or set heuristically.</para>
///
/// <para>Reference: Marfoq, O., et al. (2023). "kNN-Per: Nearest Neighbor-Based
/// Personalization for Federated Learning." ICML 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class KNNPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _k;
    private readonly double _lambda;
    private List<(T[] Features, int Label)>? _cache;

    /// <summary>
    /// Creates a new kNN-Per personalization strategy.
    /// </summary>
    /// <param name="k">Number of nearest neighbors. Default: 5.</param>
    /// <param name="lambda">Weight of kNN component in prediction. Default: 0.5.</param>
    public KNNPersonalization(int k = 5, double lambda = 0.5)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k must be at least 1.");
        }

        if (lambda < 0 || lambda > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(lambda), "Lambda must be in [0, 1].");
        }

        _k = k;
        _lambda = lambda;
    }

    /// <summary>
    /// Builds the local feature cache from training data.
    /// </summary>
    /// <param name="features">Feature vectors extracted by the global model.</param>
    /// <param name="labels">Corresponding labels.</param>
    public void BuildCache(T[][] features, int[] labels)
    {
        Guard.NotNull(features);
        Guard.NotNull(labels);
        if (features.Length != labels.Length)
        {
            throw new ArgumentException("Features and labels must have the same count.");
        }

        _cache = new List<(T[] Features, int Label)>(features.Length);
        for (int i = 0; i < features.Length; i++)
        {
            _cache.Add((features[i], labels[i]));
        }
    }

    /// <summary>
    /// Performs distance-weighted kNN lookup using cosine similarity.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of just counting which class appears most often among
    /// nearest neighbors (uniform kNN), distance-weighted kNN gives more weight to closer neighbors.
    /// We use cosine similarity (direction-based) rather than L2 distance (magnitude-based) because
    /// feature extractor representations in deep learning often have meaningful direction but arbitrary
    /// magnitude. The weights are proportional to cosine similarity, so very similar examples
    /// contribute more to the prediction.</para>
    /// </remarks>
    /// <param name="queryFeatures">Feature vector for the query input.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <returns>Probability distribution over classes from distance-weighted kNN.</returns>
    public double[] KNNPredict(T[] queryFeatures, int numClasses)
    {
        Guard.NotNull(queryFeatures);
        if (numClasses < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Must have at least 1 class.");
        }

        if (_cache == null || _cache.Count == 0)
        {
            throw new InvalidOperationException("Cache not built. Call BuildCache first.");
        }

        // Compute cosine similarities to all cache entries.
        double queryNorm = 0;
        var queryDoubles = new double[queryFeatures.Length];
        for (int j = 0; j < queryFeatures.Length; j++)
        {
            queryDoubles[j] = NumOps.ToDouble(queryFeatures[j]);
            queryNorm += queryDoubles[j] * queryDoubles[j];
        }

        queryNorm = Math.Sqrt(queryNorm);

        var similarities = new (double Similarity, int Label)[_cache.Count];
        for (int i = 0; i < _cache.Count; i++)
        {
            var cached = _cache[i].Features;
            int len = Math.Min(queryFeatures.Length, cached.Length);

            double dot = 0, cachedNorm = 0;
            for (int j = 0; j < len; j++)
            {
                double cv = NumOps.ToDouble(cached[j]);
                dot += queryDoubles[j] * cv;
                cachedNorm += cv * cv;
            }

            cachedNorm = Math.Sqrt(cachedNorm);
            double denom = queryNorm * cachedNorm;
            double sim = denom > 1e-10 ? dot / denom : 0;

            similarities[i] = (sim, _cache[i].Label);
        }

        // Sort by similarity (descending) and take top k.
        Array.Sort(similarities, (a, b) => b.Similarity.CompareTo(a.Similarity));
        int effectiveK = Math.Min(_k, similarities.Length);

        // Distance-weighted voting: weight = max(0, similarity).
        var classCounts = new double[numClasses];
        double totalWeight = 0;
        for (int i = 0; i < effectiveK; i++)
        {
            double weight = Math.Max(0, similarities[i].Similarity);
            if (similarities[i].Label >= 0 && similarities[i].Label < numClasses)
            {
                classCounts[similarities[i].Label] += weight;
                totalWeight += weight;
            }
        }

        // Normalize to probability distribution.
        if (totalWeight > 0)
        {
            for (int c = 0; c < numClasses; c++)
            {
                classCounts[c] /= totalWeight;
            }
        }

        return classCounts;
    }

    /// <summary>
    /// Combines kNN prediction with global model prediction per the kNN-Per formula:
    /// p_final = lambda * p_kNN + (1 - lambda) * p_global.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the core kNN-Per prediction formula. The global model's
    /// prediction captures general knowledge learned from all clients, while the kNN prediction
    /// captures local patterns specific to this client's data. Lambda controls the balance:
    /// higher lambda trusts the local cache more, lower lambda trusts the global model more.</para>
    /// </remarks>
    /// <param name="queryFeatures">Feature vector for the query input.</param>
    /// <param name="globalPrediction">Probability distribution from the global model.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <returns>Combined prediction: lambda * p_kNN + (1 - lambda) * p_global.</returns>
    public double[] CombinedPredict(T[] queryFeatures, double[] globalPrediction, int numClasses)
    {
        Guard.NotNull(queryFeatures);
        Guard.NotNull(globalPrediction);
        if (globalPrediction.Length != numClasses)
        {
            throw new ArgumentException(
                $"Global prediction length ({globalPrediction.Length}) must match numClasses ({numClasses}).",
                nameof(globalPrediction));
        }

        var knnPred = KNNPredict(queryFeatures, numClasses);
        var combined = new double[numClasses];

        for (int c = 0; c < numClasses; c++)
        {
            combined[c] = _lambda * knnPred[c] + (1.0 - _lambda) * globalPrediction[c];
        }

        return combined;
    }

    /// <summary>Gets the number of neighbors (k).</summary>
    public int K => _k;

    /// <summary>Gets the kNN mixing weight (lambda).</summary>
    public double Lambda => _lambda;

    /// <summary>Gets the current cache size.</summary>
    public int CacheSize => _cache?.Count ?? 0;
}
