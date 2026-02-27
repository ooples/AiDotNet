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
    /// Performs kNN lookup to find the most common label among nearest neighbors.
    /// </summary>
    /// <param name="queryFeatures">Feature vector for the query input.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <returns>Probability distribution over classes from kNN.</returns>
    public double[] KNNPredict(T[] queryFeatures, int numClasses)
    {
        if (_cache == null || _cache.Count == 0)
        {
            throw new InvalidOperationException("Cache not built. Call BuildCache first.");
        }

        // Compute distances to all cache entries.
        var distances = new (double Distance, int Label)[_cache.Count];
        for (int i = 0; i < _cache.Count; i++)
        {
            double dist = 0;
            var cached = _cache[i].Features;
            int len = Math.Min(queryFeatures.Length, cached.Length);
            for (int j = 0; j < len; j++)
            {
                double diff = NumOps.ToDouble(queryFeatures[j]) - NumOps.ToDouble(cached[j]);
                dist += diff * diff;
            }

            distances[i] = (dist, _cache[i].Label);
        }

        // Sort and take top k.
        Array.Sort(distances, (a, b) => a.Distance.CompareTo(b.Distance));
        int effectiveK = Math.Min(_k, distances.Length);

        var classCounts = new double[numClasses];
        for (int i = 0; i < effectiveK; i++)
        {
            if (distances[i].Label >= 0 && distances[i].Label < numClasses)
            {
                classCounts[distances[i].Label] += 1.0 / effectiveK;
            }
        }

        return classCounts;
    }

    /// <summary>Gets the number of neighbors (k).</summary>
    public int K => _k;

    /// <summary>Gets the kNN mixing weight (lambda).</summary>
    public double Lambda => _lambda;

    /// <summary>Gets the current cache size.</summary>
    public int CacheSize => _cache?.Count ?? 0;
}
