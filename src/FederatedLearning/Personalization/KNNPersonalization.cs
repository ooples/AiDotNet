
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <para>Prediction (Marfoq et al., 2022, Eqs. 4 and 5):</para>
/// <code>
/// p_kNN(y | x) = sum_{i in N_k(x)} 1[y_i = y] * exp(-d(phi(x_i), phi(x)) / tau)
///              / sum_{i in N_k(x)} exp(-d(phi(x_i), phi(x)) / tau)
/// p_final      = lambda * p_kNN + (1 - lambda) * p_global
/// </code>
/// <para>where N_k(x) are the k cached points closest to x under the distance d (Euclidean by
/// default, as in the paper), tau is a temperature (1 in the paper's equation) and lambda is tuned
/// by cross-validation or set heuristically.</para>
///
/// <para>Reference: Marfoq, O., Neglia, G., Kameni, L., Vidal, R. (2022). "Personalized Federated
/// Learning through Local Memorization." ICML 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class KNNPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private const double CosineNormFloor = 1e-10;

    private readonly int _k;
    private readonly double _lambda;
    private readonly double _temperature;
    private readonly DistanceMetricType _distanceMetric;

    // Cache stored as one [count, featureDim] matrix so every query is a handful of whole-tensor ops.
    private Tensor<T>? _cacheFeatures;
    private Tensor<T>? _cacheNorms;
    private int[] _cacheLabels = Array.Empty<int>();

    /// <summary>
    /// Creates a new kNN-Per personalization strategy.
    /// </summary>
    /// <param name="k">Number of nearest neighbors. Default: 5.</param>
    /// <param name="lambda">Weight of kNN component in prediction. Default: 0.5.</param>
    /// <param name="temperature">Temperature tau of the exp(-d / tau) neighbor weights. Default: 1 (the
    /// paper's Eq. 4).</param>
    /// <param name="distanceMetric">Distance between cached and query features:
    /// <see cref="DistanceMetricType.Euclidean"/> (the paper's default) or
    /// <see cref="DistanceMetricType.Cosine"/> (1 - cosine similarity).</param>
    public KNNPersonalization(
        int k = 5,
        double lambda = 0.5,
        double temperature = 1.0,
        DistanceMetricType distanceMetric = DistanceMetricType.Euclidean)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k must be at least 1.");
        }

        if (lambda < 0 || lambda > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(lambda), "Lambda must be in [0, 1].");
        }

        if (!(temperature > 0) || double.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive and finite.");
        }

        if (distanceMetric != DistanceMetricType.Euclidean && distanceMetric != DistanceMetricType.Cosine)
        {
            throw new ArgumentException(
                $"kNN-Per supports {DistanceMetricType.Euclidean} and {DistanceMetricType.Cosine} distances; got {distanceMetric}.",
                nameof(distanceMetric));
        }

        _k = k;
        _lambda = lambda;
        _temperature = temperature;
        _distanceMetric = distanceMetric;
    }

    /// <summary>
    /// Builds the local feature cache from training data.
    /// </summary>
    /// <param name="features">Feature vectors extracted by the global model; all must have the same length.</param>
    /// <param name="labels">Corresponding labels.</param>
    public void BuildCache(T[][] features, int[] labels)
    {
        Guard.NotNull(features);
        Guard.NotNull(labels);
        if (features.Length != labels.Length)
        {
            throw new ArgumentException("Features and labels must have the same count.");
        }

        if (features.Length == 0)
        {
            _cacheFeatures = null;
            _cacheNorms = null;
            _cacheLabels = Array.Empty<int>();
            return;
        }

        int featureDim = -1;
        foreach (var row in features)
        {
            if (row is null)
            {
                throw new ArgumentException("Feature vectors must not be null.", nameof(features));
            }

            if (featureDim < 0)
            {
                featureDim = row.Length;
            }
            else if (row.Length != featureDim)
            {
                throw new ArgumentException(
                    $"All feature vectors must have the same length. Expected {featureDim}, got {row.Length}.",
                    nameof(features));
            }
        }

        if (featureDim < 1)
        {
            throw new ArgumentException("Feature vectors must not be empty.", nameof(features));
        }

        // TensorStack copies every row, so later mutation of the caller's arrays cannot change the cache.
        var cache = Engine.TensorStack(
            features.Select(row => Vector<T>.Wrap(row).AsTensor()).ToArray(), 0);
        _cacheFeatures = cache;
        _cacheNorms = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(cache, cache), new[] { 1 }, false));
        _cacheLabels = (int[])labels.Clone();
    }

    /// <summary>
    /// Performs the kNN-Per lookup: exp(-distance / temperature) weighted voting over the k nearest cached points.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of just counting which class appears most often among
    /// nearest neighbors (uniform kNN), each of the k nearest neighbors votes with weight
    /// exp(-distance / temperature), so closer neighbors contribute more. The weights are then
    /// normalized into a probability distribution over classes. Neighbors whose label is outside
    /// [0, numClasses) are ignored; if none remain the result is uniform.</para>
    /// </remarks>
    /// <param name="queryFeatures">Feature vector for the query input (same length as the cached features).</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <returns>Probability distribution over classes from the weighted kNN vote.</returns>
    public T[] KNNPredict(T[] queryFeatures, int numClasses)
    {
        Guard.NotNull(queryFeatures);
        if (numClasses < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Must have at least 1 class.");
        }

        var cache = _cacheFeatures;
        var cacheNorms = _cacheNorms;
        if (cache is null || cacheNorms is null || _cacheLabels.Length == 0)
        {
            throw new InvalidOperationException("Cache not built. Call BuildCache first.");
        }

        int count = cache.Shape[0];
        int featureDim = cache.Shape[1];
        if (queryFeatures.Length != featureDim)
        {
            throw new ArgumentException(
                $"Query length ({queryFeatures.Length}) must match the cached feature length ({featureDim}).",
                nameof(queryFeatures));
        }

        var query = Vector<T>.Wrap(queryFeatures).AsTensor().Reshape(1, featureDim);
        Tensor<T> distances;
        if (_distanceMetric == DistanceMetricType.Euclidean)
        {
            var diff = Engine.TensorSubtract(cache, Engine.TensorBroadcastTo(query, new[] { count, featureDim }));
            distances = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 1 }, false));
        }
        else
        {
            var dots = Engine.TensorMatMul(cache, query.Reshape(featureDim, 1)).Reshape(count);
            T queryNorm = Engine.Norm(Vector<T>.Wrap(queryFeatures));
            var denom = Engine.TensorMax(
                Engine.TensorMultiplyScalar(cacheNorms, queryNorm), NumOps.FromDouble(CosineNormFloor));
            distances = Engine.TensorAddScalar(Engine.TensorNegate(Engine.TensorDivide(dots, denom)), NumOps.One);
        }

        // The k smallest distances are the k largest negated distances.
        int effectiveK = Math.Min(_k, count);
        var negNearest = Engine.TensorTopK(Engine.TensorNegate(distances), effectiveK, 0, out var nearest);

        // exp(-(d - d_min) / tau): shifting by the nearest distance leaves the normalized weights unchanged
        // and keeps the largest weight at exactly 1, so a distant cache cannot underflow every weight to 0.
        var shifted = Engine.TensorSubtract(
            negNearest,
            Engine.TensorBroadcastTo(Engine.ReduceMax(negNearest, new[] { 0 }, true), new[] { effectiveK }));
        var weights = Engine.TensorExp(Engine.TensorDivideScalar(shifted, NumOps.FromDouble(_temperature)));

        int[] neighborLabels = nearest.ToArray().Select(i => _cacheLabels[i]).ToArray();
        bool[] validLabel = neighborLabels.Select(label => label >= 0 && label < numClasses).ToArray();
        int[] scatterLabels = neighborLabels.Select(label => label >= 0 && label < numClasses ? label : 0).ToArray();
        weights = Engine.TensorWhere(validLabel, weights, new Tensor<T>(new[] { effectiveK }));

        T totalWeight = Engine.TensorSum(weights);
        if (!NumOps.GreaterThan(totalWeight, NumOps.Zero))
        {
            // No neighbor carries a label in [0, numClasses): fall back to the uniform distribution.
            return Enumerable.Repeat(NumOps.FromDouble(1.0 / numClasses), numClasses).ToArray();
        }

        var classMass = Engine.ScatterAdd(
            weights, new Tensor<int>(scatterLabels, new[] { effectiveK }), 0, numClasses);
        return Engine.TensorDivideScalar(classMass, totalWeight).ToArray();
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
    public T[] CombinedPredict(T[] queryFeatures, T[] globalPrediction, int numClasses)
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
        return Engine.TensorAddScaled(
            Vector<T>.Wrap(knnPred).AsTensor(),
            Vector<T>.Wrap(globalPrediction).AsTensor(),
            NumOps.FromDouble(_lambda),
            NumOps.FromDouble(1.0 - _lambda)).ToArray();
    }

    /// <summary>Gets the number of neighbors (k).</summary>
    public int K => _k;

    /// <summary>Gets the kNN mixing weight (lambda).</summary>
    public double Lambda => _lambda;

    /// <summary>Gets the temperature tau of the exp(-d / tau) neighbor weights.</summary>
    public double Temperature => _temperature;

    /// <summary>Gets the distance used to find and weight neighbors.</summary>
    public DistanceMetricType DistanceMetric => _distanceMetric;

    /// <summary>Gets the current cache size.</summary>
    public int CacheSize => _cacheLabels.Length;
}
