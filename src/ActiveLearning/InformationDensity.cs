using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements information density sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Information density measures how representative a sample is
/// of the overall data distribution by computing its average similarity to all other samples
/// in the pool. This helps select informative samples that are also typical of the data.</para>
///
/// <para><b>Formula:</b> ID(x) = Uncertainty(x) × [1/|U| × Σ sim(x, x')]^β</para>
/// <para>where sim(x, x') is the similarity between samples (typically cosine or RBF kernel).</para>
///
/// <para><b>Intuition:</b> A sample is information-dense if:</para>
/// <list type="bullet">
/// <item><description>It has high uncertainty (model is unsure).</description></item>
/// <item><description>It is similar to many other samples (representative).</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Balances informativeness and representativeness.</description></item>
/// <item><description>Avoids selecting outliers.</description></item>
/// <item><description>Configurable β parameter for tuning.</description></item>
/// </list>
///
/// <para><b>Reference:</b> McCallum, A. &amp; Nigam, K. (1998). "Employing EM and Pool-Based
/// Active Learning for Text Classification."</para>
/// </remarks>
public class InformationDensity<T> : IActiveLearningStrategy<T>
{
    /// <summary>
    /// Defines the similarity measure for computing information density.
    /// </summary>
    public enum SimilarityMeasure
    {
        /// <summary>Cosine similarity.</summary>
        Cosine,
        /// <summary>RBF (Gaussian) kernel similarity.</summary>
        RBF,
        /// <summary>Inverse Euclidean distance.</summary>
        InverseEuclidean
    }

    private readonly INumericOperations<T> _numOps;
    private readonly double _beta;
    private readonly double _rbfGamma;
    private readonly SimilarityMeasure _similarityMeasure;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the InformationDensity class.
    /// </summary>
    /// <param name="beta">Exponent for density weighting (default: 1.0).</param>
    /// <param name="similarityMeasure">Similarity measure to use (default: Cosine).</param>
    /// <param name="rbfGamma">Gamma parameter for RBF kernel (default: 1.0).</param>
    public InformationDensity(
        double beta = 1.0,
        SimilarityMeasure similarityMeasure = SimilarityMeasure.Cosine,
        double rbfGamma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = beta;
        _similarityMeasure = similarityMeasure;
        _rbfGamma = rbfGamma;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"InformationDensity-{_similarityMeasure}-beta{_beta}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var scores = ComputeInformativenessScores(model, unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        if (_useBatchDiversity)
        {
            return SelectWithDiversity(scores, unlabeledPool, batchSize);
        }
        else
        {
            return SelectTopScoring(scores, batchSize);
        }
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var predictions = model.Predict(unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        var numClasses = predictions.Length / numSamples;
        var featureSize = unlabeledPool.Length / numSamples;
        var scores = new Vector<T>(numSamples);

        // Compute uncertainty scores (using entropy)
        var uncertaintyScores = new Vector<T>(numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            var probs = ExtractProbabilities(predictions, i, numClasses);
            uncertaintyScores[i] = ComputeEntropy(probs);
        }

        // Compute average similarity to pool
        var avgSimilarities = ComputeAverageSimilarities(unlabeledPool, numSamples, featureSize);

        // Combine: ID(x) = Uncertainty(x) × AvgSimilarity(x)^β
        for (int i = 0; i < numSamples; i++)
        {
            var uncertainty = _numOps.ToDouble(uncertaintyScores[i]);
            var avgSim = _numOps.ToDouble(avgSimilarities[i]);
            var weightedSim = Math.Pow(avgSim, _beta);
            scores[i] = _numOps.FromDouble(uncertainty * weightedSim);
        }

        UpdateStatistics(scores);
        return scores;
    }

    /// <inheritdoc />
    public Dictionary<string, T> GetSelectionStatistics()
    {
        return new Dictionary<string, T>
        {
            ["MinScore"] = _lastMinScore,
            ["MaxScore"] = _lastMaxScore,
            ["MeanScore"] = _lastMeanScore
        };
    }

    /// <summary>
    /// Computes average similarity of each sample to all other samples in the pool.
    /// </summary>
    private Vector<T> ComputeAverageSimilarities(Tensor<T> pool, int numSamples, int featureSize)
    {
        var avgSimilarities = new Vector<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            var totalSim = 0.0;
            var count = 0;

            for (int j = 0; j < numSamples; j++)
            {
                if (i != j)
                {
                    var sim = ComputeSimilarity(pool, i, j, featureSize);
                    totalSim += sim;
                    count++;
                }
            }

            var avgSim = count > 0 ? totalSim / count : 0.0;
            avgSimilarities[i] = _numOps.FromDouble(avgSim);
        }

        return avgSimilarities;
    }

    /// <summary>
    /// Computes similarity between two samples based on configured measure.
    /// </summary>
    private double ComputeSimilarity(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        return _similarityMeasure switch
        {
            SimilarityMeasure.Cosine => ComputeCosineSimilarity(pool, idx1, idx2, featureSize),
            SimilarityMeasure.RBF => ComputeRBFSimilarity(pool, idx1, idx2, featureSize),
            SimilarityMeasure.InverseEuclidean => ComputeInverseEuclidean(pool, idx1, idx2, featureSize),
            _ => ComputeCosineSimilarity(pool, idx1, idx2, featureSize)
        };
    }

    /// <summary>
    /// Computes cosine similarity between two samples.
    /// </summary>
    private double ComputeCosineSimilarity(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var dotProduct = _numOps.Zero;
        var norm1 = _numOps.Zero;
        var norm2 = _numOps.Zero;
        var start1 = idx1 * featureSize;
        var start2 = idx2 * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var v1 = pool[start1 + i];
            var v2 = pool[start2 + i];
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(v1, v2));
            norm1 = _numOps.Add(norm1, _numOps.Multiply(v1, v1));
            norm2 = _numOps.Add(norm2, _numOps.Multiply(v2, v2));
        }

        var normProduct = Math.Sqrt(_numOps.ToDouble(norm1)) * Math.Sqrt(_numOps.ToDouble(norm2));

        if (normProduct < 1e-10)
            return 0.0;

        var cosineSim = _numOps.ToDouble(dotProduct) / normProduct;
        // Normalize from [-1, 1] to [0, 1]
        return (cosineSim + 1.0) / 2.0;
    }

    /// <summary>
    /// Computes RBF (Gaussian) kernel similarity.
    /// </summary>
    private double ComputeRBFSimilarity(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var dist = ComputeEuclideanDistanceDouble(pool, idx1, idx2, featureSize);
        return Math.Exp(-_rbfGamma * dist * dist);
    }

    /// <summary>
    /// Computes inverse Euclidean distance similarity.
    /// </summary>
    private double ComputeInverseEuclidean(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var dist = ComputeEuclideanDistanceDouble(pool, idx1, idx2, featureSize);
        return 1.0 / (1.0 + dist);
    }

    /// <summary>
    /// Computes Euclidean distance returning double.
    /// </summary>
    private double ComputeEuclideanDistanceDouble(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var sumSquared = 0.0;
        var start1 = idx1 * featureSize;
        var start2 = idx2 * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var diff = _numOps.ToDouble(pool[start1 + i]) - _numOps.ToDouble(pool[start2 + i]);
            sumSquared += diff * diff;
        }

        return Math.Sqrt(sumSquared);
    }

    /// <summary>
    /// Computes entropy for uncertainty estimation.
    /// </summary>
    private T ComputeEntropy(Vector<T> probabilities)
    {
        var entropy = _numOps.Zero;
        var epsilon = _numOps.FromDouble(1e-10);

        for (int i = 0; i < probabilities.Length; i++)
        {
            var p = _numOps.Add(probabilities[i], epsilon);
            var logP = _numOps.FromDouble(Math.Log(_numOps.ToDouble(p)));
            var term = _numOps.Multiply(p, logP);
            entropy = _numOps.Subtract(entropy, term);
        }

        return entropy;
    }

    /// <summary>
    /// Extracts probabilities for a single sample from batch predictions.
    /// </summary>
    private Vector<T> ExtractProbabilities(Tensor<T> predictions, int sampleIndex, int numClasses)
    {
        var probs = new Vector<T>(numClasses);
        var startIdx = sampleIndex * numClasses;

        // Apply softmax to get probabilities
        var maxLogit = _numOps.MinValue;
        for (int c = 0; c < numClasses; c++)
        {
            if (_numOps.GreaterThan(predictions[startIdx + c], maxLogit))
            {
                maxLogit = predictions[startIdx + c];
            }
        }

        var expSum = _numOps.Zero;
        for (int c = 0; c < numClasses; c++)
        {
            var shifted = _numOps.Subtract(predictions[startIdx + c], maxLogit);
            var expVal = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
            probs[c] = expVal;
            expSum = _numOps.Add(expSum, expVal);
        }

        for (int c = 0; c < numClasses; c++)
        {
            probs[c] = _numOps.Divide(probs[c], expSum);
        }

        return probs;
    }

    /// <summary>
    /// Selects top-scoring samples without diversity consideration.
    /// </summary>
    private int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        var indexedScores = new List<(int Index, T Score)>();
        for (int i = 0; i < scores.Length; i++)
        {
            indexedScores.Add((i, scores[i]));
        }

        return indexedScores
            .OrderByDescending(x => _numOps.ToDouble(x.Score))
            .Take(batchSize)
            .Select(x => x.Index)
            .ToArray();
    }

    /// <summary>
    /// Selects samples considering both information density and batch diversity.
    /// </summary>
    private int[] SelectWithDiversity(Vector<T> scores, Tensor<T> pool, int batchSize)
    {
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, scores.Length));
        var featureSize = pool.Length / pool.Shape[0];

        while (selected.Count < batchSize && remaining.Count > 0)
        {
            var best = -1;
            var bestCombinedScore = _numOps.MinValue;

            foreach (var idx in remaining)
            {
                var idScore = scores[idx];
                var diversityScore = selected.Count == 0
                    ? _numOps.One
                    : ComputeMinDistanceToSelected(pool, idx, selected, featureSize);

                var combinedScore = _numOps.Multiply(idScore, diversityScore);

                if (_numOps.GreaterThan(combinedScore, bestCombinedScore))
                {
                    bestCombinedScore = combinedScore;
                    best = idx;
                }
            }

            if (best >= 0)
            {
                selected.Add(best);
                remaining.Remove(best);
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Computes minimum distance from a sample to already selected samples.
    /// </summary>
    private T ComputeMinDistanceToSelected(Tensor<T> pool, int sampleIdx, List<int> selected, int featureSize)
    {
        var minDist = _numOps.MaxValue;

        foreach (var selIdx in selected)
        {
            var dist = ComputeEuclideanDistance(pool, sampleIdx, selIdx, featureSize);
            if (_numOps.LessThan(dist, minDist))
            {
                minDist = dist;
            }
        }

        return minDist;
    }

    /// <summary>
    /// Computes Euclidean distance between two samples.
    /// </summary>
    private T ComputeEuclideanDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var sumSquared = _numOps.Zero;
        var start1 = idx1 * featureSize;
        var start2 = idx2 * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var diff = _numOps.Subtract(pool[start1 + i], pool[start2 + i]);
            var squared = _numOps.Multiply(diff, diff);
            sumSquared = _numOps.Add(sumSquared, squared);
        }

        return _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(sumSquared)));
    }

    /// <summary>
    /// Updates selection statistics.
    /// </summary>
    private void UpdateStatistics(Vector<T> scores)
    {
        if (scores.Length == 0) return;

        _lastMinScore = scores[0];
        _lastMaxScore = scores[0];
        var sum = _numOps.Zero;

        for (int i = 0; i < scores.Length; i++)
        {
            if (_numOps.LessThan(scores[i], _lastMinScore))
                _lastMinScore = scores[i];
            if (_numOps.GreaterThan(scores[i], _lastMaxScore))
                _lastMaxScore = scores[i];
            sum = _numOps.Add(sum, scores[i]);
        }

        _lastMeanScore = _numOps.Divide(sum, _numOps.FromDouble(scores.Length));
    }
}
