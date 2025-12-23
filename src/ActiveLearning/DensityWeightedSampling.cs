using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements density-weighted sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Density-weighted sampling combines uncertainty with density
/// weighting to avoid selecting outliers. Even if a sample is uncertain, it may not be
/// informative if it's an outlier that doesn't represent the data distribution.</para>
///
/// <para><b>Formula:</b> Score(x) = Uncertainty(x) × Density(x)^β</para>
/// <para>where Density(x) is computed using average distance to k nearest neighbors.</para>
///
/// <para><b>Parameters:</b></para>
/// <list type="bullet">
/// <item><description><b>β (beta):</b> Controls density influence. β=1 gives equal weight,
/// β>1 emphasizes density, β&lt;1 emphasizes uncertainty.</description></item>
/// <item><description><b>k:</b> Number of neighbors for density estimation.</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Avoids selecting outliers with high uncertainty.</description></item>
/// <item><description>Selects samples representative of the data distribution.</description></item>
/// <item><description>Configurable balance between uncertainty and density.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Settles, B. &amp; Craven, M. (2008). "An Analysis of Active Learning
/// Strategies for Sequence Labeling Tasks."</para>
/// </remarks>
public class DensityWeightedSampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _beta;
    private readonly int _kNeighbors;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the DensityWeightedSampling class.
    /// </summary>
    /// <param name="beta">Exponent for density weighting (default: 1.0).</param>
    /// <param name="kNeighbors">Number of neighbors for density estimation (default: 10).</param>
    public DensityWeightedSampling(double beta = 1.0, int kNeighbors = 10)
    {
        if (kNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kNeighbors), "kNeighbors must be at least 1.");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = beta;
        _kNeighbors = kNeighbors;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"DensityWeightedSampling-beta{_beta}-k{_kNeighbors}";

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

        // Compute density scores
        var densityScores = ComputeDensityScores(unlabeledPool, numSamples, featureSize);

        // Combine: Score = Uncertainty × Density^β
        for (int i = 0; i < numSamples; i++)
        {
            var uncertainty = _numOps.ToDouble(uncertaintyScores[i]);
            var density = _numOps.ToDouble(densityScores[i]);
            var weightedDensity = Math.Pow(density, _beta);
            scores[i] = _numOps.FromDouble(uncertainty * weightedDensity);
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
    /// Computes density scores based on average distance to k nearest neighbors.
    /// Higher density = more neighbors nearby = more representative.
    /// </summary>
    private Vector<T> ComputeDensityScores(Tensor<T> pool, int numSamples, int featureSize)
    {
        var densityScores = new Vector<T>(numSamples);
        var k = Math.Min(_kNeighbors, numSamples - 1);

        for (int i = 0; i < numSamples; i++)
        {
            // Compute distances to all other samples
            var distances = new List<double>();
            for (int j = 0; j < numSamples; j++)
            {
                if (i != j)
                {
                    var dist = _numOps.ToDouble(ComputeEuclideanDistance(pool, i, j, featureSize));
                    distances.Add(dist);
                }
            }

            // Get average of k smallest distances
            distances.Sort();
            var kDistances = distances.Take(k).ToList();
            var avgDist = kDistances.Count > 0 ? kDistances.Average() : 1.0;

            // Convert to density (inverse of average distance)
            // Add small epsilon to avoid division by zero
            var density = 1.0 / (avgDist + 1e-10);
            densityScores[i] = _numOps.FromDouble(density);
        }

        // Normalize density scores to [0, 1]
        NormalizeDensityScores(densityScores);

        return densityScores;
    }

    /// <summary>
    /// Normalizes density scores to [0, 1] range.
    /// </summary>
    private void NormalizeDensityScores(Vector<T> scores)
    {
        var maxDensity = _numOps.Zero;
        for (int i = 0; i < scores.Length; i++)
        {
            if (_numOps.GreaterThan(scores[i], maxDensity))
            {
                maxDensity = scores[i];
            }
        }

        if (_numOps.ToDouble(maxDensity) > 1e-10)
        {
            for (int i = 0; i < scores.Length; i++)
            {
                scores[i] = _numOps.Divide(scores[i], maxDensity);
            }
        }
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
    /// Selects samples considering both density-weighted scores and diversity.
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
                var dwScore = scores[idx];
                var diversityScore = selected.Count == 0
                    ? _numOps.One
                    : ComputeMinDistanceToSelected(pool, idx, selected, featureSize);

                var combinedScore = _numOps.Multiply(dwScore, diversityScore);

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
