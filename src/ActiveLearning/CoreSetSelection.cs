using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements core-set selection using the k-center-greedy algorithm for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Core-set selection aims to select samples that best represent
/// the overall data distribution. It uses the k-center algorithm which iteratively selects
/// the sample that is farthest from all previously selected samples. This ensures good
/// coverage of the feature space.</para>
///
/// <para><b>Algorithm (k-center-greedy):</b></para>
/// <list type="number">
/// <item><description>Start with a random sample or the one farthest from the origin.</description></item>
/// <item><description>Repeat until batch is full:
///   <list type="bullet">
///   <item><description>For each remaining sample, compute distance to nearest selected sample.</description></item>
///   <item><description>Select the sample with maximum minimum-distance.</description></item>
///   </list>
/// </description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Ensures diverse coverage of feature space.</description></item>
/// <item><description>Works independently of model predictions.</description></item>
/// <item><description>Good for initial exploration before model is trained.</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n × k × d) where n=pool size, k=batch size, d=feature dimension.</para>
/// <para><b>Reference:</b> Sener &amp; Savarese, "Active Learning for Convolutional Neural Networks:
/// A Core-Set Approach" (ICLR 2018).</para>
/// </remarks>
public class CoreSetSelection<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines the distance metric to use for core-set selection.
    /// </summary>
    public enum DistanceMetric
    {
        /// <summary>Standard Euclidean (L2) distance.</summary>
        Euclidean,
        /// <summary>Manhattan (L1) distance.</summary>
        Manhattan,
        /// <summary>Cosine distance (1 - cosine similarity).</summary>
        Cosine
    }

    private readonly DistanceMetric _distanceMetric;

    /// <summary>
    /// Initializes a new instance of the CoreSetSelection class.
    /// </summary>
    /// <param name="distanceMetric">The distance metric to use (default: Euclidean).</param>
    public CoreSetSelection(DistanceMetric distanceMetric = DistanceMetric.Euclidean)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _distanceMetric = distanceMetric;
        _useBatchDiversity = true; // Core-set is inherently diversity-based
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"CoreSetSelection-{_distanceMetric}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value; // Note: Core-set always uses diversity inherently
    }

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);
        var featureSize = unlabeledPool.Length / numSamples;

        // Use k-center-greedy algorithm
        return KCenterGreedy(unlabeledPool, featureSize, batchSize);
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        var featureSize = unlabeledPool.Length / numSamples;
        var scores = new Vector<T>(numSamples);

        // Score each sample by its distance to the center of the data
        var center = ComputeDataCenter(unlabeledPool, numSamples, featureSize);

        for (int i = 0; i < numSamples; i++)
        {
            scores[i] = ComputeDistanceToCenter(unlabeledPool, i, center, featureSize);
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
    /// Implements the k-center-greedy algorithm.
    /// </summary>
    private int[] KCenterGreedy(Tensor<T> pool, int featureSize, int batchSize)
    {
        var numSamples = pool.Shape[0];
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, numSamples));

        // Track minimum distance to selected set for each remaining sample
        var minDistToSelected = new T[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            minDistToSelected[i] = _numOps.MaxValue;
        }

        // Start with the sample farthest from origin (or we could use random)
        var firstIdx = FindFarthestFromOrigin(pool, featureSize);
        selected.Add(firstIdx);
        remaining.Remove(firstIdx);

        // Update distances
        foreach (var idx in remaining)
        {
            var dist = ComputeDistance(pool, idx, firstIdx, featureSize);
            minDistToSelected[idx] = dist;
        }

        // Iteratively select sample with maximum min-distance to selected set
        while (selected.Count < batchSize && remaining.Count > 0)
        {
            var best = -1;
            var bestDist = _numOps.MinValue;

            foreach (var idx in remaining)
            {
                if (_numOps.GreaterThan(minDistToSelected[idx], bestDist))
                {
                    bestDist = minDistToSelected[idx];
                    best = idx;
                }
            }

            if (best >= 0)
            {
                selected.Add(best);
                remaining.Remove(best);

                // Update min distances for remaining samples
                foreach (var idx in remaining)
                {
                    var dist = ComputeDistance(pool, idx, best, featureSize);
                    if (_numOps.LessThan(dist, minDistToSelected[idx]))
                    {
                        minDistToSelected[idx] = dist;
                    }
                }
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Finds the sample farthest from the origin.
    /// </summary>
    private int FindFarthestFromOrigin(Tensor<T> pool, int featureSize)
    {
        var numSamples = pool.Shape[0];
        var maxDist = _numOps.MinValue;
        var maxIdx = 0;

        for (int i = 0; i < numSamples; i++)
        {
            var dist = _numOps.Zero;
            var start = i * featureSize;

            for (int j = 0; j < featureSize; j++)
            {
                var val = pool[start + j];
                dist = _numOps.Add(dist, _numOps.Multiply(val, val));
            }

            if (_numOps.GreaterThan(dist, maxDist))
            {
                maxDist = dist;
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /// <summary>
    /// Computes distance between two samples based on the configured metric.
    /// </summary>
    private T ComputeDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        return _distanceMetric switch
        {
            DistanceMetric.Euclidean => ComputeEuclideanDistance(pool, idx1, idx2, featureSize),
            DistanceMetric.Manhattan => ComputeManhattanDistance(pool, idx1, idx2, featureSize),
            DistanceMetric.Cosine => ComputeCosineDistance(pool, idx1, idx2, featureSize),
            _ => ComputeEuclideanDistance(pool, idx1, idx2, featureSize)
        };
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
    /// Computes Manhattan distance between two samples.
    /// </summary>
    private T ComputeManhattanDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var sum = _numOps.Zero;
        var start1 = idx1 * featureSize;
        var start2 = idx2 * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var diff = _numOps.Subtract(pool[start1 + i], pool[start2 + i]);
            sum = _numOps.Add(sum, _numOps.FromDouble(Math.Abs(_numOps.ToDouble(diff))));
        }

        return sum;
    }

    /// <summary>
    /// Computes cosine distance between two samples: 1 - cosine_similarity.
    /// </summary>
    private T ComputeCosineDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
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

        var normProduct = _numOps.FromDouble(
            Math.Sqrt(_numOps.ToDouble(norm1)) * Math.Sqrt(_numOps.ToDouble(norm2)));

        if (_numOps.ToDouble(normProduct) < 1e-10)
        {
            return _numOps.One;
        }

        var cosineSim = _numOps.Divide(dotProduct, normProduct);
        return _numOps.Subtract(_numOps.One, cosineSim);
    }

    /// <summary>
    /// Computes the center of the data.
    /// </summary>
    private Vector<T> ComputeDataCenter(Tensor<T> pool, int numSamples, int featureSize)
    {
        var center = new Vector<T>(featureSize);

        for (int i = 0; i < numSamples; i++)
        {
            var start = i * featureSize;
            for (int j = 0; j < featureSize; j++)
            {
                center[j] = _numOps.Add(center[j], pool[start + j]);
            }
        }

        var n = _numOps.FromDouble(numSamples);
        for (int j = 0; j < featureSize; j++)
        {
            center[j] = _numOps.Divide(center[j], n);
        }

        return center;
    }

    /// <summary>
    /// Computes distance from a sample to the data center.
    /// </summary>
    private T ComputeDistanceToCenter(Tensor<T> pool, int sampleIdx, Vector<T> center, int featureSize)
    {
        var sumSquared = _numOps.Zero;
        var start = sampleIdx * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var diff = _numOps.Subtract(pool[start + i], center[i]);
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
