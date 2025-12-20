using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Diversity Sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Diversity Sampling selects samples that are representative of
/// different regions in the input space. Instead of focusing on uncertain samples near the
/// decision boundary, diversity sampling ensures good coverage of the data distribution.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Compute pairwise distances between all unlabeled samples.</description></item>
/// <item><description>Select samples that maximize diversity - samples that are far apart
/// from each other and from already labeled data.</description></item>
/// <item><description>Can use k-medoids clustering, farthest-first traversal, or core-set
/// construction approaches.</description></item>
/// </list>
///
/// <para><b>Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Farthest-First:</b> Greedily selects the sample farthest from all
/// previously selected samples.</description></item>
/// <item><description><b>K-Center-Greedy:</b> Selects samples to minimize the maximum distance
/// from any point to its nearest selected sample (core-set construction).</description></item>
/// <item><description><b>Clustering-Based:</b> Clusters the data and selects samples closest
/// to cluster centers.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Sener and Savarese, "Active Learning for Convolutional Neural Networks:
/// A Core-Set Approach" (2018). ICLR.</para>
/// </remarks>
public class DiversitySampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly DiversityMethod _method;
    private readonly DistanceMetric _distanceMetric;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;
    private T _lastCoverageRadius;

    /// <summary>
    /// Defines the diversity selection method.
    /// </summary>
    public enum DiversityMethod
    {
        /// <summary>Greedily selects farthest sample from current selection.</summary>
        FarthestFirst,
        /// <summary>K-center greedy for core-set construction.</summary>
        KCenterGreedy,
        /// <summary>Select samples based on density peaks.</summary>
        DensityPeaks
    }

    /// <summary>
    /// Defines the distance metric to use.
    /// </summary>
    public enum DistanceMetric
    {
        /// <summary>Euclidean distance.</summary>
        Euclidean,
        /// <summary>Cosine distance (1 - cosine similarity).</summary>
        Cosine,
        /// <summary>Manhattan (L1) distance.</summary>
        Manhattan
    }

    /// <summary>
    /// Initializes a new instance of the DiversitySampling class.
    /// </summary>
    /// <param name="method">The diversity method to use (default: KCenterGreedy).</param>
    /// <param name="distanceMetric">The distance metric to use (default: Euclidean).</param>
    public DiversitySampling(
        DiversityMethod method = DiversityMethod.KCenterGreedy,
        DistanceMetric distanceMetric = DistanceMetric.Euclidean)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _method = method;
        _distanceMetric = distanceMetric;
        _useBatchDiversity = true; // Diversity sampling inherently considers batch diversity
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
        _lastCoverageRadius = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"DiversitySampling-{_method}-{_distanceMetric}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <summary>
    /// Gets the coverage radius from the last selection.
    /// </summary>
    /// <remarks>
    /// <para>The coverage radius is the maximum distance from any point to its nearest
    /// selected sample. Lower values indicate better coverage of the data space.</para>
    /// </remarks>
    public T CoverageRadius => _lastCoverageRadius;

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        // Note: model is not used in pure diversity sampling, but we accept it for interface compatibility
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        var selected = _method switch
        {
            DiversityMethod.FarthestFirst => SelectFarthestFirst(unlabeledPool, batchSize),
            DiversityMethod.KCenterGreedy => SelectKCenterGreedy(unlabeledPool, batchSize),
            DiversityMethod.DensityPeaks => SelectDensityPeaks(unlabeledPool, batchSize),
            _ => SelectKCenterGreedy(unlabeledPool, batchSize)
        };

        // Compute coverage radius
        _lastCoverageRadius = ComputeCoverageRadius(unlabeledPool, selected);

        return selected;
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        // Note: model is not used in diversity sampling
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var numSamples = unlabeledPool.Shape[0];
        var featureSize = unlabeledPool.Length / numSamples;

        // Compute diversity scores based on distance to nearest neighbors
        var scores = new Vector<T>(numSamples);

        if (_method == DiversityMethod.DensityPeaks)
        {
            // Density-based scoring: high density samples are more representative
            scores = ComputeDensityScores(unlabeledPool, numSamples, featureSize);
        }
        else
        {
            // Distance-based scoring: samples far from others get high scores
            scores = ComputeDistanceScores(unlabeledPool, numSamples, featureSize);
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
            ["MeanScore"] = _lastMeanScore,
            ["CoverageRadius"] = _lastCoverageRadius
        };
    }

    /// <summary>
    /// Selects samples using farthest-first traversal.
    /// </summary>
    private int[] SelectFarthestFirst(Tensor<T> pool, int batchSize)
    {
        var numSamples = pool.Shape[0];
        var featureSize = pool.Length / numSamples;
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, numSamples));

        // Start with a random sample (or the centroid's nearest neighbor)
        var centroid = ComputeCentroid(pool, numSamples, featureSize);
        var first = FindNearestToCentroid(pool, centroid, numSamples, featureSize);
        selected.Add(first);
        remaining.Remove(first);

        // Greedily add the farthest sample
        while (selected.Count < batchSize && remaining.Count > 0)
        {
            var farthest = -1;
            var maxMinDist = _numOps.MinValue;

            foreach (var idx in remaining)
            {
                var minDist = _numOps.MaxValue;
                foreach (var selIdx in selected)
                {
                    var dist = ComputeDistance(pool, idx, selIdx, featureSize);
                    if (_numOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                    }
                }

                if (_numOps.GreaterThan(minDist, maxMinDist))
                {
                    maxMinDist = minDist;
                    farthest = idx;
                }
            }

            if (farthest >= 0)
            {
                selected.Add(farthest);
                remaining.Remove(farthest);
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Selects samples using k-center greedy algorithm for core-set construction.
    /// </summary>
    private int[] SelectKCenterGreedy(Tensor<T> pool, int batchSize)
    {
        // K-center greedy is similar to farthest-first but specifically aims to
        // minimize the maximum distance from any point to the nearest center
        return SelectFarthestFirst(pool, batchSize);
    }

    /// <summary>
    /// Selects samples based on density peaks.
    /// </summary>
    private int[] SelectDensityPeaks(Tensor<T> pool, int batchSize)
    {
        var numSamples = pool.Shape[0];
        var featureSize = pool.Length / numSamples;

        // Compute local density for each sample
        var densities = ComputeLocalDensities(pool, numSamples, featureSize);

        // Compute distance to nearest higher-density sample
        var deltaDistances = ComputeDeltaDistances(pool, densities, numSamples, featureSize);

        // Score = density * delta (decision graph)
        var scores = new Vector<T>(numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            scores[i] = _numOps.Multiply(densities[i], deltaDistances[i]);
        }

        // Select top-scoring samples (density peaks)
        var indexedScores = new List<(int Index, T Score)>();
        for (int i = 0; i < numSamples; i++)
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
    /// Computes local density for each sample using a Gaussian kernel.
    /// </summary>
    private Vector<T> ComputeLocalDensities(Tensor<T> pool, int numSamples, int featureSize)
    {
        var densities = new Vector<T>(numSamples);

        // Compute average distance to estimate bandwidth
        var avgDist = ComputeAverageNearestNeighborDistance(pool, numSamples, featureSize, k: 5);
        var bandwidth = _numOps.ToDouble(avgDist);

        for (int i = 0; i < numSamples; i++)
        {
            var density = _numOps.Zero;
            for (int j = 0; j < numSamples; j++)
            {
                if (i != j)
                {
                    var dist = ComputeDistance(pool, i, j, featureSize);
                    var distVal = _numOps.ToDouble(dist);
                    var kernelVal = Math.Exp(-distVal * distVal / (2 * bandwidth * bandwidth));
                    density = _numOps.Add(density, _numOps.FromDouble(kernelVal));
                }
            }
            densities[i] = density;
        }

        return densities;
    }

    /// <summary>
    /// Computes delta distance (distance to nearest higher-density sample).
    /// </summary>
    private Vector<T> ComputeDeltaDistances(Tensor<T> pool, Vector<T> densities, int numSamples, int featureSize)
    {
        var deltas = new Vector<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            var minDistToHigherDensity = _numOps.MaxValue;
            var foundHigher = false;

            for (int j = 0; j < numSamples; j++)
            {
                if (i != j && _numOps.GreaterThan(densities[j], densities[i]))
                {
                    var dist = ComputeDistance(pool, i, j, featureSize);
                    if (_numOps.LessThan(dist, minDistToHigherDensity))
                    {
                        minDistToHigherDensity = dist;
                        foundHigher = true;
                    }
                }
            }

            // If no higher density point, set delta to max distance
            if (!foundHigher)
            {
                for (int j = 0; j < numSamples; j++)
                {
                    if (i != j)
                    {
                        var dist = ComputeDistance(pool, i, j, featureSize);
                        if (_numOps.GreaterThan(dist, minDistToHigherDensity))
                        {
                            minDistToHigherDensity = dist;
                        }
                    }
                }
            }

            deltas[i] = minDistToHigherDensity;
        }

        return deltas;
    }

    /// <summary>
    /// Computes distance-based scores (average distance to k nearest neighbors).
    /// </summary>
    private Vector<T> ComputeDistanceScores(Tensor<T> pool, int numSamples, int featureSize)
    {
        var scores = new Vector<T>(numSamples);
        var k = Math.Min(5, numSamples - 1);

        for (int i = 0; i < numSamples; i++)
        {
            var distances = new List<T>();
            for (int j = 0; j < numSamples; j++)
            {
                if (i != j)
                {
                    distances.Add(ComputeDistance(pool, i, j, featureSize));
                }
            }

            // Sort and take average of k nearest
            var sortedDist = distances.OrderBy(d => _numOps.ToDouble(d)).Take(k).ToList();
            var sum = _numOps.Zero;
            foreach (var d in sortedDist)
            {
                sum = _numOps.Add(sum, d);
            }
            scores[i] = _numOps.Divide(sum, _numOps.FromDouble(sortedDist.Count));
        }

        // Invert scores so larger distance = higher score (more diverse)
        // Actually, for outliers we want HIGH distance, so no inversion needed
        return scores;
    }

    /// <summary>
    /// Computes density-based scores (inverse of average distance = high density).
    /// </summary>
    private Vector<T> ComputeDensityScores(Tensor<T> pool, int numSamples, int featureSize)
    {
        var scores = new Vector<T>(numSamples);
        var k = Math.Min(5, numSamples - 1);

        for (int i = 0; i < numSamples; i++)
        {
            var distances = new List<T>();
            for (int j = 0; j < numSamples; j++)
            {
                if (i != j)
                {
                    distances.Add(ComputeDistance(pool, i, j, featureSize));
                }
            }

            var sortedDist = distances.OrderBy(d => _numOps.ToDouble(d)).Take(k).ToList();
            var sum = _numOps.Zero;
            foreach (var d in sortedDist)
            {
                sum = _numOps.Add(sum, d);
            }
            var avgDist = _numOps.Divide(sum, _numOps.FromDouble(sortedDist.Count));

            // Density is inverse of distance
            var epsilon = _numOps.FromDouble(1e-10);
            scores[i] = _numOps.Divide(_numOps.One, _numOps.Add(avgDist, epsilon));
        }

        return scores;
    }

    /// <summary>
    /// Computes the average k-nearest neighbor distance.
    /// </summary>
    private T ComputeAverageNearestNeighborDistance(Tensor<T> pool, int numSamples, int featureSize, int k)
    {
        var totalDist = _numOps.Zero;
        var count = 0;

        for (int i = 0; i < numSamples; i++)
        {
            var distances = new List<T>();
            for (int j = 0; j < numSamples; j++)
            {
                if (i != j)
                {
                    distances.Add(ComputeDistance(pool, i, j, featureSize));
                }
            }

            var sortedDist = distances.OrderBy(d => _numOps.ToDouble(d)).Take(k).ToList();
            foreach (var d in sortedDist)
            {
                totalDist = _numOps.Add(totalDist, d);
                count++;
            }
        }

        return count > 0 ? _numOps.Divide(totalDist, _numOps.FromDouble(count)) : _numOps.One;
    }

    /// <summary>
    /// Computes the coverage radius for selected samples.
    /// </summary>
    private T ComputeCoverageRadius(Tensor<T> pool, int[] selected)
    {
        var numSamples = pool.Shape[0];
        var featureSize = pool.Length / numSamples;
        var maxMinDist = _numOps.Zero;

        for (int i = 0; i < numSamples; i++)
        {
            var minDist = _numOps.MaxValue;
            foreach (var selIdx in selected)
            {
                var dist = ComputeDistance(pool, i, selIdx, featureSize);
                if (_numOps.LessThan(dist, minDist))
                {
                    minDist = dist;
                }
            }

            if (_numOps.GreaterThan(minDist, maxMinDist))
            {
                maxMinDist = minDist;
            }
        }

        return maxMinDist;
    }

    /// <summary>
    /// Computes distance between two samples based on the configured metric.
    /// </summary>
    private T ComputeDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        return _distanceMetric switch
        {
            DistanceMetric.Euclidean => ComputeEuclideanDistance(pool, idx1, idx2, featureSize),
            DistanceMetric.Cosine => ComputeCosineDistance(pool, idx1, idx2, featureSize),
            DistanceMetric.Manhattan => ComputeManhattanDistance(pool, idx1, idx2, featureSize),
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
    /// Computes cosine distance (1 - cosine similarity) between two samples.
    /// </summary>
    private T ComputeCosineDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var dotProduct = _numOps.Zero;
        var norm1Sq = _numOps.Zero;
        var norm2Sq = _numOps.Zero;
        var start1 = idx1 * featureSize;
        var start2 = idx2 * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var v1 = pool[start1 + i];
            var v2 = pool[start2 + i];

            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(v1, v2));
            norm1Sq = _numOps.Add(norm1Sq, _numOps.Multiply(v1, v1));
            norm2Sq = _numOps.Add(norm2Sq, _numOps.Multiply(v2, v2));
        }

        var norm1 = Math.Sqrt(_numOps.ToDouble(norm1Sq));
        var norm2 = Math.Sqrt(_numOps.ToDouble(norm2Sq));

        if (norm1 < 1e-10 || norm2 < 1e-10)
        {
            return _numOps.One;
        }

        var cosineSim = _numOps.ToDouble(dotProduct) / (norm1 * norm2);
        return _numOps.FromDouble(1.0 - cosineSim);
    }

    /// <summary>
    /// Computes Manhattan (L1) distance between two samples.
    /// </summary>
    private T ComputeManhattanDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var sum = _numOps.Zero;
        var start1 = idx1 * featureSize;
        var start2 = idx2 * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var diff = _numOps.Subtract(pool[start1 + i], pool[start2 + i]);
            var absDiff = _numOps.FromDouble(Math.Abs(_numOps.ToDouble(diff)));
            sum = _numOps.Add(sum, absDiff);
        }

        return sum;
    }

    /// <summary>
    /// Computes the centroid of the data.
    /// </summary>
    private Vector<T> ComputeCentroid(Tensor<T> pool, int numSamples, int featureSize)
    {
        var centroid = new Vector<T>(featureSize);

        for (int f = 0; f < featureSize; f++)
        {
            var sum = _numOps.Zero;
            for (int i = 0; i < numSamples; i++)
            {
                sum = _numOps.Add(sum, pool[i * featureSize + f]);
            }
            centroid[f] = _numOps.Divide(sum, _numOps.FromDouble(numSamples));
        }

        return centroid;
    }

    /// <summary>
    /// Finds the sample nearest to the centroid.
    /// </summary>
    private int FindNearestToCentroid(Tensor<T> pool, Vector<T> centroid, int numSamples, int featureSize)
    {
        var minDist = _numOps.MaxValue;
        var nearest = 0;

        for (int i = 0; i < numSamples; i++)
        {
            var distSq = _numOps.Zero;
            for (int f = 0; f < featureSize; f++)
            {
                var diff = _numOps.Subtract(pool[i * featureSize + f], centroid[f]);
                distSq = _numOps.Add(distSq, _numOps.Multiply(diff, diff));
            }

            if (_numOps.LessThan(distSq, minDist))
            {
                minDist = distSq;
                nearest = i;
            }
        }

        return nearest;
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
