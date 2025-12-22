using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Batch;

/// <summary>
/// Clustering-based batch selection strategy using k-means clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This strategy first groups similar samples into clusters,
/// then selects the most informative sample from each cluster. This ensures the batch
/// covers different regions of the data space.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Apply k-means clustering to group samples</description></item>
/// <item><description>For each cluster, find the most informative sample</description></item>
/// <item><description>Select samples round-robin from clusters until batch is full</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Guarantees coverage of different data regions</description></item>
/// <item><description>Balances informativeness within diversity constraints</description></item>
/// <item><description>Interpretable cluster-based selection</description></item>
/// </list>
///
/// <para><b>Considerations:</b></para>
/// <list type="bullet">
/// <item><description>Requires choosing number of clusters (usually = batch size)</description></item>
/// <item><description>K-means assumes Euclidean distances</description></item>
/// <item><description>May need many iterations for convergence</description></item>
/// </list>
/// </remarks>
public class ClusteredBatchStrategy<T, TInput, TOutput> : IClusteringBatchStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numClusters;
    private readonly int _maxIterations;
    private readonly T _convergenceThreshold;
    private T _diversityTradeoff;

    private int[]? _lastClusterAssignments;

    /// <inheritdoc/>
    public string Name => $"Clustered Batch Selection (k={_numClusters})";

    /// <inheritdoc/>
    public int NumClusters => _numClusters;

    /// <inheritdoc/>
    public T DiversityTradeoff
    {
        get => _diversityTradeoff;
        set => _diversityTradeoff = value;
    }

    /// <summary>
    /// Initializes a new ClusteredBatchStrategy with default settings.
    /// </summary>
    public ClusteredBatchStrategy()
        : this(numClusters: 10, maxIterations: 100, diversityTradeoff: 0.5)
    {
    }

    /// <summary>
    /// Initializes a new ClusteredBatchStrategy with specified parameters.
    /// </summary>
    /// <param name="numClusters">Number of clusters to use (typically equals batch size).</param>
    /// <param name="maxIterations">Maximum k-means iterations.</param>
    /// <param name="diversityTradeoff">Trade-off between informativeness and diversity.</param>
    public ClusteredBatchStrategy(int numClusters, int maxIterations = 100, double diversityTradeoff = 0.5)
    {
        _numClusters = Math.Max(1, numClusters);
        _maxIterations = Math.Max(1, maxIterations);
        _convergenceThreshold = NumOps.FromDouble(1e-6);
        _diversityTradeoff = NumOps.FromDouble(diversityTradeoff);
    }

    /// <inheritdoc/>
    public int[] SelectBatch(
        int[] candidateIndices,
        Vector<T> scores,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        if (candidateIndices.Length == 0)
        {
            return Array.Empty<int>();
        }

        int effectiveBatchSize = Math.Min(batchSize, candidateIndices.Length);
        int effectiveClusters = Math.Min(_numClusters, candidateIndices.Length);

        // Extract features for clustering
        var features = ExtractFeatures(candidateIndices, unlabeledPool);

        // Perform k-means clustering
        var clusterAssignments = KMeansClustering(features, effectiveClusters);
        _lastClusterAssignments = clusterAssignments;

        // Map scores to candidates
        var candidateScores = new T[candidateIndices.Length];
        for (int i = 0; i < candidateIndices.Length; i++)
        {
            candidateScores[i] = scores[i];
        }

        // Select from clusters
        var selected = SelectFromClusters(clusterAssignments, new Vector<T>(candidateScores), effectiveBatchSize);

        // Map back to pool indices
        var poolIndices = new int[selected.Length];
        for (int i = 0; i < selected.Length; i++)
        {
            poolIndices[i] = candidateIndices[selected[i]];
        }

        return poolIndices;
    }

    /// <inheritdoc/>
    public int[] ClusterSamples(IDataset<T, TInput, TOutput> unlabeledPool)
    {
        var indices = unlabeledPool.GetIndices();
        var features = ExtractFeatures(indices, unlabeledPool);
        int effectiveClusters = Math.Min(_numClusters, unlabeledPool.Count);
        return KMeansClustering(features, effectiveClusters);
    }

    /// <inheritdoc/>
    public int[] SelectFromClusters(int[] clusterAssignments, Vector<T> scores, int batchSize)
    {
        if (clusterAssignments.Length == 0)
        {
            return Array.Empty<int>();
        }

        // Group samples by cluster
        var clusterGroups = new Dictionary<int, List<(int Index, T Score)>>();

        for (int i = 0; i < clusterAssignments.Length; i++)
        {
            int cluster = clusterAssignments[i];
            if (!clusterGroups.TryGetValue(cluster, out var group))
            {
                group = new List<(int Index, T Score)>();
                clusterGroups[cluster] = group;
            }
            group.Add((i, scores[i]));
        }

        // Sort each cluster by score (descending)
        foreach (var group in clusterGroups.Values)
        {
            group.Sort((a, b) => NumOps.Compare(b.Score, a.Score));
        }

        // Round-robin selection from clusters
        var selected = new List<int>();
        var clusterPointers = clusterGroups.Keys.ToDictionary(k => k, _ => 0);
        var clusterOrder = clusterGroups.Keys.ToList();

        int clusterIdx = 0;
        while (selected.Count < batchSize && clusterGroups.Values.Any(g => clusterPointers.Values.Any(p => p < g.Count)))
        {
            int cluster = clusterOrder[clusterIdx % clusterOrder.Count];
            var group = clusterGroups[cluster];
            int pointer = clusterPointers[cluster];

            if (pointer < group.Count)
            {
                selected.Add(group[pointer].Index);
                clusterPointers[cluster] = pointer + 1;
            }

            clusterIdx++;

            // Break if we've cycled through all clusters without adding
            if (clusterIdx >= clusterOrder.Count * (batchSize + 1))
            {
                break;
            }
        }

        return selected.Take(batchSize).ToArray();
    }

    /// <inheritdoc/>
    public T ComputeDiversity(TInput sample1, TInput sample2)
    {
        if (sample1 is Vector<T> vec1 && sample2 is Vector<T> vec2)
        {
            return ComputeEuclideanDistance(vec1, vec2);
        }

        if (sample1 is T[] arr1 && sample2 is T[] arr2)
        {
            return ComputeEuclideanDistance(new Vector<T>(arr1), new Vector<T>(arr2));
        }

        return NumOps.One;
    }

    #region Private Methods

    private List<Vector<T>> ExtractFeatures(int[] indices, IDataset<T, TInput, TOutput> pool)
    {
        var features = new List<Vector<T>>();

        foreach (var idx in indices)
        {
            var input = pool.GetInput(idx);
            var feature = ConvertToVector(input);
            features.Add(feature);
        }

        return features;
    }

    private Vector<T> ConvertToVector(TInput input)
    {
        if (input is Vector<T> vec)
        {
            return vec;
        }

        if (input is T[] arr)
        {
            return new Vector<T>(arr);
        }

        if (input is IReadOnlyList<T> list)
        {
            return new Vector<T>(list.ToArray());
        }

        // Single value
        if (input is T val)
        {
            return new Vector<T>(new[] { val });
        }

        return new Vector<T>(new[] { NumOps.Zero });
    }

    private int[] KMeansClustering(List<Vector<T>> features, int k)
    {
        if (features.Count == 0 || k <= 0)
        {
            return Array.Empty<int>();
        }

        int n = features.Count;
        int dim = features[0].Length;

        // Initialize centroids using k-means++
        var centroids = InitializeCentroids(features, k);
        var assignments = new int[n];

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Assign samples to nearest centroid
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int nearest = FindNearestCentroid(features[i], centroids);
                if (nearest != assignments[i])
                {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }

            // Update centroids
            centroids = UpdateCentroids(features, assignments, k, dim);
        }

        return assignments;
    }

    private List<Vector<T>> InitializeCentroids(List<Vector<T>> features, int k)
    {
        var centroids = new List<Vector<T>>();
        var random = RandomHelper.Shared;

        // First centroid: random sample
        int firstIdx = random.Next(features.Count);
        centroids.Add(features[firstIdx].Clone());

        // Remaining centroids: k-means++ initialization
        for (int c = 1; c < k; c++)
        {
            var distances = new T[features.Count];
            T totalDistance = NumOps.Zero;

            for (int i = 0; i < features.Count; i++)
            {
                T minDist = NumOps.MaxValue;
                foreach (var centroid in centroids)
                {
                    var dist = ComputeSquaredDistance(features[i], centroid);
                    if (NumOps.Compare(dist, minDist) < 0)
                    {
                        minDist = dist;
                    }
                }
                distances[i] = minDist;
                totalDistance = NumOps.Add(totalDistance, minDist);
            }

            // Sample proportional to distance squared
            if (NumOps.Compare(totalDistance, NumOps.Zero) <= 0)
            {
                // All distances are zero, pick randomly
                int randIdx = random.Next(features.Count);
                centroids.Add(features[randIdx].Clone());
            }
            else
            {
                T threshold = NumOps.Multiply(NumOps.FromDouble(random.NextDouble()), totalDistance);
                T cumulative = NumOps.Zero;
                int selectedIdx = 0;

                for (int i = 0; i < features.Count; i++)
                {
                    cumulative = NumOps.Add(cumulative, distances[i]);
                    if (NumOps.Compare(cumulative, threshold) >= 0)
                    {
                        selectedIdx = i;
                        break;
                    }
                }

                centroids.Add(features[selectedIdx].Clone());
            }
        }

        return centroids;
    }

    private int FindNearestCentroid(Vector<T> sample, List<Vector<T>> centroids)
    {
        int nearest = 0;
        T minDist = NumOps.MaxValue;

        for (int c = 0; c < centroids.Count; c++)
        {
            var dist = ComputeSquaredDistance(sample, centroids[c]);
            if (NumOps.Compare(dist, minDist) < 0)
            {
                minDist = dist;
                nearest = c;
            }
        }

        return nearest;
    }

    private List<Vector<T>> UpdateCentroids(List<Vector<T>> features, int[] assignments, int k, int dim)
    {
        var newCentroids = new List<Vector<T>>();
        var counts = new int[k];

        // Initialize centroid sums
        var sums = new T[k][];
        for (int c = 0; c < k; c++)
        {
            sums[c] = new T[dim];
            for (int d = 0; d < dim; d++)
            {
                sums[c][d] = NumOps.Zero;
            }
        }

        // Accumulate sums
        for (int i = 0; i < features.Count; i++)
        {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < dim; d++)
            {
                sums[cluster][d] = NumOps.Add(sums[cluster][d], features[i][d]);
            }
        }

        // Compute new centroids
        for (int c = 0; c < k; c++)
        {
            var centroid = new T[dim];
            if (counts[c] > 0)
            {
                for (int d = 0; d < dim; d++)
                {
                    centroid[d] = NumOps.Divide(sums[c][d], NumOps.FromDouble(counts[c]));
                }
            }
            else
            {
                // Empty cluster: keep old centroid or reinitialize
                for (int d = 0; d < dim; d++)
                {
                    centroid[d] = NumOps.Zero;
                }
            }
            newCentroids.Add(new Vector<T>(centroid));
        }

        return newCentroids;
    }

    private T ComputeSquaredDistance(Vector<T> a, Vector<T> b)
    {
        int length = Math.Min(a.Length, b.Length);
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return sum;
    }

    private T ComputeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        return NumOps.Sqrt(ComputeSquaredDistance(a, b));
    }

    #endregion
}
