using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Active Learning Literature Survey", "https://minds.wisconsin.edu/handle/1793/60660", Year = 2009, Authors = "Burr Settles")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
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
        var candidateScores = scores.GetSubVector(0, candidateIndices.Length);

        // Select from clusters
        var selected = SelectFromClusters(clusterAssignments, candidateScores, effectiveBatchSize);

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
        var memberLists = new Dictionary<int, List<int>>();

        for (int i = 0; i < clusterAssignments.Length; i++)
        {
            int cluster = clusterAssignments[i];
            if (!memberLists.TryGetValue(cluster, out var members))
            {
                members = new List<int>();
                memberLists[cluster] = members;
            }
            members.Add(i);
        }

        // Sort each cluster by score (descending), with the same comparison sort as before
        var engine = AiDotNetEngine.Current;
        var scoreTensor = Tensor<T>.FromVector(scores);
        var clusterGroups = new Dictionary<int, List<(int Index, int Rank)>>();
        foreach (var entry in memberLists)
        {
            var members = entry.Value.ToArray();
            var memberScores = engine.TensorIndexSelect(scoreTensor, new Tensor<int>(members, new[] { members.Length }), 0);
            var order = engine.ArgSort(memberScores, -1, descending: true).ToArray();
            var group = new List<(int Index, int Rank)>(members.Length);
            for (int r = 0; r < order.Length; r++)
            {
                group.Add((members[order[r]], r));
            }

            clusterGroups[entry.Key] = group;
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
            return VectorHelper.EuclideanDistance(vec1, vec2);
        }

        if (sample1 is T[] arr1 && sample2 is T[] arr2)
        {
            return VectorHelper.EuclideanDistance(new Vector<T>(arr1), new Vector<T>(arr2));
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
        var engine = AiDotNetEngine.Current;
        var samples = ActiveLearningEngineOps.StackVectors(features);

        // Initialize centroids using k-means++
        var centroids = InitializeCentroids(samples, k);
        var assignments = new int[n];

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Assign samples to nearest centroid (first one on ties)
            var nearest = engine.TensorArgMin(ActiveLearningEngineOps.SquaredDistances(samples, centroids), 1).ToArray();
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                if (nearest[i] != assignments[i])
                {
                    changed = true;
                    break;
                }
            }

            assignments = nearest;

            if (!changed)
            {
                break;
            }

            // Update centroids
            centroids = UpdateCentroids(samples, assignments, k);
        }

        return assignments;
    }

    /// <summary>
    /// k-means++ initialization over the [n, d] samples.
    /// </summary>
    /// <returns>A [k, d] tensor of centroids.</returns>
    private Tensor<T> InitializeCentroids(Tensor<T> samples, int k)
    {
        var engine = AiDotNetEngine.Current;
        int n = samples.Shape[0];
        int dim = samples.Shape[1];
        var random = RandomHelper.Shared;
        var centroids = new List<Tensor<T>>(k);

        // First centroid: random sample
        int firstIdx = random.Next(n);
        centroids.Add(engine.TensorSlice(samples, new[] { firstIdx, 0 }, new[] { 1, dim }));

        // Squared distance from every sample to its nearest centroid so far
        var minDist = new Tensor<T>(new[] { n });
        minDist.Fill(NumOps.MaxValue);

        // Remaining centroids: k-means++ initialization
        for (int c = 1; c < k; c++)
        {
            var newest = centroids[centroids.Count - 1];
            minDist = engine.TensorMin(minDist, ActiveLearningEngineOps.SquaredDistances(samples, newest).Reshape(new[] { n }));
            var cumulative = engine.TensorCumSum(minDist, 0);
            var totalDistance = engine.TensorSlice(cumulative, new[] { n - 1 }, new[] { 1 });

            // Sample proportional to distance squared
            if (engine.TensorCountNonzero(engine.TensorGreaterThan(totalDistance, NumOps.Zero)) == 0)
            {
                // All distances are zero, pick randomly
                int randIdx = random.Next(n);
                centroids.Add(engine.TensorSlice(samples, new[] { randIdx, 0 }, new[] { 1, dim }));
            }
            else
            {
                // First sample whose cumulative distance reaches the threshold
                var threshold = engine.TensorMultiplyScalar(totalDistance, NumOps.FromDouble(random.NextDouble()));
                int selectedIdx = engine.TensorSearchSorted(cumulative, threshold).GetFlat(0);
                if (selectedIdx >= n)
                {
                    selectedIdx = 0;
                }

                centroids.Add(engine.TensorSlice(samples, new[] { selectedIdx, 0 }, new[] { 1, dim }));
            }
        }

        return engine.TensorConcatenate(centroids.ToArray(), 0);
    }

    /// <summary>
    /// Mean of the samples assigned to each cluster; empty clusters get a zero centroid.
    /// </summary>
    /// <returns>A [k, d] tensor of centroids.</returns>
    private Tensor<T> UpdateCentroids(Tensor<T> samples, int[] assignments, int k)
    {
        var engine = AiDotNetEngine.Current;
        int n = samples.Shape[0];
        int dim = samples.Shape[1];
        var clusterIndex = new Tensor<int>(assignments, new[] { n });

        // Accumulate sums and counts
        var sums = engine.TensorIndexAdd(new Tensor<T>(new[] { k, dim }), 0, clusterIndex, samples);
        var ones = new Tensor<T>(new[] { n, 1 });
        ones.Fill(NumOps.One);
        var counts = engine.TensorIndexAdd(new Tensor<T>(new[] { k, 1 }), 0, clusterIndex, ones);

        // Compute new centroids
        var shape = new[] { k, dim };
        var countsWide = engine.TensorBroadcastTo(counts, shape);
        var means = engine.TensorDivide(sums, countsWide);
        var nonEmpty = engine.TensorGreaterThan(countsWide, NumOps.Zero);
        return engine.TensorWhere(nonEmpty, means, new Tensor<T>(shape));
    }


    #endregion
}
