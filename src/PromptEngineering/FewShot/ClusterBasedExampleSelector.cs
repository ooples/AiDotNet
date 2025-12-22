using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PromptEngineering.FewShot;

/// <summary>
/// Selects examples using a clustering approach to ensure broad coverage.
/// </summary>
/// <typeparam name="T">The type of numeric data used for similarity scoring.</typeparam>
/// <remarks>
/// <para>
/// This selector groups similar examples into clusters and selects representative examples from
/// each cluster. This ensures broad coverage across different types of examples.
/// </para>
/// <para><b>For Beginners:</b> Groups similar examples and picks one from each group.
///
/// Think of organizing photos:
/// - Group 1: Beach photos
/// - Group 2: Mountain photos
/// - Group 3: City photos
///
/// Instead of showing all beach photos, you show one from each group.
///
/// Example:
/// <code>
/// var selector = new ClusterBasedExampleSelector&lt;double&gt;(embeddingFunction, clusterCount: 5);
///
/// // Add many examples (they'll be automatically clustered)
/// selector.AddExample(new FewShotExample { Input = "Billing question", Output = "Contact billing" });
/// selector.AddExample(new FewShotExample { Input = "Technical issue", Output = "Contact support" });
/// selector.AddExample(new FewShotExample { Input = "Account problem", Output = "Reset password" });
/// // ... more examples
///
/// // Get examples from different clusters
/// var examples = selector.SelectExamples("Help me", 3);
/// // Returns one billing, one technical, one account example
/// </code>
///
/// Use this when:
/// - Examples naturally fall into categories
/// - You want guaranteed coverage of all categories
/// - Building a general-purpose system
/// </para>
/// </remarks>
public class ClusterBasedExampleSelector<T> : FewShotExampleSelectorBase<T>
{
    private readonly Func<string, Vector<T>> _embeddingFunction;
    private readonly Dictionary<FewShotExample, Vector<T>> _exampleEmbeddings;
    private readonly int _clusterCount;
    private readonly Random _random;
    private List<List<FewShotExample>> _clusters;
    private bool _clustersDirty;

    /// <summary>
    /// Initializes a new instance of the ClusterBasedExampleSelector class.
    /// </summary>
    /// <param name="embeddingFunction">Function to convert text to embedding vectors.</param>
    /// <param name="clusterCount">Number of clusters to create.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The cluster count determines how many groups to create.
    ///
    /// Guidelines:
    /// - More clusters = more fine-grained grouping
    /// - Fewer clusters = broader grouping
    /// - A good starting point: sqrt(number of examples)
    ///
    /// For 100 examples, try 10 clusters.
    /// For 1000 examples, try 30-50 clusters.
    /// </para>
    /// </remarks>
    public ClusterBasedExampleSelector(IEmbeddingModel<T> embeddingModel, int clusterCount = 5, int? seed = null)
        : this(embeddingModel is null ? throw new ArgumentNullException(nameof(embeddingModel)) : embeddingModel.Embed, clusterCount, seed)
    {
    }

    /// <summary>
    /// Initializes a new instance of the ClusterBasedExampleSelector class.
    /// </summary>
    /// <param name="embeddingFunction">Function to convert text to embedding vectors.</param>
    /// <param name="clusterCount">Number of clusters to create.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ClusterBasedExampleSelector(Func<string, Vector<T>> embeddingFunction, int clusterCount = 5, int? seed = null)
    {
        _embeddingFunction = embeddingFunction ?? throw new ArgumentNullException(nameof(embeddingFunction));
        _clusterCount = Math.Max(1, clusterCount);
        _exampleEmbeddings = new Dictionary<FewShotExample, Vector<T>>();
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _clusters = new List<List<FewShotExample>>();
        _clustersDirty = true;
    }

    /// <summary>
    /// Gets the number of clusters used for selection.
    /// </summary>
    public int ClusterCount => _clusterCount;

    /// <summary>
    /// Called when an example is added. Pre-computes the embedding.
    /// </summary>
    protected override void OnExampleAdded(FewShotExample example)
    {
        _exampleEmbeddings[example] = _embeddingFunction(example.Input);
        _clustersDirty = true;
    }

    /// <summary>
    /// Called when an example is removed. Removes the cached embedding.
    /// </summary>
    protected override void OnExampleRemoved(FewShotExample example)
    {
        _exampleEmbeddings.Remove(example);
        _clustersDirty = true;
    }

    /// <summary>
    /// Selects examples from different clusters.
    /// </summary>
    protected override IReadOnlyList<FewShotExample> SelectExamplesCore(string query, int count)
    {
        if (_clustersDirty)
        {
            RebuildClusters();
            _clustersDirty = false;
        }

        var queryEmbedding = _embeddingFunction(query);
        var selected = new List<FewShotExample>(count);

        var clustersByRelevance = _clusters
            .Where(cluster => cluster.Count > 0)
            .Select(cluster => new { Cluster = cluster, Relevance = GetClusterRelevance(cluster, queryEmbedding) })
            .ToList();

        clustersByRelevance.Sort((a, b) => CompareDescending(a.Relevance, b.Relevance));

        foreach (var clusterInfo in clustersByRelevance)
        {
            if (selected.Count >= count)
            {
                break;
            }

            var bestExample = GetMostRelevantExample(clusterInfo.Cluster, queryEmbedding, selected);
            if (bestExample is not null)
            {
                selected.Add(bestExample);
            }
        }

        // Second pass: if we need more, cycle through clusters again
        int cycleIndex = 0;
        while (selected.Count < count && clustersByRelevance.Count > 0 && cycleIndex < clustersByRelevance.Count * 10)
        {
            var clusterInfo = clustersByRelevance[cycleIndex % clustersByRelevance.Count];
            var nextBest = GetMostRelevantExample(clusterInfo.Cluster, queryEmbedding, selected);
            if (nextBest is not null)
            {
                selected.Add(nextBest);
            }

            cycleIndex++;
        }

        return selected.AsReadOnly();
    }

    private FewShotExample? GetMostRelevantExample(
        List<FewShotExample> cluster,
        Vector<T> queryEmbedding,
        List<FewShotExample> excluded)
    {
        FewShotExample? best = null;
        T bestScore = NumOps.Zero;
        bool hasBestScore = false;

        foreach (var example in cluster)
        {
            if (excluded.Contains(example))
            {
                continue;
            }

            var score = CosineSimilarity(queryEmbedding, _exampleEmbeddings[example]);
            if (!hasBestScore || NumOps.GreaterThan(score, bestScore))
            {
                best = example;
                bestScore = score;
                hasBestScore = true;
            }
        }

        return best;
    }

    /// <summary>
    /// Rebuilds the clusters using k-means clustering.
    /// </summary>
    private void RebuildClusters()
    {
        var examples = Examples.ToList();
        if (examples.Count == 0)
        {
            _clusters = new List<List<FewShotExample>>();
            return;
        }

        var actualClusterCount = Math.Min(_clusterCount, examples.Count);
        var embeddings = examples.Select(ex => _exampleEmbeddings[ex]).ToList();
        var dimension = embeddings[0].Length;

        // Initialize centroids randomly (k-means++)
        var centroids = InitializeCentroids(embeddings, actualClusterCount, dimension);

        // Run k-means iterations
        var assignments = new int[examples.Count];
        const int maxIterations = 100;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Assign each example to nearest centroid
            bool changed = false;
            for (int i = 0; i < examples.Count; i++)
            {
                var nearestCluster = FindNearestCentroid(embeddings[i], centroids);
                if (assignments[i] != nearestCluster)
                {
                    assignments[i] = nearestCluster;
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }

            // Update centroids
            centroids = UpdateCentroids(embeddings, assignments, actualClusterCount, dimension);
        }

        // Build cluster lists
        _clusters = Enumerable.Range(0, actualClusterCount)
            .Select(_ => new List<FewShotExample>())
            .ToList();

        for (int i = 0; i < examples.Count; i++)
        {
            _clusters[assignments[i]].Add(examples[i]);
        }
    }

    /// <summary>
    /// Initializes cluster centroids using k-means++ algorithm.
    /// </summary>
    private List<Vector<T>> InitializeCentroids(List<Vector<T>> embeddings, int k, int dimension)
    {
        var centroids = new List<Vector<T>>();

        // First centroid: random
        var firstIndex = _random.Next(embeddings.Count);
        centroids.Add(new Vector<T>(embeddings[firstIndex]));

        // Remaining centroids: k-means++ selection
        while (centroids.Count < k)
        {
            var distances = new T[embeddings.Count];
            T totalDistance = NumOps.Zero;

            for (int i = 0; i < embeddings.Count; i++)
            {
                T minDist = NumOps.Zero;
                bool hasMinDist = false;

                foreach (var centroid in centroids)
                {
                    var dist = EuclideanDistanceSquared(embeddings[i], centroid);
                    if (!hasMinDist || NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                        hasMinDist = true;
                    }
                }

                if (!hasMinDist)
                {
                    minDist = NumOps.Zero;
                }

                distances[i] = minDist;
                totalDistance = NumOps.Add(totalDistance, minDist);
            }

            if (NumOps.Equals(totalDistance, NumOps.Zero))
            {
                var randomIndex = _random.Next(embeddings.Count);
                centroids.Add(new Vector<T>(embeddings[randomIndex]));
                continue;
            }

            // Select next centroid with probability proportional to distance squared
            var threshold = NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), totalDistance);
            T cumulative = NumOps.Zero;
            bool selected = false;

            for (int i = 0; i < embeddings.Count; i++)
            {
                cumulative = NumOps.Add(cumulative, distances[i]);
                if (NumOps.GreaterThanOrEquals(cumulative, threshold))
                {
                    centroids.Add(new Vector<T>(embeddings[i]));
                    selected = true;
                    break;
                }
            }

            if (!selected)
            {
                centroids.Add(new Vector<T>(embeddings[embeddings.Count - 1]));
            }
        }

        return centroids;
    }

    /// <summary>
    /// Finds the nearest centroid for an embedding.
    /// </summary>
    private int FindNearestCentroid(Vector<T> embedding, List<Vector<T>> centroids)
    {
        int nearest = 0;
        T minDistance = NumOps.Zero;
        bool hasMinDistance = false;

        for (int i = 0; i < centroids.Count; i++)
        {
            var distance = EuclideanDistanceSquared(embedding, centroids[i]);
            if (!hasMinDistance || NumOps.LessThan(distance, minDistance))
            {
                minDistance = distance;
                nearest = i;
                hasMinDistance = true;
            }
        }

        return nearest;
    }

    /// <summary>
    /// Updates centroids based on current assignments.
    /// </summary>
    private List<Vector<T>> UpdateCentroids(List<Vector<T>> embeddings, int[] assignments, int k, int dimension)
    {
        var newCentroids = new List<T[]>();
        var counts = new int[k];

        // Initialize new centroids
        for (int i = 0; i < k; i++)
        {
            newCentroids.Add(new T[dimension]);
        }

        // Sum up embeddings for each cluster
        for (int i = 0; i < embeddings.Count; i++)
        {
            var cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < dimension; d++)
            {
                newCentroids[cluster][d] = NumOps.Add(newCentroids[cluster][d], embeddings[i][d]);
            }
        }

        // Average
        for (int i = 0; i < k; i++)
        {
            if (counts[i] > 0)
            {
                var countValue = NumOps.FromDouble(counts[i]);
                for (int d = 0; d < dimension; d++)
                {
                    newCentroids[i][d] = NumOps.Divide(newCentroids[i][d], countValue);
                }
            }
        }

        return newCentroids.Select(arr => new Vector<T>(arr)).ToList();
    }

    /// <summary>
    /// Gets the average relevance of a cluster to the query.
    /// </summary>
    private T GetClusterRelevance(List<FewShotExample> cluster, Vector<T> queryEmbedding)
    {
        if (cluster.Count == 0)
        {
            return NumOps.Zero;
        }

        T sum = NumOps.Zero;
        foreach (var example in cluster)
        {
            sum = NumOps.Add(sum, CosineSimilarity(queryEmbedding, _exampleEmbeddings[example]));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(cluster.Count));
    }
}
