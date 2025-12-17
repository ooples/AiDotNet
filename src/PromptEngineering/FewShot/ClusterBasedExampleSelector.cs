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
    private readonly Func<string, double[]> _embeddingFunction;
    private readonly Dictionary<FewShotExample, double[]> _exampleEmbeddings;
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
    public ClusterBasedExampleSelector(Func<string, double[]> embeddingFunction, int clusterCount = 5, int? seed = null)
    {
        _embeddingFunction = embeddingFunction ?? throw new ArgumentNullException(nameof(embeddingFunction));
        _clusterCount = Math.Max(1, clusterCount);
        _exampleEmbeddings = new Dictionary<FewShotExample, double[]>();
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
        // Rebuild clusters if needed
        if (_clustersDirty)
        {
            RebuildClusters();
            _clustersDirty = false;
        }

        var queryEmbedding = _embeddingFunction(query);
        var selected = new List<FewShotExample>();

        // First pass: select one example from each cluster (round-robin)
        // Sort clusters by relevance to query
        var clustersByRelevance = _clusters
            .Select((cluster, index) => new { Cluster = cluster, Index = index })
            .Where(c => c.Cluster.Count > 0)
            .OrderByDescending(c => GetClusterRelevance(c.Cluster, queryEmbedding))
            .ToList();

        foreach (var clusterInfo in clustersByRelevance)
        {
            if (selected.Count >= count)
            {
                break;
            }

            // Select the most relevant example from this cluster
            var bestExample = clusterInfo.Cluster
                .OrderByDescending(ex => CosineSimilarity(queryEmbedding, _exampleEmbeddings[ex]))
                .First();

            selected.Add(bestExample);
        }

        // Second pass: if we need more, cycle through clusters again
        int cycleIndex = 0;
        while (selected.Count < count && cycleIndex < clustersByRelevance.Count * 10)
        {
            var clusterInfo = clustersByRelevance[cycleIndex % clustersByRelevance.Count];
            var remaining = clusterInfo.Cluster.Where(ex => !selected.Contains(ex)).ToList();

            if (remaining.Count > 0)
            {
                var nextBest = remaining
                    .OrderByDescending(ex => CosineSimilarity(queryEmbedding, _exampleEmbeddings[ex]))
                    .First();
                selected.Add(nextBest);
            }

            cycleIndex++;
        }

        return selected.Take(count).ToList().AsReadOnly();
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
    private List<double[]> InitializeCentroids(List<double[]> embeddings, int k, int dimension)
    {
        var centroids = new List<double[]>();

        // First centroid: random
        var firstIndex = _random.Next(embeddings.Count);
        centroids.Add((double[])embeddings[firstIndex].Clone());

        // Remaining centroids: k-means++ selection
        while (centroids.Count < k)
        {
            var distances = new double[embeddings.Count];
            double totalDistance = 0;

            for (int i = 0; i < embeddings.Count; i++)
            {
                var minDist = centroids.Min(c => EuclideanDistanceSquared(embeddings[i], c));
                distances[i] = minDist;
                totalDistance += minDist;
            }

            // Select next centroid with probability proportional to distance squared
            var threshold = _random.NextDouble() * totalDistance;
            double cumulative = 0;
            for (int i = 0; i < embeddings.Count; i++)
            {
                cumulative += distances[i];
                if (cumulative >= threshold)
                {
                    centroids.Add((double[])embeddings[i].Clone());
                    break;
                }
            }

            // Fallback: if we didn't select one, pick random
            if (centroids.Count < k && centroids.Count == centroids.Distinct().Count())
            {
                var randomIndex = _random.Next(embeddings.Count);
                centroids.Add((double[])embeddings[randomIndex].Clone());
            }
        }

        return centroids;
    }

    /// <summary>
    /// Finds the nearest centroid for an embedding.
    /// </summary>
    private int FindNearestCentroid(double[] embedding, List<double[]> centroids)
    {
        int nearest = 0;
        double minDistance = double.MaxValue;

        for (int i = 0; i < centroids.Count; i++)
        {
            var distance = EuclideanDistanceSquared(embedding, centroids[i]);
            if (distance < minDistance)
            {
                minDistance = distance;
                nearest = i;
            }
        }

        return nearest;
    }

    /// <summary>
    /// Updates centroids based on current assignments.
    /// </summary>
    private List<double[]> UpdateCentroids(List<double[]> embeddings, int[] assignments, int k, int dimension)
    {
        var newCentroids = new List<double[]>();
        var counts = new int[k];

        // Initialize new centroids
        for (int i = 0; i < k; i++)
        {
            newCentroids.Add(new double[dimension]);
        }

        // Sum up embeddings for each cluster
        for (int i = 0; i < embeddings.Count; i++)
        {
            var cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < dimension; d++)
            {
                newCentroids[cluster][d] += embeddings[i][d];
            }
        }

        // Average
        for (int i = 0; i < k; i++)
        {
            if (counts[i] > 0)
            {
                for (int d = 0; d < dimension; d++)
                {
                    newCentroids[i][d] /= counts[i];
                }
            }
        }

        return newCentroids;
    }

    /// <summary>
    /// Gets the average relevance of a cluster to the query.
    /// </summary>
    private double GetClusterRelevance(List<FewShotExample> cluster, double[] queryEmbedding)
    {
        if (cluster.Count == 0)
        {
            return 0;
        }

        return cluster.Average(ex => CosineSimilarity(queryEmbedding, _exampleEmbeddings[ex]));
    }

    /// <summary>
    /// Calculates squared Euclidean distance between two vectors.
    /// </summary>
    private static double EuclideanDistanceSquared(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    /// <summary>
    /// Calculates cosine similarity between two vectors.
    /// </summary>
    private static double CosineSimilarity(double[] a, double[] b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length.");
        }

        double dotProduct = 0;
        double magnitudeA = 0;
        double magnitudeB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        double magnitude = Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB);
        return magnitude > 0 ? dotProduct / magnitude : 0;
    }
}
