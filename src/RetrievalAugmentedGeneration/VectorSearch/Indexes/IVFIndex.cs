using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Inverted File (IVF) index that partitions the vector space for faster search.
    /// </summary>
    /// <remarks>
    /// IVF partitions vectors into clusters (cells) and only searches the most relevant
    /// clusters during query time. This is an approximate nearest neighbor (ANN) method
    /// that trades some accuracy for significant speed improvements.
    /// Search complexity: O(n/m + k) where n is total vectors, m is number of clusters, k is result size.
    /// Best for medium to large datasets (10K - 10M vectors).
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class IVFIndex<T> : IVectorIndex<T>
    {
        private readonly Dictionary<string, Vector<T>> _vectors;
        private readonly ISimilarityMetric<T> _metric;
        private readonly int _numClusters;
        private readonly int _numProbes;
        private readonly Dictionary<int, List<string>> _clusters;
        private Vector<T>[]? _centroids;

        /// <inheritdoc/>
        public int Count => _vectors.Count;

        /// <summary>
        /// Initializes a new instance of the IVFIndex class.
        /// </summary>
        /// <param name="metric">The similarity metric to use.</param>
        /// <param name="numClusters">Number of clusters to partition vectors into.</param>
        /// <param name="numProbes">Number of clusters to search during query (default: 1).</param>
        public IVFIndex(ISimilarityMetric<T> metric, int numClusters = 100, int numProbes = 1)
        {
            _metric = metric ?? throw new ArgumentNullException(nameof(metric));
            if (numClusters <= 0)
                throw new ArgumentException("Number of clusters must be positive", nameof(numClusters));
            if (numProbes <= 0)
                throw new ArgumentException("Number of probes must be positive", nameof(numProbes));

            _numClusters = numClusters;
            _numProbes = Math.Min(numProbes, numClusters);
            _vectors = new Dictionary<string, Vector<T>>();
            _clusters = new Dictionary<int, List<string>>();
        }

        /// <inheritdoc/>
        public void Add(string id, Vector<T> vector)
        {
            if (string.IsNullOrEmpty(id))
                throw new ArgumentException("ID cannot be null or empty", nameof(id));
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            _vectors[id] = vector;
            _centroids = null; // Invalidate centroids, will rebuild on next search
        }

        /// <inheritdoc/>
        public void AddBatch(Dictionary<string, Vector<T>> vectors)
        {
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            // Validate all entries before adding any
            var invalidEntries = vectors.Where(kvp => string.IsNullOrEmpty(kvp.Key) || kvp.Value == null).ToList();
            if (invalidEntries.Any())
            {
                var firstInvalid = invalidEntries.First();
                if (string.IsNullOrEmpty(firstInvalid.Key))
                    throw new ArgumentException("ID cannot be null or empty", nameof(vectors));
                throw new ArgumentNullException(nameof(vectors), "Vector cannot be null");
            }

            // Add all vectors directly from key-value pairs
            foreach (var kvp in vectors)
            {
                _vectors[kvp.Key] = kvp.Value;
            }
            _centroids = null; // Invalidate centroids, will rebuild on next search
        }

        /// <inheritdoc/>
        public List<(string Id, T Score)> Search(Vector<T> query, int k)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            if (k <= 0)
                throw new ArgumentException("k must be positive", nameof(k));

            if (_vectors.Count == 0)
                return new List<(string Id, T Score)>();

            // Build or rebuild index if needed
            if (_centroids == null)
                BuildIndex();

            // Find nearest clusters
            var nearestClusters = FindNearestClusters(query, _numProbes);

            // Search within those clusters using LINQ for cleaner mapping
            var scores = nearestClusters
                .Where(clusterId => _clusters.ContainsKey(clusterId))
                .SelectMany(clusterId => _clusters[clusterId])
                .Select(id => (Id: id, Score: _metric.Calculate(query, _vectors[id])))
                .ToList();

            var sorted = _metric.HigherIsBetter
                ? scores.OrderByDescending(x => x.Score)
                : scores.OrderBy(x => x.Score);

            return sorted.Take(Math.Min(k, scores.Count)).ToList();
        }

        /// <inheritdoc/>
        public bool Remove(string id)
        {
            var removed = _vectors.Remove(id);
            if (removed)
            {
                _centroids = null; // Invalidate index
            }
            return removed;
        }

        /// <inheritdoc/>
        public void Clear()
        {
            _vectors.Clear();
            _clusters.Clear();
            _centroids = null;
        }

        private void BuildIndex()
        {
            if (_vectors.Count == 0)
                return;

            // Simple k-means clustering (simplified for testing)
            var actualClusters = Math.Min(_numClusters, _vectors.Count);
            _centroids = new Vector<T>[actualClusters];
            _clusters.Clear();

            // Initialize centroids with random vectors
            var vectorList = _vectors.Values.ToList();
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < actualClusters; i++)
            {
                _centroids[i] = vectorList[random.Next(vectorList.Count)];
                _clusters[i] = new List<string>();
            }

            // Assign vectors to nearest centroid using LINQ grouping
            var vectorClusterGroups = _vectors
                .Select(kvp => new { VectorId = kvp.Key, ClusterId = FindNearestClusters(kvp.Value, 1)[0] })
                .GroupBy(x => x.ClusterId);

            foreach (var group in vectorClusterGroups)
            {
                _clusters[group.Key].AddRange(group.Select(x => x.VectorId));
            }
        }

        private List<int> FindNearestClusters(Vector<T> query, int numProbes)
        {
            if (_centroids == null || _centroids.Length == 0)
                return new List<int>();

            var clusterScores = _centroids
                .Select((centroid, i) => (ClusterId: i, Score: _metric.Calculate(query, centroid)))
                .ToList();

            var sorted = _metric.HigherIsBetter
                ? clusterScores.OrderByDescending(x => x.Score)
                : clusterScores.OrderBy(x => x.Score);

            return sorted.Take(numProbes).Select(x => x.ClusterId).ToList();
        }
    }
}
