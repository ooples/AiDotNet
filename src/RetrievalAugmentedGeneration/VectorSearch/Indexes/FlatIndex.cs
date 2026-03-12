using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Flat index that performs exact brute-force similarity search.
    /// </summary>
    /// <remarks>
    /// This is the simplest and most accurate index type, comparing the query
    /// against every vector in the index. It guarantees exact nearest neighbors
    /// but has O(n) search complexity where n is the number of vectors.
    /// Best for small datasets (< 10K vectors) where accuracy is critical.
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class FlatIndex<T> : IVectorIndex<T>
    {
        private readonly Dictionary<string, Vector<T>> _vectors;
        private readonly ISimilarityMetric<T> _metric;

        /// <inheritdoc/>
        public int Count => _vectors.Count;

        /// <summary>
        /// Initializes a new instance of the FlatIndex class.
        /// </summary>
        /// <param name="metric">The similarity metric to use for search.</param>
        public FlatIndex(ISimilarityMetric<T> metric)
        {
            Guard.NotNull(metric);
            _metric = metric;
            _vectors = new Dictionary<string, Vector<T>>();
        }

        /// <inheritdoc/>
        public void Add(string id, Vector<T> vector)
        {
            if (string.IsNullOrEmpty(id))
                throw new ArgumentException("ID cannot be null or empty", nameof(id));
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            _vectors[id] = vector;
        }

        /// <inheritdoc/>
        public void AddBatch(Dictionary<string, Vector<T>> vectors)
        {
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            foreach (var kvp in vectors)
            {
                Add(kvp.Key, kvp.Value);
            }
        }

        /// <inheritdoc/>
        public List<(string Id, T Score)> Search(Vector<T> query, int k)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            if (k <= 0)
                throw new ArgumentException("k must be positive", nameof(k));

            var scores = new List<(string Id, T Score)>();

            foreach (var kvp in _vectors)
            {
                var score = _metric.Calculate(query, kvp.Value);
                scores.Add((kvp.Key, score));
            }

            // Sort based on whether higher or lower is better
            var sorted = _metric.HigherIsBetter
                ? scores.OrderByDescending(x => x.Score)
                : scores.OrderBy(x => x.Score);

            return sorted.Take(Math.Min(k, scores.Count)).ToList();
        }

        /// <inheritdoc/>
        public bool Remove(string id)
        {
            return _vectors.Remove(id);
        }

        /// <inheritdoc/>
        public void Clear()
        {
            _vectors.Clear();
        }
    }
}
