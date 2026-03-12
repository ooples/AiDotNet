using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Locality-Sensitive Hashing (LSH) index for approximate nearest neighbor search.
    /// </summary>
    /// <remarks>
    /// LSH uses hash functions that map similar vectors to the same hash buckets.
    /// Multiple hash tables are used to improve recall. During search, only vectors
    /// in the same buckets as the query are considered, providing sublinear search time.
    /// Search complexity: O(n^ρ) where ρ &lt; 1, depending on parameters.
    /// Best for high-dimensional sparse data and when approximate results are acceptable.
    /// This is a simplified implementation using random projection for testing.
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations. Must be a numeric type
    /// that implements IConvertible (e.g., float, double, decimal). Using non-numeric types
    /// will result in InvalidCastException at runtime.</typeparam>
    public class LSHIndex<T> : IVectorIndex<T>
    {
        private readonly Dictionary<string, Vector<T>> _vectors;
        private readonly ISimilarityMetric<T> _metric;
        private readonly int _numHashTables;
        private readonly int _numHashFunctions;
        private readonly List<Dictionary<int, List<string>>> _hashTables;
        private readonly List<Vector<T>[]> _randomVectors;
        private readonly Random _random;
        private int _dimension;

        /// <inheritdoc/>
        public int Count => _vectors.Count;

        /// <summary>
        /// Initializes a new instance of the LSHIndex class.
        /// </summary>
        /// <param name="metric">The similarity metric to use.</param>
        /// <param name="numHashTables">Number of hash tables to use (default: 10).</param>
        /// <param name="numHashFunctions">Number of hash functions per table (default: 4).</param>
        /// <param name="seed">Random seed for reproducibility (default: 42).</param>
        public LSHIndex(ISimilarityMetric<T> metric, int numHashTables = 10, int numHashFunctions = 4, int seed = 42)
        {
            Guard.NotNull(metric);
            _metric = metric;
            if (numHashTables <= 0)
                throw new ArgumentException("Number of hash tables must be positive", nameof(numHashTables));
            if (numHashFunctions <= 0)
                throw new ArgumentException("Number of hash functions must be positive", nameof(numHashFunctions));

            _numHashTables = numHashTables;
            _numHashFunctions = numHashFunctions;
            _vectors = new Dictionary<string, Vector<T>>();
            _hashTables = new List<Dictionary<int, List<string>>>();
            _randomVectors = new List<Vector<T>[]>();
            _random = RandomHelper.CreateSeededRandom(seed);
            _dimension = 0;

            for (int i = 0; i < _numHashTables; i++)
            {
                _hashTables.Add(new Dictionary<int, List<string>>());
            }
        }

        /// <inheritdoc/>
        public void Add(string id, Vector<T> vector)
        {
            if (string.IsNullOrEmpty(id))
                throw new ArgumentException("ID cannot be null or empty", nameof(id));
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            if (_vectors.Count == 0)
            {
                _dimension = vector.Length;
                InitializeRandomVectors();
            }
            else if (vector.Length != _dimension)
            {
                throw new ArgumentException($"Vector dimension {vector.Length} does not match index dimension {_dimension}");
            }

            // If ID already exists, remove old hash entries first
            if (_vectors.TryGetValue(id, out var existingVector))
            {
                RemoveFromHashTables(id, existingVector);
            }

            _vectors[id] = vector;

            // Add to hash tables
            for (int tableIdx = 0; tableIdx < _numHashTables; tableIdx++)
            {
                var hash = ComputeHash(vector, tableIdx);
                if (!_hashTables[tableIdx].ContainsKey(hash))
                {
                    _hashTables[tableIdx][hash] = new List<string>();
                }
                _hashTables[tableIdx][hash].Add(id);
            }
        }

        /// <summary>
        /// Removes an ID from all hash tables based on the given vector's hash.
        /// </summary>
        private void RemoveFromHashTables(string id, Vector<T> vector)
        {
            for (int tableIdx = 0; tableIdx < _numHashTables; tableIdx++)
            {
                var hash = ComputeHash(vector, tableIdx);
                if (_hashTables[tableIdx].TryGetValue(hash, out var bucket))
                {
                    bucket.Remove(id);
                    if (bucket.Count == 0)
                    {
                        _hashTables[tableIdx].Remove(hash);
                    }
                }
            }
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

            if (_vectors.Count == 0)
                return new List<(string Id, T Score)>();

            // Collect candidate vectors from all hash tables
            var candidates = new HashSet<string>();
            for (int tableIdx = 0; tableIdx < _numHashTables; tableIdx++)
            {
                var hash = ComputeHash(query, tableIdx);
                if (_hashTables[tableIdx].TryGetValue(hash, out var bucket))
                {
                    foreach (var id in bucket)
                    {
                        candidates.Add(id);
                    }
                }
            }

            // If no candidates found in hash buckets, fall back to full search
            if (candidates.Count == 0)
            {
                candidates = new HashSet<string>(_vectors.Keys);
            }

            // Score all candidates
            var scores = new List<(string Id, T Score)>();
            foreach (var id in candidates)
            {
                var score = _metric.Calculate(query, _vectors[id]);
                scores.Add((id, score));
            }

            var sorted = _metric.HigherIsBetter
                ? scores.OrderByDescending(x => x.Score)
                : scores.OrderBy(x => x.Score);

            return sorted.Take(Math.Min(k, scores.Count)).ToList();
        }

        /// <inheritdoc/>
        public bool Remove(string id)
        {
            if (!_vectors.TryGetValue(id, out var vector))
                return false;

            _vectors.Remove(id);
            RemoveFromHashTables(id, vector);

            return true;
        }

        /// <inheritdoc/>
        public void Clear()
        {
            _vectors.Clear();
            foreach (var table in _hashTables)
            {
                table.Clear();
            }
            _randomVectors.Clear();
            _dimension = 0;
        }

        private void InitializeRandomVectors()
        {
            _randomVectors.Clear();
            for (int tableIdx = 0; tableIdx < _numHashTables; tableIdx++)
            {
                var vectors = new Vector<T>[_numHashFunctions];
                for (int i = 0; i < _numHashFunctions; i++)
                {
                    var components = new T[_dimension];
                    for (int j = 0; j < _dimension; j++)
                    {
                        // Generate random values between -1 and 1
                        var value = _random.NextDouble() * 2 - 1;
                        components[j] = (T)Convert.ChangeType(value, typeof(T));
                    }
                    vectors[i] = new Vector<T>(components);
                }
                _randomVectors.Add(vectors);
            }
        }

        private int ComputeHash(Vector<T> vector, int tableIdx)
        {
            int hash = 0;
            for (int i = 0; i < _numHashFunctions; i++)
            {
                var dotProduct = vector.DotProduct(_randomVectors[tableIdx][i]);
                var bit = Convert.ToDouble(dotProduct) >= 0 ? 1 : 0;
                hash = (hash << 1) | bit;
            }
            return hash;
        }
    }
}
