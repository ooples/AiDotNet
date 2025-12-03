using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Hierarchical Navigable Small World (HNSW) graph-based index for approximate nearest neighbor search.
    /// </summary>
    /// <remarks>
    /// <para>
    /// HNSW builds a multi-layer graph structure where each layer is a proximity graph.
    /// Search starts at the top layer and progressively refines results by moving down layers.
    /// This provides excellent recall with logarithmic search complexity.
    /// </para>
    /// <para>
    /// Search complexity: O(log n) on average where n is the number of vectors.
    /// Best for large datasets (100K+ vectors) requiring high recall and fast search.
    /// </para>
    /// <para>
    /// Based on the paper: "Efficient and robust approximate nearest neighbor search using
    /// Hierarchical Navigable Small World graphs" by Malkov and Yashunin (2018).
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HNSWIndex<T> : IVectorIndex<T>
    {
        private readonly Dictionary<string, Vector<T>> _vectors;
        private readonly ISimilarityMetric<T> _metric;
        private readonly INumericOperations<T> _numOps;
        private readonly int _maxConnections;        // M parameter
        private readonly int _maxConnectionsLayer0;  // M0 = 2 * M for layer 0
        private readonly int _efConstruction;        // ef during construction
        private readonly int _efSearch;              // ef during search
        private readonly double _levelMultiplier;    // mL = 1 / ln(M)
        private readonly Random _random;

        // Multi-layer graph: layer -> (nodeId -> list of neighbor nodeIds)
        private readonly List<Dictionary<string, List<string>>> _layers;
        private readonly Dictionary<string, int> _nodeMaxLayer;  // Maximum layer for each node
        private string? _entryPoint;
        private int _maxLevel;

        /// <inheritdoc/>
        public int Count => _vectors.Count;

        /// <summary>
        /// Initializes a new instance of the HNSWIndex class.
        /// </summary>
        /// <param name="metric">The similarity metric to use.</param>
        /// <param name="maxConnections">Maximum number of connections per node (M parameter). Default: 16.</param>
        /// <param name="efConstruction">Size of dynamic candidate list during construction. Default: 200.</param>
        /// <param name="efSearch">Size of dynamic candidate list during search. Default: 50.</param>
        /// <param name="seed">Random seed for reproducibility. Default: 42.</param>
        public HNSWIndex(ISimilarityMetric<T> metric, int maxConnections = 16, int efConstruction = 200, int efSearch = 50, int seed = 42)
        {
            _metric = metric ?? throw new ArgumentNullException(nameof(metric));
            if (maxConnections < 2)
                throw new ArgumentException("Max connections (M) must be at least 2", nameof(maxConnections));
            if (efConstruction <= 0)
                throw new ArgumentException("EF construction must be positive", nameof(efConstruction));
            if (efSearch <= 0)
                throw new ArgumentException("EF search must be positive", nameof(efSearch));

            _numOps = MathHelper.GetNumericOperations<T>();
            _maxConnections = maxConnections;
            _maxConnectionsLayer0 = maxConnections * 2;  // Layer 0 has more connections
            _efConstruction = efConstruction;
            _efSearch = efSearch;
            _levelMultiplier = 1.0 / Math.Log(maxConnections);
            _random = new Random(seed);

            _vectors = new Dictionary<string, Vector<T>>();
            _layers = new List<Dictionary<string, List<string>>>();
            _nodeMaxLayer = new Dictionary<string, int>();
            _entryPoint = null;
            _maxLevel = -1;
        }

        /// <inheritdoc/>
        public void Add(string id, Vector<T> vector)
        {
            if (string.IsNullOrEmpty(id))
                throw new ArgumentException("ID cannot be null or empty", nameof(id));
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            // If updating existing vector, remove it first
            if (_vectors.ContainsKey(id))
            {
                Remove(id);
            }

            _vectors[id] = vector;

            // Calculate random level for this node
            int nodeLevel = GetRandomLevel();
            _nodeMaxLayer[id] = nodeLevel;

            // Ensure we have enough layers
            while (_layers.Count <= nodeLevel)
            {
                _layers.Add(new Dictionary<string, List<string>>());
            }

            // Initialize empty neighbor lists for this node at all its levels
            for (int level = 0; level <= nodeLevel; level++)
            {
                _layers[level][id] = new List<string>();
            }

            // If this is the first node, set it as entry point
            if (_entryPoint == null)
            {
                _entryPoint = id;
                _maxLevel = nodeLevel;
                return;
            }

            // Find entry point for insertion
            string currentNode = _entryPoint;

            // Traverse from top to the node's level + 1, finding closest node at each level
            for (int level = _maxLevel; level > nodeLevel; level--)
            {
                currentNode = GreedySearchClosest(vector, currentNode, level);
            }

            // For levels from nodeLevel down to 0, find neighbors and connect
            for (int level = Math.Min(nodeLevel, _maxLevel); level >= 0; level--)
            {
                int maxConn = level == 0 ? _maxConnectionsLayer0 : _maxConnections;

                // Search for candidates at this level
                var candidates = SearchLayer(vector, currentNode, _efConstruction, level);

                // Select best neighbors
                var neighbors = SelectNeighbors(vector, candidates, maxConn);

                // Connect the new node to its neighbors (bidirectional)
                foreach (var neighbor in neighbors)
                {
                    // Add edge from new node to neighbor
                    _layers[level][id].Add(neighbor.Id);

                    // Add edge from neighbor to new node
                    if (_layers[level].TryGetValue(neighbor.Id, out var neighborConnections))
                    {
                        neighborConnections.Add(id);

                        // Prune neighbor's connections if exceeding max
                        if (neighborConnections.Count > maxConn)
                        {
                            PruneConnections(neighbor.Id, level, maxConn);
                        }
                    }
                }

                // Use the closest candidate as entry point for next level
                if (candidates.Count > 0)
                {
                    currentNode = candidates[0].Id;
                }
            }

            // Update entry point if new node has higher level
            if (nodeLevel > _maxLevel)
            {
                _entryPoint = id;
                _maxLevel = nodeLevel;
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

            if (_vectors.Count == 0 || _entryPoint == null)
                return new List<(string Id, T Score)>();

            // Start from entry point at top level
            string currentNode = _entryPoint;

            // Traverse from top level down to level 1, finding closest node
            for (int level = _maxLevel; level > 0; level--)
            {
                currentNode = GreedySearchClosest(query, currentNode, level);
            }

            // Search at level 0 with efSearch candidates
            int ef = Math.Max(_efSearch, k);
            var candidates = SearchLayer(query, currentNode, ef, 0);

            // Return top k results
            return candidates.Take(k).ToList();
        }

        /// <inheritdoc/>
        public bool Remove(string id)
        {
            if (!_vectors.Remove(id))
                return false;

            if (!_nodeMaxLayer.TryGetValue(id, out int nodeLevel))
                return true;

            _nodeMaxLayer.Remove(id);

            // Remove from all layers
            for (int level = 0; level <= nodeLevel && level < _layers.Count; level++)
            {
                if (_layers[level].TryGetValue(id, out var connections))
                {
                    // Remove this node from all its neighbors' connection lists
                    foreach (var neighborId in connections)
                    {
                        if (_layers[level].TryGetValue(neighborId, out var neighborConnections))
                        {
                            neighborConnections.Remove(id);
                        }
                    }
                    _layers[level].Remove(id);
                }
            }

            // Update entry point if we removed it
            if (_entryPoint == id)
            {
                _entryPoint = _vectors.Keys.FirstOrDefault();
                if (_entryPoint != null && _nodeMaxLayer.TryGetValue(_entryPoint, out int newMaxLevel))
                {
                    _maxLevel = newMaxLevel;
                }
                else
                {
                    _maxLevel = -1;
                }
            }

            return true;
        }

        /// <inheritdoc/>
        public void Clear()
        {
            _vectors.Clear();
            _layers.Clear();
            _nodeMaxLayer.Clear();
            _entryPoint = null;
            _maxLevel = -1;
        }

        /// <summary>
        /// Generates a random level for a new node using exponential distribution.
        /// </summary>
        private int GetRandomLevel()
        {
            double r = _random.NextDouble();
            return (int)Math.Floor(-Math.Log(r) * _levelMultiplier);
        }

        /// <summary>
        /// Performs greedy search to find the single closest node at a given level.
        /// </summary>
        private string GreedySearchClosest(Vector<T> query, string entryPoint, int level)
        {
            string current = entryPoint;
            T currentDist = _metric.Calculate(query, _vectors[current]);

            bool improved = true;
            while (improved)
            {
                improved = false;

                if (!_layers[level].TryGetValue(current, out var neighbors))
                    break;

                foreach (var neighborId in neighbors)
                {
                    if (!_vectors.TryGetValue(neighborId, out var neighborVector))
                        continue;

                    T neighborDist = _metric.Calculate(query, neighborVector);

                    if (IsBetterScore(neighborDist, currentDist))
                    {
                        current = neighborId;
                        currentDist = neighborDist;
                        improved = true;
                    }
                }
            }

            return current;
        }

        /// <summary>
        /// Searches a single layer for ef nearest neighbors using beam search.
        /// </summary>
        private List<(string Id, T Score)> SearchLayer(Vector<T> query, string entryPoint, int ef, int level)
        {
            var visited = new HashSet<string> { entryPoint };
            T entryDist = _metric.Calculate(query, _vectors[entryPoint]);

            // Use lists and manual sorting for candidates and results
            var candidates = new List<(string Id, T Score)> { (entryPoint, entryDist) };
            var results = new List<(string Id, T Score)> { (entryPoint, entryDist) };

            while (candidates.Count > 0)
            {
                // Get best candidate (first in sorted list)
                var nearest = candidates[0];
                candidates.RemoveAt(0);

                // Get worst result
                var furthest = results[results.Count - 1];

                // If nearest candidate is worse than worst result, we're done
                if (results.Count >= ef && !IsBetterScore(nearest.Score, furthest.Score))
                    break;

                // Explore neighbors
                if (!_layers[level].TryGetValue(nearest.Id, out var neighbors))
                    continue;

                foreach (var neighborId in neighbors)
                {
                    if (visited.Contains(neighborId))
                        continue;

                    visited.Add(neighborId);

                    if (!_vectors.TryGetValue(neighborId, out var neighborVector))
                        continue;

                    T neighborDist = _metric.Calculate(query, neighborVector);
                    furthest = results[results.Count - 1];

                    // Add to results if better than worst result or we have room
                    if (results.Count < ef || IsBetterScore(neighborDist, furthest.Score))
                    {
                        // Insert into candidates maintaining sorted order
                        InsertSorted(candidates, (neighborId, neighborDist));

                        // Insert into results maintaining sorted order
                        InsertSorted(results, (neighborId, neighborDist));

                        // Remove worst if we exceed ef
                        if (results.Count > ef)
                        {
                            results.RemoveAt(results.Count - 1);
                        }
                    }
                }
            }

            return results;
        }

        /// <summary>
        /// Inserts an item into a sorted list maintaining sort order.
        /// </summary>
        private void InsertSorted(List<(string Id, T Score)> list, (string Id, T Score) item)
        {
            int index = 0;
            while (index < list.Count && IsBetterScore(list[index].Score, item.Score))
            {
                index++;
            }
            list.Insert(index, item);
        }

        /// <summary>
        /// Selects the best neighbors from candidates using simple selection.
        /// </summary>
        private List<(string Id, T Score)> SelectNeighbors(Vector<T> query, List<(string Id, T Score)> candidates, int maxNeighbors)
        {
            // Already sorted by score, just take top maxNeighbors
            return candidates.Take(maxNeighbors).ToList();
        }

        /// <summary>
        /// Prunes connections for a node to keep only the best ones.
        /// </summary>
        private void PruneConnections(string nodeId, int level, int maxConnections)
        {
            if (!_layers[level].TryGetValue(nodeId, out var connections) ||
                !_vectors.TryGetValue(nodeId, out var nodeVector))
                return;

            // Score all connections and keep the best
            var scored = connections
                .Where(neighborId => _vectors.ContainsKey(neighborId))
                .Select(neighborId => (
                    Id: neighborId,
                    Score: _metric.Calculate(nodeVector, _vectors[neighborId])
                ))
                .ToList();

            // Sort by best score first
            if (_metric.HigherIsBetter)
                scored = scored.OrderByDescending(x => x.Score).ToList();
            else
                scored = scored.OrderBy(x => x.Score).ToList();

            _layers[level][nodeId] = scored.Take(maxConnections).Select(x => x.Id).ToList();
        }

        /// <summary>
        /// Returns true if score a is better than score b according to the metric.
        /// </summary>
        private bool IsBetterScore(T a, T b)
        {
            if (_metric.HigherIsBetter)
                return _numOps.GreaterThan(a, b);
            else
                return _numOps.LessThan(a, b);
        }
    }
}
