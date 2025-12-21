using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

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
            _random = RandomHelper.CreateSeededRandom(seed);

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

            // Initialize node in the graph
            int nodeLevel = InitializeNode(id);

            // If this is the first node, set it as entry point
            if (_entryPoint == null)
            {
                _entryPoint = id;
                _maxLevel = nodeLevel;
                return;
            }

            // Find entry point for insertion and connect at each level
            string currentNode = FindInsertionEntryPoint(vector, nodeLevel);
            ConnectNodeAtAllLevels(id, vector, nodeLevel, currentNode);

            // Update entry point if new node has higher level
            if (nodeLevel > _maxLevel)
            {
                _entryPoint = id;
                _maxLevel = nodeLevel;
            }
        }

        /// <summary>
        /// Initializes a new node in the graph with random level assignment.
        /// </summary>
        private int InitializeNode(string id)
        {
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

            return nodeLevel;
        }

        /// <summary>
        /// Finds the entry point for inserting a new node by traversing from top level.
        /// </summary>
        /// <remarks>This method should only be called when _entryPoint is not null.</remarks>
        private string FindInsertionEntryPoint(Vector<T> vector, int nodeLevel)
        {
            // _entryPoint is guaranteed non-null when this method is called (checked in Add)
            string currentNode = _entryPoint ?? throw new InvalidOperationException("Entry point is null");

            // Traverse from top to the node's level + 1, finding closest node at each level
            for (int level = _maxLevel; level > nodeLevel; level--)
            {
                currentNode = GreedySearchClosest(vector, currentNode, level);
            }

            return currentNode;
        }

        /// <summary>
        /// Connects a new node to neighbors at all applicable levels.
        /// </summary>
        private void ConnectNodeAtAllLevels(string id, Vector<T> vector, int nodeLevel, string entryNode)
        {
            string currentNode = entryNode;
            for (int level = Math.Min(nodeLevel, _maxLevel); level >= 0; level--)
            {
                int maxConn = level == 0 ? _maxConnectionsLayer0 : _maxConnections;
                var candidates = SearchLayer(vector, currentNode, _efConstruction, level);
                var neighbors = SelectNeighbors(vector, candidates, maxConn);

                ConnectNodeToNeighbors(id, neighbors, level, maxConn);

                // Use the closest candidate as entry point for next level (skip on last iteration)
                if (level > 0)
                {
                    currentNode = candidates.Count > 0 ? candidates[0].Id : entryNode;
                }
            }
        }

        /// <summary>
        /// Creates bidirectional connections between a node and its neighbors at a specific level.
        /// </summary>
        private void ConnectNodeToNeighbors(string id, List<(string Id, T Score)> neighbors, int level, int maxConn)
        {
            foreach (var neighbor in neighbors)
            {
                AddBidirectionalEdge(id, neighbor.Id, level, maxConn);
            }
        }

        /// <summary>
        /// Adds a bidirectional edge between two nodes at a specific level.
        /// </summary>
        private void AddBidirectionalEdge(string nodeId, string neighborId, int level, int maxConn)
        {
            _layers[level][nodeId].Add(neighborId);
            AddReverseEdgeAndPrune(neighborId, nodeId, level, maxConn);
        }

        /// <summary>
        /// Adds reverse edge from neighbor to node and prunes if needed.
        /// </summary>
        private void AddReverseEdgeAndPrune(string neighborId, string nodeId, int level, int maxConn)
        {
            if (!_layers[level].TryGetValue(neighborId, out var connections))
                return;

            connections.Add(nodeId);

            if (connections.Count > maxConn)
                PruneConnections(neighborId, level, maxConn);
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
                    var layerAtLevel = _layers[level];
                    var neighborConnections = connections
                        .Where(neighborId => layerAtLevel.ContainsKey(neighborId))
                        .Select(neighborId => layerAtLevel[neighborId]);

                    foreach (var connections_list in neighborConnections)
                    {
                        connections_list.Remove(id);
                    }
                    _layers[level].Remove(id);
                }
            }

            // Update entry point if we removed it - must select node with highest level to maintain HNSW invariant
            if (_entryPoint == id)
            {
                var highestLevelNode = _nodeMaxLayer
                    .OrderByDescending(kvp => kvp.Value)
                    .Select(kvp => kvp.Key)
                    .FirstOrDefault();

                _entryPoint = highestLevelNode;
                _maxLevel = highestLevelNode != null && _nodeMaxLayer.TryGetValue(highestLevelNode, out int newMaxLevel)
                    ? newMaxLevel
                    : -1;
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
            // Guard against r being too close to zero which would cause -Math.Log(0) = PositiveInfinity
            // Use Math.Max to ensure a safe minimum value without floating point comparison
            const double MinThreshold = 1e-10;
            double safeValue = Math.Max(r, MinThreshold);
            return (int)Math.Floor(-Math.Log(safeValue) * _levelMultiplier);
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

                var validNeighbors = neighbors
                    .Where(nId => _vectors.ContainsKey(nId))
                    .Select(nId => (nId, _vectors[nId]));

                foreach (var (neighborId, neighborVector) in validNeighbors)
                {
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

                var unvisitedNeighbors = neighbors
                    .Where(nId => !visited.Contains(nId) && _vectors.ContainsKey(nId))
                    .Select(nId => (nId, _vectors[nId]));

                foreach (var (neighborId, neighborVector) in unvisitedNeighbors)
                {
                    visited.Add(neighborId);

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

            // Score all connections and keep the best, sorted by best score first
            var scored = connections
                .Where(neighborId => _vectors.ContainsKey(neighborId))
                .Select(neighborId => (
                    Id: neighborId,
                    Score: _metric.Calculate(nodeVector, _vectors[neighborId])
                ));

            var sortedScored = _metric.HigherIsBetter
                ? scored.OrderByDescending(x => x.Score)
                : scored.OrderBy(x => x.Score);

            _layers[level][nodeId] = sortedScored.Take(maxConnections).Select(x => x.Id).ToList();
        }

        /// <summary>
        /// Returns true if score a is better than score b according to the metric.
        /// </summary>
        private bool IsBetterScore(T a, T b) =>
            _metric.HigherIsBetter ? _numOps.GreaterThan(a, b) : _numOps.LessThan(a, b);
    }
}
