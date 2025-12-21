using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Knowledge graph for storing and querying entity relationships using a pluggable storage backend.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// A knowledge graph stores entities (nodes) and their relationships (edges) to enable structured information retrieval.
/// This implementation delegates storage operations to an <see cref="IGraphStore{T}"/> implementation,
/// allowing you to swap between in-memory, file-based, or database-backed storage.
/// </para>
/// <para><b>For Beginners:</b> A knowledge graph is like a map of how information connects together.
///
/// Imagine Wikipedia as a graph:
/// - Each article is a node (Albert Einstein, Physics, Germany, etc.)
/// - Links between articles are edges (Einstein STUDIED Physics, Einstein BORN_IN Germany)
/// - You can traverse the graph to find related information
///
/// This class lets you:
/// 1. Add entities and relationships
/// 2. Find connections between entities
/// 3. Traverse the graph to discover related information
/// 4. Query based on entity types or relationships
///
/// For example, to answer "Who worked at Princeton?":
/// 1. Find all edges with type "WORKED_AT"
/// 2. Filter for target = "Princeton University"
/// 3. Return the source entities (people who worked there)
///
/// Storage backends you can use:
/// - MemoryGraphStore: Fast, in-memory (default)
/// - FileGraphStore: Persistent, disk-based
/// - Neo4jGraphStore: Professional graph database (future)
/// </para>
/// </remarks>
public class KnowledgeGraph<T>
{
    private readonly IGraphStore<T> _store;

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _store.NodeCount;

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public int EdgeCount => _store.EdgeCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeGraph{T}"/> class with a custom graph store.
    /// </summary>
    /// <param name="store">The graph store implementation to use for storage.</param>
    public KnowledgeGraph(IGraphStore<T> store)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeGraph{T}"/> class with default in-memory storage.
    /// </summary>
    public KnowledgeGraph() : this(new MemoryGraphStore<T>())
    {
    }

    /// <summary>
    /// Adds a node to the graph or updates it if it already exists.
    /// </summary>
    /// <param name="node">The node to add.</param>
    public void AddNode(GraphNode<T> node)
    {
        _store.AddNode(node);
    }

    /// <summary>
    /// Adds an edge to the graph.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    /// <exception cref="InvalidOperationException">Thrown when source or target nodes don't exist.</exception>
    public void AddEdge(GraphEdge<T> edge)
    {
        _store.AddEdge(edge);
    }

    /// <summary>
    /// Gets a node by its ID.
    /// </summary>
    /// <param name="nodeId">The node ID.</param>
    /// <returns>The node, or null if not found.</returns>
    public GraphNode<T>? GetNode(string nodeId)
    {
        return _store.GetNode(nodeId);
    }

    /// <summary>
    /// Gets all nodes with a specific label.
    /// </summary>
    /// <param name="label">The node label to filter by.</param>
    /// <returns>Collection of nodes with the specified label.</returns>
    public IEnumerable<GraphNode<T>> GetNodesByLabel(string label)
    {
        return _store.GetNodesByLabel(label);
    }

    /// <summary>
    /// Gets all outgoing edges from a node.
    /// </summary>
    /// <param name="nodeId">The source node ID.</param>
    /// <returns>Collection of outgoing edges.</returns>
    public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
    {
        return _store.GetOutgoingEdges(nodeId);
    }

    /// <summary>
    /// Gets all incoming edges to a node.
    /// </summary>
    /// <param name="nodeId">The target node ID.</param>
    /// <returns>Collection of incoming edges.</returns>
    public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId)
    {
        return _store.GetIncomingEdges(nodeId);
    }

    /// <summary>
    /// Gets all neighbors of a node (nodes connected by outgoing edges).
    /// </summary>
    /// <param name="nodeId">The node ID.</param>
    /// <returns>Collection of neighbor nodes.</returns>
    public IEnumerable<GraphNode<T>> GetNeighbors(string nodeId)
    {
        var edges = GetOutgoingEdges(nodeId);
        return edges
            .Select(e => _store.GetNode(e.TargetId))
            .OfType<GraphNode<T>>();
    }

    /// <summary>
    /// Performs breadth-first search traversal starting from a node.
    /// </summary>
    /// <param name="startNodeId">The starting node ID.</param>
    /// <param name="maxDepth">Maximum traversal depth (default: unlimited).</param>
    /// <returns>Collection of nodes in BFS order.</returns>
    public IEnumerable<GraphNode<T>> BreadthFirstTraversal(string startNodeId, int maxDepth = int.MaxValue)
    {
        if (_store.GetNode(startNodeId) == null)
            yield break;

        var visited = new HashSet<string>();
        var queue = new Queue<(string nodeId, int depth)>();
        queue.Enqueue((startNodeId, 0));
        visited.Add(startNodeId);

        while (queue.Count > 0)
        {
            var (nodeId, depth) = queue.Dequeue();
            var node = _store.GetNode(nodeId);

            // Skip if node was removed between queue add and dequeue
            if (node == null)
                continue;

            yield return node;

            if (depth >= maxDepth)
                continue;

            foreach (var edge in GetOutgoingEdges(nodeId))
            {
                if (!visited.Contains(edge.TargetId))
                {
                    visited.Add(edge.TargetId);
                    queue.Enqueue((edge.TargetId, depth + 1));
                }
            }
        }
    }

    /// <summary>
    /// Finds the shortest path between two nodes using BFS.
    /// </summary>
    /// <param name="startNodeId">The starting node ID.</param>
    /// <param name="endNodeId">The target node ID.</param>
    /// <returns>List of node IDs representing the path, or empty if no path exists.</returns>
    public List<string> FindShortestPath(string startNodeId, string endNodeId)
    {
        if (_store.GetNode(startNodeId) == null || _store.GetNode(endNodeId) == null)
            return new List<string>();

        var visited = new HashSet<string>();
        var parent = new Dictionary<string, string>();
        var queue = new Queue<string>();

        queue.Enqueue(startNodeId);
        visited.Add(startNodeId);

        while (queue.Count > 0)
        {
            var nodeId = queue.Dequeue();

            if (nodeId == endNodeId)
            {
                // Reconstruct path
                var path = new List<string>();
                var current = endNodeId;
                while (current != startNodeId)
                {
                    path.Add(current);
                    current = parent[current];
                }
                path.Add(startNodeId);
                path.Reverse();
                return path;
            }

            foreach (var edge in GetOutgoingEdges(nodeId))
            {
                if (!visited.Contains(edge.TargetId))
                {
                    visited.Add(edge.TargetId);
                    parent[edge.TargetId] = nodeId;
                    queue.Enqueue(edge.TargetId);
                }
            }
        }

        return new List<string>(); // No path found
    }

    /// <summary>
    /// Finds nodes related to a query by entity name or property matching.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <returns>Collection of matching nodes.</returns>
    public IEnumerable<GraphNode<T>> FindRelatedNodes(string query, int topK = 10)
    {
        var queryLower = query.ToLowerInvariant();

        return _store.GetAllNodes()
            .Where(node =>
            {
                var name = node.GetProperty<string>("name") ?? node.Id;
                return name.ToLowerInvariant().Contains(queryLower) ||
                       node.Label.ToLowerInvariant().Contains(queryLower);
            })
            .Take(topK);
    }

    /// <summary>
    /// Clears all nodes and edges from the graph.
    /// </summary>
    public void Clear()
    {
        _store.Clear();
    }

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    /// <returns>Collection of all nodes.</returns>
    public IEnumerable<GraphNode<T>> GetAllNodes()
    {
        return _store.GetAllNodes();
    }

    /// <summary>
    /// Gets all edges in the graph.
    /// </summary>
    /// <returns>Collection of all edges.</returns>
    public IEnumerable<GraphEdge<T>> GetAllEdges()
    {
        return _store.GetAllEdges();
    }
}
