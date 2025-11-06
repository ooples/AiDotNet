using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// In-memory knowledge graph for storing and querying entity relationships.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// A knowledge graph stores entities (nodes) and their relationships (edges) to enable structured information retrieval.
/// This implementation uses efficient in-memory data structures optimized for graph traversal and querying.
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
/// </para>
/// </remarks>
public class KnowledgeGraph<T>
{
    private readonly Dictionary<string, GraphNode<T>> _nodes;
    private readonly Dictionary<string, GraphEdge<T>> _edges;
    private readonly Dictionary<string, HashSet<string>> _outgoingEdges; // nodeId -> edge IDs going out
    private readonly Dictionary<string, HashSet<string>> _incomingEdges; // nodeId -> edge IDs coming in
    private readonly Dictionary<string, HashSet<string>> _nodesByLabel; // label -> node IDs
    
    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _nodes.Count;
    
    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public int EdgeCount => _edges.Count;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeGraph{T}"/> class.
    /// </summary>
    public KnowledgeGraph()
    {
        _nodes = new Dictionary<string, GraphNode<T>>();
        _edges = new Dictionary<string, GraphEdge<T>>();
        _outgoingEdges = new Dictionary<string, HashSet<string>>();
        _incomingEdges = new Dictionary<string, HashSet<string>>();
        _nodesByLabel = new Dictionary<string, HashSet<string>>();
    }
    
    /// <summary>
    /// Adds a node to the graph or updates it if it already exists.
    /// </summary>
    /// <param name="node">The node to add.</param>
    public void AddNode(GraphNode<T> node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));
            
        _nodes[node.Id] = node;
        
        if (!_nodesByLabel.ContainsKey(node.Label))
            _nodesByLabel[node.Label] = new HashSet<string>();
        _nodesByLabel[node.Label].Add(node.Id);
        
        if (!_outgoingEdges.ContainsKey(node.Id))
            _outgoingEdges[node.Id] = new HashSet<string>();
        if (!_incomingEdges.ContainsKey(node.Id))
            _incomingEdges[node.Id] = new HashSet<string>();
    }
    
    /// <summary>
    /// Adds an edge to the graph.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    /// <exception cref="InvalidOperationException">Thrown when source or target nodes don't exist.</exception>
    public void AddEdge(GraphEdge<T> edge)
    {
        if (edge == null)
            throw new ArgumentNullException(nameof(edge));
        if (!_nodes.ContainsKey(edge.SourceId))
            throw new InvalidOperationException($"Source node '{edge.SourceId}' does not exist");
        if (!_nodes.ContainsKey(edge.TargetId))
            throw new InvalidOperationException($"Target node '{edge.TargetId}' does not exist");
            
        _edges[edge.Id] = edge;
        _outgoingEdges[edge.SourceId].Add(edge.Id);
        _incomingEdges[edge.TargetId].Add(edge.Id);
    }
    
    /// <summary>
    /// Gets a node by its ID.
    /// </summary>
    /// <param name="nodeId">The node ID.</param>
    /// <returns>The node, or null if not found.</returns>
    public GraphNode<T>? GetNode(string nodeId)
    {
        return _nodes.TryGetValue(nodeId, out var node) ? node : null;
    }
    
    /// <summary>
    /// Gets all nodes with a specific label.
    /// </summary>
    /// <param name="label">The node label to filter by.</param>
    /// <returns>Collection of nodes with the specified label.</returns>
    public IEnumerable<GraphNode<T>> GetNodesByLabel(string label)
    {
        if (!_nodesByLabel.TryGetValue(label, out var nodeIds))
            return Enumerable.Empty<GraphNode<T>>();
            
        return nodeIds.Select(id => _nodes[id]);
    }
    
    /// <summary>
    /// Gets all outgoing edges from a node.
    /// </summary>
    /// <param name="nodeId">The source node ID.</param>
    /// <returns>Collection of outgoing edges.</returns>
    public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
    {
        if (!_outgoingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();
            
        return edgeIds.Select(id => _edges[id]);
    }
    
    /// <summary>
    /// Gets all incoming edges to a node.
    /// </summary>
    /// <param name="nodeId">The target node ID.</param>
    /// <returns>Collection of incoming edges.</returns>
    public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId)
    {
        if (!_incomingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();
            
        return edgeIds.Select(id => _edges[id]);
    }
    
    /// <summary>
    /// Gets all neighbors of a node (nodes connected by outgoing edges).
    /// </summary>
    /// <param name="nodeId">The node ID.</param>
    /// <returns>Collection of neighbor nodes.</returns>
    public IEnumerable<GraphNode<T>> GetNeighbors(string nodeId)
    {
        var edges = GetOutgoingEdges(nodeId);
        return edges.Select(e => _nodes[e.TargetId]);
    }
    
    /// <summary>
    /// Performs breadth-first search traversal starting from a node.
    /// </summary>
    /// <param name="startNodeId">The starting node ID.</param>
    /// <param name="maxDepth">Maximum traversal depth (default: unlimited).</param>
    /// <returns>Collection of nodes in BFS order.</returns>
    public IEnumerable<GraphNode<T>> BreadthFirstTraversal(string startNodeId, int maxDepth = int.MaxValue)
    {
        if (!_nodes.ContainsKey(startNodeId))
            yield break;
            
        var visited = new HashSet<string>();
        var queue = new Queue<(string nodeId, int depth)>();
        queue.Enqueue((startNodeId, 0));
        visited.Add(startNodeId);
        
        while (queue.Count > 0)
        {
            var (nodeId, depth) = queue.Dequeue();
            yield return _nodes[nodeId];
            
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
        if (!_nodes.ContainsKey(startNodeId) || !_nodes.ContainsKey(endNodeId))
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
        
        return _nodes.Values
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
        _nodes.Clear();
        _edges.Clear();
        _outgoingEdges.Clear();
        _incomingEdges.Clear();
        _nodesByLabel.Clear();
    }
    
    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    /// <returns>Collection of all nodes.</returns>
    public IEnumerable<GraphNode<T>> GetAllNodes()
    {
        return _nodes.Values;
    }
    
    /// <summary>
    /// Gets all edges in the graph.
    /// </summary>
    /// <returns>Collection of all edges.</returns>
    public IEnumerable<GraphEdge<T>> GetAllEdges()
    {
        return _edges.Values;
    }
}
