using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// In-memory implementation of <see cref="IGraphStore{T}"/> using dictionaries for fast lookups.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// This implementation provides high-performance graph storage entirely in RAM.
/// All operations are O(1) or O(degree) complexity. Data is lost when the application stops.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is NOT thread-safe. Callers must ensure proper
/// synchronization when accessing from multiple threads. For thread-safe operations,
/// use external locking or consider using <see cref="FileGraphStore{T}"/> which provides
/// thread-safe access via ConcurrentDictionary.
/// </para>
/// <para><b>For Beginners:</b> This stores your graph in the computer's memory (RAM).
///
/// Pros:
/// - Very fast (everything in RAM)
/// - Simple to use (no setup required)
///
/// Cons:
/// - Data lost when app closes
/// - Limited by available RAM
/// - Not thread-safe (single-threaded use only)
///
/// Good for:
/// - Development and testing
/// - Small to medium graphs (&lt;100K nodes)
/// - Temporary graphs that don't need persistence
///
/// Not good for:
/// - Production systems requiring persistence
/// - Very large graphs (&gt;1M nodes)
/// - Multi-process or multi-threaded access to the same graph
///
/// For persistent storage, use FileGraphStore or Neo4jGraphStore instead.
/// </para>
/// </remarks>
public class MemoryGraphStore<T> : IGraphStore<T>
{
    private readonly Dictionary<string, GraphNode<T>> _nodes;
    private readonly Dictionary<string, GraphEdge<T>> _edges;
    private readonly Dictionary<string, HashSet<string>> _outgoingEdges; // nodeId -> edge IDs going out
    private readonly Dictionary<string, HashSet<string>> _incomingEdges; // nodeId -> edge IDs coming in
    private readonly Dictionary<string, HashSet<string>> _nodesByLabel; // label -> node IDs

    /// <inheritdoc/>
    public int NodeCount => _nodes.Count;

    /// <inheritdoc/>
    public int EdgeCount => _edges.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryGraphStore{T}"/> class.
    /// </summary>
    public MemoryGraphStore()
    {
        _nodes = new Dictionary<string, GraphNode<T>>();
        _edges = new Dictionary<string, GraphEdge<T>>();
        _outgoingEdges = new Dictionary<string, HashSet<string>>();
        _incomingEdges = new Dictionary<string, HashSet<string>>();
        _nodesByLabel = new Dictionary<string, HashSet<string>>();
    }

    /// <inheritdoc/>
    public void AddNode(GraphNode<T> node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        // Remove old label index if node exists with different label
        if (_nodes.TryGetValue(node.Id, out var existingNode) && existingNode.Label != node.Label)
        {
            if (_nodesByLabel.TryGetValue(existingNode.Label, out var oldLabelNodeIds))
            {
                oldLabelNodeIds.Remove(node.Id);
                if (oldLabelNodeIds.Count == 0)
                    _nodesByLabel.Remove(existingNode.Label);
            }
        }

        _nodes[node.Id] = node;

        if (!_nodesByLabel.ContainsKey(node.Label))
            _nodesByLabel[node.Label] = new HashSet<string>();
        _nodesByLabel[node.Label].Add(node.Id);

        if (!_outgoingEdges.ContainsKey(node.Id))
            _outgoingEdges[node.Id] = new HashSet<string>();
        if (!_incomingEdges.ContainsKey(node.Id))
            _incomingEdges[node.Id] = new HashSet<string>();
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    public GraphNode<T>? GetNode(string nodeId)
    {
        return _nodes.TryGetValue(nodeId, out var node) ? node : null;
    }

    /// <inheritdoc/>
    public GraphEdge<T>? GetEdge(string edgeId)
    {
        return _edges.TryGetValue(edgeId, out var edge) ? edge : null;
    }

    /// <inheritdoc/>
    public bool RemoveNode(string nodeId)
    {
        if (!_nodes.TryGetValue(nodeId, out var node))
            return false;

        // Remove all outgoing edges
        if (_outgoingEdges.TryGetValue(nodeId, out var outgoing))
        {
            foreach (var edgeId in outgoing.ToList())
            {
                if (_edges.TryGetValue(edgeId, out var edge))
                {
                    _edges.Remove(edgeId);
                    _incomingEdges[edge.TargetId].Remove(edgeId);
                }
            }
            _outgoingEdges.Remove(nodeId);
        }

        // Remove all incoming edges
        if (_incomingEdges.TryGetValue(nodeId, out var incoming))
        {
            foreach (var edgeId in incoming.ToList())
            {
                if (_edges.TryGetValue(edgeId, out var edge))
                {
                    _edges.Remove(edgeId);
                    _outgoingEdges[edge.SourceId].Remove(edgeId);
                }
            }
            _incomingEdges.Remove(nodeId);
        }

        // Remove from label index
        if (_nodesByLabel.TryGetValue(node.Label, out var nodeIds))
        {
            nodeIds.Remove(nodeId);
            if (nodeIds.Count == 0)
                _nodesByLabel.Remove(node.Label);
        }

        // Remove the node itself
        _nodes.Remove(nodeId);
        return true;
    }

    /// <inheritdoc/>
    public bool RemoveEdge(string edgeId)
    {
        if (!_edges.TryGetValue(edgeId, out var edge))
            return false;

        _edges.Remove(edgeId);
        _outgoingEdges[edge.SourceId].Remove(edgeId);
        _incomingEdges[edge.TargetId].Remove(edgeId);
        return true;
    }

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
    {
        if (!_outgoingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        // Use TryGetValue to safely handle edges that may have been removed
        return edgeIds
            .Select(id => _edges.TryGetValue(id, out var edge) ? edge : null)
            .OfType<GraphEdge<T>>();
    }

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId)
    {
        if (!_incomingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        // Use TryGetValue to safely handle edges that may have been removed
        return edgeIds
            .Select(id => _edges.TryGetValue(id, out var edge) ? edge : null)
            .OfType<GraphEdge<T>>();
    }

    /// <inheritdoc/>
    public IEnumerable<GraphNode<T>> GetNodesByLabel(string label)
    {
        if (!_nodesByLabel.TryGetValue(label, out var nodeIds))
            return Enumerable.Empty<GraphNode<T>>();

        // Use TryGetValue to safely handle nodes that may have been removed
        return nodeIds
            .Select(id => _nodes.TryGetValue(id, out var node) ? node : null)
            .OfType<GraphNode<T>>();
    }

    /// <inheritdoc/>
    public IEnumerable<GraphNode<T>> GetAllNodes()
    {
        return _nodes.Values;
    }

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetAllEdges()
    {
        return _edges.Values;
    }

    /// <inheritdoc/>
    public void Clear()
    {
        _nodes.Clear();
        _edges.Clear();
        _outgoingEdges.Clear();
        _incomingEdges.Clear();
        _nodesByLabel.Clear();
    }

    // Async methods (for MemoryGraphStore, these wrap synchronous operations)

    /// <inheritdoc/>
    public Task AddNodeAsync(GraphNode<T> node)
    {
        AddNode(node);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task AddEdgeAsync(GraphEdge<T> edge)
    {
        AddEdge(edge);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<GraphNode<T>?> GetNodeAsync(string nodeId)
    {
        return Task.FromResult(GetNode(nodeId));
    }

    /// <inheritdoc/>
    public Task<GraphEdge<T>?> GetEdgeAsync(string edgeId)
    {
        return Task.FromResult(GetEdge(edgeId));
    }

    /// <inheritdoc/>
    public Task<bool> RemoveNodeAsync(string nodeId)
    {
        return Task.FromResult(RemoveNode(nodeId));
    }

    /// <inheritdoc/>
    public Task<bool> RemoveEdgeAsync(string edgeId)
    {
        return Task.FromResult(RemoveEdge(edgeId));
    }

    /// <inheritdoc/>
    public Task<IEnumerable<GraphEdge<T>>> GetOutgoingEdgesAsync(string nodeId)
    {
        return Task.FromResult(GetOutgoingEdges(nodeId));
    }

    /// <inheritdoc/>
    public Task<IEnumerable<GraphEdge<T>>> GetIncomingEdgesAsync(string nodeId)
    {
        return Task.FromResult(GetIncomingEdges(nodeId));
    }

    /// <inheritdoc/>
    public Task<IEnumerable<GraphNode<T>>> GetNodesByLabelAsync(string label)
    {
        return Task.FromResult(GetNodesByLabel(label));
    }

    /// <inheritdoc/>
    public Task<IEnumerable<GraphNode<T>>> GetAllNodesAsync()
    {
        return Task.FromResult(GetAllNodes());
    }

    /// <inheritdoc/>
    public Task<IEnumerable<GraphEdge<T>>> GetAllEdgesAsync()
    {
        return Task.FromResult(GetAllEdges());
    }

    /// <inheritdoc/>
    public Task ClearAsync()
    {
        Clear();
        return Task.CompletedTask;
    }
}
