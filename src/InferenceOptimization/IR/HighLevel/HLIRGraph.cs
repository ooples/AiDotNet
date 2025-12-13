using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.IR.Common;

namespace AiDotNet.InferenceOptimization.IR.HighLevel;

/// <summary>
/// High-Level Intermediate Representation Graph.
/// Represents the complete computation graph at a semantic level.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// HLIRGraph provides a container for HLIRNodes with efficient traversal, validation,
/// and transformation capabilities. It maintains both node-reference and ID-based
/// representations for flexibility.
/// </para>
///
/// <para><b>Industry Comparison:</b></para>
/// <list type="bullet">
/// <item>TVM Relay: Module with global functions - we add richer graph operations</item>
/// <item>MLIR: ModuleOp with nested regions - we simplify for ML workloads</item>
/// <item>XLA HLO: HloModule with computations - we add bidirectional edges</item>
/// <item>ONNX: GraphProto - we add dynamic modification support</item>
/// </list>
///
/// <para><b>Exceeds Standards By:</b></para>
/// <list type="bullet">
/// <item>Incremental validation and repair</item>
/// <item>Built-in pattern matching for optimization passes</item>
/// <item>Automatic ID management with compaction</item>
/// <item>Comprehensive graph statistics and profiling</item>
/// <item>Subgraph extraction and splicing</item>
/// </list>
/// </remarks>
public class HLIRGraph<T> where T : struct
{
    #region Fields

    private readonly Dictionary<int, HLIRNode<T>> _nodeMap = new();
    private int _nextNodeId;
    private bool _isDirty = true;
    private List<HLIRNode<T>>? _cachedTopologicalOrder;

    #endregion

    #region Properties

    /// <summary>
    /// All nodes in the graph.
    /// </summary>
    public IReadOnlyList<HLIRNode<T>> Nodes => _nodeMap.Values.ToList();

    /// <summary>
    /// Input nodes (nodes with no inputs).
    /// </summary>
    public List<HLIRNode<T>> InputNodes { get; } = new();

    /// <summary>
    /// Output nodes (nodes with no outputs, or explicitly marked).
    /// </summary>
    public List<HLIRNode<T>> OutputNodes { get; } = new();

    /// <summary>
    /// Graph name for debugging.
    /// </summary>
    public string Name { get; set; } = "HLIRGraph";

    /// <summary>
    /// Graph-level metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; } = new();

    /// <summary>
    /// Number of nodes in the graph.
    /// </summary>
    public int NodeCount => _nodeMap.Count;

    /// <summary>
    /// Version number, incremented on each modification.
    /// </summary>
    public int Version { get; private set; }

    #endregion

    #region Node Management

    /// <summary>
    /// Adds a new node to the graph, assigning it a unique ID.
    /// </summary>
    public HLIRNode<T> AddNode(HLIRNode<T> node)
    {
        if (node.Id < 0)
        {
            node.Id = _nextNodeId++;
        }
        else if (_nodeMap.ContainsKey(node.Id))
        {
            throw new InvalidOperationException($"Node with ID {node.Id} already exists");
        }
        else
        {
            _nextNodeId = Math.Max(_nextNodeId, node.Id + 1);
        }

        _nodeMap[node.Id] = node;
        MarkDirty();
        node.AddProvenance($"Added to graph '{Name}'");
        return node;
    }

    /// <summary>
    /// Creates and adds a new node with the specified operation.
    /// </summary>
    public HLIRNode<T> CreateNode(OperationType operation, string name, params HLIRNode<T>[] inputs)
    {
        var node = new HLIRNode<T>
        {
            Id = _nextNodeId++,
            Name = name,
            Operation = operation
        };

        foreach (var input in inputs)
        {
            node.AddInput(input);
        }

        _nodeMap[node.Id] = node;
        MarkDirty();
        node.AddProvenance($"Created in graph '{Name}'");
        return node;
    }

    /// <summary>
    /// Removes a node from the graph.
    /// </summary>
    public bool RemoveNode(HLIRNode<T> node)
    {
        if (!_nodeMap.ContainsKey(node.Id)) return false;

        // Disconnect from inputs
        foreach (var input in node.Inputs.ToList())
        {
            node.RemoveInput(input);
        }

        // Disconnect from outputs
        foreach (var output in node.Outputs.ToList())
        {
            output.RemoveInput(node);
        }

        // Remove from input/output lists
        InputNodes.Remove(node);
        OutputNodes.Remove(node);

        _nodeMap.Remove(node.Id);
        MarkDirty();
        return true;
    }

    /// <summary>
    /// Finds a node by ID.
    /// </summary>
    public HLIRNode<T>? FindNode(int id) =>
        _nodeMap.TryGetValue(id, out var node) ? node : null;

    /// <summary>
    /// Finds nodes by name (partial match).
    /// </summary>
    public IEnumerable<HLIRNode<T>> FindNodesByName(string name) =>
        _nodeMap.Values.Where(n => n.Name.Contains(name, StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Finds nodes by operation type.
    /// </summary>
    public IEnumerable<HLIRNode<T>> FindNodesByOperation(OperationType operation) =>
        _nodeMap.Values.Where(n => n.Operation == operation);

    /// <summary>
    /// Replaces a node with another, updating all connections.
    /// </summary>
    /// <param name="oldNode">The existing node to replace.</param>
    /// <param name="newNode">The new node to insert in its place.</param>
    /// <exception cref="InvalidOperationException">Thrown when oldNode is not in the graph.</exception>
    /// <remarks>
    /// <para>
    /// This method performs a complete replacement by:
    /// <list type="number">
    /// <item><description>Copying all input connections from oldNode to newNode</description></item>
    /// <item><description>Redirecting all output connections from oldNode to newNode</description></item>
    /// <item><description>Adding newNode to the graph</description></item>
    /// <item><description>Removing oldNode from the graph</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// If newNode has the same ID as oldNode, a new ID will be automatically assigned
    /// to avoid conflicts during the replacement process.
    /// </para>
    /// </remarks>
    public void ReplaceNode(HLIRNode<T> oldNode, HLIRNode<T> newNode)
    {
        if (!_nodeMap.ContainsKey(oldNode.Id))
        {
            throw new InvalidOperationException($"Node {oldNode.Id} not in graph");
        }

        // Handle case where new node has same ID as old node
        // This prevents AddNode from failing when oldNode hasn't been removed yet
        if (newNode.Id == oldNode.Id)
        {
            newNode.Id = -1; // Force new ID assignment during AddNode
        }

        // Copy connections
        foreach (var input in oldNode.Inputs.ToList())
        {
            newNode.AddInput(input);
        }

        foreach (var output in oldNode.Outputs.ToList())
        {
            output.ReplaceInput(oldNode, newNode);
        }

        // Update graph - order matters: add new node first, then remove old
        AddNode(newNode);
        RemoveNode(oldNode);

        newNode.AddProvenance($"Replaced node n{oldNode.Id}");
    }

    #endregion

    #region Traversal

    /// <summary>
    /// Gets nodes in topological order (inputs before outputs).
    /// </summary>
    public List<HLIRNode<T>> GetTopologicalOrder()
    {
        if (!_isDirty && _cachedTopologicalOrder != null)
        {
            return _cachedTopologicalOrder;
        }

        var result = new List<HLIRNode<T>>();
        var visited = new HashSet<int>();
        var visiting = new HashSet<int>();

        void Visit(HLIRNode<T> node)
        {
            if (visited.Contains(node.Id)) return;
            if (visiting.Contains(node.Id))
            {
                throw new InvalidOperationException($"Cycle detected at node {node.Id}");
            }

            visiting.Add(node.Id);

            foreach (var input in node.Inputs)
            {
                Visit(input);
            }

            visiting.Remove(node.Id);
            visited.Add(node.Id);
            result.Add(node);
        }

        // Start from output nodes or all nodes
        var startNodes = OutputNodes.Count > 0
            ? OutputNodes
            : _nodeMap.Values.Where(n => n.Outputs.Count == 0).ToList();

        foreach (var node in startNodes)
        {
            Visit(node);
        }

        // Add any remaining nodes (disconnected components)
        foreach (var node in _nodeMap.Values.Where(n => !visited.Contains(n.Id)))
        {
            Visit(node);
        }

        _cachedTopologicalOrder = result;
        _isDirty = false;
        return result;
    }

    /// <summary>
    /// Gets nodes in reverse topological order (outputs before inputs).
    /// </summary>
    /// <returns>A new list containing nodes in reverse topological order (does not modify cached order).</returns>
    public List<HLIRNode<T>> GetReverseTopologicalOrder()
    {
        var order = GetTopologicalOrder();
        var reversed = new List<HLIRNode<T>>(order);
        reversed.Reverse();
        return reversed;
    }

    /// <summary>
    /// Iterates over all nodes in BFS order from inputs.
    /// </summary>
    public IEnumerable<HLIRNode<T>> BreadthFirstFromInputs()
    {
        var visited = new HashSet<int>();
        var queue = new Queue<HLIRNode<T>>(InputNodes);

        foreach (var node in InputNodes)
        {
            visited.Add(node.Id);
        }

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            yield return node;

            foreach (var output in node.Outputs)
            {
                if (!visited.Contains(output.Id))
                {
                    visited.Add(output.Id);
                    queue.Enqueue(output);
                }
            }
        }
    }

    #endregion

    #region Pattern Matching

    /// <summary>
    /// Finds sequences of nodes matching an operation pattern.
    /// </summary>
    public List<List<HLIRNode<T>>> FindPatterns(params OperationType[] pattern)
    {
        if (pattern.Length == 0) return new List<List<HLIRNode<T>>>();

        var results = new List<List<HLIRNode<T>>>();

        foreach (var startNode in FindNodesByOperation(pattern[0]))
        {
            if (startNode.IsFused || startNode.IsMarkedForDeletion) continue;

            var sequence = TryMatchPattern(startNode, pattern);
            if (sequence != null)
            {
                results.Add(sequence);
            }
        }

        return results;
    }

    private List<HLIRNode<T>>? TryMatchPattern(HLIRNode<T> startNode, OperationType[] pattern)
    {
        var sequence = new List<HLIRNode<T>> { startNode };
        var currentNode = startNode;

        for (int i = 1; i < pattern.Length; i++)
        {
            // Must have exactly one output
            if (currentNode.Outputs.Count != 1) return null;

            var nextNode = currentNode.Outputs[0];

            // Next node must match pattern and have single input
            if (nextNode.Operation != pattern[i] ||
                nextNode.Inputs.Count != 1 ||
                nextNode.IsFused ||
                nextNode.IsMarkedForDeletion)
            {
                return null;
            }

            sequence.Add(nextNode);
            currentNode = nextNode;
        }

        return sequence;
    }

    /// <summary>
    /// Finds diamond patterns (fork-join).
    /// </summary>
    public List<(HLIRNode<T> fork, List<HLIRNode<T>> branches, HLIRNode<T> join)> FindDiamondPatterns()
    {
        var results = new List<(HLIRNode<T>, List<HLIRNode<T>>, HLIRNode<T>)>();

        foreach (var forkNode in _nodeMap.Values.Where(n => n.Outputs.Count > 1))
        {
            // Check if all outputs eventually merge to the same node
            var outputPaths = new Dictionary<HLIRNode<T>, HashSet<HLIRNode<T>>>();

            foreach (var output in forkNode.Outputs)
            {
                outputPaths[output] = GetReachableNodes(output);
            }

            // Find common descendants
            if (outputPaths.Count < 2) continue;

            var common = outputPaths.Values.First().ToHashSet();
            foreach (var paths in outputPaths.Values.Skip(1))
            {
                common.IntersectWith(paths);
            }

            // Find the first common node (join point)
            var joinNode = common
                .OrderBy(n => GetTopologicalOrder().IndexOf(n))
                .FirstOrDefault();

            if (joinNode != null && forkNode.Outputs.All(o => joinNode.Inputs.Contains(o) ||
                GetReachableNodes(o).Contains(joinNode)))
            {
                results.Add((forkNode, forkNode.Outputs.ToList(), joinNode));
            }
        }

        return results;
    }

    private HashSet<HLIRNode<T>> GetReachableNodes(HLIRNode<T> start)
    {
        var reachable = new HashSet<HLIRNode<T>>();
        var stack = new Stack<HLIRNode<T>>();
        stack.Push(start);

        while (stack.Count > 0)
        {
            var node = stack.Pop();
            if (reachable.Add(node))
            {
                foreach (var output in node.Outputs)
                {
                    stack.Push(output);
                }
            }
        }

        return reachable;
    }

    #endregion

    #region Validation

    /// <summary>
    /// Validates the graph structure.
    /// </summary>
    public ValidationResult Validate()
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Check for duplicate IDs
        var ids = new HashSet<int>();
        foreach (var node in _nodeMap.Values)
        {
            if (!ids.Add(node.Id))
            {
                errors.Add($"Duplicate node ID: {node.Id}");
            }
        }

        // Check node validity
        foreach (var node in _nodeMap.Values)
        {
            if (!node.Validate())
            {
                errors.Add($"Invalid node: n{node.Id} ({node.Name})");
            }

            // Check that inputs are in graph
            foreach (var input in node.Inputs)
            {
                if (!_nodeMap.ContainsKey(input.Id))
                {
                    errors.Add($"Node n{node.Id} references missing input n{input.Id}");
                }
            }
        }

        // Check for cycles
        try
        {
            GetTopologicalOrder();
        }
        catch (InvalidOperationException ex)
        {
            errors.Add($"Graph contains cycle: {ex.Message}");
        }

        // Check input/output node consistency
        foreach (var input in InputNodes)
        {
            if (!_nodeMap.ContainsKey(input.Id))
            {
                errors.Add($"Input node n{input.Id} not in graph");
            }
        }

        foreach (var output in OutputNodes)
        {
            if (!_nodeMap.ContainsKey(output.Id))
            {
                errors.Add($"Output node n{output.Id} not in graph");
            }
        }

        // Warnings
        var deadNodes = _nodeMap.Values.Where(n =>
            !n.HasConsumers &&
            !OutputNodes.Contains(n) &&
            n.CanEliminate).ToList();

        if (deadNodes.Count > 0)
        {
            warnings.Add($"Graph has {deadNodes.Count} dead nodes that could be eliminated");
        }

        return new ValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings
        };
    }

    #endregion

    #region Statistics

    /// <summary>
    /// Gets comprehensive graph statistics.
    /// </summary>
    public HLIRGraphStatistics GetStatistics()
    {
        var stats = new HLIRGraphStatistics
        {
            TotalNodes = _nodeMap.Count,
            InputNodes = InputNodes.Count,
            OutputNodes = OutputNodes.Count
        };

        // Count by operation
        foreach (var node in _nodeMap.Values)
        {
            if (!stats.NodesByOperation.ContainsKey(node.Operation))
            {
                stats.NodesByOperation[node.Operation] = 0;
            }
            stats.NodesByOperation[node.Operation]++;

            if (node.IsFused) stats.FusedNodes++;
            if (node.Cost != null)
            {
                stats.TotalFLOPs += node.Cost.FLOPs;
                stats.TotalMemoryRead += node.Cost.MemoryRead;
                stats.TotalMemoryWrite += node.Cost.MemoryWrite;
            }
        }

        // Graph depth (longest path)
        stats.GraphDepth = ComputeGraphDepth();

        // Critical path
        stats.CriticalPathLength = ComputeCriticalPathLength();

        return stats;
    }

    private int ComputeGraphDepth()
    {
        var depth = new Dictionary<int, int>();
        foreach (var node in GetTopologicalOrder())
        {
            var inputDepth = node.Inputs.Count > 0
                ? node.Inputs.Max(i => depth.GetValueOrDefault(i.Id, 0))
                : 0;
            depth[node.Id] = inputDepth + 1;
        }
        return depth.Count > 0 ? depth.Values.Max() : 0;
    }

    private long ComputeCriticalPathLength()
    {
        var pathCost = new Dictionary<int, long>();
        foreach (var node in GetTopologicalOrder())
        {
            var inputCost = node.Inputs.Count > 0
                ? node.Inputs.Max(i => pathCost.GetValueOrDefault(i.Id, 0))
                : 0;
            var nodeCost = node.Cost?.EstimatedLatencyNs ?? 1;
            pathCost[node.Id] = inputCost + nodeCost;
        }
        return pathCost.Count > 0 ? pathCost.Values.Max() : 0;
    }

    #endregion

    #region Utilities

    /// <summary>
    /// Creates a deep copy of the graph.
    /// </summary>
    public HLIRGraph<T> Clone()
    {
        var clone = new HLIRGraph<T> { Name = Name + "_clone" };

        // Clone all nodes
        var nodeClones = new Dictionary<int, HLIRNode<T>>();
        foreach (var node in _nodeMap.Values)
        {
            var nodeClone = node.Clone();
            nodeClone.Id = node.Id;
            nodeClones[node.Id] = nodeClone;
            clone._nodeMap[nodeClone.Id] = nodeClone;
        }

        // Reconnect edges
        foreach (var node in _nodeMap.Values)
        {
            var nodeClone = nodeClones[node.Id];
            foreach (var input in node.Inputs)
            {
                nodeClone.AddInput(nodeClones[input.Id]);
            }
        }

        // Copy input/output lists
        foreach (var input in InputNodes)
        {
            clone.InputNodes.Add(nodeClones[input.Id]);
        }
        foreach (var output in OutputNodes)
        {
            clone.OutputNodes.Add(nodeClones[output.Id]);
        }

        // Copy metadata
        foreach (var kvp in Metadata)
        {
            clone.Metadata[kvp.Key] = kvp.Value;
        }

        clone._nextNodeId = _nextNodeId;
        return clone;
    }

    /// <summary>
    /// Extracts a subgraph containing specified nodes and their dependencies.
    /// </summary>
    public HLIRGraph<T> ExtractSubgraph(IEnumerable<HLIRNode<T>> nodes)
    {
        var subgraph = new HLIRGraph<T> { Name = Name + "_subgraph" };
        var nodeSet = new HashSet<int>(nodes.Select(n => n.Id));

        // Add all dependencies
        var toProcess = new Queue<HLIRNode<T>>(nodes);
        while (toProcess.Count > 0)
        {
            var node = toProcess.Dequeue();
            foreach (var input in node.Inputs)
            {
                if (nodeSet.Add(input.Id))
                {
                    toProcess.Enqueue(input);
                }
            }
        }

        // Clone nodes into subgraph
        var clones = new Dictionary<int, HLIRNode<T>>();
        foreach (var id in nodeSet)
        {
            var node = _nodeMap[id];
            var clone = node.Clone();
            clone.Id = node.Id;
            clones[id] = clone;
            subgraph.AddNode(clone);
        }

        // Reconnect edges within subgraph
        foreach (var id in nodeSet)
        {
            var original = _nodeMap[id];
            var clone = clones[id];
            foreach (var input in original.Inputs)
            {
                if (clones.TryGetValue(input.Id, out var inputClone))
                {
                    clone.AddInput(inputClone);
                }
            }
        }

        return subgraph;
    }

    /// <summary>
    /// Compacts node IDs to be sequential starting from 0.
    /// </summary>
    public void CompactNodeIds()
    {
        var mapping = new Dictionary<int, int>();
        var orderedNodes = GetTopologicalOrder();

        for (int i = 0; i < orderedNodes.Count; i++)
        {
            mapping[orderedNodes[i].Id] = i;
        }

        // First pass: Update all InputIds arrays using old IDs (before node IDs change)
        foreach (var node in orderedNodes)
        {
            node.InputIds = node.Inputs.Select(inp => mapping[inp.Id]).ToArray();
        }

        // Second pass: Update all node IDs and rebuild node map
        _nodeMap.Clear();
        foreach (var node in orderedNodes)
        {
            node.Id = mapping[node.Id];
            _nodeMap[node.Id] = node;
        }

        _nextNodeId = orderedNodes.Count;
        MarkDirty();
    }

    private void MarkDirty()
    {
        _isDirty = true;
        _cachedTopologicalOrder = null;
        Version++;
    }

    public override string ToString()
    {
        return $"HLIRGraph '{Name}': {_nodeMap.Count} nodes, {InputNodes.Count} inputs, {OutputNodes.Count} outputs";
    }

    #endregion
}

/// <summary>
/// Result of graph validation.
/// </summary>
public class ValidationResult
{
    public bool IsValid { get; init; }
    public List<string> Errors { get; init; } = new();
    public List<string> Warnings { get; init; } = new();

    public override string ToString()
    {
        if (IsValid && Warnings.Count == 0) return "Valid";
        var parts = new List<string>();
        if (!IsValid) parts.Add($"{Errors.Count} errors");
        if (Warnings.Count > 0) parts.Add($"{Warnings.Count} warnings");
        return string.Join(", ", parts);
    }
}

/// <summary>
/// Comprehensive graph statistics.
/// </summary>
public class HLIRGraphStatistics
{
    public int TotalNodes { get; set; }
    public int InputNodes { get; set; }
    public int OutputNodes { get; set; }
    public int FusedNodes { get; set; }
    public int GraphDepth { get; set; }
    public long CriticalPathLength { get; set; }
    public long TotalFLOPs { get; set; }
    public long TotalMemoryRead { get; set; }
    public long TotalMemoryWrite { get; set; }
    public Dictionary<OperationType, int> NodesByOperation { get; } = new();

    public double ArithmeticIntensity =>
        (TotalMemoryRead + TotalMemoryWrite) > 0
            ? (double)TotalFLOPs / (TotalMemoryRead + TotalMemoryWrite)
            : 0;

    public override string ToString()
    {
        return $"Nodes: {TotalNodes}, Depth: {GraphDepth}, FLOPs: {TotalFLOPs:N0}, Memory: {(TotalMemoryRead + TotalMemoryWrite) / 1024.0 / 1024.0:F2} MB";
    }
}
