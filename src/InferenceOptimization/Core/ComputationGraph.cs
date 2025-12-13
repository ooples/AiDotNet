using AiDotNet.Enums;

namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Represents a computation graph for neural network inference.
/// The graph consists of nodes (operations) and edges (data dependencies).
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class ComputationGraph<T> : IComputationGraph<T> where T : struct
{
    public List<ComputationNode<T>> Nodes { get; private set; }
    public List<ComputationNode<T>> InputNodes { get; private set; }
    public List<ComputationNode<T>> OutputNodes { get; private set; }

    private readonly Dictionary<string, ComputationNode<T>> _nodeIndex;

    public ComputationGraph()
    {
        Nodes = new List<ComputationNode<T>>();
        InputNodes = new List<ComputationNode<T>>();
        OutputNodes = new List<ComputationNode<T>>();
        _nodeIndex = new Dictionary<string, ComputationNode<T>>();
    }

    public void AddNode(ComputationNode<T> node)
    {
        if (node == null)
        {
            throw new ArgumentNullException(nameof(node));
        }

        if (!_nodeIndex.ContainsKey(node.Id))
        {
            Nodes.Add(node);
            _nodeIndex[node.Id] = node;

            // Track input/output nodes
            if (node.OperationType == OperationType.Input)
            {
                InputNodes.Add(node);
            }
            else if (node.OperationType == OperationType.Output)
            {
                OutputNodes.Add(node);
            }
        }
    }

    public void RemoveNode(ComputationNode<T> node)
    {
        if (node == null) return;

        // Remove connections
        foreach (var input in node.Inputs.ToList())
        {
            input.Outputs.Remove(node);
        }

        foreach (var output in node.Outputs.ToList())
        {
            output.Inputs.Remove(node);
        }

        // Remove from collections
        Nodes.Remove(node);
        InputNodes.Remove(node);
        OutputNodes.Remove(node);
        _nodeIndex.Remove(node.Id);
    }

    public ComputationNode<T>? FindNodeById(string id)
    {
        return _nodeIndex.TryGetValue(id, out var node) ? node : null;
    }

    public List<ComputationNode<T>> FindNodesByName(string name)
    {
        return Nodes.Where(n => n.Name == name).ToList();
    }

    public List<ComputationNode<T>> GetTopologicalOrder()
    {
        var visited = new HashSet<ComputationNode<T>>();
        var result = new List<ComputationNode<T>>();
        var inStack = new HashSet<ComputationNode<T>>();

        foreach (var node in Nodes.Where(node => !visited.Contains(node)))
        {
            if (!TopologicalSortUtil(node, visited, inStack, result))
            {
                throw new InvalidOperationException("Graph contains a cycle");
            }
        }

        return result;
    }

    private bool TopologicalSortUtil(
        ComputationNode<T> node,
        HashSet<ComputationNode<T>> visited,
        HashSet<ComputationNode<T>> inStack,
        List<ComputationNode<T>> result)
    {
        if (inStack.Contains(node))
        {
            return false; // Cycle detected
        }

        if (visited.Contains(node))
        {
            return true; // Already processed
        }

        visited.Add(node);
        inStack.Add(node);

        foreach (var input in node.Inputs)
        {
            if (!TopologicalSortUtil(input, visited, inStack, result))
            {
                return false;
            }
        }

        inStack.Remove(node);
        result.Add(node);

        return true;
    }

    public bool Validate()
    {
        try
        {
            // Check for cycles (GetTopologicalOrder throws if there's a cycle)
            GetTopologicalOrder();

            // Check that all nodes are reachable from inputs
            var reachable = new HashSet<ComputationNode<T>>();
            var queue = new Queue<ComputationNode<T>>(InputNodes);

            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                if (reachable.Contains(node)) continue;

                reachable.Add(node);

                foreach (var output in node.Outputs)
                {
                    queue.Enqueue(output);
                }
            }

            // All nodes should be reachable from inputs
            var unreachable = Nodes.Where(n => !reachable.Contains(n) && n.OperationType != OperationType.Constant).ToList();
            if (unreachable.Any())
            {
                return false;
            }

            return true;
        }
        catch (InvalidOperationException)
        {
            // Graph contains a cycle or other structural issue
            return false;
        }
    }

    public IComputationGraph<T> Clone()
    {
        var clonedGraph = new ComputationGraph<T>();
        var nodeMapping = new Dictionary<string, ComputationNode<T>>();

        // Clone all nodes
        foreach (var node in Nodes)
        {
            var clonedNode = node.Clone();
            clonedGraph.AddNode(clonedNode);
            nodeMapping[node.Id] = clonedNode;
        }

        // Rebuild connections
        foreach (var node in Nodes)
        {
            var clonedNode = nodeMapping[node.Id];

            foreach (var input in node.Inputs)
            {
                var clonedInput = nodeMapping[input.Id];
                clonedNode.AddInput(clonedInput);
            }
        }

        return clonedGraph;
    }

    /// <summary>
    /// Gets statistics about the graph.
    /// </summary>
    public GraphStatistics GetStatistics()
    {
        var stats = new GraphStatistics
        {
            TotalNodes = Nodes.Count,
            InputNodes = InputNodes.Count,
            OutputNodes = OutputNodes.Count,
            FusedNodes = Nodes.Count(n => n.IsFused),
            OperationTypeCounts = new Dictionary<OperationType, int>()
        };

        foreach (var node in Nodes)
        {
            if (!stats.OperationTypeCounts.ContainsKey(node.OperationType))
            {
                stats.OperationTypeCounts[node.OperationType] = 0;
            }
            stats.OperationTypeCounts[node.OperationType]++;
        }

        return stats;
    }

    public override string ToString()
    {
        return $"ComputationGraph: {Nodes.Count} nodes, {InputNodes.Count} inputs, {OutputNodes.Count} outputs";
    }
}

/// <summary>
/// Statistics about a computation graph.
/// </summary>
public class GraphStatistics
{
    public int TotalNodes { get; set; }
    public int InputNodes { get; set; }
    public int OutputNodes { get; set; }
    public int FusedNodes { get; set; }
    public Dictionary<OperationType, int> OperationTypeCounts { get; set; } = new();

    public override string ToString()
    {
        var opsString = string.Join(", ", OperationTypeCounts.Select(kv => $"{kv.Key}: {kv.Value}"));
        return $"Total: {TotalNodes}, Inputs: {InputNodes}, Outputs: {OutputNodes}, Fused: {FusedNodes}\nOperations: {opsString}";
    }
}
