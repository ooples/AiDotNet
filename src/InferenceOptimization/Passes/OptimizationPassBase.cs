using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Base class for optimization passes.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public abstract class OptimizationPassBase<T> : IOptimizationPass<T> where T : struct
{
    public abstract OptimizationPassType PassType { get; }
    public abstract string Name { get; }

    public abstract bool Apply(IComputationGraph<T> graph);

    public virtual bool CanApply(IComputationGraph<T> graph)
    {
        return graph != null && graph.Nodes.Count > 0;
    }

    /// <summary>
    /// Helper method to find fusion candidates in the graph.
    /// </summary>
    protected List<List<ComputationNode<T>>> FindFusionCandidates(
        IComputationGraph<T> graph,
        params OperationType[] pattern)
    {
        var candidates = new List<List<ComputationNode<T>>>();

        foreach (var node in graph.Nodes.Where(n => n.OperationType == pattern[0] && !n.IsFused))
        {
            var sequence = TryMatchPattern(node, pattern);
            if (sequence != null)
            {
                candidates.Add(sequence);
            }
        }

        return candidates;
    }

    /// <summary>
    /// Tries to match a pattern starting from a given node.
    /// </summary>
    protected List<ComputationNode<T>>? TryMatchPattern(
        ComputationNode<T> startNode,
        OperationType[] pattern)
    {
        var sequence = new List<ComputationNode<T>> { startNode };
        var currentNode = startNode;

        for (int i = 1; i < pattern.Length; i++)
        {
            // Check if current node has exactly one output
            if (currentNode.Outputs.Count != 1)
            {
                return null;
            }

            var nextNode = currentNode.Outputs[0];

            // Check if next node matches the pattern and has exactly one input
            if (nextNode.OperationType != pattern[i] || nextNode.Inputs.Count != 1)
            {
                return null;
            }

            // Check if next node is not already fused
            if (nextNode.IsFused)
            {
                return null;
            }

            sequence.Add(nextNode);
            currentNode = nextNode;
        }

        return sequence;
    }

    /// <summary>
    /// Replaces a sequence of nodes with a fused node.
    /// </summary>
    protected ComputationNode<T> FuseNodes(
        IComputationGraph<T> graph,
        List<ComputationNode<T>> nodesToFuse,
        OperationType fusedOperationType)
    {
        if (nodesToFuse.Count == 0)
        {
            throw new ArgumentException("No nodes to fuse");
        }

        var firstNode = nodesToFuse[0];
        var lastNode = nodesToFuse[nodesToFuse.Count - 1];

        // Create fused node
        var fusedNode = new ComputationNode<T>
        {
            OperationType = fusedOperationType,
            Name = $"{firstNode.Name}_fused",
            OutputShape = lastNode.OutputShape,
            IsFused = true,
            FusedFrom = new List<ComputationNode<T>>(nodesToFuse)
        };

        // Copy parameters from all nodes
        foreach (var node in nodesToFuse)
        {
            foreach (var param in node.Parameters)
            {
                fusedNode.Parameters[$"{node.Name}_{param.Key}"] = param.Value;
            }

            foreach (var meta in node.Metadata)
            {
                fusedNode.Metadata[$"{node.Name}_{meta.Key}"] = meta.Value;
            }
        }

        // Connect inputs from first node
        foreach (var input in firstNode.Inputs)
        {
            fusedNode.AddInput(input);
            input.Outputs.Remove(firstNode);
        }

        // Connect outputs from last node
        foreach (var output in lastNode.Outputs)
        {
            output.ReplaceInput(lastNode, fusedNode);
        }

        // Add fused node to graph
        graph.AddNode(fusedNode);

        // Remove original nodes
        foreach (var node in nodesToFuse)
        {
            graph.RemoveNode(node);
        }

        return fusedNode;
    }
}
