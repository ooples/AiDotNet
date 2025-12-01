using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Eliminates nodes that don't contribute to the output (dead code).
/// This includes nodes with no consumers and nodes that are not reachable from outputs.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class DeadCodeEliminationPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.DeadCodeElimination;
    public override string Name => "Dead Code Elimination";

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Mark all nodes reachable from outputs
        var reachable = MarkReachableNodes(graph);

        // Remove unreachable nodes
        var nodesToRemove = graph.Nodes
            .Where(n => !reachable.Contains(n) &&
                       n.OperationType != OperationType.Input &&
                       n.OperationType != OperationType.Output &&
                       n.CanEliminate)
            .ToList();

        foreach (var node in nodesToRemove)
        {
            graph.RemoveNode(node);
            modified = true;
        }

        // Also remove nodes with no consumers (unless they're outputs)
        var noConsumerNodes = graph.Nodes
            .Where(n => n.Outputs.Count == 0 &&
                       n.OperationType != OperationType.Output &&
                       n.CanEliminate)
            .ToList();

        foreach (var node in noConsumerNodes)
        {
            graph.RemoveNode(node);
            modified = true;
        }

        return modified;
    }

    private HashSet<ComputationNode<T>> MarkReachableNodes(IComputationGraph<T> graph)
    {
        var reachable = new HashSet<ComputationNode<T>>();
        var queue = new Queue<ComputationNode<T>>();

        // Start from output nodes and work backwards
        foreach (var output in graph.OutputNodes)
        {
            queue.Enqueue(output);
        }

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();

            if (reachable.Contains(node))
            {
                continue;
            }

            reachable.Add(node);

            // Add all inputs to the queue
            foreach (var input in node.Inputs)
            {
                if (!reachable.Contains(input))
                {
                    queue.Enqueue(input);
                }
            }
        }

        return reachable;
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) && graph.OutputNodes.Count > 0;
    }
}
