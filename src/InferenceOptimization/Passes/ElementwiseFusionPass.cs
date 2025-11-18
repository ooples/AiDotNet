using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Fuses consecutive elementwise operations into a single operation.
/// For example: (x + y) * z can be computed in a single fused kernel.
/// This reduces memory bandwidth by avoiding intermediate results.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class ElementwiseFusionPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.ElementwiseFusion;
    public override string Name => "Elementwise Operation Fusion";

    private static readonly HashSet<OperationType> ElementwiseOps = new()
    {
        OperationType.Add,
        OperationType.Subtract,
        OperationType.Multiply,
        OperationType.Divide,
        OperationType.Power,
        OperationType.Sqrt,
        OperationType.Exp,
        OperationType.Log,
        OperationType.ReLU,
        OperationType.Sigmoid,
        OperationType.Tanh
    };

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Find chains of elementwise operations
        foreach (var node in graph.Nodes.Where(n => ElementwiseOps.Contains(n.OperationType) && !n.IsFused).ToList())
        {
            var chain = FindElementwiseChain(node);

            // Only fuse if we have at least 2 operations
            if (chain.Count >= 2)
            {
                FuseElementwiseChain(graph, chain);
                modified = true;
            }
        }

        return modified;
    }

    private List<ComputationNode<T>> FindElementwiseChain(ComputationNode<T> startNode)
    {
        var chain = new List<ComputationNode<T>> { startNode };
        var current = startNode;

        // Follow the chain forward
        while (current.Outputs.Count == 1)
        {
            var next = current.Outputs[0];

            // Stop if:
            // 1. Not an elementwise op
            // 2. Already fused
            // 3. Has multiple inputs (more complex than simple chain)
            // 4. The input has multiple consumers (would break other paths)
            if (!ElementwiseOps.Contains(next.OperationType) ||
                next.IsFused ||
                current.Outputs.Count > 1)
            {
                break;
            }

            chain.Add(next);
            current = next;
        }

        return chain;
    }

    private void FuseElementwiseChain(IComputationGraph<T> graph, List<ComputationNode<T>> chain)
    {
        var firstNode = chain[0];
        var lastNode = chain[chain.Count - 1];

        // Create fused elementwise node
        var fusedNode = new ComputationNode<T>
        {
            OperationType = OperationType.Custom,
            Name = $"{firstNode.Name}_elementwise_fused",
            OutputShape = lastNode.OutputShape,
            IsFused = true,
            CanOperateInPlace = chain.All(n => n.CanOperateInPlace),
            FusedFrom = new List<ComputationNode<T>>(chain)
        };

        // Store the operation sequence for code generation
        fusedNode.Metadata["OperationSequence"] = chain.Select(n => n.OperationType).ToList();

        // Collect all unique inputs from the chain
        var allInputs = new HashSet<ComputationNode<T>>();
        foreach (var node in chain)
        {
            foreach (var input in node.Inputs)
            {
                if (!chain.Contains(input))
                {
                    allInputs.Add(input);
                }
            }
        }

        // Connect inputs
        foreach (var input in allInputs)
        {
            fusedNode.AddInput(input);

            // Remove connections to chain nodes
            foreach (var chainNode in chain)
            {
                input.Outputs.Remove(chainNode);
            }
        }

        // Connect outputs
        foreach (var output in lastNode.Outputs)
        {
            output.ReplaceInput(lastNode, fusedNode);
        }

        // Add fused node
        graph.AddNode(fusedNode);

        // Remove chain nodes
        foreach (var node in chain)
        {
            graph.RemoveNode(node);
        }
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => ElementwiseOps.Contains(n.OperationType));
    }
}
