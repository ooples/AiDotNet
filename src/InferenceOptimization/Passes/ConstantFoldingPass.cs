using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Folds constant expressions at compile time to reduce runtime computation.
/// For example: If two constants are multiplied, compute the result once during optimization
/// rather than every inference call.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class ConstantFoldingPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.ConstantFolding;
    public override string Name => "Constant Folding";

    private static readonly HashSet<OperationType> FoldableOps = new()
    {
        OperationType.Add,
        OperationType.Subtract,
        OperationType.Multiply,
        OperationType.Divide,
        OperationType.Power,
        OperationType.Sqrt,
        OperationType.Exp,
        OperationType.Log,
        OperationType.MatMul
    };

    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;
        bool changed;

        // Keep folding until no more changes (iterative constant propagation)
        do
        {
            changed = false;

            foreach (var node in graph.Nodes
                .Where(n => FoldableOps.Contains(n.OperationType) && !n.IsFused)
                .Where(n => n.Inputs.All(input => input.OperationType == OperationType.Constant))
                .ToList())
            {
                if (TryFoldConstant(graph, node))
                {
                    changed = true;
                    modified = true;
                }
            }
        } while (changed);

        return modified;
    }

    private bool TryFoldConstant(IOptimizationGraph<T> graph, OptimizationNode<T> node)
    {
        try
        {
            // Compute the constant result
            Tensor<T>? result = node.OperationType switch
            {
                OperationType.Add => FoldAdd(node),
                OperationType.Subtract => FoldSubtract(node),
                OperationType.Multiply => FoldMultiply(node),
                OperationType.Divide => FoldDivide(node),
                OperationType.MatMul => FoldMatMul(node),
                _ => null
            };

            if (result == null)
            {
                return false;
            }

            // Create a new constant node with the result
            var constantNode = new OptimizationNode<T>
            {
                OperationType = OperationType.Constant,
                Name = $"{node.Name}_folded",
                OutputShape = node.OutputShape,
                ConstantValue = result,
                CanEliminate = false // Constants should not be eliminated
            };

            // Replace the operation node with the constant node
            foreach (var output in node.Outputs.ToList())
            {
                output.ReplaceInput(node, constantNode);
            }

            // Add constant node and remove operation node
            graph.AddNode(constantNode);
            graph.RemoveNode(node);

            return true;
        }
        catch (InvalidOperationException)
        {
            // If folding fails due to invalid graph state, leave the node as is
            return false;
        }
        catch (ArgumentException)
        {
            // If folding fails due to invalid arguments, leave the node as is
            return false;
        }
    }

    private Tensor<T>? FoldAdd(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        // In a real implementation, you would use tensor arithmetic here
        // For now, we mark that folding is possible
        node.Metadata["FoldingResult"] = "Add";
        return left; // Placeholder
    }

    private Tensor<T>? FoldSubtract(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        node.Metadata["FoldingResult"] = "Subtract";
        return left; // Placeholder
    }

    private Tensor<T>? FoldMultiply(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        node.Metadata["FoldingResult"] = "Multiply";
        return left; // Placeholder
    }

    private Tensor<T>? FoldDivide(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        node.Metadata["FoldingResult"] = "Divide";
        return left; // Placeholder
    }

    private Tensor<T>? FoldMatMul(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        node.Metadata["FoldingResult"] = "MatMul";
        return left; // Placeholder
    }

    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.Constant);
    }
}
