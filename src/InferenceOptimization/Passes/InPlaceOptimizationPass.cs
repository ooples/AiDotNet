using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Marks operations that can be performed in-place to reduce memory allocation.
/// Operations like ReLU, Dropout, and some elementwise operations can modify
/// their input tensors directly instead of allocating new memory.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class InPlaceOptimizationPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.InPlaceOptimization;
    public override string Name => "In-Place Operation Optimization";

    private static readonly HashSet<OperationType> InPlaceCandidates = new()
    {
        OperationType.ReLU,
        OperationType.LeakyReLU,
        OperationType.ELU,
        OperationType.SELU,
        OperationType.Sigmoid,
        OperationType.Tanh,
        OperationType.Dropout,
        OperationType.Add,      // Can be in-place if one input won't be used again
        OperationType.Multiply, // Same as Add
        OperationType.Clip
    };

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        foreach (var node in graph.Nodes)
        {
            if (InPlaceCandidates.Contains(node.OperationType) && CanBeInPlace(node))
            {
                node.CanOperateInPlace = true;
                node.Metadata["InPlaceOptimized"] = true;
                modified = true;
            }
        }

        return modified;
    }

    private bool CanBeInPlace(ComputationNode<T> node)
    {
        // Check if this operation can safely be performed in-place
        // An operation can be in-place if:
        // 1. It has exactly one input (for unary operations)
        // 2. OR for binary operations, one of the inputs is not used elsewhere

        if (IsUnaryOperation(node.OperationType))
        {
            // For unary operations, check if input has only one consumer (this node)
            if (node.Inputs.Count == 1)
            {
                var input = node.Inputs[0];
                return input.Outputs.Count == 1; // Only this node uses the input
            }
        }
        else if (IsBinaryOperation(node.OperationType))
        {
            // For binary operations, at least one input should have only this node as consumer
            return node.Inputs.Any(input => input.Outputs.Count == 1);
        }

        return false;
    }

    private bool IsUnaryOperation(OperationType opType)
    {
        return opType == OperationType.ReLU ||
               opType == OperationType.LeakyReLU ||
               opType == OperationType.ELU ||
               opType == OperationType.SELU ||
               opType == OperationType.Sigmoid ||
               opType == OperationType.Tanh ||
               opType == OperationType.Dropout ||
               opType == OperationType.Clip;
    }

    private bool IsBinaryOperation(OperationType opType)
    {
        return opType == OperationType.Add ||
               opType == OperationType.Multiply ||
               opType == OperationType.Subtract ||
               opType == OperationType.Divide;
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => InPlaceCandidates.Contains(n.OperationType));
    }
}
