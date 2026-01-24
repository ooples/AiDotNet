using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Eliminates common subexpressions by sharing computation.
/// If the same operation with the same inputs appears multiple times,
/// compute it once and reuse the result.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class CommonSubexpressionEliminationPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.CommonSubexpressionElimination;
    public override string Name => "Common Subexpression Elimination";

    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;

        // Build a signature for each node based on its operation and inputs
        var signatureToNode = new Dictionary<string, OptimizationNode<T>>();

        foreach (var node in graph.GetTopologicalOrder())
        {
            // Skip certain types that shouldn't be CSE'd
            if (node.OperationType == OperationType.Input ||
                node.OperationType == OperationType.Output ||
                node.OperationType == OperationType.Dropout || // Non-deterministic
                !node.CanEliminate)
            {
                continue;
            }

            var signature = ComputeSignature(node);

            if (signatureToNode.TryGetValue(signature, out var existingNode))
            {
                // Found a common subexpression!
                // Replace all uses of this node with the existing one
                foreach (var output in node.Outputs.ToList())
                {
                    output.ReplaceInput(node, existingNode);
                }

                graph.RemoveNode(node);
                modified = true;
            }
            else
            {
                signatureToNode[signature] = node;
            }
        }

        return modified;
    }

    private string ComputeSignature(OptimizationNode<T> node)
    {
        // Create a signature based on:
        // 1. Operation type
        // 2. Input node IDs (sorted only for commutative operations)
        // 3. Key parameters

        var parts = new List<string>
        {
            node.OperationType.ToString()
        };

        // For commutative operations (Add, Multiply), sort input IDs so a+b == b+a
        // For non-commutative operations (Subtract, Divide, MatMul, Power), preserve order
        var inputIds = IsCommutativeOperation(node.OperationType)
            ? node.Inputs.Select(n => n.Id).OrderBy(id => id)
            : node.Inputs.Select(n => n.Id);
        parts.AddRange(inputIds);

        // Add key parameters (if any)
        foreach (var param in node.Parameters.OrderBy(kv => kv.Key))
        {
            parts.Add($"{param.Key}={param.Value}");
        }

        // Add key metadata (if any)
        // Only include metadata that affects computation
        foreach (var meta in node.Metadata.Where(kv => IsComputationalMetadata(kv.Key)).OrderBy(kv => kv.Key))
        {
            parts.Add($"{meta.Key}={meta.Value}");
        }

        return string.Join("|", parts);
    }

    private bool IsComputationalMetadata(string key)
    {
        // These metadata keys affect the computation and should be part of the signature
        var computationalKeys = new HashSet<string>
        {
            "stride",
            "padding",
            "kernel_size",
            "dilation",
            "groups",
            "transpose",
            "alpha", // For LeakyReLU, etc.
            "beta"
        };

        return computationalKeys.Contains(key.ToLower());
    }

    /// <summary>
    /// Determines if an operation is commutative (operand order doesn't matter).
    /// For commutative operations like Add and Multiply, a+b == b+a, so input IDs can be sorted.
    /// For non-commutative operations like Subtract and Divide, a-b != b-a, so order must be preserved.
    /// </summary>
    private static bool IsCommutativeOperation(OperationType opType)
    {
        return opType switch
        {
            // Commutative operations (order doesn't matter)
            OperationType.Add => true,
            OperationType.Multiply => true,
            // Non-commutative operations (order matters)
            OperationType.Subtract => false,
            OperationType.Divide => false,
            OperationType.Power => false,
            OperationType.MatMul => false,
            OperationType.Convolution2D => false,
            OperationType.BatchNorm => false,
            // Default to non-commutative (safer - preserves operand order)
            _ => false
        };
    }

    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) && graph.Nodes.Count > 1;
    }
}
