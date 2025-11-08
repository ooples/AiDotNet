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

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Build a signature for each node based on its operation and inputs
        var signatureToNode = new Dictionary<string, ComputationNode<T>>();

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

    private string ComputeSignature(ComputationNode<T> node)
    {
        // Create a signature based on:
        // 1. Operation type
        // 2. Input node IDs (sorted for determinism)
        // 3. Key parameters

        var parts = new List<string>
        {
            node.OperationType.ToString()
        };

        // Add sorted input IDs
        var inputIds = node.Inputs.Select(n => n.Id).OrderBy(id => id);
        parts.AddRange(inputIds);

        // Add key parameters (if any)
        foreach (var param in node.Parameters.OrderBy(kv => kv.Key))
        {
            parts.Add($"{param.Key}={param.Value}");
        }

        // Add key metadata (if any)
        foreach (var meta in node.Metadata.OrderBy(kv => kv.Key))
        {
            // Only include metadata that affects computation
            if (IsComputationalMetadata(meta.Key))
            {
                parts.Add($"{meta.Key}={meta.Value}");
            }
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

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) && graph.Nodes.Count > 1;
    }
}
