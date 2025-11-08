using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Fuses multi-head attention components into a single optimized operation.
/// Multi-head attention consists of multiple Q, K, V projections, attention computation,
/// and output projection. Fusing these provides significant speedup in transformers.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class MultiHeadAttentionFusionPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.AttentionFusion;
    public override string Name => "Multi-Head Attention Fusion";

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Find multi-head attention layers
        var attentionNodes = graph.Nodes.Where(n =>
            n.OperationType == OperationType.MultiHeadAttention && !n.IsFused).ToList();

        foreach (var attentionNode in attentionNodes)
        {
            // Check if this attention node is followed by operations that can be fused
            if (CanFuseAttention(attentionNode))
            {
                FuseAttention(graph, attentionNode);
                modified = true;
            }
        }

        return modified;
    }

    private bool CanFuseAttention(ComputationNode<T> attentionNode)
    {
        // Check if the attention pattern is suitable for fusion
        // In a real implementation, we'd check for:
        // 1. Q, K, V projection matrices
        // 2. Attention computation (softmax, dropout)
        // 3. Output projection
        return attentionNode.Outputs.Count > 0;
    }

    private void FuseAttention(IComputationGraph<T> graph, ComputationNode<T> attentionNode)
    {
        // Create a fused multi-head attention node
        var fusedNode = new ComputationNode<T>
        {
            OperationType = OperationType.FusedMultiHeadAttention,
            Name = $"{attentionNode.Name}_fused",
            OutputShape = attentionNode.OutputShape,
            IsFused = true,
            FusedFrom = new List<ComputationNode<T>> { attentionNode }
        };

        // Copy all parameters
        foreach (var param in attentionNode.Parameters)
        {
            fusedNode.Parameters[param.Key] = param.Value;
        }

        // Set metadata for optimized attention computation
        fusedNode.Metadata["UseFlashAttention"] = true; // Enable flash attention if available
        fusedNode.Metadata["ScaledDotProduct"] = true;

        // Connect inputs
        foreach (var input in attentionNode.Inputs)
        {
            fusedNode.AddInput(input);
            input.Outputs.Remove(attentionNode);
        }

        // Connect outputs
        foreach (var output in attentionNode.Outputs)
        {
            output.ReplaceInput(attentionNode, fusedNode);
        }

        // Replace in graph
        graph.AddNode(fusedNode);
        graph.RemoveNode(attentionNode);
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.MultiHeadAttention ||
                                   n.OperationType == OperationType.Attention);
    }
}
