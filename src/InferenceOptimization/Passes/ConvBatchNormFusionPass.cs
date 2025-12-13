using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Fuses Convolution + BatchNormalization into a single operation.
/// This is a critical optimization that eliminates the normalization overhead during inference.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class ConvBatchNormFusionPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.ConvBatchNormFusion;
    public override string Name => "Conv + BatchNorm Fusion";

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Find Conv -> BatchNorm patterns
        var candidates = FindFusionCandidates(
            graph,
            OperationType.Convolution,
            OperationType.BatchNormalization
        );

        // Also check for Convolution2D variant
        candidates.AddRange(FindFusionCandidates(
            graph,
            OperationType.Convolution2D,
            OperationType.BatchNormalization
        ));

        foreach (var sequence in candidates)
        {
            FuseConvBatchNorm(graph, sequence);
            modified = true;
        }

        return modified;
    }

    private void FuseConvBatchNorm(IComputationGraph<T> graph, List<ComputationNode<T>> nodes)
    {
        // Create fused node
        var fusedNode = FuseNodes(graph, nodes, OperationType.FusedConvBatchNorm);

        // Note: In a real implementation, you would fold the BatchNorm parameters
        // (mean, variance, gamma, beta) into the convolution weights and bias.
        // This requires numerical computation which would be done at optimization time.

        // Mark for special handling during execution
        fusedNode.Metadata["RequiresWeightFolding"] = true;
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.Convolution ||
                                   n.OperationType == OperationType.Convolution2D);
    }
}
