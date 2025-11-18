using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Fuses Convolution + BatchNormalization + ReLU into a single operation.
/// This is one of the most common patterns in CNNs (ResNet, VGG, etc.) and provides
/// significant speedup by reducing memory traffic and kernel launches.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class ConvBatchNormReLUFusionPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.ConvBatchNormReLUFusion;
    public override string Name => "Conv + BatchNorm + ReLU Fusion";

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Find Conv -> BatchNorm -> ReLU patterns
        var candidates = FindFusionCandidates(
            graph,
            OperationType.Convolution,
            OperationType.BatchNormalization,
            OperationType.ReLU
        );

        // Also check for Convolution2D variant
        candidates.AddRange(FindFusionCandidates(
            graph,
            OperationType.Convolution2D,
            OperationType.BatchNormalization,
            OperationType.ReLU
        ));

        // Also check for LeakyReLU and other ReLU variants
        candidates.AddRange(FindFusionCandidates(
            graph,
            OperationType.Convolution,
            OperationType.BatchNormalization,
            OperationType.LeakyReLU
        ));

        foreach (var sequence in candidates)
        {
            FuseConvBatchNormReLU(graph, sequence);
            modified = true;
        }

        return modified;
    }

    private void FuseConvBatchNormReLU(IComputationGraph<T> graph, List<ComputationNode<T>> nodes)
    {
        // Create fused node
        var fusedNode = FuseNodes(graph, nodes, OperationType.FusedConvBatchNormReLU);

        // This fusion provides maximum benefit:
        // 1. Fold BatchNorm into Conv weights
        // 2. Apply ReLU in-place
        // 3. Single kernel launch instead of 3
        fusedNode.Metadata["RequiresWeightFolding"] = true;
        fusedNode.CanOperateInPlace = true; // ReLU can be in-place
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.Convolution ||
                                   n.OperationType == OperationType.Convolution2D) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.BatchNormalization) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.ReLU ||
                                   n.OperationType == OperationType.LeakyReLU);
    }
}
