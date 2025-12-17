using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Fuses MatMul + Bias + Activation (ReLU, GELU, etc.) into a single operation.
/// This is the most common pattern in transformer feed-forward networks and MLPs.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
/// <remarks>
/// <para>
/// Matrix multiplication followed by bias addition and activation is the fundamental
/// building block of neural networks. Fusing these operations provides significant
/// performance benefits by:
/// <list type="bullet">
/// <item><description>Reducing memory bandwidth (no intermediate tensor writes)</description></item>
/// <item><description>Enabling hardware-optimized fused kernels (cuBLAS, oneDNN)</description></item>
/// <item><description>Reducing kernel launch overhead</description></item>
/// </list>
/// </para>
/// <para><b>Fusion Patterns:</b></para>
/// <list type="bullet">
/// <item><description>MatMul + Add (bias) + ReLU → FusedMatMulBiasReLU</description></item>
/// <item><description>MatMul + Add (bias) + GELU → FusedMatMulBiasGELU</description></item>
/// <item><description>FusedMatMulBias + Activation → FusedMatMulBias{Activation}</description></item>
/// </list>
/// <para><b>Performance Impact:</b> Typically 30-50% speedup for transformer feed-forward layers.</para>
/// </remarks>
public class MatMulBiasActivationFusionPass<T> : OptimizationPassBase<T> where T : struct
{
    /// <inheritdoc/>
    public override OptimizationPassType PassType => OptimizationPassType.MatMulBiasActivationFusion;

    /// <inheritdoc/>
    public override string Name => "MatMul + Bias + Activation Fusion";

    private static readonly HashSet<OperationType> SupportedActivations = new()
    {
        OperationType.ReLU,
        OperationType.GELU,
        OperationType.LeakyReLU,
        OperationType.Tanh,
        OperationType.Sigmoid,
        OperationType.Swish,
        OperationType.Mish
    };

    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;

        // Look for already fused MatMulBias nodes followed by activation
        foreach (var fusedNode in graph.Nodes.Where(n =>
            n.OperationType == OperationType.FusedMatMulBias).ToList())
        {
            if (fusedNode.Outputs.Count == 1)
            {
                var activationNode = fusedNode.Outputs[0];

                if (SupportedActivations.Contains(activationNode.OperationType) &&
                    activationNode.Inputs.Count == 1)
                {
                    FuseMatMulBiasActivation(graph, fusedNode, activationNode);
                    modified = true;
                }
            }
        }

        // Also look for unfused MatMul -> Add -> Activation
        foreach (var matmulNode in graph.Nodes.Where(n =>
            (n.OperationType == OperationType.MatMul ||
             n.OperationType == OperationType.Dense ||
             n.OperationType == OperationType.FullyConnected) && !n.IsFused).ToList())
        {
            if (matmulNode.Outputs.Count == 1)
            {
                var addNode = matmulNode.Outputs[0];

                if (addNode.OperationType == OperationType.Add &&
                    addNode.Inputs.Count == 2 &&
                    addNode.Outputs.Count == 1)
                {
                    var activationNode = addNode.Outputs[0];

                    if (SupportedActivations.Contains(activationNode.OperationType) &&
                        activationNode.Inputs.Count == 1)
                    {
                        // Check if Add has a constant bias
                        var biasNode = addNode.Inputs.FirstOrDefault(n => n != matmulNode);

                        if (biasNode != null && biasNode.OperationType == OperationType.Constant)
                        {
                            FuseMatMulBiasActivationFromScratch(
                                graph,
                                matmulNode,
                                addNode,
                                biasNode,
                                activationNode);
                            modified = true;
                        }
                    }
                }
            }
        }

        return modified;
    }

    private void FuseMatMulBiasActivation(
        IOptimizationGraph<T> graph,
        OptimizationNode<T> fusedMatMulBias,
        OptimizationNode<T> activation)
    {
        // Determine the fused operation type based on activation
        var fusedType = activation.OperationType switch
        {
            OperationType.ReLU => OperationType.FusedMatMulBiasReLU,
            OperationType.GELU => OperationType.FusedMatMulBiasGELU,
            _ => OperationType.FusedMatMulBias
        };

        // Create new fused node
        var newFusedNode = new OptimizationNode<T>
        {
            OperationType = fusedType,
            Name = $"{fusedMatMulBias.Name}_{activation.OperationType.ToString().ToLower()}",
            OutputShape = activation.OutputShape,
            IsFused = true,
            CanOperateInPlace = true,
            FusedFrom = new List<OptimizationNode<T>> { fusedMatMulBias, activation }
        };

        // Copy all parameters from fused MatMulBias
        foreach (var param in fusedMatMulBias.Parameters)
        {
            newFusedNode.Parameters[param.Key] = param.Value;
        }

        // Copy activation parameters if any
        foreach (var param in activation.Parameters)
        {
            newFusedNode.Parameters[$"activation_{param.Key}"] = param.Value;
        }

        // Connect inputs
        foreach (var input in fusedMatMulBias.Inputs)
        {
            newFusedNode.AddInput(input);
            input.Outputs.Remove(fusedMatMulBias);
        }

        // Connect outputs
        foreach (var output in activation.Outputs)
        {
            output.ReplaceInput(activation, newFusedNode);
        }

        // Add new fused node and remove old ones
        graph.AddNode(newFusedNode);
        graph.RemoveNode(fusedMatMulBias);
        graph.RemoveNode(activation);
    }

    private void FuseMatMulBiasActivationFromScratch(
        IOptimizationGraph<T> graph,
        OptimizationNode<T> matmul,
        OptimizationNode<T> add,
        OptimizationNode<T> bias,
        OptimizationNode<T> activation)
    {
        var fusedType = activation.OperationType switch
        {
            OperationType.ReLU => OperationType.FusedMatMulBiasReLU,
            OperationType.GELU => OperationType.FusedMatMulBiasGELU,
            _ => OperationType.FusedMatMulBias
        };

        var fusedNode = new OptimizationNode<T>
        {
            OperationType = fusedType,
            Name = $"{matmul.Name}_fused",
            OutputShape = activation.OutputShape,
            IsFused = true,
            CanOperateInPlace = true,
            FusedFrom = new List<OptimizationNode<T>> { matmul, add, activation }
        };

        // Copy matmul parameters
        foreach (var param in matmul.Parameters)
        {
            fusedNode.Parameters[param.Key] = param.Value;
        }

        // Add bias
        fusedNode.Parameters["bias"] = bias.ConstantValue!;

        // Copy activation parameters
        foreach (var param in activation.Parameters)
        {
            fusedNode.Parameters[$"activation_{param.Key}"] = param.Value;
        }

        // Connect inputs
        foreach (var input in matmul.Inputs)
        {
            fusedNode.AddInput(input);
            input.Outputs.Remove(matmul);
        }

        // Connect outputs
        foreach (var output in activation.Outputs)
        {
            output.ReplaceInput(activation, fusedNode);
        }

        // Add fused node and remove originals
        graph.AddNode(fusedNode);
        graph.RemoveNode(matmul);
        graph.RemoveNode(add);

        // Only remove bias if it's not used elsewhere (shared biases should be kept)
        // The bias was consumed by the Add node, but if it has other consumers, keep it
        if (bias.Outputs.Count <= 1)
        {
            graph.RemoveNode(bias);
        }

        graph.RemoveNode(activation);
    }

    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.MatMul ||
                                   n.OperationType == OperationType.Dense ||
                                   n.OperationType == OperationType.FullyConnected ||
                                   n.OperationType == OperationType.FusedMatMulBias);
    }
}
