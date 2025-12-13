using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Fuses MatMul + Bias (Add) operations into a single Gemm operation.
/// This is extremely common in fully connected layers and transformers.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class MatMulBiasFusionPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.MatMulBiasFusion;
    public override string Name => "MatMul + Bias Fusion";

    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;

        // Find MatMul -> Add patterns where Add is adding a bias
        foreach (var matmulNode in graph.Nodes.Where(n =>
            (n.OperationType == OperationType.MatMul ||
             n.OperationType == OperationType.Dense ||
             n.OperationType == OperationType.FullyConnected) && !n.IsFused && n.Outputs.Count == 1).ToList())
        {
            var addNode = matmulNode.Outputs[0];

            // Check if it's an Add with a constant bias
            if (addNode.OperationType == OperationType.Add &&
                addNode.Inputs.Count == 2 &&
                !addNode.IsFused)
            {
                // One input should be matmul, other should be constant
                var otherInput = addNode.Inputs.FirstOrDefault(n => n != matmulNode);

                if (otherInput != null && otherInput.OperationType == OperationType.Constant)
                {
                    FuseMatMulBias(graph, new List<OptimizationNode<T>> { matmulNode, addNode, otherInput });
                    modified = true;
                }
            }
        }

        return modified;
    }

    private void FuseMatMulBias(IOptimizationGraph<T> graph, List<OptimizationNode<T>> nodes)
    {
        var matmulNode = nodes[0];
        var addNode = nodes[1];
        var biasNode = nodes[2];

        // Create fused Gemm node (General Matrix Multiplication with bias)
        var fusedNode = new OptimizationNode<T>
        {
            OperationType = OperationType.FusedMatMulBias,
            Name = $"{matmulNode.Name}_gemm",
            OutputShape = addNode.OutputShape,
            IsFused = true,
            FusedFrom = new List<OptimizationNode<T>> { matmulNode, addNode }
        };

        // Copy parameters
        foreach (var param in matmulNode.Parameters)
        {
            fusedNode.Parameters[param.Key] = param.Value;
        }

        // Add bias as a parameter
        fusedNode.Parameters["bias"] = biasNode.ConstantValue!;

        // Connect inputs (from matmul, excluding the bias constant)
        foreach (var input in matmulNode.Inputs)
        {
            fusedNode.AddInput(input);
            input.Outputs.Remove(matmulNode);
        }

        // Connect outputs
        foreach (var output in addNode.Outputs)
        {
            output.ReplaceInput(addNode, fusedNode);
        }

        // Add fused node
        graph.AddNode(fusedNode);

        // Remove original nodes
        graph.RemoveNode(matmulNode);
        graph.RemoveNode(addNode);
        graph.RemoveNode(biasNode);
    }

    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.MatMul ||
                                   n.OperationType == OperationType.Dense ||
                                   n.OperationType == OperationType.FullyConnected);
    }
}
