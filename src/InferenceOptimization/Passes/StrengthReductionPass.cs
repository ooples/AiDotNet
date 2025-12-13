using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Replaces expensive operations with cheaper equivalent operations.
/// Examples:
/// - x^2 -> x * x
/// - x * 2 -> x + x
/// - x / 2 -> x * 0.5
/// - sqrt(x^2) -> abs(x)
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class StrengthReductionPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.StrengthReduction;
    public override string Name => "Strength Reduction";

    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;

        var reducibleOps = new HashSet<OperationType>
        {
            OperationType.Power, OperationType.Divide, OperationType.Multiply
        };
        foreach (var node in graph.Nodes.Where(n => reducibleOps.Contains(n.OperationType)).ToList())
        {
            if (TryReduceStrength(graph, node))
            {
                modified = true;
            }
        }

        return modified;
    }

    private bool TryReduceStrength(IOptimizationGraph<T> graph, OptimizationNode<T> node)
    {
        return node.OperationType switch
        {
            OperationType.Power => ReducePower(graph, node),
            OperationType.Divide => ReduceDivide(graph, node),
            OperationType.Multiply => ReduceMultiply(graph, node),
            _ => false
        };
    }

    private bool ReducePower(IOptimizationGraph<T> graph, OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        var exponent = node.Inputs[1];

        // x^2 -> x * x (multiplication is faster than power)
        if (IsConstantWithValue(exponent, 2))
        {
            var multiplyNode = new OptimizationNode<T>
            {
                OperationType = OperationType.Multiply,
                Name = $"{node.Name}_strength_reduced",
                OutputShape = node.OutputShape
            };

            var input = node.Inputs[0];
            multiplyNode.AddInput(input);
            multiplyNode.AddInput(input);

            ReplaceNode(graph, node, multiplyNode);
            return true;
        }

        return false;
    }

    private bool ReduceDivide(IOptimizationGraph<T> graph, OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        var divisor = node.Inputs[1];

        // x / constant -> x * (1/constant) (multiplication is faster than division)
        if (divisor.OperationType == OperationType.Constant)
        {
            var multiplyNode = new OptimizationNode<T>
            {
                OperationType = OperationType.Multiply,
                Name = $"{node.Name}_strength_reduced",
                OutputShape = node.OutputShape
            };

            // Create reciprocal constant
            var reciprocalNode = new OptimizationNode<T>
            {
                OperationType = OperationType.Constant,
                Name = $"{divisor.Name}_reciprocal",
                OutputShape = divisor.OutputShape,
                Metadata = new Dictionary<string, object>
                {
                    ["IsReciprocal"] = true,
                    ["OriginalConstant"] = divisor
                }
            };

            graph.AddNode(reciprocalNode);

            multiplyNode.AddInput(node.Inputs[0]);
            multiplyNode.AddInput(reciprocalNode);

            ReplaceNode(graph, node, multiplyNode);
            return true;
        }

        return false;
    }

    private bool ReduceMultiply(IOptimizationGraph<T> graph, OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        // x * 2 -> x + x (addition might be faster on some hardware)
        var left = node.Inputs[0];
        var right = node.Inputs[1];

        if (IsConstantWithValue(right, 2))
        {
            var addNode = new OptimizationNode<T>
            {
                OperationType = OperationType.Add,
                Name = $"{node.Name}_strength_reduced",
                OutputShape = node.OutputShape
            };

            addNode.AddInput(left);
            addNode.AddInput(left);

            // Note: Only apply this if beneficial on target hardware
            if (IsAdditionFasterThanMultiplication())
            {
                ReplaceNode(graph, node, addNode);
                return true;
            }
        }

        return false;
    }

    private bool IsConstantWithValue(OptimizationNode<T> node, double value)
    {
        if (node.OperationType != OperationType.Constant)
        {
            return false;
        }

        // In a real implementation, check the actual constant value
        return node.Metadata.TryGetValue("Value", out var val) &&
               Math.Abs((double)val - value) < 1e-6;
    }

    private bool IsAdditionFasterThanMultiplication()
    {
        // This would depend on the target hardware
        // For most modern CPUs, multiplication and addition have similar latency
        // So we return false by default
        return false;
    }

    private void ReplaceNode(IOptimizationGraph<T> graph, OptimizationNode<T> oldNode, OptimizationNode<T> newNode)
    {
        // Replace all outputs
        foreach (var output in oldNode.Outputs.ToList())
        {
            output.ReplaceInput(oldNode, newNode);
        }

        // Add new node and remove old one
        graph.AddNode(newNode);
        graph.RemoveNode(oldNode);
    }

    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               (graph.Nodes.Any(n => n.OperationType == OperationType.Power) ||
                graph.Nodes.Any(n => n.OperationType == OperationType.Divide));
    }
}
