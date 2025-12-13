using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Applies algebraic simplification rules to reduce computational complexity.
/// Examples:
/// - x * 1 = x
/// - x + 0 = x
/// - x * 0 = 0
/// - x / 1 = x
/// - x^1 = x
/// - x^0 = 1
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class AlgebraicSimplificationPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.AlgebraicSimplification;
    public override string Name => "Algebraic Simplification";

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;
        bool changed;

        do
        {
            changed = false;

            var simplifiableOps = new HashSet<OperationType>
            {
                OperationType.Multiply, OperationType.Add, OperationType.Subtract,
                OperationType.Divide, OperationType.Power
            };
            foreach (var node in graph.Nodes.Where(n => simplifiableOps.Contains(n.OperationType)).ToList())
            {
                if (TrySimplifyNode(graph, node))
                {
                    changed = true;
                    modified = true;
                }
            }
        } while (changed);

        return modified;
    }

    private bool TrySimplifyNode(IComputationGraph<T> graph, ComputationNode<T> node)
    {
        return node.OperationType switch
        {
            OperationType.Multiply => SimplifyMultiply(graph, node),
            OperationType.Add => SimplifyAdd(graph, node),
            OperationType.Subtract => SimplifySubtract(graph, node),
            OperationType.Divide => SimplifyDivide(graph, node),
            OperationType.Power => SimplifyPower(graph, node),
            _ => false
        };
    }

    private bool SimplifyMultiply(IComputationGraph<T> graph, ComputationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        var left = node.Inputs[0];
        var right = node.Inputs[1];

        // x * 0 = 0
        if (IsZeroConstant(left) || IsZeroConstant(right))
        {
            ReplaceWithConstant(graph, node, CreateZeroConstant(node));
            return true;
        }

        // x * 1 = x
        if (IsOneConstant(left))
        {
            ReplaceWithNode(graph, node, right);
            return true;
        }

        if (IsOneConstant(right))
        {
            ReplaceWithNode(graph, node, left);
            return true;
        }

        return false;
    }

    private bool SimplifyAdd(IComputationGraph<T> graph, ComputationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        var left = node.Inputs[0];
        var right = node.Inputs[1];

        // x + 0 = x
        if (IsZeroConstant(left))
        {
            ReplaceWithNode(graph, node, right);
            return true;
        }

        if (IsZeroConstant(right))
        {
            ReplaceWithNode(graph, node, left);
            return true;
        }

        return false;
    }

    private bool SimplifySubtract(IComputationGraph<T> graph, ComputationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        var right = node.Inputs[1];

        // x - 0 = x
        if (IsZeroConstant(right))
        {
            ReplaceWithNode(graph, node, node.Inputs[0]);
            return true;
        }

        // x - x = 0 (if inputs are the same)
        if (node.Inputs[0] == node.Inputs[1])
        {
            ReplaceWithConstant(graph, node, CreateZeroConstant(node));
            return true;
        }

        return false;
    }

    private bool SimplifyDivide(IComputationGraph<T> graph, ComputationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        var right = node.Inputs[1];

        // x / 1 = x
        if (IsOneConstant(right))
        {
            ReplaceWithNode(graph, node, node.Inputs[0]);
            return true;
        }

        return false;
    }

    private bool SimplifyPower(IComputationGraph<T> graph, ComputationNode<T> node)
    {
        if (node.Inputs.Count != 2) return false;

        var right = node.Inputs[1];

        // x^0 = 1
        if (IsZeroConstant(right))
        {
            ReplaceWithConstant(graph, node, CreateOneConstant(node));
            return true;
        }

        // x^1 = x
        if (IsOneConstant(right))
        {
            ReplaceWithNode(graph, node, node.Inputs[0]);
            return true;
        }

        return false;
    }

    private bool IsZeroConstant(ComputationNode<T> node)
    {
        return node.OperationType == OperationType.Constant &&
               node.Metadata.TryGetValue("IsZero", out var isZero) &&
               (bool)isZero;
    }

    private bool IsOneConstant(ComputationNode<T> node)
    {
        return node.OperationType == OperationType.Constant &&
               node.Metadata.TryGetValue("IsOne", out var isOne) &&
               (bool)isOne;
    }

    private ComputationNode<T> CreateZeroConstant(ComputationNode<T> templateNode)
    {
        return new ComputationNode<T>
        {
            OperationType = OperationType.Constant,
            Name = "const_zero",
            OutputShape = templateNode.OutputShape,
            Metadata = new Dictionary<string, object> { ["IsZero"] = true }
        };
    }

    private ComputationNode<T> CreateOneConstant(ComputationNode<T> templateNode)
    {
        return new ComputationNode<T>
        {
            OperationType = OperationType.Constant,
            Name = "const_one",
            OutputShape = templateNode.OutputShape,
            Metadata = new Dictionary<string, object> { ["IsOne"] = true }
        };
    }

    private void ReplaceWithNode(IComputationGraph<T> graph, ComputationNode<T> oldNode, ComputationNode<T> newNode)
    {
        // Replace all uses of oldNode with newNode
        foreach (var output in oldNode.Outputs.ToList())
        {
            output.ReplaceInput(oldNode, newNode);
        }

        graph.RemoveNode(oldNode);
    }

    private void ReplaceWithConstant(IComputationGraph<T> graph, ComputationNode<T> oldNode, ComputationNode<T> constantNode)
    {
        // Add constant to graph
        graph.AddNode(constantNode);

        // Replace uses
        foreach (var output in oldNode.Outputs.ToList())
        {
            output.ReplaceInput(oldNode, constantNode);
        }

        graph.RemoveNode(oldNode);
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph);
    }
}
