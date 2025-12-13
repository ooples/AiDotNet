using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Folds constant expressions at compile time to reduce runtime computation.
/// For example: If two constants are multiplied, compute the result once during optimization
/// rather than every inference call.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class ConstantFoldingPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.ConstantFolding;
    public override string Name => "Constant Folding";

    private static readonly HashSet<OperationType> FoldableOps = new()
    {
        OperationType.Add,
        OperationType.Subtract,
        OperationType.Multiply,
        OperationType.Divide,
        OperationType.Power,
        OperationType.Sqrt,
        OperationType.Exp,
        OperationType.Log,
        OperationType.MatMul
    };

    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;
        bool changed;

        // Keep folding until no more changes (iterative constant propagation)
        do
        {
            changed = false;

            foreach (var node in graph.Nodes
                .Where(n => FoldableOps.Contains(n.OperationType) && !n.IsFused)
                .Where(n => n.Inputs.All(input => input.OperationType == OperationType.Constant))
                .ToList())
            {
                if (TryFoldConstant(graph, node))
                {
                    changed = true;
                    modified = true;
                }
            }
        } while (changed);

        return modified;
    }

    /// <summary>
    /// Attempts to fold a constant expression node into a single constant value.
    /// </summary>
    /// <param name="graph">The optimization graph containing the node.</param>
    /// <param name="node">The operation node whose inputs are all constants.</param>
    /// <returns>
    /// True if the operation was successfully folded into a constant node;
    /// false if folding could not be performed.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method computes the result of the operation at optimization time and replaces
    /// the operation node with a constant node containing the precomputed result. This
    /// eliminates runtime computation for operations involving only constants.
    /// </para>
    /// <para>
    /// The folding process:
    /// <list type="number">
    /// <item><description>Compute the operation result using vectorized Engine operations</description></item>
    /// <item><description>Create a new constant node with the computed result</description></item>
    /// <item><description>Update all output connections to reference the new constant</description></item>
    /// <item><description>Remove the original operation node from the graph</description></item>
    /// </list>
    /// </para>
    /// <para><b>Thread Safety:</b> This method modifies the graph structure and is not thread-safe.</para>
    /// </remarks>
    private bool TryFoldConstant(IOptimizationGraph<T> graph, OptimizationNode<T> node)
    {
        try
        {
            // Compute the constant result
            Tensor<T>? result = node.OperationType switch
            {
                OperationType.Add => FoldAdd(node),
                OperationType.Subtract => FoldSubtract(node),
                OperationType.Multiply => FoldMultiply(node),
                OperationType.Divide => FoldDivide(node),
                OperationType.MatMul => FoldMatMul(node),
                _ => null
            };

            if (result == null)
            {
                return false;
            }

            // Create a new constant node with the result
            var constantNode = new OptimizationNode<T>
            {
                OperationType = OperationType.Constant,
                Name = $"{node.Name}_folded",
                OutputShape = node.OutputShape,
                ConstantValue = result,
                CanEliminate = false // Constants should not be eliminated
            };

            // Replace the operation node with the constant node
            foreach (var output in node.Outputs.ToList())
            {
                output.ReplaceInput(node, constantNode);
            }

            // Add constant node and remove operation node
            graph.AddNode(constantNode);
            graph.RemoveNode(node);

            return true;
        }
        catch (InvalidOperationException)
        {
            // If folding fails due to invalid graph state, leave the node as is
            return false;
        }
        catch (ArgumentException)
        {
            // If folding fails due to invalid arguments, leave the node as is
            return false;
        }
    }

    /// <summary>
    /// Folds an addition operation by computing the elementwise sum of two constant tensors.
    /// </summary>
    /// <param name="node">The optimization node representing the addition operation.</param>
    /// <returns>
    /// A tensor containing the elementwise sum of the two input tensors,
    /// or null if the operation cannot be folded.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method uses the Engine's vectorized TensorAdd operation for optimal performance.
    /// The operation requires both input tensors to have identical shapes since broadcasting
    /// is not supported during constant folding.
    /// </para>
    /// <para><b>Performance:</b> Uses hardware-accelerated SIMD operations when available.</para>
    /// </remarks>
    private Tensor<T>? FoldAdd(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        // Perform vectorized tensor addition using Engine operations
        try
        {
            // Use elementwise addition - tensors must have compatible shapes
            if (!left.Shape.SequenceEqual(right.Shape))
            {
                // Shape mismatch - cannot fold without broadcasting support
                return null;
            }

            // Use Engine's vectorized addition for optimal performance
            var engine = AiDotNetEngine.Current;
            return engine.TensorAdd(left, right);
        }
        catch
        {
            // If tensor arithmetic fails, don't fold
            return null;
        }
    }

    /// <summary>
    /// Folds a subtraction operation by computing the elementwise difference of two constant tensors.
    /// </summary>
    /// <param name="node">The optimization node representing the subtraction operation.</param>
    /// <returns>
    /// A tensor containing the elementwise difference of the two input tensors (left - right),
    /// or null if the operation cannot be folded.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method uses the Engine's vectorized TensorSubtract operation for optimal performance.
    /// The operation requires both input tensors to have identical shapes since broadcasting
    /// is not supported during constant folding.
    /// </para>
    /// <para><b>Performance:</b> Uses hardware-accelerated SIMD operations when available.</para>
    /// </remarks>
    private Tensor<T>? FoldSubtract(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        // Perform vectorized tensor subtraction using Engine operations
        try
        {
            if (!left.Shape.SequenceEqual(right.Shape))
            {
                // Shape mismatch - cannot fold without broadcasting support
                return null;
            }

            // Use Engine's vectorized subtraction for optimal performance
            var engine = AiDotNetEngine.Current;
            return engine.TensorSubtract(left, right);
        }
        catch
        {
            // If tensor arithmetic fails, don't fold
            return null;
        }
    }

    /// <summary>
    /// Folds a multiplication operation by computing the elementwise product of two constant tensors.
    /// </summary>
    /// <param name="node">The optimization node representing the multiplication operation.</param>
    /// <returns>
    /// A tensor containing the elementwise product (Hadamard product) of the two input tensors,
    /// or null if the operation cannot be folded.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method uses the Engine's vectorized TensorMultiply operation for optimal performance.
    /// The operation requires both input tensors to have identical shapes since broadcasting
    /// is not supported during constant folding.
    /// </para>
    /// <para><b>Note:</b> This performs elementwise multiplication (Hadamard product), not
    /// matrix multiplication. For matrix multiplication, use <see cref="FoldMatMul"/>.</para>
    /// <para><b>Performance:</b> Uses hardware-accelerated SIMD operations when available.</para>
    /// </remarks>
    private Tensor<T>? FoldMultiply(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        // Perform vectorized tensor multiplication (elementwise) using Engine operations
        try
        {
            if (!left.Shape.SequenceEqual(right.Shape))
            {
                // Shape mismatch - cannot fold without broadcasting support
                return null;
            }

            // Use Engine's vectorized multiplication for optimal performance
            var engine = AiDotNetEngine.Current;
            return engine.TensorMultiply(left, right);
        }
        catch
        {
            // If tensor arithmetic fails, don't fold
            return null;
        }
    }

    /// <summary>
    /// Folds a division operation by computing the elementwise quotient of two constant tensors.
    /// </summary>
    /// <param name="node">The optimization node representing the division operation.</param>
    /// <returns>
    /// A tensor containing the elementwise quotient of the two input tensors,
    /// or null if the operation cannot be folded.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method uses the Engine's vectorized TensorDivide operation for optimal performance.
    /// The operation requires both input tensors to have identical shapes since broadcasting
    /// is not supported during constant folding.
    /// </para>
    /// <para><b>Performance:</b> Uses hardware-accelerated SIMD operations when available.</para>
    /// </remarks>
    private Tensor<T>? FoldDivide(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        // Perform vectorized tensor division using Engine operations
        try
        {
            if (!left.Shape.SequenceEqual(right.Shape))
            {
                // Shape mismatch - cannot fold without broadcasting support
                return null;
            }

            // Use Engine's vectorized division for optimal performance
            var engine = AiDotNetEngine.Current;
            return engine.TensorDivide(left, right);
        }
        catch
        {
            // If tensor arithmetic fails (e.g., division by zero), don't fold
            return null;
        }
    }

    /// <summary>
    /// Folds a matrix multiplication operation by computing the product of two constant tensors.
    /// </summary>
    /// <param name="node">The optimization node representing the matrix multiplication operation.</param>
    /// <returns>
    /// A tensor containing the matrix product of the two input tensors,
    /// or null if the operation cannot be folded.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method uses the Engine's vectorized TensorMatMul operation for optimal performance.
    /// Matrix multiplication requires 2D tensors with compatible dimensions:
    /// left[M,K] × right[K,N] = result[M,N].
    /// </para>
    /// <para><b>Performance:</b> Uses optimized BLAS-like operations when available,
    /// with cache-friendly memory access patterns and potential GPU acceleration.</para>
    /// </remarks>
    private Tensor<T>? FoldMatMul(OptimizationNode<T> node)
    {
        if (node.Inputs.Count != 2) return null;

        var left = node.Inputs[0].ConstantValue;
        var right = node.Inputs[1].ConstantValue;

        if (left == null || right == null) return null;

        // Matrix multiplication requires 2D tensors with compatible dimensions
        try
        {
            if (left.Shape.Length != 2 || right.Shape.Length != 2)
            {
                return null;
            }

            int k = left.Shape[1];

            // Check dimension compatibility: left[m,k] @ right[k,n] = result[m,n]
            if (k != right.Shape[0])
            {
                return null;
            }

            // Use Engine's vectorized matrix multiplication for optimal performance
            var engine = AiDotNetEngine.Current;
            return engine.TensorMatMul(left, right);
        }
        catch
        {
            // If matrix multiplication fails, don't fold
            return null;
        }
    }

    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.Constant);
    }
}
