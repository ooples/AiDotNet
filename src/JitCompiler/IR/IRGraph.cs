using System.Linq;

namespace AiDotNet.JitCompiler.IR;

/// <summary>
/// Represents a computation graph in intermediate representation form.
/// </summary>
/// <remarks>
/// <para>
/// An IRGraph is a structured representation of a sequence of tensor operations
/// that have been recorded during autodiff execution. It serves as an intermediate
/// format between the high-level ComputationNode graph and the low-level compiled code.
/// </para>
/// <para><b>For Beginners:</b> Think of an IRGraph as a recipe for computations.
///
/// Just like a recipe lists ingredients and steps:
/// - InputIds are the ingredients (input tensors)
/// - Operations are the cooking steps (add, multiply, etc.)
/// - OutputIds are the final dishes (output tensors)
/// - TensorShapes tells us the "size" of each intermediate result
///
/// The IR graph makes it easier to optimize the computation (like combining steps)
/// and then compile it to fast executable code.
///
/// Example:
/// If your model does: result = ReLU(MatMul(input, weights) + bias)
/// The IR graph would have 3 operations: MatMul, Add, ReLU
/// Each operation knows its inputs and produces an output.
/// </para>
/// </remarks>
public class IRGraph
{
    /// <summary>
    /// Gets or sets the list of operations in this graph, in execution order.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Operations are stored in topological order, meaning each operation appears
    /// after all operations that produce its inputs. This ensures correct execution order.
    /// </para>
    /// <para><b>For Beginners:</b> This is the ordered list of computation steps.
    ///
    /// The order matters! You can't add two numbers before you've computed them.
    /// Each operation in the list uses results from earlier operations.
    /// </para>
    /// </remarks>
    public List<IROp> Operations { get; set; } = new();

    /// <summary>
    /// Gets or sets the mapping from tensor IDs to their shapes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every tensor in the graph (inputs, outputs, and intermediates) has a unique ID
    /// and a known shape (represented as int[] matching Tensor&lt;T&gt;.Shape).
    /// This dictionary provides that mapping.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a table that tells us the size of each value.
    ///
    /// For example:
    /// - Tensor 0 might be [32, 784] (a batch of 32 images, each with 784 pixels)
    /// - Tensor 1 might be [784, 128] (weights connecting 784 inputs to 128 outputs)
    /// - Tensor 2 might be [32, 128] (the result of multiplying tensor 0 and 1)
    ///
    /// Knowing shapes helps us:
    /// - Allocate the right amount of memory
    /// - Check that operations are valid (can't multiply incompatible shapes)
    /// - Optimize operations for specific sizes
    /// </para>
    /// </remarks>
    public Dictionary<int, int[]> TensorShapes { get; set; } = new();

    /// <summary>
    /// Gets or sets the IDs of input tensors to this graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input tensors are provided by the caller and are not computed within the graph.
    /// They serve as the starting point for all computations.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "ingredients" that you provide to start the computation.
    ///
    /// For a neural network, inputs might be:
    /// - The input data (like an image)
    /// - Model parameters (weights and biases)
    ///
    /// The graph will process these inputs through all its operations to produce outputs.
    /// </para>
    /// </remarks>
    public List<int> InputIds { get; set; } = new();

    /// <summary>
    /// Gets or sets the IDs of output tensors produced by this graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Output tensors are the final results of the graph computation and are
    /// returned to the caller.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "final dishes" - the results you care about.
    ///
    /// For a neural network, outputs might be:
    /// - Predictions (class probabilities)
    /// - Loss value
    /// - Intermediate features (for visualization)
    ///
    /// Everything else in the graph is just intermediate calculations to get to these outputs.
    /// </para>
    /// </remarks>
    public List<int> OutputIds { get; set; } = new();

    /// <summary>
    /// Gets or sets optional metadata about the graph.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Validates the graph structure for correctness.
    /// </summary>
    /// <returns>True if the graph is valid, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// Validation checks include:
    /// - All input tensor IDs are defined in TensorShapes
    /// - All operation inputs reference valid tensor IDs
    /// - No cycles in the graph (it's a DAG)
    /// - All output IDs are produced by operations or are inputs
    /// </para>
    /// <para><b>For Beginners:</b> This checks that the "recipe" makes sense.
    ///
    /// It verifies:
    /// - You're not using an ingredient that doesn't exist
    /// - Steps are in the right order (don't use results before computing them)
    /// - The final outputs are actually produced by the recipe
    ///
    /// If validation fails, something is wrong with how the graph was constructed.
    /// </para>
    /// </remarks>
    public bool Validate()
    {
        // Check that all inputs have shapes defined
        foreach (var inputId in InputIds.Where(id => !TensorShapes.ContainsKey(id)))
        {
            return false;
        }

        // Track which tensors have been produced
        var producedTensors = new HashSet<int>(InputIds);

        // Check each operation
        foreach (var op in Operations)
        {
            // Validate the operation itself
            if (!op.Validate())
            {
                return false;
            }

            // Check that all inputs have been produced
            foreach (var inputId in op.InputIds.Where(id => !producedTensors.Contains(id)))
            {
                return false; // Using a tensor before it's produced
            }

            // Mark ALL outputs as produced (support multi-output operations like gradient ops)
            // Use per-output shapes from OutputShapes array for proper multi-output support
            var outputShapes = op.OutputShapes;
            for (int i = 0; i < op.OutputIds.Length; i++)
            {
                var outputId = op.OutputIds[i];
                producedTensors.Add(outputId);

                // Ensure output shape is defined for each output using per-output shape
                if (!TensorShapes.ContainsKey(outputId))
                {
                    TensorShapes[outputId] = i < outputShapes.Length ? outputShapes[i] : op.OutputShape;
                }
            }
        }

        // Check that all outputs have been produced
        foreach (var outputId in OutputIds.Where(id => !producedTensors.Contains(id)))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Gets a string representation of the graph for debugging and visualization.
    /// </summary>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"IR Graph:");
        sb.AppendLine($"  Inputs: {string.Join(", ", InputIds.Select(id => $"t{id}"))}");
        sb.AppendLine($"  Operations ({Operations.Count}):");
        foreach (var op in Operations)
        {
            sb.AppendLine($"    {op}");
        }
        sb.AppendLine($"  Outputs: {string.Join(", ", OutputIds.Select(id => $"t{id}"))}");
        return sb.ToString();
    }

    /// <summary>
    /// Computes a hash code for this graph structure (ignoring tensor values).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hash is based on the graph structure: operation types, shapes, and connectivity.
    /// This is used for caching compiled graphs - graphs with the same structure can reuse
    /// the same compiled code even if the actual tensor values are different.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a "fingerprint" for the graph structure.
    ///
    /// Two graphs with the same fingerprint have the same structure (same operations,
    /// same shapes) even if the actual numbers in the tensors are different.
    ///
    /// This lets us reuse compiled code:
    /// - First time: Compile the graph (slow)
    /// - Next time with same structure: Reuse compiled code (fast!)
    ///
    /// It's like having a pre-cooked recipe that you can use with different ingredients.
    /// </para>
    /// </remarks>
    public int ComputeStructureHash()
    {
        int hash = 17;

        // Hash input shapes
        foreach (var inputId in InputIds.OrderBy(id => id))
        {
            hash = hash * 31 + inputId.GetHashCode();
            if (TensorShapes.TryGetValue(inputId, out var shape))
            {
                hash = hash * 31 + shape.GetShapeHashCode();
            }
        }

        // Hash operations
        foreach (var op in Operations)
        {
            hash = hash * 31 + op.OpType.GetHashCode();

            // Hash ALL output IDs (support multi-output operations)
            foreach (var outputId in op.OutputIds)
            {
                hash = hash * 31 + outputId;
            }

            hash = hash * 31 + op.OutputType.GetHashCode();
            hash = hash * 31 + op.OutputShape.GetShapeHashCode();

            foreach (var inputId in op.InputIds)
            {
                hash = hash * 31 + inputId;
            }
        }

        // Hash output IDs
        foreach (var outputId in OutputIds.OrderBy(id => id))
        {
            hash = hash * 31 + outputId.GetHashCode();
        }

        return hash;
    }
}
