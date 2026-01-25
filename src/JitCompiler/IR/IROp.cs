namespace AiDotNet.JitCompiler.IR;

/// <summary>
/// Base class for all IR operations.
/// </summary>
/// <remarks>
/// <para>
/// IROp represents a single operation in the intermediate representation graph.
/// Each operation has inputs (tensor IDs), produces an output (tensor ID), and
/// has metadata about types and shapes.
/// </para>
/// <para><b>For Beginners:</b> An IROp is like a single step in a recipe.
///
/// Each operation:
/// - Takes some inputs (the tensor IDs it needs)
/// - Performs a calculation (add, multiply, etc.)
/// - Produces an output (a new tensor ID)
/// - Knows what type and shape the output will be
///
/// For example, an "Add" operation might:
/// - Take inputs: tensor 0 and tensor 1
/// - Perform: element-wise addition
/// - Produce: tensor 2
/// - Know: output has the same shape as the inputs
///
/// The JIT compiler uses this information to generate optimized code.
/// </para>
/// </remarks>
public abstract class IROp
{
    /// <summary>
    /// Gets or sets the identifiers for the outputs of this operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output IDs identify the tensors produced by this operation.
    /// They're used by subsequent operations to reference these results.
    /// Most operations produce a single output, but gradient operations
    /// may produce multiple outputs (one gradient per input).
    /// </para>
    /// <para><b>For Beginners:</b> These are like variable names for the results.
    ///
    /// For example, if this operation computes "c = a + b":
    /// - OutputIds might be [2] (representing "c")
    /// - InputIds might be [0, 1] (representing "a" and "b")
    ///
    /// For gradient operations like "grad(a * b)":
    /// - OutputIds might be [3, 4] (representing "d_a" and "d_b")
    /// - InputIds might be [0, 1, 2] (representing "a", "b", "d_c")
    ///
    /// Later operations can use these tensor IDs as their inputs.
    /// </para>
    /// </remarks>
    public int[] OutputIds { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the primary output ID (first element of OutputIds).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a convenience property for operations that produce a single output.
    /// For multi-output operations like gradient ops, use OutputIds directly.
    /// </para>
    /// <para><b>For Beginners:</b> Most operations produce just one result.
    /// This property makes it easy to work with the common case.
    /// </para>
    /// </remarks>
    public int OutputId
    {
        get => OutputIds.Length > 0 ? OutputIds[0] : -1;
        set => OutputIds = [value];
    }

    /// <summary>
    /// Gets or sets the identifiers of the input tensors to this operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input IDs reference tensors that must be computed before this operation.
    /// They can be graph inputs, constants, or outputs from earlier operations.
    /// </para>
    /// <para><b>For Beginners:</b> These are the inputs this operation needs.
    ///
    /// For a binary operation like addition:
    /// - InputIds = [0, 1] means "add tensor 0 and tensor 1"
    ///
    /// For a unary operation like ReLU:
    /// - InputIds = [5] means "apply ReLU to tensor 5"
    ///
    /// The order matters! For subtraction, [0, 1] means "0 - 1", not "1 - 0".
    /// </para>
    /// </remarks>
    public int[] InputIds { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the data type of the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output type determines what numeric type (float, double, int, etc.)
    /// the result tensor will use. This affects memory usage and precision.
    /// </para>
    /// <para><b>For Beginners:</b> This tells us what kind of numbers the result contains.
    ///
    /// Common types:
    /// - Float32: Single-precision floating point (most common for neural networks)
    /// - Float64: Double-precision floating point (higher precision, more memory)
    /// - Int32: 32-bit integers
    ///
    /// The type affects:
    /// - Memory usage (float32 uses half the memory of float64)
    /// - Precision (how accurate calculations are)
    /// - Performance (some operations are faster with certain types)
    /// </para>
    /// </remarks>
    public IRType OutputType { get; set; }

    /// <summary>
    /// Gets or sets the shape of the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output shape is represented as an int[] array matching the existing
    /// Tensor&lt;T&gt;.Shape format. Each element is the size of that dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This tells us the size and dimensions of the result.
    ///
    /// Examples:
    /// - [] = scalar (single number)
    /// - [10] = vector with 10 elements
    /// - [3, 4] = 3×4 matrix
    /// - [32, 3, 224, 224] = batch of 32 RGB images, each 224×224 pixels
    ///
    /// The shape is determined by the operation:
    /// - Adding [3, 4] + [3, 4] → [3, 4] (same shape)
    /// - Matrix multiply [3, 4] × [4, 5] → [3, 5] (rows from left, cols from right)
    /// - Sum [3, 4] along axis 1 → [3] (reduces one dimension)
    /// </para>
    /// </remarks>
    public int[] OutputShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets the shapes of all output tensors, indexed by position in OutputIds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For multi-output operations (e.g., split, gradient ops), each output may have
    /// a different shape. This property returns an array where OutputShapes[i]
    /// corresponds to OutputIds[i].
    /// </para>
    /// <para>
    /// If not explicitly set, defaults to using OutputShape for all outputs
    /// (backward compatibility for single-output operations).
    /// </para>
    /// <para><b>For Beginners:</b> When an operation produces multiple outputs,
    /// each output can have its own shape. For example, a "split" operation that
    /// splits a [6, 4] tensor into two parts might produce:
    /// - OutputIds = [1, 2]
    /// - OutputShapes = [[3, 4], [3, 4]] (each half)
    /// </para>
    /// </remarks>
    public virtual int[][] OutputShapes
    {
        get
        {
            // Default: use OutputShape for all outputs (backward compatibility)
            var count = OutputIds.Length;
            if (count == 0) return Array.Empty<int[]>();

            var shapes = new int[count][];
            for (int i = 0; i < count; i++)
            {
                shapes[i] = OutputShape;
            }
            return shapes;
        }
    }

    /// <summary>
    /// Gets the operation type name for debugging and visualization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// By default, this returns the class name without the "Op" suffix.
    /// For example, "MatMulOp" becomes "MatMul".
    /// </para>
    /// <para><b>For Beginners:</b> This is a human-readable name for the operation.
    ///
    /// Used for:
    /// - Debugging (see what operations are in the graph)
    /// - Visualization (draw a graph diagram)
    /// - Logging (track what the compiler is doing)
    ///
    /// Examples: "Add", "MatMul", "ReLU", "Conv2D"
    /// </para>
    /// </remarks>
    public virtual string OpType => GetType().Name.Replace("Op", "");

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// Basic validation checks that the operation has required information.
    /// Derived classes can override to add operation-specific validation.
    /// </para>
    /// <para><b>For Beginners:</b> This checks that the operation makes sense.
    ///
    /// Basic checks:
    /// - Output ID is valid (non-negative)
    /// - Has the right number of inputs
    /// - Shapes are compatible
    ///
    /// Specific operations add their own checks:
    /// - MatMul: inner dimensions must match
    /// - Conv2D: kernel size must be valid
    /// - Reshape: total elements must be preserved
    ///
    /// If validation fails, the operation can't be compiled.
    /// </para>
    /// </remarks>
    public virtual bool Validate()
    {
        // Basic validation: must have at least one output
        if (OutputIds == null || OutputIds.Length == 0)
            return false;

        // All output IDs should be non-negative
        foreach (var outputId in OutputIds)
        {
            if (outputId < 0)
                return false;
        }

        // Output shape should be valid
        if (OutputShape == null || !OutputShape.IsValidShape())
            return false;

        return true;
    }

    /// <summary>
    /// Gets a string representation of this operation for debugging.
    /// </summary>
    /// <returns>A string describing this operation.</returns>
    /// <remarks>
    /// <para>
    /// The string format is: "tOutput = OpType(tInput1, tInput2, ...) : Type [Shape]"
    /// </para>
    /// <para><b>For Beginners:</b> This creates a readable description of the operation.
    ///
    /// Example outputs:
    /// - "t2 = Add(t0, t1) : Float32 [3, 4]"
    /// - "t5 = MatMul(t3, t4) : Float32 [128, 256]"
    /// - "t8 = ReLU(t7) : Float32 [32, 128]"
    ///
    /// This is super helpful for debugging - you can see exactly what each
    /// operation does and what shape tensors flow through the graph.
    /// </para>
    /// </remarks>
    public override string ToString()
    {
        var inputs = string.Join(", ", InputIds.Select(id => $"t{id}"));
        var outputs = string.Join(", ", OutputIds.Select(id => $"t{id}"));
        return $"{outputs} = {OpType}({inputs}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Interface for optimization passes that transform IR graphs.
/// </summary>
/// <remarks>
/// <para>
/// Optimization passes take an IR graph and transform it to an equivalent
/// but more efficient version. Examples include constant folding, dead code
/// elimination, and operation fusion.
/// </para>
/// <para><b>For Beginners:</b> An optimization pass improves the graph without changing what it computes.
///
/// Think of it like optimizing a recipe:
/// - Original: "Add 1 cup flour. Add another 1 cup flour."
/// - Optimized: "Add 2 cups flour."
/// - Result is the same, but simpler!
///
/// Common optimizations:
/// - Constant folding: Compute constant expressions at compile time
/// - Dead code elimination: Remove operations whose results aren't used
/// - Operation fusion: Combine multiple operations into one
/// - Common subexpression elimination: Compute repeated expressions only once
///
/// These make the compiled code faster by:
/// - Doing less work
/// - Using less memory
/// - Better utilizing CPU/GPU resources
/// </para>
/// </remarks>
public interface IOptimizationPass
{
    /// <summary>
    /// Applies this optimization pass to an IR graph.
    /// </summary>
    /// <param name="graph">The graph to optimize.</param>
    /// <returns>The optimized graph (may be the same instance or a new one).</returns>
    /// <remarks>
    /// <para>
    /// The optimization must preserve the semantics of the graph - it should
    /// produce the same results for the same inputs, just more efficiently.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms the graph to make it faster.
    ///
    /// The pass:
    /// - Examines the graph to find optimization opportunities
    /// - Creates a new, more efficient version
    /// - Returns the optimized graph
    ///
    /// The optimized graph computes the same results but runs faster.
    ///
    /// Multiple passes can be chained:
    /// - Original graph
    /// - → Constant folding
    /// - → Dead code elimination
    /// - → Operation fusion
    /// - → Optimized graph (much faster!)
    /// </para>
    /// </remarks>
    IRGraph Optimize(IRGraph graph);

    /// <summary>
    /// Gets the name of this optimization pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The name is used for logging and debugging to track which optimizations
    /// have been applied to a graph.
    /// </para>
    /// <para><b>For Beginners:</b> A human-readable name for this optimization.
    ///
    /// Examples:
    /// - "Constant Folding"
    /// - "Dead Code Elimination"
    /// - "Operation Fusion"
    ///
    /// Used when printing optimization logs like:
    /// "Applied Constant Folding: reduced 150 ops to 142 ops"
    /// </para>
    /// </remarks>
    string Name { get; }
}
