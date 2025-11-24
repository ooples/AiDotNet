using AiDotNet.Helpers;

namespace AiDotNet.Autodiff;

/// <summary>
/// Represents a node in the automatic differentiation computation graph.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A ComputationNode is a fundamental building block of automatic differentiation.
/// It represents a value in a computational graph, along with information about
/// how to compute gradients with respect to that value. Each node stores its value,
/// gradient, parent nodes (inputs), and a backward function for gradient computation.
/// </para>
/// <para><b>For Beginners:</b> This represents a single step in a calculation that can be differentiated.
///
/// Think of it like this:
/// - A node stores a value (like the output of adding two numbers)
/// - It remembers what inputs were used to create this value (the two numbers)
/// - It knows how to calculate gradients (derivatives) with respect to its inputs
/// - Connecting nodes together forms a graph that tracks the entire calculation
///
/// This enables automatic differentiation, where gradients can be computed
/// automatically for complex operations by chaining together simple derivatives.
/// </para>
/// </remarks>
public class ComputationNode<T>
{
    /// <summary>
    /// Gets or sets the value stored in this node.
    /// </summary>
    /// <value>A tensor containing the computed value.</value>
    /// <remarks>
    /// This is the forward pass result - the actual output of the operation
    /// that this node represents.
    /// </remarks>
    public Tensor<T> Value { get; set; }

    /// <summary>
    /// Gets or sets the gradient accumulated at this node.
    /// </summary>
    /// <value>A tensor containing the gradient, or null if not yet computed.</value>
    /// <remarks>
    /// <para>
    /// The gradient represents the derivative of the loss with respect to this node's value.
    /// It's computed during the backward pass and accumulated from all paths in the graph
    /// that use this node's output.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how much the final result changes when this value changes.
    ///
    /// The gradient tells you:
    /// - How sensitive the final output is to this value
    /// - Which direction to adjust this value to reduce the loss
    /// - How much to adjust it (larger gradient = bigger adjustment needed)
    /// </para>
    /// </remarks>
    public Tensor<T>? Gradient { get; set; }

    /// <summary>
    /// Gets or sets the parent nodes (inputs) that were used to compute this node's value.
    /// </summary>
    /// <value>A list of parent computation nodes.</value>
    /// <remarks>
    /// <para>
    /// Parent nodes are the inputs to the operation that produced this node's value.
    /// During backpropagation, gradients flow from this node back to its parents.
    /// </para>
    /// <para><b>For Beginners:</b> These are the inputs that were combined to create this node's value.
    ///
    /// For example, if this node represents c = a + b:
    /// - a and b would be the parent nodes
    /// - When computing gradients, we need to know how c's gradient affects a and b
    /// - This parent list lets us trace back through the calculation
    /// </para>
    /// </remarks>
    public List<ComputationNode<T>> Parents { get; set; }

    /// <summary>
    /// Gets or sets the backward function that computes gradients for parent nodes.
    /// </summary>
    /// <value>A function that takes the current gradient and computes parent gradients.</value>
    /// <remarks>
    /// <para>
    /// The backward function implements the chain rule of calculus. Given the gradient
    /// at this node, it computes how much gradient should flow to each parent node.
    /// This is the core of automatic differentiation.
    /// </para>
    /// <para><b>For Beginners:</b> This function knows how to pass gradients backwards through this operation.
    ///
    /// The backward function:
    /// - Takes the gradient at this node as input
    /// - Calculates what gradients should go to each parent
    /// - Implements the mathematical rule for differentiating this operation
    ///
    /// For example, for addition (c = a + b):
    /// - The backward function would pass the gradient of c equally to both a and b
    /// - For multiplication (c = a * b), it would multiply c's gradient by the other input's value
    /// </para>
    /// </remarks>
    public Action<Tensor<T>>? BackwardFunction { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether this node requires gradient computation.
    /// </summary>
    /// <value>True if gradients should be computed for this node; false otherwise.</value>
    /// <remarks>
    /// <para>
    /// This flag controls whether this node participates in gradient computation.
    /// Setting it to false can improve performance for constants or intermediate
    /// values that don't need gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This controls whether to track gradients for this value.
    ///
    /// Set to true for:
    /// - Model parameters that need to be trained
    /// - Values you want to optimize
    ///
    /// Set to false for:
    /// - Constants that never change
    /// - Intermediate values you don't need gradients for (saves memory)
    /// </para>
    /// </remarks>
    public bool RequiresGradient { get; set; }

    /// <summary>
    /// Gets or sets an optional name for this node (useful for debugging).
    /// </summary>
    /// <value>A string name for the node, or null if not named.</value>
    /// <remarks>
    /// Node names help with debugging and visualization of the computation graph.
    /// They're optional but recommended for important nodes like parameters.
    /// </remarks>
    public string? Name { get; set; }

    /// <summary>
    /// Gets or sets the type of operation that created this node (used for JIT compilation).
    /// </summary>
    /// <value>A string identifying the operation type (e.g., "Add", "MatMul", "ReLU"), or null if not set.</value>
    /// <remarks>
    /// <para>
    /// This property is used by the JIT compiler to convert ComputationNode graphs to IR operations.
    /// It stores the name of the operation that produced this node's value, enabling the compiler
    /// to reconstruct the operation graph and optimize it for faster execution.
    /// </para>
    /// <para><b>For Beginners:</b> This records what operation created this node's value.
    ///
    /// For example:
    /// - If this node was created by adding two tensors, OperationType would be "Add"
    /// - If created by matrix multiplication, OperationType would be "MatMul"
    /// - If created by ReLU activation, OperationType would be "ReLU"
    ///
    /// This information allows the JIT compiler to:
    /// - Understand what operations are in the graph
    /// - Optimize sequences of operations
    /// - Generate fast compiled code
    ///
    /// This is optional and only needed when using JIT compilation.
    /// </para>
    /// </remarks>
    public string? OperationType { get; set; }

    /// <summary>
    /// Gets or sets additional operation-specific parameters (used for JIT compilation).
    /// </summary>
    /// <value>A dictionary of parameter names to values, or null if not set.</value>
    /// <remarks>
    /// <para>
    /// Some operations require additional parameters beyond their inputs. For example,
    /// convolution needs stride and padding, softmax needs an axis, etc. This dictionary
    /// stores those parameters for use by the JIT compiler.
    /// </para>
    /// <para><b>For Beginners:</b> This stores extra settings for operations.
    ///
    /// For example:
    /// - A Power operation might store {"Exponent": 2.0}
    /// - A Softmax operation might store {"Axis": -1}
    /// - A Conv2D operation might store {"Stride": [1, 1], "Padding": [0, 0]}
    ///
    /// These parameters tell the JIT compiler exactly how the operation should behave,
    /// enabling it to generate the correct optimized code.
    ///
    /// This is optional and only needed when using JIT compilation.
    /// </para>
    /// </remarks>
    public Dictionary<string, object>? OperationParams { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ComputationNode{T}"/> class.
    /// </summary>
    /// <param name="value">The value stored in this node.</param>
    /// <param name="requiresGradient">Whether this node requires gradient computation.</param>
    /// <param name="parents">The parent nodes that were used to compute this value.</param>
    /// <param name="backwardFunction">The function to compute gradients during backpropagation.</param>
    /// <param name="name">Optional name for this node.</param>
    /// <remarks>
    /// <para>
    /// Creates a new computation node with the specified properties. If requiresGradient
    /// is true and a backward function is provided, this node will participate in
    /// automatic differentiation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new node in the computation graph.
    ///
    /// When creating a node:
    /// - Provide the computed value
    /// - Specify if it needs gradients (usually true for parameters, false for constants)
    /// - List the parent nodes (inputs) if any
    /// - Provide a backward function if gradients are needed
    /// - Optionally give it a name for debugging
    /// </para>
    /// </remarks>
    public ComputationNode(
        Tensor<T> value,
        bool requiresGradient = false,
        List<ComputationNode<T>>? parents = null,
        Action<Tensor<T>>? backwardFunction = null,
        string? name = null)
    {
        Value = value;
        RequiresGradient = requiresGradient;
        Parents = parents ?? new List<ComputationNode<T>>();
        BackwardFunction = backwardFunction;
        Name = name;
        Gradient = null; // Initialized during backward pass
    }

    /// <summary>
    /// Performs backward propagation from this node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method triggers backpropagation through the computation graph starting
    /// from this node. It performs a topological sort to determine the correct order
    /// for computing gradients, then calls each node's backward function in reverse
    /// topological order.
    /// </para>
    /// <para><b>For Beginners:</b> This computes gradients for all nodes that led to this one.
    ///
    /// The backward process:
    /// 1. Starts with this node's gradient (usually set to 1 for the final loss)
    /// 2. Works backwards through the graph in the right order
    /// 3. Each node passes gradients to its parents
    /// 4. Gradients accumulate if a node has multiple children
    ///
    /// This is how neural networks learn - by computing gradients and using them
    /// to update parameters.
    /// </para>
    /// </remarks>
    public void Backward()
    {
        // Build topological order
        var topoOrder = TopologicalSort();

        // Clear all gradients in the topological order to ensure clean state
        // This is critical for multiple backward passes (persistent tapes, higher-order gradients)
        foreach (var node in topoOrder)
        {
            node.Gradient = null;
        }

        // Initialize root gradient to ones (for final node)
        Gradient = new Tensor<T>(Value.Shape);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < Gradient.Length; i++)
        {
            Gradient[i] = numOps.One;
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }
    }

    /// <summary>
    /// Performs a topological sort of the computation graph rooted at this node.
    /// </summary>
    /// <returns>A list of nodes in topological order.</returns>
    /// <remarks>
    /// <para>
    /// Topological sorting ensures that each node appears before all nodes that depend on it.
    /// This ordering is essential for correct gradient computation during backpropagation.
    /// The algorithm uses depth-first search (DFS) with cycle detection.
    /// </para>
    /// <para><b>For Beginners:</b> This determines the correct order to process nodes during backprop.
    ///
    /// Why ordering matters:
    /// - We need to compute gradients from outputs back to inputs
    /// - A node's gradient must be computed before we can compute its parents' gradients
    /// - Topological sort finds an order where this always works
    ///
    /// Think of it like following a recipe backwards - you need to know all the ways
    /// an ingredient was used before you can figure out how much you need of it.
    /// </para>
    /// </remarks>
    private List<ComputationNode<T>> TopologicalSort()
    {
        var visited = new HashSet<ComputationNode<T>>();
        var result = new List<ComputationNode<T>>();

        // Use iterative DFS with explicit stack to avoid stack overflow for deep graphs
        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
        stack.Push((this, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
            {
                continue;
            }

            if (processed)
            {
                // All parents have been visited, add to result
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                // Mark for processing after parents
                stack.Push((node, true));

                // Push parents onto stack (they will be processed first)
                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        stack.Push((parent, false));
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Zeros out the gradient for this node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the gradient to zero. It should be called between training
    /// iterations to prevent gradient accumulation across batches (unless intentional).
    /// </para>
    /// <para><b>For Beginners:</b> This clears the gradient to prepare for the next calculation.
    ///
    /// When to use:
    /// - Before each new training iteration
    /// - When you want to start fresh gradient computation
    /// - To prevent gradients from adding up across multiple backward passes
    ///
    /// In most training loops, you zero gradients at the start of each batch.
    /// </para>
    /// </remarks>
    public void ZeroGradient()
    {
        if (Gradient != null)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < Gradient.Length; i++)
            {
                Gradient[i] = numOps.Zero;
            }
        }
    }

    /// <summary>
    /// Recursively zeros out gradients for this node and all its ancestors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method traverses the computation graph and zeros all gradients.
    /// It's useful for clearing the entire graph between training iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This clears gradients for this node and everything it depends on.
    ///
    /// Use this when:
    /// - You want to reset the entire computation graph
    /// - Starting a new training iteration
    /// - You need to ensure no old gradients remain
    ///
    /// This is more thorough than ZeroGradient() as it clears the whole graph.
    /// </para>
    /// </remarks>
    public void ZeroGradientRecursive()
    {
        var visited = new HashSet<ComputationNode<T>>();

        void ZeroRecursive(ComputationNode<T> node)
        {
            if (visited.Contains(node))
            {
                return;
            }

            visited.Add(node);
            node.ZeroGradient();

            foreach (var parent in node.Parents)
            {
                ZeroRecursive(parent);
            }
        }

        ZeroRecursive(this);
    }
}
