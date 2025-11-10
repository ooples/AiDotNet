namespace AiDotNet.Autodiff;

/// <summary>
/// Provides automatic differentiation support for tensor operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TensorOperations is a helper class that integrates automatic differentiation with tensor operations.
/// It records operations performed on tensors to an active GradientTape (if present) and creates
/// the computation graph needed for backpropagation.
/// </para>
/// <para>
/// This class follows the opt-in pattern: tensor operations only record to the gradient tape
/// when explicitly used within a GradientTape context. Outside of a GradientTape context,
/// operations work normally without any overhead.
/// </para>
/// <para><b>For Beginners:</b> This class bridges regular tensor operations with automatic differentiation.
///
/// Think of it like adding a "recording mode" to your calculations:
/// - When you're inside a GradientTape context, operations are recorded
/// - The recording remembers how each value was computed
/// - Later, you can "play it backwards" to compute gradients
/// - When not recording, operations work exactly as before
///
/// This enables features like:
/// - Automatic gradient computation for neural network training
/// - Computing derivatives without writing manual backward passes
/// - Building complex computational graphs automatically
///
/// Example usage:
/// <code>
/// using (var tape = new GradientTape&lt;double&gt;())
/// {
///     var x = TensorOperations&lt;double&gt;.Variable(inputTensor, "x");
///     var y = TensorOperations&lt;double&gt;.Variable(parameterTensor, "y");
///     tape.Watch(x);
///     tape.Watch(y);
///
///     var z = TensorOperations&lt;double&gt;.Add(x, y); // Recorded to tape
///     var gradients = tape.Gradient(z, new[] { x, y });
/// }
/// </code>
/// </para>
/// </remarks>
public static class TensorOperations<T>
{
    /// <summary>
    /// Creates a computation node from a tensor value.
    /// </summary>
    /// <param name="value">The tensor value.</param>
    /// <param name="name">Optional name for the node.</param>
    /// <param name="requiresGradient">Whether this node requires gradient computation.</param>
    /// <returns>A computation node wrapping the tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a leaf node in the computation graph - a node with no parents.
    /// Leaf nodes typically represent inputs or parameters that gradients will be computed with respect to.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a starting point in your calculation graph.
    ///
    /// Use this to wrap:
    /// - Model parameters (weights, biases) that need gradients
    /// - Input data that you want to compute gradients for
    /// - Constants (with requiresGradient=false)
    ///
    /// The returned ComputationNode tracks the tensor's value and will accumulate gradients
    /// during backpropagation.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Variable(
        Tensor<T> value,
        string? name = null,
        bool requiresGradient = true)
    {
        return new ComputationNode<T>(
            value: value,
            requiresGradient: requiresGradient,
            parents: null,
            backwardFunction: null,
            name: name);
    }

    /// <summary>
    /// Creates a constant computation node from a tensor value.
    /// </summary>
    /// <param name="value">The tensor value.</param>
    /// <param name="name">Optional name for the node.</param>
    /// <returns>A computation node that doesn't require gradients.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a constant node - a value that won't have gradients computed.
    /// Use this for constants, hyperparameters, or intermediate values you don't need gradients for.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a value that won't be adjusted during training.
    ///
    /// Use this for:
    /// - Constants (like pi, e, or fixed multipliers)
    /// - Hyperparameters that don't change during training
    /// - Any value you don't need gradients for (saves memory)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Constant(Tensor<T> value, string? name = null)
    {
        return Variable(value, name, requiresGradient: false);
    }

    /// <summary>
    /// Performs element-wise addition of two computation nodes.
    /// </summary>
    /// <param name="a">The first node.</param>
    /// <param name="b">The second node.</param>
    /// <returns>A new computation node containing the sum.</returns>
    /// <remarks>
    /// <para>
    /// This method performs element-wise addition and records the operation to any active GradientTape.
    /// The backward function distributes gradients equally to both inputs (since ∂(a+b)/∂a = 1 and ∂(a+b)/∂b = 1).
    /// </para>
    /// <para><b>For Beginners:</b> This adds two tensors together and remembers how to compute gradients.
    ///
    /// For addition (c = a + b):
    /// - The forward pass computes the sum element-wise
    /// - The backward pass sends gradients to both inputs unchanged
    /// - This is because changing 'a' by 1 changes the sum by 1, same for 'b'
    ///
    /// Example:
    /// If the gradient flowing back to c is [1, 2, 3], then both 'a' and 'b' receive [1, 2, 3]
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Add(ComputationNode<T> a, ComputationNode<T> b)
    {
        // Forward pass: compute the sum
        var result = a.Value.Add(b.Value);

        // Create backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            // Distribute gradient to both parents
            // d(a+b)/da = 1, so gradient flows unchanged to 'a'
            // d(a+b)/db = 1, so gradient flows unchanged to 'b'

            if (a.RequiresGradient)
            {
                if (a.Gradient == null)
                {
                    a.Gradient = gradient;
                }
                else
                {
                    // Accumulate gradients (for nodes used multiple times)
                    a.Gradient = a.Gradient.Add(gradient);
                }
            }

            if (b.RequiresGradient)
            {
                if (b.Gradient == null)
                {
                    b.Gradient = gradient;
                }
                else
                {
                    // Accumulate gradients (for nodes used multiple times)
                    b.Gradient = b.Gradient.Add(gradient);
                }
            }
        }

        // Create the result node
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        // Record to active tape if present
        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
        {
            tape.RecordOperation(node);
        }

        return node;
    }

    /// <summary>
    /// Performs element-wise subtraction of two computation nodes.
    /// </summary>
    /// <param name="a">The node to subtract from.</param>
    /// <param name="b">The node to subtract.</param>
    /// <returns>A new computation node containing the difference.</returns>
    /// <remarks>
    /// <para>
    /// This method performs element-wise subtraction and records the operation to any active GradientTape.
    /// The backward function sends gradient to 'a' unchanged and negated gradient to 'b'
    /// (since ∂(a-b)/∂a = 1 and ∂(a-b)/∂b = -1).
    /// </para>
    /// <para><b>For Beginners:</b> This subtracts one tensor from another and tracks gradients.
    ///
    /// For subtraction (c = a - b):
    /// - The forward pass computes a minus b element-wise
    /// - The backward pass sends the gradient to 'a' unchanged
    /// - But sends the *negative* gradient to 'b'
    /// - This is because increasing 'b' by 1 *decreases* the result by 1
    ///
    /// Example:
    /// If the gradient flowing to c is [1, 2, 3]:
    /// - 'a' receives [1, 2, 3]
    /// - 'b' receives [-1, -2, -3]
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Subtract(ComputationNode<T> a, ComputationNode<T> b)
    {
        // Forward pass: compute the difference
        var result = a.Value.ElementwiseSubtract(b.Value);
        var numOps = INumericOperations<T>.GetOperations();

        // Create backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            // d(a-b)/da = 1, so gradient flows unchanged to 'a'
            if (a.RequiresGradient)
            {
                if (a.Gradient == null)
                {
                    a.Gradient = gradient;
                }
                else
                {
                    a.Gradient = a.Gradient.Add(gradient);
                }
            }

            // d(a-b)/db = -1, so negative gradient flows to 'b'
            if (b.RequiresGradient)
            {
                var negGradient = gradient.Transform((x, _) => numOps.Negate(x));
                if (b.Gradient == null)
                {
                    b.Gradient = negGradient;
                }
                else
                {
                    b.Gradient = b.Gradient.Add(negGradient);
                }
            }
        }

        // Create the result node
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        // Record to active tape if present
        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
        {
            tape.RecordOperation(node);
        }

        return node;
    }

    /// <summary>
    /// Performs element-wise multiplication of two computation nodes.
    /// </summary>
    /// <param name="a">The first node.</param>
    /// <param name="b">The second node.</param>
    /// <returns>A new computation node containing the element-wise product.</returns>
    /// <remarks>
    /// <para>
    /// This method performs element-wise (Hadamard) multiplication and records the operation.
    /// The backward function uses the product rule: ∂(a*b)/∂a = b and ∂(a*b)/∂b = a.
    /// </para>
    /// <para><b>For Beginners:</b> This multiplies two tensors element-wise and tracks gradients.
    ///
    /// For element-wise multiplication (c = a * b):
    /// - The forward pass multiplies corresponding elements
    /// - The backward pass uses the product rule from calculus
    /// - Gradient to 'a' is: incoming gradient * b's value
    /// - Gradient to 'b' is: incoming gradient * a's value
    ///
    /// Example:
    /// If a=[2,3], b=[4,5], c=[8,15]
    /// If gradient to c is [1,1]:
    /// - 'a' receives [1*4, 1*5] = [4, 5]
    /// - 'b' receives [1*2, 1*3] = [2, 3]
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ElementwiseMultiply(ComputationNode<T> a, ComputationNode<T> b)
    {
        // Forward pass: element-wise multiplication
        var result = a.Value.ElementwiseMultiply(b.Value);

        // Create backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            // d(a*b)/da = b, so gradient * b flows to 'a'
            if (a.RequiresGradient)
            {
                var gradA = gradient.ElementwiseMultiply(b.Value);
                if (a.Gradient == null)
                {
                    a.Gradient = gradA;
                }
                else
                {
                    a.Gradient = a.Gradient.Add(gradA);
                }
            }

            // d(a*b)/db = a, so gradient * a flows to 'b'
            if (b.RequiresGradient)
            {
                var gradB = gradient.ElementwiseMultiply(a.Value);
                if (b.Gradient == null)
                {
                    b.Gradient = gradB;
                }
                else
                {
                    b.Gradient = b.Gradient.Add(gradB);
                }
            }
        }

        // Create the result node
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        // Record to active tape if present
        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
        {
            tape.RecordOperation(node);
        }

        return node;
    }
}
