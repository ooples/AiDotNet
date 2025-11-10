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
        var numOps = MathHelper.GetNumericOperations<T>();

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

    /// <summary>
    /// Performs element-wise division of two computation nodes.
    /// </summary>
    /// <param name="a">The numerator node.</param>
    /// <param name="b">The denominator node.</param>
    /// <returns>A new computation node containing the element-wise quotient.</returns>
    /// <remarks>
    /// <para>
    /// This method performs element-wise division and records the operation to any active GradientTape.
    /// The backward function uses the quotient rule: ∂(a/b)/∂a = 1/b and ∂(a/b)/∂b = -a/b².
    /// </para>
    /// <para><b>For Beginners:</b> This divides one tensor by another element-wise and tracks gradients.
    ///
    /// For element-wise division (c = a / b):
    /// - The forward pass divides corresponding elements
    /// - The backward pass uses the quotient rule from calculus
    /// - Gradient to 'a' is: incoming gradient * (1/b)
    /// - Gradient to 'b' is: incoming gradient * (-a/b²)
    ///
    /// Example:
    /// If a=[6,8], b=[2,4], c=[3,2]
    /// If gradient to c is [1,1]:
    /// - 'a' receives [1/2, 1/4] = [0.5, 0.25]
    /// - 'b' receives [-6/4, -8/16] = [-1.5, -0.5]
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Divide(ComputationNode<T> a, ComputationNode<T> b)
    {
        var result = a.Value.ElementwiseDivide(b.Value);
        var numOps = MathHelper.GetNumericOperations<T>();

        void BackwardFunction(Tensor<T> gradient)
        {
            // ∂(a/b)/∂a = 1/b
            if (a.RequiresGradient)
            {
                var gradA = gradient.ElementwiseDivide(b.Value);
                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }

            // ∂(a/b)/∂b = -a/b²
            if (b.RequiresGradient)
            {
                var bSquared = b.Value.ElementwiseMultiply(b.Value);
                var gradB = gradient.ElementwiseMultiply(a.Value).ElementwiseDivide(bSquared);
                gradB = gradB.Transform((x, _) => numOps.Negate(x));
                if (b.Gradient == null)
                    b.Gradient = gradB;
                else
                    b.Gradient = b.Gradient.Add(gradB);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Raises a computation node to a power.
    /// </summary>
    /// <param name="a">The base node.</param>
    /// <param name="exponent">The exponent value.</param>
    /// <returns>A new computation node containing the power operation result.</returns>
    /// <remarks>
    /// <para>
    /// This method raises each element to a power and records the operation.
    /// The backward function uses the power rule: ∂(a^n)/∂a = n * a^(n-1).
    /// </para>
    /// <para><b>For Beginners:</b> This raises a tensor to a power and tracks gradients.
    ///
    /// For power operation (c = a^n):
    /// - The forward pass raises each element to the power
    /// - The backward pass uses the power rule from calculus
    /// - Gradient to 'a' is: incoming gradient * n * a^(n-1)
    ///
    /// Example:
    /// If a=[2,3], n=2, c=[4,9]
    /// If gradient to c is [1,1]:
    /// - 'a' receives [1*2*2^1, 1*2*3^1] = [4, 6]
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Power(ComputationNode<T> a, double exponent)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var expValue = numOps.FromDouble(exponent);
        var result = a.Value.Transform((x, _) => numOps.Power(x, expValue));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(a^n)/∂a = n * a^(n-1)
                var expMinusOne = numOps.FromDouble(exponent - 1);
                var gradA = a.Value.Transform((x, _) =>
                {
                    var powered = numOps.Power(x, expMinusOne);
                    return numOps.Multiply(numOps.Multiply(expValue, powered), numOps.One);
                });
                gradA = gradA.ElementwiseMultiply(gradient);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the exponential function (e^x) for a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the exponential result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes e raised to each element and records the operation.
    /// The backward function uses: ∂(e^a)/∂a = e^a.
    /// </para>
    /// <para><b>For Beginners:</b> This computes e^x for each element and tracks gradients.
    ///
    /// For exponential (c = e^a):
    /// - The forward pass computes e^x for each element
    /// - The backward pass has a special property: the derivative equals the output!
    /// - Gradient to 'a' is: incoming gradient * e^a (which is just the output)
    ///
    /// This is used in softmax, sigmoid, and many activation functions.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Exp(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => numOps.Exp(x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(e^a)/∂a = e^a = result
                var gradA = gradient.ElementwiseMultiply(result);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the natural logarithm for a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the logarithm result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the natural logarithm of each element and records the operation.
    /// The backward function uses: ∂(log(a))/∂a = 1/a.
    /// </para>
    /// <para><b>For Beginners:</b> This computes the natural log and tracks gradients.
    ///
    /// For logarithm (c = log(a)):
    /// - The forward pass computes log for each element
    /// - The backward pass uses: gradient to 'a' is incoming gradient * (1/a)
    ///
    /// Logarithms are used in loss functions like cross-entropy.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Log(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => numOps.Log(x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(log(a))/∂a = 1/a
                var gradA = gradient.ElementwiseDivide(a.Value);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the square root for a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the square root result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the square root of each element and records the operation.
    /// The backward function uses: ∂(√a)/∂a = 1/(2√a).
    /// </para>
    /// <para><b>For Beginners:</b> This computes square root and tracks gradients.
    ///
    /// For square root (c = √a):
    /// - The forward pass computes √x for each element
    /// - The backward pass: gradient to 'a' is incoming gradient * 1/(2√a)
    /// - Which simplifies to: incoming gradient / (2 * output)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Sqrt(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => numOps.Sqrt(x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(√a)/∂a = 1/(2√a) = 1/(2*result)
                var two = numOps.FromDouble(2.0);
                var twoTimesResult = result.Transform((x, _) => numOps.Multiply(two, x));
                var gradA = gradient.ElementwiseDivide(twoTimesResult);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the hyperbolic tangent (tanh) for a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the tanh result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes tanh of each element and records the operation.
    /// The backward function uses: ∂(tanh(a))/∂a = 1 - tanh²(a).
    /// </para>
    /// <para><b>For Beginners:</b> Tanh is a common activation function in neural networks.
    ///
    /// For tanh (c = tanh(a)):
    /// - The forward pass computes tanh for each element (outputs between -1 and 1)
    /// - The backward pass: gradient to 'a' is incoming gradient * (1 - output²)
    ///
    /// Tanh is popular because it's centered around 0 (unlike sigmoid which is 0 to 1).
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Tanh(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => MathHelper.Tanh(x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(tanh(a))/∂a = 1 - tanh²(a) = 1 - result²
                var resultSquared = result.ElementwiseMultiply(result);
                var oneMinusSquared = resultSquared.Transform((x, _) => numOps.Subtract(numOps.One, x));
                var gradA = gradient.ElementwiseMultiply(oneMinusSquared);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the sigmoid function for a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the sigmoid result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes sigmoid (σ(x) = 1/(1+e^(-x))) and records the operation.
    /// The backward function uses: ∂σ(a)/∂a = σ(a) * (1 - σ(a)).
    /// </para>
    /// <para><b>For Beginners:</b> Sigmoid squashes values to be between 0 and 1.
    ///
    /// For sigmoid (c = σ(a)):
    /// - The forward pass computes 1/(1+e^(-x)) for each element
    /// - The backward pass: gradient to 'a' is incoming gradient * output * (1 - output)
    ///
    /// Sigmoid is used in binary classification and as a gate in LSTM networks.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Sigmoid(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => MathHelper.Sigmoid(x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂σ(a)/∂a = σ(a) * (1 - σ(a)) = result * (1 - result)
                var oneMinusResult = result.Transform((x, _) => numOps.Subtract(numOps.One, x));
                var derivative = result.ElementwiseMultiply(oneMinusResult);
                var gradA = gradient.ElementwiseMultiply(derivative);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the ReLU (Rectified Linear Unit) activation for a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the ReLU result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes ReLU (max(0, x)) and records the operation.
    /// The backward function uses: ∂ReLU(a)/∂a = 1 if a > 0, else 0.
    /// </para>
    /// <para><b>For Beginners:</b> ReLU is the most popular activation function in deep learning.
    ///
    /// For ReLU (c = max(0, a)):
    /// - The forward pass keeps positive values, zeros out negative values
    /// - The backward pass: gradient flows through if input was positive, blocked if negative
    ///
    /// ReLU is popular because:
    /// - Very fast to compute
    /// - Helps avoid vanishing gradients
    /// - Works well in practice for deep networks
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ReLU(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) =>
            numOps.GreaterThan(x, numOps.Zero) ? x : numOps.Zero);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂ReLU(a)/∂a = 1 if a > 0, else 0
                var mask = a.Value.Transform((x, _) =>
                    numOps.GreaterThan(x, numOps.Zero) ? numOps.One : numOps.Zero);
                var gradA = gradient.ElementwiseMultiply(mask);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Negates a computation node (computes -a).
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the negated result.</returns>
    /// <remarks>
    /// <para>
    /// This method negates each element and records the operation.
    /// The backward function simply negates the incoming gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This flips the sign of each element.
    ///
    /// For negation (c = -a):
    /// - The forward pass flips signs (positive becomes negative, vice versa)
    /// - The backward pass also flips the gradient sign
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Negate(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => numOps.Negate(x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(-a)/∂a = -1
                var gradA = gradient.Transform((x, _) => numOps.Negate(x));

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Performs matrix multiplication on two computation nodes.
    /// </summary>
    /// <param name="a">The left matrix (must be 2D).</param>
    /// <param name="b">The right matrix (must be 2D).</param>
    /// <returns>A computation node representing the matrix product.</returns>
    /// <remarks>
    /// <para>
    /// Computes C = A·B where A has shape [m, n] and B has shape [n, p], resulting in C with shape [m, p].
    /// </para>
    /// <para><b>Gradient computation:</b>
    /// - ∂(A·B)/∂A = gradOut·B^T
    /// - ∂(A·B)/∂B = A^T·gradOut
    /// </para>
    /// </remarks>
    public static ComputationNode<T> MatrixMultiply(ComputationNode<T> a, ComputationNode<T> b)
    {
        var result = a.Value.MatrixMultiply(b.Value);

        void BackwardFunction(Tensor<T> gradient)
        {
            // ∂(A·B)/∂A = gradOut·B^T
            if (a.RequiresGradient)
            {
                var bTransposed = b.Value.Transpose();
                var gradA = gradient.MatrixMultiply(bTransposed);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }

            // ∂(A·B)/∂B = A^T·gradOut
            if (b.RequiresGradient)
            {
                var aTransposed = a.Value.Transpose();
                var gradB = aTransposed.MatrixMultiply(gradient);

                if (b.Gradient == null)
                    b.Gradient = gradB;
                else
                    b.Gradient = b.Gradient.Add(gradB);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Transposes a 2D computation node (matrix).
    /// </summary>
    /// <param name="a">The matrix to transpose (must be 2D).</param>
    /// <returns>A computation node representing the transposed matrix.</returns>
    /// <remarks>
    /// <para>
    /// For a 2D tensor, swaps rows and columns: if A has shape [m, n], result has shape [n, m].
    /// </para>
    /// <para><b>Gradient computation:</b>
    /// - ∂(A^T)/∂A = gradOut^T (transpose the gradient back)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Transpose(ComputationNode<T> a)
    {
        var result = a.Value.Transpose();

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(A^T)/∂A = gradOut^T
                var gradA = gradient.Transpose();

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Sums elements of a computation node along specified axes.
    /// </summary>
    /// <param name="a">The computation node to sum.</param>
    /// <param name="axes">The axes along which to sum. If null, sums all elements.</param>
    /// <param name="keepDims">Whether to keep the reduced dimensions with size 1. Default is false.</param>
    /// <returns>A computation node representing the sum.</returns>
    /// <remarks>
    /// <para>
    /// Reduces the tensor by summing along specified axes.
    /// </para>
    /// <para><b>Gradient computation:</b>
    /// - The gradient is broadcast back to the original shape, as each element contributed equally to the sum.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Sum(ComputationNode<T> a, int[]? axes = null, bool keepDims = false)
    {
        var result = a.Value.Sum(axes);

        // Store original shape for gradient computation
        var originalShape = a.Value.Shape;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Gradient needs to be broadcast back to original shape
                // Each element in the original tensor contributed to the sum,
                // so each gets a copy of the gradient

                Tensor<T> gradA;
                if (axes == null || axes.Length == 0)
                {
                    // Summed all elements - broadcast scalar gradient to full shape
                    gradA = new Tensor<T>(originalShape);
                    var gradValue = gradient[0]; // Scalar result
                    for (int i = 0; i < gradA.Length; i++)
                    {
                        gradA[i] = gradValue;
                    }
                }
                else
                {
                    // Summed along specific axes - need to broadcast back
                    // For now, handle common case of summing over last axis
                    if (axes.Length == 1 && axes[0] == originalShape.Length - 1)
                    {
                        // Summing over last axis - repeat gradient along that axis
                        gradA = new Tensor<T>(originalShape);
                        int outerSize = gradient.Length;
                        int innerSize = originalShape[originalShape.Length - 1];

                        for (int i = 0; i < outerSize; i++)
                        {
                            for (int j = 0; j < innerSize; j++)
                            {
                                gradA[i * innerSize + j] = gradient[i];
                            }
                        }
                    }
                    else
                    {
                        // General case - reshape gradient and broadcast
                        gradA = new Tensor<T>(originalShape);
                        // Simple broadcast: copy gradient value to all positions
                        for (int i = 0; i < gradA.Length; i++)
                        {
                            // For general case, map back through reduction
                            gradA[i] = gradient[0];
                        }
                    }
                }

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the mean of elements in a computation node.
    /// </summary>
    /// <param name="a">The computation node to compute mean of.</param>
    /// <returns>A computation node representing the mean (scalar).</returns>
    /// <remarks>
    /// <para>
    /// Computes the average of all elements in the tensor.
    /// </para>
    /// <para><b>Gradient computation:</b>
    /// - ∂(mean(A))/∂A = gradOut / count
    /// - Each element gets an equal share of the gradient, divided by the total count.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Mean(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var mean = a.Value.Mean();
        var result = new Tensor<T>(new int[] { 1 });
        result[0] = mean;

        var originalShape = a.Value.Shape;
        var count = a.Value.Length;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(mean(A))/∂A = gradOut / count
                var gradA = new Tensor<T>(originalShape);
                var gradValue = numOps.Divide(gradient[0], numOps.FromInt(count));

                for (int i = 0; i < gradA.Length; i++)
                {
                    gradA[i] = gradValue;
                }

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Reshapes a computation node to a new shape.
    /// </summary>
    /// <param name="a">The computation node to reshape.</param>
    /// <param name="newShape">The new shape (must have same total number of elements).</param>
    /// <returns>A computation node with the new shape.</returns>
    /// <remarks>
    /// <para>
    /// Changes the shape of the tensor without changing the underlying data.
    /// The total number of elements must remain the same.
    /// </para>
    /// <para><b>Gradient computation:</b>
    /// - ∂(Reshape(A))/∂A = Reshape(gradOut, A.Shape)
    /// - Simply reshape the gradient back to the original shape.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Reshape(ComputationNode<T> a, params int[] newShape)
    {
        var result = a.Value.Reshape(newShape);
        var originalShape = a.Value.Shape;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(Reshape(A))/∂A = Reshape(gradOut, originalShape)
                var gradA = gradient.Reshape(originalShape);

                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }
}
