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
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Value.Shape);
        for (int i = 0; i < a.Value.Length; i++)
        {
            result[i] = numOps.Divide(a.Value[i], b.Value[i]);
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            // ∂(a/b)/∂a = 1/b
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(gradient.Shape);
                for (int i = 0; i < gradient.Length; i++)
                {
                    gradA[i] = numOps.Divide(gradient[i], b.Value[i]);
                }
                if (a.Gradient == null)
                    a.Gradient = gradA;
                else
                    a.Gradient = a.Gradient.Add(gradA);
            }

            // ∂(a/b)/∂b = -a/b²
            if (b.RequiresGradient)
            {
                var bSquared = b.Value.ElementwiseMultiply(b.Value);
                var gradB = new Tensor<T>(gradient.Shape);
                for (int i = 0; i < gradient.Length; i++)
                {
                    var numerator = numOps.Multiply(gradient[i], a.Value[i]);
                    gradB[i] = numOps.Negate(numOps.Divide(numerator, bSquared[i]));
                }
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
                var gradA = new Tensor<T>(gradient.Shape);
                for (int i = 0; i < gradient.Length; i++)
                {
                    gradA[i] = numOps.Divide(gradient[i], a.Value[i]);
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
                var gradA = new Tensor<T>(gradient.Shape);
                for (int i = 0; i < gradient.Length; i++)
                {
                    var twoTimesResult = numOps.Multiply(two, result[i]);
                    gradA[i] = numOps.Divide(gradient[i], twoTimesResult);
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
                        // General case - proper gradient broadcasting for arbitrary axes
                        gradA = new Tensor<T>(originalShape);

                        // Create reduced shape (original shape with summed dimensions set to 1)
                        int[] reducedShape = new int[originalShape.Length];
                        for (int d = 0; d < originalShape.Length; d++)
                        {
                            reducedShape[d] = axes.Contains(d) ? 1 : originalShape[d];
                        }

                        // Compute strides for original shape
                        int[] strides = new int[originalShape.Length];
                        strides[originalShape.Length - 1] = 1;
                        for (int d = originalShape.Length - 2; d >= 0; d--)
                        {
                            strides[d] = strides[d + 1] * originalShape[d + 1];
                        }

                        // Compute strides for reduced shape
                        int[] reducedStrides = new int[reducedShape.Length];
                        reducedStrides[reducedShape.Length - 1] = 1;
                        for (int d = reducedShape.Length - 2; d >= 0; d--)
                        {
                            reducedStrides[d] = reducedStrides[d + 1] * reducedShape[d + 1];
                        }

                        // Broadcast gradient back to original shape
                        for (int i = 0; i < gradA.Length; i++)
                        {
                            // Compute multi-dimensional index for original tensor
                            int[] indices = new int[originalShape.Length];
                            int remaining = i;
                            for (int d = 0; d < originalShape.Length; d++)
                            {
                                indices[d] = remaining / strides[d];
                                remaining %= strides[d];
                            }

                            // Map to reduced index by setting summed dimensions to 0
                            int reducedIndex = 0;
                            for (int d = 0; d < originalShape.Length; d++)
                            {
                                int idx = axes.Contains(d) ? 0 : indices[d];
                                reducedIndex += idx * reducedStrides[d];
                            }

                            gradA[i] = gradient[reducedIndex];
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
                var gradValue = numOps.Divide(gradient[0], numOps.FromDouble(count));

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

    /// <summary>
    /// Computes the softmax function for a computation node along a specified axis.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <param name="axis">The axis along which to compute softmax. Default is -1 (last axis).</param>
    /// <returns>A new computation node containing the softmax result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes softmax (σ(x_i) = exp(x_i) / Σexp(x_j)) along the specified axis.
    /// Uses numerical stability trick: subtract max before exponentiating.
    /// The backward function uses: ∂softmax/∂x = softmax(x) * (grad - Σ(grad * softmax(x))).
    /// </para>
    /// <para><b>For Beginners:</b> Softmax converts a vector of numbers into probabilities.
    ///
    /// For softmax:
    /// - The forward pass exponentiates each element, then normalizes so they sum to 1
    /// - The result is a probability distribution (all values between 0 and 1, summing to 1)
    /// - The backward pass is complex but efficient: uses the Jacobian of softmax
    ///
    /// Softmax is crucial for:
    /// - Multi-class classification (final layer outputs)
    /// - Attention mechanisms (computing attention weights)
    /// - Anywhere you need to convert scores to probabilities
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Softmax(ComputationNode<T> a, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        // Normalize axis to positive index
        if (axis < 0)
            axis = shape.Length + axis;

        // For simplicity, handle 2D case (batch, features) with axis=-1
        if (shape.Length == 2 && axis == 1)
        {
            int batchSize = shape[0];
            int features = shape[1];
            var result = new Tensor<T>(shape);

            // Compute softmax for each row
            for (int b = 0; b < batchSize; b++)
            {
                // Find max for numerical stability
                var maxVal = a.Value[b, 0];
                for (int f = 1; f < features; f++)
                {
                    if (numOps.GreaterThan(a.Value[b, f], maxVal))
                        maxVal = a.Value[b, f];
                }

                // Compute exp(x - max) and sum
                var expSum = numOps.Zero;
                var expValues = new T[features];
                for (int f = 0; f < features; f++)
                {
                    var shifted = numOps.Subtract(a.Value[b, f], maxVal);
                    expValues[f] = numOps.Exp(shifted);
                    expSum = numOps.Add(expSum, expValues[f]);
                }

                // Normalize
                for (int f = 0; f < features; f++)
                {
                    result[b, f] = numOps.Divide(expValues[f], expSum);
                }
            }

            void BackwardFunction(Tensor<T> gradient)
            {
                if (a.RequiresGradient)
                {
                    // ∂softmax/∂x_i = softmax_i * (∂L/∂y_i - Σ_j(∂L/∂y_j * softmax_j))
                    var gradA = new Tensor<T>(shape);

                    for (int b = 0; b < batchSize; b++)
                    {
                        // Compute sum of (gradient * softmax)
                        var dotProduct = numOps.Zero;
                        for (int f = 0; f < features; f++)
                        {
                            dotProduct = numOps.Add(dotProduct,
                                numOps.Multiply(gradient[b, f], result[b, f]));
                        }

                        // Compute gradient for each element
                        for (int f = 0; f < features; f++)
                        {
                            var gradMinusDot = numOps.Subtract(gradient[b, f], dotProduct);
                            gradA[b, f] = numOps.Multiply(result[b, f], gradMinusDot);
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
        else
        {
            throw new NotImplementedException(
                $"Softmax is currently only implemented for 2D tensors along axis=-1. " +
                $"Got shape=[{string.Join(", ", shape)}], axis={axis}");
        }
    }

    /// <summary>
    /// Concatenates multiple computation nodes along a specified axis.
    /// </summary>
    /// <param name="nodes">The list of nodes to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate. Default is 0.</param>
    /// <returns>A new computation node containing the concatenated result.</returns>
    /// <remarks>
    /// <para>
    /// This method concatenates tensors along the specified axis.
    /// All tensors must have the same shape except along the concatenation axis.
    /// The backward function splits the gradient and sends each portion to the corresponding input.
    /// </para>
    /// <para><b>For Beginners:</b> Concat stacks tensors together along a dimension.
    ///
    /// For concatenation:
    /// - The forward pass combines multiple tensors into one larger tensor
    /// - The backward pass splits the gradient back to each input
    /// - Think of it like gluing arrays together end-to-end
    ///
    /// Used in:
    /// - Skip connections (concatenating features from different layers)
    /// - Multi-input architectures
    /// - Feature fusion in neural networks
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Concat(List<ComputationNode<T>> nodes, int axis = 0)
    {
        if (nodes.Count == 0)
            throw new ArgumentException("Cannot concatenate empty list of nodes");

        var numOps = MathHelper.GetNumericOperations<T>();
        var firstShape = nodes[0].Value.Shape;

        // Normalize axis
        if (axis < 0)
            axis = firstShape.Length + axis;

        // Validate shapes match except on concat axis
        for (int i = 1; i < nodes.Count; i++)
        {
            var shape = nodes[i].Value.Shape;
            if (shape.Length != firstShape.Length)
                throw new ArgumentException("All tensors must have the same rank");

            for (int d = 0; d < firstShape.Length; d++)
            {
                if (d != axis && shape[d] != firstShape[d])
                    throw new ArgumentException(
                        $"Shape mismatch at dimension {d}: {shape[d]} vs {firstShape[d]}");
            }
        }

        // Compute output shape
        int[] outputShape = (int[])firstShape.Clone();
        for (int i = 1; i < nodes.Count; i++)
        {
            outputShape[axis] += nodes[i].Value.Shape[axis];
        }

        // Perform concatenation (handle 2D case for simplicity)
        Tensor<T> result;
        if (firstShape.Length == 2 && axis == 1)
        {
            // Concatenate along columns (features)
            int rows = firstShape[0];
            int totalCols = outputShape[1];
            result = new Tensor<T>(new int[] { rows, totalCols });

            int colOffset = 0;
            foreach (var node in nodes)
            {
                int cols = node.Value.Shape[1];
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        result[r, colOffset + c] = node.Value[r, c];
                    }
                }
                colOffset += cols;
            }
        }
        else if (firstShape.Length == 2 && axis == 0)
        {
            // Concatenate along rows (batch)
            int cols = firstShape[1];
            int totalRows = outputShape[0];
            result = new Tensor<T>(new int[] { totalRows, cols });

            int rowOffset = 0;
            foreach (var node in nodes)
            {
                int rows = node.Value.Shape[0];
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        result[rowOffset + r, c] = node.Value[r, c];
                    }
                }
                rowOffset += rows;
            }
        }
        else
        {
            throw new NotImplementedException(
                $"Concat is currently only implemented for 2D tensors. " +
                $"Got shape=[{string.Join(", ", firstShape)}]");
        }

        // Store sizes for gradient splitting
        var sizes = nodes.Select(n => n.Value.Shape[axis]).ToList();

        void BackwardFunction(Tensor<T> gradient)
        {
            // Split gradient along concat axis and distribute to inputs
            if (firstShape.Length == 2 && axis == 1)
            {
                int rows = firstShape[0];
                int colOffset = 0;
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (!nodes[i].RequiresGradient)
                    {
                        colOffset += sizes[i];
                        continue;
                    }

                    int cols = sizes[i];
                    var gradPart = new Tensor<T>(new int[] { rows, cols });

                    for (int r = 0; r < rows; r++)
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            gradPart[r, c] = gradient[r, colOffset + c];
                        }
                    }

                    if (nodes[i].Gradient == null)
                        nodes[i].Gradient = gradPart;
                    else
                        nodes[i].Gradient = nodes[i].Gradient.Add(gradPart);

                    colOffset += cols;
                }
            }
            else if (firstShape.Length == 2 && axis == 0)
            {
                int cols = firstShape[1];
                int rowOffset = 0;
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (!nodes[i].RequiresGradient)
                    {
                        rowOffset += sizes[i];
                        continue;
                    }

                    int rows = sizes[i];
                    var gradPart = new Tensor<T>(new int[] { rows, cols });

                    for (int r = 0; r < rows; r++)
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            gradPart[r, c] = gradient[rowOffset + r, c];
                        }
                    }

                    if (nodes[i].Gradient == null)
                        nodes[i].Gradient = gradPart;
                    else
                        nodes[i].Gradient = nodes[i].Gradient.Add(gradPart);

                    rowOffset += rows;
                }
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: nodes.Any(n => n.RequiresGradient),
            parents: nodes,
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Pads a tensor with a constant value along specified dimensions.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <param name="padWidth">Padding width for each dimension as (before, after) pairs.</param>
    /// <param name="value">The value to use for padding. Default is zero.</param>
    /// <returns>A new computation node containing the padded result.</returns>
    /// <remarks>
    /// <para>
    /// This method adds padding around the tensor.
    /// The backward function simply crops the gradient back to the original size (gradients for padding are zero).
    /// </para>
    /// <para><b>For Beginners:</b> Pad adds extra elements around a tensor.
    ///
    /// For padding:
    /// - The forward pass adds border elements with a constant value
    /// - The backward pass removes those border gradients (they don't affect the original tensor)
    /// - Think of it like adding margins to an image
    ///
    /// Used in:
    /// - Convolutional layers (to maintain spatial dimensions)
    /// - Handling variable-length sequences
    /// - Data augmentation
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Pad(ComputationNode<T> a, int[,] padWidth, T? value = default)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var padValue = value ?? numOps.Zero;
        var shape = a.Value.Shape;

        // Validate padWidth dimensions
        if (padWidth.GetLength(0) != shape.Length)
            throw new ArgumentException("padWidth must have same number of dimensions as input tensor");

        // Compute output shape
        var outputShape = new int[shape.Length];
        for (int d = 0; d < shape.Length; d++)
        {
            outputShape[d] = shape[d] + padWidth[d, 0] + padWidth[d, 1];
        }

        // Handle 2D case
        if (shape.Length == 2)
        {
            int inputRows = shape[0];
            int inputCols = shape[1];
            int padTop = padWidth[0, 0];
            int padBottom = padWidth[0, 1];
            int padLeft = padWidth[1, 0];
            int padRight = padWidth[1, 1];

            var result = new Tensor<T>(outputShape);

            // Initialize with pad value
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = padValue;
            }

            // Copy input data to center
            for (int r = 0; r < inputRows; r++)
            {
                for (int c = 0; c < inputCols; c++)
                {
                    result[padTop + r, padLeft + c] = a.Value[r, c];
                }
            }

            void BackwardFunction(Tensor<T> gradient)
            {
                if (a.RequiresGradient)
                {
                    // Extract gradient for original (unpadded) region
                    var gradA = new Tensor<T>(shape);

                    for (int r = 0; r < inputRows; r++)
                    {
                        for (int c = 0; c < inputCols; c++)
                        {
                            gradA[r, c] = gradient[padTop + r, padLeft + c];
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
        else
        {
            throw new NotImplementedException(
                $"Pad is currently only implemented for 2D tensors. " +
                $"Got shape=[{string.Join(", ", shape)}]");
        }
    }

    /// <summary>
    /// Performs 2D max pooling on a 4D tensor (batch, channels, height, width).
    /// </summary>
    /// <param name="a">The input node with shape [batch, channels, height, width].</param>
    /// <param name="poolSize">The size of the pooling window [poolH, poolW].</param>
    /// <param name="strides">The stride for the pooling operation [strideH, strideW]. If null, uses poolSize.</param>
    /// <returns>A new computation node containing the max pooled result.</returns>
    /// <remarks>
    /// <para>
    /// This method performs max pooling over 2D spatial dimensions.
    /// During forward pass, it tracks which element was the max for routing gradients during backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> MaxPool downsamples by taking the maximum value in each window.
    ///
    /// For max pooling:
    /// - The forward pass slides a window and takes the max value in each position
    /// - This reduces spatial dimensions (downsampling)
    /// - The backward pass routes gradients only to the positions that were max
    /// - Other positions get zero gradient (they didn't contribute to the output)
    ///
    /// Used in:
    /// - CNNs for translation invariance
    /// - Reducing spatial resolution
    /// - Building hierarchical features
    /// </para>
    /// </remarks>
    public static ComputationNode<T> MaxPool2D(
        ComputationNode<T> a,
        int[] poolSize,
        int[]? strides = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (shape.Length != 4)
            throw new ArgumentException("MaxPool2D requires 4D input [batch, channels, height, width]");

        strides ??= poolSize;

        int batch = shape[0];
        int channels = shape[1];
        int inH = shape[2];
        int inW = shape[3];
        int poolH = poolSize[0];
        int poolW = poolSize[1];
        int strideH = strides[0];
        int strideW = strides[1];

        int outH = (inH - poolH) / strideH + 1;
        int outW = (inW - poolW) / strideW + 1;

        var result = new Tensor<T>(new int[] { batch, channels, outH, outW });
        // Store max positions for backprop
        var maxPositions = new int[batch, channels, outH, outW, 2]; // [h_offset, w_offset]

        // Forward pass: compute max pooling and track positions
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int hStart = oh * strideH;
                        int wStart = ow * strideW;

                        var maxVal = a.Value[b * channels * inH * inW +
                                           c * inH * inW +
                                           hStart * inW +
                                           wStart];
                        int maxHOffset = 0;
                        int maxWOffset = 0;

                        // Find max in pooling window
                        for (int ph = 0; ph < poolH; ph++)
                        {
                            for (int pw = 0; pw < poolW; pw++)
                            {
                                int h = hStart + ph;
                                int w = wStart + pw;
                                if (h < inH && w < inW)
                                {
                                    var val = a.Value[b * channels * inH * inW +
                                                     c * inH * inW +
                                                     h * inW +
                                                     w];
                                    if (numOps.GreaterThan(val, maxVal))
                                    {
                                        maxVal = val;
                                        maxHOffset = ph;
                                        maxWOffset = pw;
                                    }
                                }
                            }
                        }

                        result[b, c, oh, ow] = maxVal;
                        maxPositions[b, c, oh, ow, 0] = maxHOffset;
                        maxPositions[b, c, oh, ow, 1] = maxWOffset;
                    }
                }
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(shape);

                // Route gradients to max positions
                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                int hStart = oh * strideH;
                                int wStart = ow * strideW;
                                int maxHOffset = maxPositions[b, c, oh, ow, 0];
                                int maxWOffset = maxPositions[b, c, oh, ow, 1];

                                int maxH = hStart + maxHOffset;
                                int maxW = wStart + maxWOffset;

                                int gradIdx = b * channels * inH * inW +
                                             c * inH * inW +
                                            maxH * inW +
                                             maxW;

                                gradA[gradIdx] = numOps.Add(gradA[gradIdx], gradient[b, c, oh, ow]);
                            }
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
    /// Performs 2D average pooling on a 4D tensor (batch, channels, height, width).
    /// </summary>
    /// <param name="a">The input node with shape [batch, channels, height, width].</param>
    /// <param name="poolSize">The size of the pooling window [poolH, poolW].</param>
    /// <param name="strides">The stride for the pooling operation [strideH, strideW]. If null, uses poolSize.</param>
    /// <returns>A new computation node containing the average pooled result.</returns>
    /// <remarks>
    /// <para>
    /// This method performs average pooling over 2D spatial dimensions.
    /// The backward function distributes gradients equally across the pooling window.
    /// </para>
    /// <para><b>For Beginners:</b> AvgPool downsamples by taking the average value in each window.
    ///
    /// For average pooling:
    /// - The forward pass slides a window and computes the average
    /// - This smoothly reduces spatial dimensions
    /// - The backward pass distributes gradients equally to all elements in the window
    /// - Each element gets gradient / pool_area
    ///
    /// Used in:
    /// - CNNs for smoother downsampling than max pooling
    /// - Global average pooling (replacing fully connected layers)
    /// - Reducing overfitting
    /// </para>
    /// </remarks>
    public static ComputationNode<T> AvgPool2D(
        ComputationNode<T> a,
        int[] poolSize,
        int[]? strides = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (shape.Length != 4)
            throw new ArgumentException("AvgPool2D requires 4D input [batch, channels, height, width]");

        strides ??= poolSize;

        int batch = shape[0];
        int channels = shape[1];
        int inH = shape[2];
        int inW = shape[3];
        int poolH = poolSize[0];
        int poolW = poolSize[1];
        int strideH = strides[0];
        int strideW = strides[1];

        int outH = (inH - poolH) / strideH + 1;
        int outW = (inW - poolW) / strideW + 1;

        var result = new Tensor<T>(new int[] { batch, channels, outH, outW });
        var poolArea = numOps.FromDouble(poolH * poolW);

        // Forward pass: compute average pooling
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int hStart = oh * strideH;
                        int wStart = ow * strideW;

                        var sum = numOps.Zero;

                        // Sum values in pooling window
                        for (int ph = 0; ph < poolH; ph++)
                        {
                            for (int pw = 0; pw < poolW; pw++)
                            {
                                int h = hStart + ph;
                                int w = wStart + pw;
                                if (h < inH && w < inW)
                                {
                                    sum = numOps.Add(sum, a.Value[b, c, h, w]);
                                }
                            }
                        }

                        result[b, c, oh, ow] = numOps.Divide(sum, poolArea);
                    }
                }
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(shape);

                // Distribute gradients equally across pooling windows
                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                int hStart = oh * strideH;
                                int wStart = ow * strideW;

                                var gradValue = numOps.Divide(gradient[b, c, oh, ow], poolArea);

                                // Distribute to all elements in window
                                for (int ph = 0; ph < poolH; ph++)
                                {
                                    for (int pw = 0; pw < poolW; pw++)
                                    {
                                        int h = hStart + ph;
                                        int w = wStart + pw;
                                        if (h < inH && w < inW)
                                        {
                                            gradA[b, c, h, w] = numOps.Add(gradA[b, c, h, w], gradValue);
                                        }
                                    }
                                }
                            }
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
    /// Applies layer normalization to a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <param name="normalizedShape">The shape over which to normalize (typically the feature dimensions).</param>
    /// <param name="gamma">Optional scale parameter (learnable). If null, uses ones.</param>
    /// <param name="beta">Optional shift parameter (learnable). If null, uses zeros.</param>
    /// <param name="epsilon">Small constant for numerical stability. Default is 1e-5.</param>
    /// <returns>A new computation node containing the layer normalized result.</returns>
    /// <remarks>
    /// <para>
    /// Layer normalization normalizes inputs across the feature dimension for each sample independently.
    /// Formula: y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
    /// Unlike batch normalization, this doesn't depend on batch statistics.
    /// </para>
    /// <para><b>For Beginners:</b> LayerNorm standardizes features for each sample independently.
    ///
    /// For layer normalization:
    /// - Computes mean and variance for each sample's features
    /// - Normalizes: (x - mean) / sqrt(variance)
    /// - Scales and shifts: result * gamma + beta
    /// - Works the same during training and inference (no batch dependency)
    ///
    /// Used in:
    /// - Transformers (critical component)
    /// - RNNs (stabilizes training)
    /// - Any architecture needing sample-independent normalization
    /// </para>
    /// </remarks>
    public static ComputationNode<T> LayerNorm(
        ComputationNode<T> a,
        int[] normalizedShape,
        ComputationNode<T>? gamma = null,
        ComputationNode<T>? beta = null,
        double epsilon = 1e-5)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;
        var eps = numOps.FromDouble(epsilon);

        // For 2D input [batch, features], normalize over features
        if (shape.Length == 2 && normalizedShape.Length == 1 && normalizedShape[0] == shape[1])
        {
            int batchSize = shape[0];
            int features = shape[1];

            // Create default gamma (ones) and beta (zeros) if not provided
            if (gamma == null)
            {
                var gammaTensor = new Tensor<T>(new int[] { features });
                for (int i = 0; i < features; i++)
                    gammaTensor[i] = numOps.One;
                gamma = Variable(gammaTensor, requiresGradient: false);
            }

            if (beta == null)
            {
                var betaTensor = new Tensor<T>(new int[] { features });
                for (int i = 0; i < features; i++)
                    betaTensor[i] = numOps.Zero;
                beta = Variable(betaTensor, requiresGradient: false);
            }

            var result = new Tensor<T>(shape);
            var means = new T[batchSize];
            var variances = new T[batchSize];
            var normalized = new Tensor<T>(shape);

            // Forward pass: compute mean and variance per sample
            for (int b = 0; b < batchSize; b++)
            {
                // Compute mean
                var sum = numOps.Zero;
                for (int f = 0; f < features; f++)
                {
                    sum = numOps.Add(sum, a.Value[b, f]);
                }
                means[b] = numOps.Divide(sum, numOps.FromDouble(features));

                // Compute variance
                var varSum = numOps.Zero;
                for (int f = 0; f < features; f++)
                {
                    var diff = numOps.Subtract(a.Value[b, f], means[b]);
                    varSum = numOps.Add(varSum, numOps.Multiply(diff, diff));
                }
                variances[b] = numOps.Divide(varSum, numOps.FromDouble(features));

                // Normalize and scale
                var std = numOps.Sqrt(numOps.Add(variances[b], eps));
                for (int f = 0; f < features; f++)
                {
                    var norm = numOps.Divide(
                        numOps.Subtract(a.Value[b, f], means[b]),
                        std);
                    normalized[b, f] = norm;
                    result[b, f] = numOps.Add(
                        numOps.Multiply(norm, gamma.Value[f]),
                        beta.Value[f]);
                }
            }

            void BackwardFunction(Tensor<T> gradient)
            {
                // Gradients for gamma and beta
                if (gamma.RequiresGradient)
                {
                    var gradGamma = new Tensor<T>(new int[] { features });
                    for (int f = 0; f < features; f++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            sum = numOps.Add(sum,
                                numOps.Multiply(gradient[b, f], normalized[b, f]));
                        }
                        gradGamma[f] = sum;
                    }

                    if (gamma.Gradient == null)
                        gamma.Gradient = gradGamma;
                    else
                        gamma.Gradient = gamma.Gradient.Add(gradGamma);
                }

                if (beta.RequiresGradient)
                {
                    var gradBeta = new Tensor<T>(new int[] { features });
                    for (int f = 0; f < features; f++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            sum = numOps.Add(sum, gradient[b, f]);
                        }
                        gradBeta[f] = sum;
                    }

                    if (beta.Gradient == null)
                        beta.Gradient = gradBeta;
                    else
                        beta.Gradient = beta.Gradient.Add(gradBeta);
                }

                // Gradient for input
                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(shape);

                    for (int b = 0; b < batchSize; b++)
                    {
                        var std = numOps.Sqrt(numOps.Add(variances[b], eps));
                        var invStd = numOps.Divide(numOps.One, std);

                        // Compute gradient components
                        var gradNormSum = numOps.Zero;
                        var gradNormDotNorm = numOps.Zero;

                        for (int f = 0; f < features; f++)
                        {
                            var gradNorm = numOps.Multiply(gradient[b, f], gamma.Value[f]);
                            gradNormSum = numOps.Add(gradNormSum, gradNorm);
                            gradNormDotNorm = numOps.Add(gradNormDotNorm,
                                numOps.Multiply(gradNorm, normalized[b, f]));
                        }

                        // Apply gradient formula
                        var featuresT = numOps.FromDouble(features);
                        for (int f = 0; f < features; f++)
                        {
                            var gradNorm = numOps.Multiply(gradient[b, f], gamma.Value[f]);

                            var term1 = gradNorm;
                            var term2 = numOps.Divide(gradNormSum, featuresT);
                            var term3 = numOps.Divide(
                                numOps.Multiply(normalized[b, f], gradNormDotNorm),
                                featuresT);

                            var grad = numOps.Multiply(
                                numOps.Subtract(numOps.Subtract(term1, term2), term3),
                                invStd);

                            gradA[b, f] = grad;
                        }
                    }

                    if (a.Gradient == null)
                        a.Gradient = gradA;
                    else
                        a.Gradient = a.Gradient.Add(gradA);
                }
            }

            var parents = new List<ComputationNode<T>> { a };
            if (gamma != null) parents.Add(gamma);
            if (beta != null) parents.Add(beta);

            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || (gamma?.RequiresGradient ?? false) || (beta?.RequiresGradient ?? false),
                parents: parents,
                backwardFunction: BackwardFunction,
                name: null);

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);

            return node;
        }
        else
        {
            throw new NotImplementedException(
                $"LayerNorm is currently only implemented for 2D tensors normalizing over last dimension. " +
                $"Got shape=[{string.Join(", ", shape)}], normalizedShape=[{string.Join(", ", normalizedShape)}]");
        }
    }

    /// <summary>
    /// Applies batch normalization to a computation node.
    /// </summary>
    /// <param name="a">The input node with shape [batch, features].</param>
    /// <param name="gamma">Optional scale parameter (learnable). If null, uses ones.</param>
    /// <param name="beta">Optional shift parameter (learnable). If null, uses zeros.</param>
    /// <param name="runningMean">Running mean for inference (not updated during this operation).</param>
    /// <param name="runningVar">Running variance for inference (not updated during this operation).</param>
    /// <param name="training">Whether in training mode (uses batch statistics) or inference mode (uses running statistics).</param>
    /// <param name="epsilon">Small constant for numerical stability. Default is 1e-5.</param>
    /// <returns>A new computation node containing the batch normalized result.</returns>
    /// <remarks>
    /// <para>
    /// Batch normalization normalizes inputs across the batch dimension.
    /// During training: Uses batch statistics (mean and variance computed from current batch).
    /// During inference: Uses running statistics (accumulated during training).
    /// </para>
    /// <para><b>For Beginners:</b> BatchNorm standardizes features across the batch.
    ///
    /// For batch normalization:
    /// - Training mode: Uses current batch's mean and variance
    /// - Inference mode: Uses running mean/variance from training
    /// - Normalizes: (x - mean) / sqrt(variance)
    /// - Scales and shifts: result * gamma + beta
    ///
    /// Benefits:
    /// - Stabilizes training (reduces internal covariate shift)
    /// - Allows higher learning rates
    /// - Acts as regularization
    ///
    /// Used in:
    /// - CNNs (after convolutional layers)
    /// - Deep feedforward networks
    /// - GANs and many other architectures
    /// </para>
    /// </remarks>
    public static ComputationNode<T> BatchNorm(
        ComputationNode<T> a,
        ComputationNode<T>? gamma = null,
        ComputationNode<T>? beta = null,
        Tensor<T>? runningMean = null,
        Tensor<T>? runningVar = null,
        bool training = true,
        double epsilon = 1e-5)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;
        var eps = numOps.FromDouble(epsilon);

        // Handle 2D case [batch, features]
        if (shape.Length == 2)
        {
            int batchSize = shape[0];
            int features = shape[1];

            // Create default gamma and beta if not provided
            if (gamma == null)
            {
                var gammaTensor = new Tensor<T>(new int[] { features });
                for (int i = 0; i < features; i++)
                    gammaTensor[i] = numOps.One;
                gamma = Variable(gammaTensor, requiresGradient: false);
            }

            if (beta == null)
            {
                var betaTensor = new Tensor<T>(new int[] { features });
                for (int i = 0; i < features; i++)
                    betaTensor[i] = numOps.Zero;
                beta = Variable(betaTensor, requiresGradient: false);
            }

            var result = new Tensor<T>(shape);
            T[] batchMean;
            T[] batchVar;
            var normalized = new Tensor<T>(shape);

            if (training)
            {
                // Compute batch statistics
                batchMean = new T[features];
                batchVar = new T[features];

                // Compute mean per feature
                for (int f = 0; f < features; f++)
                {
                    var sum = numOps.Zero;
                    for (int b = 0; b < batchSize; b++)
                    {
                        sum = numOps.Add(sum, a.Value[b, f]);
                    }
                    batchMean[f] = numOps.Divide(sum, numOps.FromDouble(batchSize));
                }

                // Compute variance per feature
                for (int f = 0; f < features; f++)
                {
                    var varSum = numOps.Zero;
                    for (int b = 0; b < batchSize; b++)
                    {
                        var diff = numOps.Subtract(a.Value[b, f], batchMean[f]);
                        varSum = numOps.Add(varSum, numOps.Multiply(diff, diff));
                    }
                    batchVar[f] = numOps.Divide(varSum, numOps.FromDouble(batchSize));
                }
            }
            else
            {
                // Use running statistics for inference
                if (runningMean == null || runningVar == null)
                    throw new ArgumentException("Running statistics required for inference mode");

                batchMean = new T[features];
                batchVar = new T[features];
                for (int f = 0; f < features; f++)
                {
                    batchMean[f] = runningMean[f];
                    batchVar[f] = runningVar[f];
                }
            }

            // Normalize and scale
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    var std = numOps.Sqrt(numOps.Add(batchVar[f], eps));
                    var norm = numOps.Divide(
                        numOps.Subtract(a.Value[b, f], batchMean[f]),
                        std);
                    normalized[b, f] = norm;
                    result[b, f] = numOps.Add(
                        numOps.Multiply(norm, gamma.Value[f]),
                        beta.Value[f]);
                }
            }

            void BackwardFunction(Tensor<T> gradient)
            {
                if (!training)
                {
                    // Inference mode: simpler gradient (no batch statistics gradient)
                    if (a.RequiresGradient)
                    {
                        var gradA = new Tensor<T>(shape);
                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int f = 0; f < features; f++)
                            {
                                var std = numOps.Sqrt(numOps.Add(batchVar[f], eps));
                                var invStd = numOps.Divide(numOps.One, std);
                                gradA[b, f] = numOps.Multiply(
                                    numOps.Multiply(gradient[b, f], gamma.Value[f]),
                                    invStd);
                            }
                        }

                        if (a.Gradient == null)
                            a.Gradient = gradA;
                        else
                            a.Gradient = a.Gradient.Add(gradA);
                    }
                    return;
                }

                // Training mode: full gradient computation
                // Gradients for gamma and beta
                if (gamma.RequiresGradient)
                {
                    var gradGamma = new Tensor<T>(new int[] { features });
                    for (int f = 0; f < features; f++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            sum = numOps.Add(sum,
                                numOps.Multiply(gradient[b, f], normalized[b, f]));
                        }
                        gradGamma[f] = sum;
                    }

                    if (gamma.Gradient == null)
                        gamma.Gradient = gradGamma;
                    else
                        gamma.Gradient = gamma.Gradient.Add(gradGamma);
                }

                if (beta.RequiresGradient)
                {
                    var gradBeta = new Tensor<T>(new int[] { features });
                    for (int f = 0; f < features; f++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            sum = numOps.Add(sum, gradient[b, f]);
                        }
                        gradBeta[f] = sum;
                    }

                    if (beta.Gradient == null)
                        beta.Gradient = gradBeta;
                    else
                        beta.Gradient = beta.Gradient.Add(gradBeta);
                }

                // Gradient for input (complex due to batch statistics)
                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(shape);
                    var batchSizeT = numOps.FromDouble(batchSize);

                    for (int f = 0; f < features; f++)
                    {
                        var std = numOps.Sqrt(numOps.Add(batchVar[f], eps));
                        var invStd = numOps.Divide(numOps.One, std);

                        // Sum of gradients and gradient*normalized
                        var gradSum = numOps.Zero;
                        var gradNormSum = numOps.Zero;

                        for (int b = 0; b < batchSize; b++)
                        {
                            var grad = numOps.Multiply(gradient[b, f], gamma.Value[f]);
                            gradSum = numOps.Add(gradSum, grad);
                            gradNormSum = numOps.Add(gradNormSum,
                                numOps.Multiply(grad, normalized[b, f]));
                        }

                        // Apply gradient formula
                        for (int b = 0; b < batchSize; b++)
                        {
                            var grad = numOps.Multiply(gradient[b, f], gamma.Value[f]);

                            var term1 = grad;
                            var term2 = numOps.Divide(gradSum, batchSizeT);
                            var term3 = numOps.Divide(
                                numOps.Multiply(normalized[b, f], gradNormSum),
                                batchSizeT);

                            var gradInput = numOps.Multiply(
                                numOps.Subtract(numOps.Subtract(term1, term2), term3),
                                invStd);

                            gradA[b, f] = gradInput;
                        }
                    }

                    if (a.Gradient == null)
                        a.Gradient = gradA;
                    else
                        a.Gradient = a.Gradient.Add(gradA);
                }
            }

            var parents = new List<ComputationNode<T>> { a };
            if (gamma != null) parents.Add(gamma);
            if (beta != null) parents.Add(beta);

            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || (gamma?.RequiresGradient ?? false) || (beta?.RequiresGradient ?? false),
                parents: parents,
                backwardFunction: BackwardFunction,
                name: null);

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);

            return node;
        }
        else
        {
            throw new NotImplementedException(
                $"BatchNorm is currently only implemented for 2D tensors [batch, features]. " +
                $"Got shape=[{string.Join(", ", shape)}]");
        }
    }

    /// <summary>
    /// Performs 2D convolution on a 4D tensor (batch, channels, height, width).
    /// </summary>
    /// <param name="input">The input node with shape [batch, inChannels, height, width].</param>
    /// <param name="kernel">The kernel/filter with shape [outChannels, inChannels, kernelH, kernelW].</param>
    /// <param name="bias">Optional bias with shape [outChannels]. If null, no bias is added.</param>
    /// <param name="stride">The stride [strideH, strideW]. Default is [1, 1].</param>
    /// <param name="padding">The padding [padH, padW]. Default is [0, 0].</param>
    /// <returns>A new computation node containing the convolution result.</returns>
    /// <remarks>
    /// <para>
    /// This method performs 2D convolution, the fundamental operation in CNNs.
    /// Forward: Slides the kernel over the input computing dot products.
    /// Backward: Computes gradients for both input and kernel using transposed convolutions.
    /// </para>
    /// <para><b>For Beginners:</b> Conv2D is the core operation of convolutional neural networks.
    ///
    /// For 2D convolution:
    /// - The kernel "slides" over the input, computing weighted sums
    /// - Each output position is a dot product of the kernel with input patch
    /// - Stride controls how far the kernel moves each step
    /// - Padding adds borders to control output size
    ///
    /// Gradient computation:
    /// - Gradient w.r.t. input: "full" convolution with flipped kernel
    /// - Gradient w.r.t. kernel: cross-correlation between input and output gradient
    ///
    /// Used in:
    /// - All CNNs (image classification, object detection, segmentation)
    /// - Feature extraction in vision models
    /// - Learning spatial hierarchies
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Conv2D(
        ComputationNode<T> input,
        ComputationNode<T> kernel,
        ComputationNode<T>? bias = null,
        int[]? stride = null,
        int[]? padding = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var kernelShape = kernel.Value.Shape;

        if (inputShape.Length != 4)
            throw new ArgumentException("Conv2D requires 4D input [batch, inChannels, height, width]");
        if (kernelShape.Length != 4)
            throw new ArgumentException("Conv2D requires 4D kernel [outChannels, inChannels, kernelH, kernelW]");

        stride ??= new int[] { 1, 1 };
        padding ??= new int[] { 0, 0 };

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];

        int outChannels = kernelShape[0];
        int kernelInChannels = kernelShape[1];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        if (inChannels != kernelInChannels)
            throw new ArgumentException($"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels})");

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];

        int outH = (inH + 2 * padH - kernelH) / strideH + 1;
        int outW = (inW + 2 * padW - kernelW) / strideW + 1;

        var result = new Tensor<T>(new int[] { batch, outChannels, outH, outW });

        // Forward pass: convolution
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        var sum = numOps.Zero;

                        // Convolve kernel over input
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int ih = oh * strideH + kh - padH;
                                    int iw = ow * strideW + kw - padW;

                                    // Check bounds (padding)
                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    {
                                        var inputVal = input.Value[b, ic, ih, iw];
                                        var kernelVal = kernel.Value[oc, ic, kh, kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, kernelVal));
                                    }
                                }
                            }
                        }

                        // Add bias if provided
                        if (bias != null)
                        {
                            sum = numOps.Add(sum, bias.Value[oc]);
                        }

                        result[b, oc, oh, ow] = sum;
                    }
                }
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient w.r.t. input
            if (input.RequiresGradient)
            {
                var gradInput = new Tensor<T>(inputShape);

                // Full convolution with flipped kernel
                for (int b = 0; b < batch; b++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int ih = 0; ih < inH; ih++)
                        {
                            for (int iw = 0; iw < inW; iw++)
                            {
                                var sum = numOps.Zero;

                                // Iterate over all output positions that used this input position
                                for (int oc = 0; oc < outChannels; oc++)
                                {
                                    for (int kh = 0; kh < kernelH; kh++)
                                    {
                                        for (int kw = 0; kw < kernelW; kw++)
                                        {
                                            // Compute output position
                                            int ohShifted = ih + padH - kh;
                                            int owShifted = iw + padW - kw;

                                            if (ohShifted % strideH == 0 && owShifted % strideW == 0)
                                            {
                                                int oh = ohShifted / strideH;
                                                int ow = owShifted / strideW;

                                                if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                                {
                                                    var gradVal = gradient[b, oc, oh, ow];
                                                    var kernelVal = kernel.Value[oc, ic, kh, kw];
                                                    sum = numOps.Add(sum, numOps.Multiply(gradVal, kernelVal));
                                                }
                                            }
                                        }
                                    }
                                }

                                gradInput[b, ic, ih, iw] = sum;
                            }
                        }
                    }
                }

                if (input.Gradient == null)
                    input.Gradient = gradInput;
                else
                    input.Gradient = input.Gradient.Add(gradInput);
            }

            // Gradient w.r.t. kernel
            if (kernel.RequiresGradient)
            {
                var gradKernel = new Tensor<T>(kernelShape);

                // Cross-correlation between input and output gradient
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                var sum = numOps.Zero;

                                for (int b = 0; b < batch; b++)
                                {
                                    for (int oh = 0; oh < outH; oh++)
                                    {
                                        for (int ow = 0; ow < outW; ow++)
                                        {
                                            int ih = oh * strideH + kh - padH;
                                            int iw = ow * strideW + kw - padW;

                                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                            {
                                                var gradVal = gradient[b, oc, oh, ow];
                                                var inputVal = input.Value[b, ic, ih, iw];
                                                sum = numOps.Add(sum, numOps.Multiply(gradVal, inputVal));
                                            }
                                        }
                                    }
                                }

                                gradKernel[oc, ic, kh, kw] = sum;
                            }
                        }
                    }
                }

                if (kernel.Gradient == null)
                    kernel.Gradient = gradKernel;
                else
                    kernel.Gradient = kernel.Gradient.Add(gradKernel);
            }

            // Gradient w.r.t. bias
            if (bias != null && bias.RequiresGradient)
            {
                var gradBias = new Tensor<T>(new int[] { outChannels });

                for (int oc = 0; oc < outChannels; oc++)
                {
                    var sum = numOps.Zero;
                    for (int b = 0; b < batch; b++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                sum = numOps.Add(sum, gradient[b, oc, oh, ow]);
                            }
                        }
                    }
                    gradBias[oc] = sum;
                }

                if (bias.Gradient == null)
                    bias.Gradient = gradBias;
                else
                    bias.Gradient = bias.Gradient.Add(gradBias);
            }
        }

        var parents = new List<ComputationNode<T>> { input, kernel };
        if (bias != null) parents.Add(bias);

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient || kernel.RequiresGradient || (bias?.RequiresGradient ?? false),
            parents: parents,
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Performs 2D transposed convolution (deconvolution) on a 4D tensor.
    /// </summary>
    /// <param name="input">The input node with shape [batch, inChannels, height, width].</param>
    /// <param name="kernel">The kernel with shape [inChannels, outChannels, kernelH, kernelW] (note: reversed from Conv2D).</param>
    /// <param name="bias">Optional bias with shape [outChannels]. If null, no bias is added.</param>
    /// <param name="stride">The stride [strideH, strideW]. Default is [1, 1].</param>
    /// <param name="padding">The padding [padH, padW]. Default is [0, 0].</param>
    /// <param name="outputPadding">Output padding [outPadH, outPadW] for size adjustment. Default is [0, 0].</param>
    /// <returns>A new computation node containing the transposed convolution result.</returns>
    /// <remarks>
    /// <para>
    /// Transposed convolution (often called deconvolution) upsamples the input.
    /// It's the gradient of Conv2D with respect to its input, used as a forward operation.
    /// </para>
    /// <para><b>For Beginners:</b> ConvTranspose2D upsamples spatial dimensions.
    ///
    /// For transposed convolution:
    /// - Inserts zeros between input elements according to stride
    /// - Applies regular convolution to the expanded input
    /// - Results in larger spatial dimensions (upsampling)
    ///
    /// Used in:
    /// - Image generation (GANs, VAEs)
    /// - Semantic segmentation (U-Net decoder)
    /// - Super-resolution
    /// - Any task requiring upsampling
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ConvTranspose2D(
        ComputationNode<T> input,
        ComputationNode<T> kernel,
        ComputationNode<T>? bias = null,
        int[]? stride = null,
        int[]? padding = null,
        int[]? outputPadding = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var kernelShape = kernel.Value.Shape;

        if (inputShape.Length != 4)
            throw new ArgumentException("ConvTranspose2D requires 4D input [batch, inChannels, height, width]");
        if (kernelShape.Length != 4)
            throw new ArgumentException("ConvTranspose2D requires 4D kernel [inChannels, outChannels, kernelH, kernelW]");

        stride ??= new int[] { 1, 1 };
        padding ??= new int[] { 0, 0 };
        outputPadding ??= new int[] { 0, 0 };

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];

        int kernelInChannels = kernelShape[0];
        int outChannels = kernelShape[1];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        if (inChannels != kernelInChannels)
            throw new ArgumentException($"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels})");

        int strideH = stride[0];
        int strideW = stride[1];
        int padH = padding[0];
        int padW = padding[1];
        int outPadH = outputPadding[0];
        int outPadW = outputPadding[1];

        int outH = (inH - 1) * strideH - 2 * padH + kernelH + outPadH;
        int outW = (inW - 1) * strideW - 2 * padW + kernelW + outPadW;

        var result = new Tensor<T>(new int[] { batch, outChannels, outH, outW });

        // Forward pass: transposed convolution
        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int ih = 0; ih < inH; ih++)
                {
                    for (int iw = 0; iw < inW; iw++)
                    {
                        var inputVal = input.Value[b, ic, ih, iw];

                        // Distribute this input value to output using kernel
                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int oh = ih * strideH + kh - padH;
                                    int ow = iw * strideW + kw - padW;

                                    if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                    {
                                        var kernelVal = kernel.Value[ic, oc, kh, kw];
                                        var contribution = numOps.Multiply(inputVal, kernelVal);
                                        result[b, oc, oh, ow] = numOps.Add(result[b, oc, oh, ow], contribution);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add bias if provided
            if (bias != null)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            result[b, oc, oh, ow] = numOps.Add(result[b, oc, oh, ow], bias.Value[oc]);
                        }
                    }
                }
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient w.r.t. input (this is a forward Conv2D!)
            if (input.RequiresGradient)
            {
                var gradInput = new Tensor<T>(inputShape);

                for (int b = 0; b < batch; b++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int ih = 0; ih < inH; ih++)
                        {
                            for (int iw = 0; iw < inW; iw++)
                            {
                                var sum = numOps.Zero;

                                for (int oc = 0; oc < outChannels; oc++)
                                {
                                    for (int kh = 0; kh < kernelH; kh++)
                                    {
                                        for (int kw = 0; kw < kernelW; kw++)
                                        {
                                            int oh = ih * strideH + kh - padH;
                                            int ow = iw * strideW + kw - padW;

                                            if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                            {
                                                var gradVal = gradient[b, oc, oh, ow];
                                                var kernelVal = kernel.Value[ic, oc, kh, kw];
                                                sum = numOps.Add(sum, numOps.Multiply(gradVal, kernelVal));
                                            }
                                        }
                                    }
                                }

                                gradInput[b, ic, ih, iw] = sum;
                            }
                        }
                    }
                }

                if (input.Gradient == null)
                    input.Gradient = gradInput;
                else
                    input.Gradient = input.Gradient.Add(gradInput);
            }

            // Gradient w.r.t. kernel
            if (kernel.RequiresGradient)
            {
                var gradKernel = new Tensor<T>(kernelShape);

                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                var sum = numOps.Zero;

                                for (int b = 0; b < batch; b++)
                                {
                                    for (int ih = 0; ih < inH; ih++)
                                    {
                                        for (int iw = 0; iw < inW; iw++)
                                        {
                                            int oh = ih * strideH + kh - padH;
                                            int ow = iw * strideW + kw - padW;

                                            if (oh >= 0 && oh < outH && ow >= 0 && ow < outW)
                                            {
                                                var inputVal = input.Value[b, ic, ih, iw];
                                                var gradVal = gradient[b, oc, oh, ow];
                                                sum = numOps.Add(sum, numOps.Multiply(inputVal, gradVal));
                                            }
                                        }
                                    }
                                }

                                gradKernel[ic, oc, kh, kw] = sum;
                            }
                        }
                    }
                }

                if (kernel.Gradient == null)
                    kernel.Gradient = gradKernel;
                else
                    kernel.Gradient = kernel.Gradient.Add(gradKernel);
            }

            // Gradient w.r.t. bias
            if (bias != null && bias.RequiresGradient)
            {
                var gradBias = new Tensor<T>(new int[] { outChannels });

                for (int oc = 0; oc < outChannels; oc++)
                {
                    var sum = numOps.Zero;
                    for (int b = 0; b < batch; b++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                sum = numOps.Add(sum, gradient[b, oc, oh, ow]);
                            }
                        }
                    }
                    gradBias[oc] = sum;
                }

                if (bias.Gradient == null)
                    bias.Gradient = gradBias;
                else
                    bias.Gradient = bias.Gradient.Add(gradBias);
            }
        }

        var parents = new List<ComputationNode<T>> { input, kernel };
        if (bias != null) parents.Add(bias);

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient || kernel.RequiresGradient || (bias?.RequiresGradient ?? false),
            parents: parents,
            backwardFunction: BackwardFunction,
            name: null);

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }
}
