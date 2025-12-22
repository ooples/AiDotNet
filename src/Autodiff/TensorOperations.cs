using AiDotNet.Engines;
using AiDotNet.Tensors.LinearAlgebra;

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
        var node = new ComputationNode<T>(
            value: value,
            requiresGradient: requiresGradient,
            parents: null,
            backwardFunction: null,
            name: name);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Input;
        node.OperationParams = null;

        return node;
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
        var node = Variable(value, name, requiresGradient: false);

        // Set JIT compiler metadata for constant
        node.OperationType = OperationType.Constant;
        node.OperationParams = null;

        return node;
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
        // Forward pass: compute the sum with broadcasting support
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use direct addition if shapes are equal, otherwise broadcast
        Tensor<T> result = a.Value.Shape.SequenceEqual(b.Value.Shape)
            ? engine.TensorAdd(a.Value, b.Value)
            : BroadcastAdd(a.Value, b.Value, numOps);

        // Store original shapes for gradient reduction
        var aShape = a.Value.Shape;
        var bShape = b.Value.Shape;

        // Create backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            // Distribute gradient to both parents with reduction for broadcasted dimensions
            if (a.RequiresGradient)
            {
                var aGrad = ReduceGradient(gradient, aShape);
                if (a.Gradient == null)
                {
                    a.Gradient = aGrad;
                }
                else
                {
                    a.Gradient = engine.TensorAdd(a.Gradient, aGrad);
                }
            }
            if (b.RequiresGradient)
            {
                var bGrad = ReduceGradient(gradient, bShape);
                if (b.Gradient == null)
                {
                    b.Gradient = bGrad;
                }
                else
                {
                    b.Gradient = engine.TensorAdd(b.Gradient, bGrad);
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

        // Set JIT compiler metadata
        node.OperationType = OperationType.Add;
        node.OperationParams = null;

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

        // Set JIT compiler metadata
        node.OperationType = OperationType.Subtract;
        node.OperationParams = null;

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
        // Forward pass: element-wise multiplication (with limited broadcasting support)
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        Tensor<T> result = a.Value.Shape.SequenceEqual(b.Value.Shape)
            ? engine.TensorMultiply(a.Value, b.Value)
            : BroadcastMultiply(a.Value, b.Value, numOps);

        // Store original shapes for gradient reduction
        var aShape = a.Value.Shape;
        var bShape = b.Value.Shape;

        // Create backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            // d(a*b)/da = b, so gradient * b flows to 'a'
            if (a.RequiresGradient)
            {
                var gradA = gradient.Shape.SequenceEqual(b.Value.Shape)
                    ? engine.TensorMultiply(gradient, b.Value)
                    : BroadcastMultiply(gradient, b.Value, numOps);

                gradA = ReduceGradient(gradA, aShape);

                if (a.Gradient == null)
                {
                    a.Gradient = gradA;
                }
                else
                {
                    a.Gradient = engine.TensorAdd(a.Gradient, gradA);
                }
            }

            // d(a*b)/db = a, so gradient * a flows to 'b'
            if (b.RequiresGradient)
            {
                var gradB = gradient.Shape.SequenceEqual(a.Value.Shape)
                    ? engine.TensorMultiply(gradient, a.Value)
                    : BroadcastMultiply(gradient, a.Value, numOps);

                gradB = ReduceGradient(gradB, bShape);

                if (b.Gradient == null)
                {
                    b.Gradient = gradB;
                }
                else
                {
                    b.Gradient = engine.TensorAdd(b.Gradient, gradB);
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

        // Set JIT compiler metadata
        node.OperationType = OperationType.Multiply;
        node.OperationParams = null;

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
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
                var existingGrad = b.Gradient;
                b.Gradient = existingGrad == null ? gradB : existingGrad.Add(gradB);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Divide;
        node.OperationParams = null;

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Power;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Exponent", exponent }
        };

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Exp;
        node.OperationParams = null;

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Log;
        node.OperationParams = null;

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Sqrt;
        node.OperationParams = null;

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
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.Tanh(a.Value);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(tanh(a))/∂a = 1 - tanh²(a) = 1 - result²
                var resultSquared = engine.TensorMultiply(result, result);
                var oneMinusSquared = resultSquared.Transform((x, _) => numOps.Subtract(numOps.One, x));
                var gradA = engine.TensorMultiply(gradient, oneMinusSquared);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Tanh;
        node.OperationParams = null;

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
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.Sigmoid(a.Value);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂σ(a)/∂a = σ(a) * (1 - σ(a)) = result * (1 - result)
                var oneMinusResult = result.Transform((x, _) => numOps.Subtract(numOps.One, x));
                var derivative = engine.TensorMultiply(result, oneMinusResult);
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Sigmoid;
        node.OperationParams = null;

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
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.ReLU(a.Value);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂ReLU(a)/∂a = 1 if a > 0, else 0
                var mask = a.Value.Transform((x, _) =>
                    numOps.GreaterThan(x, numOps.Zero) ? numOps.One : numOps.Zero);
                var gradA = engine.TensorMultiply(gradient, mask);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.ReLU;
        node.OperationParams = null;

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Negate;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Computes the absolute value of each element in a computation node.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the absolute values.</returns>
    /// <remarks>
    /// <para>
    /// This method computes |x| for each element and records the operation.
    /// The backward function uses the sign of the original values for gradient computation.
    /// </para>
    /// <para><b>For Beginners:</b> This makes all values positive (removes the sign).
    ///
    /// For absolute value (c = |a|):
    /// - The forward pass removes the sign of each element
    /// - The backward pass uses sign(a) to route gradients correctly
    /// - For positive values, gradient passes through unchanged
    /// - For negative values, gradient is negated
    ///
    /// Note: At x = 0, the gradient is technically undefined, but we use 0 as a convention.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Abs(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => numOps.Abs(x));

        // Store the original values for backward pass
        var originalValues = a.Value;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂|a|/∂a = sign(a) = 1 if a > 0, -1 if a < 0, 0 if a = 0
                var gradA = gradient.Transform((g, indices) =>
                {
                    var origVal = originalValues[indices];
                    // sign(x): 1 if x > 0, -1 if x < 0, 0 if x = 0
                    if (numOps.GreaterThan(origVal, numOps.Zero))
                        return g;
                    else if (numOps.LessThan(origVal, numOps.Zero))
                        return numOps.Negate(g);
                    else
                        return numOps.Zero; // Gradient at 0 is undefined, use 0
                });

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Abs;
        node.OperationParams = null;

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
        var engine = AiDotNetEngine.Current;
        var result = engine.TensorMatMul(a.Value, b.Value);
        void BackwardFunction(Tensor<T> gradient)
        {
            // ∂(A·B)/∂A = gradOut·B^T
            if (a.RequiresGradient)
            {
                var bTransposed = engine.TensorTranspose(b.Value);
                var gradA = engine.TensorMatMul(gradient, bTransposed);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
            // ∂(A·B)/∂B = A^T·gradOut
            if (b.RequiresGradient)
            {
                var aTransposed = engine.TensorTranspose(a.Value);
                var gradB = engine.TensorMatMul(aTransposed, gradient);
                var existingGrad = b.Gradient;
                b.Gradient = existingGrad == null ? gradB : engine.TensorAdd(existingGrad, gradB);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.MatMul;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Performs a matrix-vector multiplication (2D x 1D) by reshaping the vector into a column matrix.
    /// </summary>
    /// <param name="matrix">The left matrix (must be 2D).</param>
    /// <param name="vector">The right vector (must be 1D).</param>
    /// <returns>A computation node representing the vector result.</returns>
    public static ComputationNode<T> MatrixVectorMultiply(ComputationNode<T> matrix, ComputationNode<T> vector)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix));
        }

        if (vector == null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        if (matrix.Value.Shape.Length != 2)
        {
            throw new ArgumentException("MatrixVectorMultiply requires a 2D matrix input.", nameof(matrix));
        }

        if (vector.Value.Shape.Length != 1)
        {
            throw new ArgumentException("MatrixVectorMultiply requires a 1D vector input.", nameof(vector));
        }

        var rows = matrix.Value.Shape[0];
        var cols = matrix.Value.Shape[1];
        if (vector.Value.Shape[0] != cols)
        {
            throw new ArgumentException("MatrixVectorMultiply requires matching inner dimensions.", nameof(vector));
        }

        var vector2d = Reshape(vector, cols, 1);
        var result2d = MatrixMultiply(matrix, vector2d);
        return Reshape(result2d, rows);
    }

    /// <summary>
    /// Performs batched matrix multiplication of two 3D computation nodes.
    /// </summary>
    /// <param name="a">The first 3D tensor with shape [Batch, M, K].</param>
    /// <param name="b">The second 3D tensor with shape [Batch, K, N].</param>
    /// <returns>A computation node representing the batched matrix multiplication with shape [Batch, M, N].</returns>
    /// <remarks>
    /// <para>
    /// For 3D tensors, performs element-wise matrix multiplication across the batch dimension:
    /// result[i] = a[i] @ b[i] for each batch index i.
    /// </para>
    /// <para><b>Gradient computation:</b>
    /// - ∂(A·B)/∂A = gradOut·B^T (batch-wise)
    /// - ∂(A·B)/∂B = A^T·gradOut (batch-wise)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> BatchMatrixMultiply(ComputationNode<T> a, ComputationNode<T> b)
    {
        var engine = AiDotNetEngine.Current;
        var result = engine.BatchMatMul(a.Value, b.Value);
        void BackwardFunction(Tensor<T> gradient)
        {
            // For batched matmul C[i] = A[i] @ B[i]:
            // ∂L/∂A[i] = ∂L/∂C[i] @ B[i]^T
            // ∂L/∂B[i] = A[i]^T @ ∂L/∂C[i]
            if (a.RequiresGradient)
            {
                // Transpose B along last two dims: [Batch, K, N] -> [Batch, N, K]
                var bTransposed = b.Value.Transpose([0, 2, 1]);
                var gradA = engine.BatchMatMul(gradient, bTransposed);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
            if (b.RequiresGradient)
            {
                // Transpose A along last two dims: [Batch, M, K] -> [Batch, K, M]
                var aTransposed = a.Value.Transpose([0, 2, 1]);
                var gradB = engine.BatchMatMul(aTransposed, gradient);
                var existingGrad = b.Gradient;
                b.Gradient = existingGrad == null ? gradB : engine.TensorAdd(existingGrad, gradB);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.MatMul;
        node.OperationParams = null;

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
        var engine = AiDotNetEngine.Current;
        var result = engine.TensorTranspose(a.Value);
        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(A^T)/∂A = gradOut^T
                var gradA = engine.TensorTranspose(gradient);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Transpose;
        node.OperationParams = null;

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.ReduceSum;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axes", axes! },
            { "KeepDims", keepDims }
        };

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Mean;
        node.OperationParams = null;

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
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Reshape;
        node.OperationParams = new Dictionary<string, object>
        {
            { "NewShape", newShape }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Permutes the dimensions of a computation node (general transpose).
    /// </summary>
    /// <param name="a">The computation node to permute.</param>
    /// <param name="axes">The new order of dimensions.</param>
    /// <returns>A computation node with permuted dimensions.</returns>
    /// <remarks>
    /// <para>
    /// Rearranges dimensions according to the axes array.
    /// Equivalent to Transpose but for N dimensions.
    /// </para>
    /// <para><b>Gradient computation:</b>
    /// - ∂(Permute(A))/∂A = Permute(gradOut, inverseAxes)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Permute(ComputationNode<T> a, params int[] axes)
    {
        var result = a.Value.Transpose(axes);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Calculate inverse permutation
                var inverseAxes = new int[axes.Length];
                for (int i = 0; i < axes.Length; i++)
                {
                    inverseAxes[axes[i]] = i;
                }

                var gradA = gradient.Transpose(inverseAxes);
                var engine = AiDotNetEngine.Current;
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Permute;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axes", axes }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Broadcasts a 1D tensor to a 2D tensor by tiling along the batch dimension.
    /// </summary>
    /// <param name="a">The input 1D tensor node with shape [N].</param>
    /// <param name="targetShape">The target 2D shape [batchSize, N].</param>
    /// <returns>A new computation node with the broadcasted tensor.</returns>
    /// <remarks>
    /// <para>
    /// This operation broadcasts a 1D tensor (e.g., biases with shape [outputSize]) to a 2D tensor
    /// (e.g., [batchSize, outputSize]) by replicating values along the batch dimension.
    /// The backward pass correctly sums gradients along the broadcasted dimension.
    /// </para>
    /// <para><b>For Beginners:</b> Broadcasting is like copying a row multiple times to create a matrix.
    ///
    /// For example, if you have biases [b1, b2, b3] and need to add them to a batch of outputs:
    /// - Input: [b1, b2, b3] (shape [3])
    /// - Target shape: [batchSize=2, 3]
    /// - Output: [[b1, b2, b3], [b1, b2, b3]] (each row is a copy)
    ///
    /// During backpropagation, gradients from all rows are summed back to the original biases,
    /// because each bias contributed to all batch elements.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Broadcast(ComputationNode<T> a, int[] targetShape)
    {
        var engine = AiDotNetEngine.Current;
        var originalShape = a.Value.Shape;

        // Validate: we support broadcasting 1D [N] to 2D [M, N]
        if (originalShape.Length != 1 || targetShape.Length != 2 || originalShape[0] != targetShape[1])
        {
            throw new ArgumentException(
                $"Broadcast currently supports 1D [N] to 2D [M, N]. " +
                $"Got input shape [{string.Join(", ", originalShape)}] and target shape [{string.Join(", ", targetShape)}].");
        }

        int batchSize = targetShape[0];
        int outputSize = originalShape[0];

        // Forward pass: tile the 1D tensor along the batch dimension
        var result = new Tensor<T>(targetShape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputSize; i++)
            {
                result[b, i] = a.Value[i];
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Sum gradients along the batch dimension (axis 0) to get back to original shape
                // gradient has shape [batchSize, outputSize], we need shape [outputSize]
                var gradA = gradient.Sum(new int[] { 0 });
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Broadcast;
        node.OperationParams = new Dictionary<string, object>
        {
            { "TargetShape", targetShape }
        };

        var broadcastTape = GradientTape<T>.Current;
        if (broadcastTape != null && broadcastTape.IsRecording)
            broadcastTape.RecordOperation(node);
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
        var engine = AiDotNetEngine.Current;
        var shape = a.Value.Shape;

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.Softmax(a.Value, axis);

        // Capture the axis value for backward
        int capturedAxis = axis;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Use IEngine for GPU-accelerated backward pass
                var gradA = engine.SoftmaxBackward(gradient, result, capturedAxis);

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Softmax;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axis", axis }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Exponential Linear Unit (ELU) activation function to a computation node.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="alpha">The alpha parameter controlling the negative saturation value. Default is 1.0.</param>
    /// <returns>A new computation node with ELU applied.</returns>
    /// <remarks>
    /// <para>
    /// ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// ELU helps prevent "dying neurons" and pushes mean activations closer to zero.
    /// </para>
    /// <para><b>Gradient:</b> d(ELU)/dx = 1 if x > 0, alpha * exp(x) = ELU(x) + alpha otherwise.</para>
    /// </remarks>
    public static ComputationNode<T> ELU(ComputationNode<T> a, double alpha = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var alphaT = numOps.FromDouble(alpha);

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.ELU(a.Value, alpha);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(ELU)/dx = 1 if x > 0, alpha * exp(x) = ELU(x) + alpha if x <= 0
                var derivative = a.Value.Transform((x, idx) =>
                {
                    if (numOps.GreaterThan(x, numOps.Zero))
                        return numOps.One;
                    else
                        return numOps.Add(result[idx], alphaT);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.ELU;
        node.OperationParams = new Dictionary<string, object> { { "Alpha", alpha } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Leaky Rectified Linear Unit (LeakyReLU) activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="alpha">The slope for negative values. Default is 0.01.</param>
    /// <returns>A new computation node with LeakyReLU applied.</returns>
    /// <remarks>
    /// <para>
    /// LeakyReLU(x) = x if x > 0, alpha * x otherwise.
    /// Unlike ReLU, LeakyReLU allows a small gradient for negative inputs, preventing dying neurons.
    /// </para>
    /// <para><b>Gradient:</b> d(LeakyReLU)/dx = 1 if x > 0, alpha otherwise.</para>
    /// </remarks>
    public static ComputationNode<T> LeakyReLU(ComputationNode<T> a, double alpha = 0.01)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var alphaT = numOps.FromDouble(alpha);

        // Forward pass: max(alpha * x, x)
        var result = a.Value.Transform((x, _) =>
            numOps.GreaterThan(x, numOps.Zero) ? x : numOps.Multiply(alphaT, x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(LeakyReLU)/dx = 1 if x > 0, alpha otherwise
                var derivative = a.Value.Transform((x, _) =>
                    numOps.GreaterThan(x, numOps.Zero) ? numOps.One : alphaT);
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.LeakyReLU;
        node.OperationParams = new Dictionary<string, object> { { "Alpha", alpha } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Gaussian Error Linear Unit (GELU) activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with GELU applied.</returns>
    /// <remarks>
    /// <para>
    /// GELU(x) = x * Φ(x) where Φ is the standard Gaussian cumulative distribution function.
    /// Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// </para>
    /// <para>
    /// GELU is widely used in transformers (BERT, GPT) and modern architectures.
    /// </para>
    /// <para><b>Gradient:</b> d(GELU)/dx = Φ(x) + x * φ(x) where φ is the Gaussian PDF.</para>
    /// </remarks>
    public static ComputationNode<T> GELU(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.GELU(a.Value);

        // Constants for approximation
        var sqrt2OverPi = numOps.FromDouble(Math.Sqrt(2.0 / Math.PI)); // ~0.7978845608
        var c = numOps.FromDouble(0.044715);
        var half = numOps.FromDouble(0.5);
        var one = numOps.One;
        var three = numOps.FromDouble(3.0);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Approximate gradient of GELU using tanh approximation:
                // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                // d(GELU)/dx = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * √(2/π) * (1 + 3 * 0.044715 * x²)
                var derivative = a.Value.Transform((x, _) =>
                {
                    var x2 = numOps.Multiply(x, x);
                    var x3 = numOps.Multiply(x2, x);
                    var inner = numOps.Multiply(sqrt2OverPi, numOps.Add(x, numOps.Multiply(c, x3)));
                    var tanhInner = MathHelper.Tanh(inner);
                    var sech2 = numOps.Subtract(one, numOps.Multiply(tanhInner, tanhInner));

                    // 0.5 * (1 + tanh(...))
                    var term1 = numOps.Multiply(half, numOps.Add(one, tanhInner));

                    // 0.5 * x * sech²(...) * √(2/π) * (1 + 3 * 0.044715 * x²)
                    var innerDeriv = numOps.Add(one, numOps.Multiply(numOps.Multiply(three, c), x2));
                    var term2 = numOps.Multiply(numOps.Multiply(numOps.Multiply(half, x), sech2),
                                                numOps.Multiply(sqrt2OverPi, innerDeriv));

                    return numOps.Add(term1, term2);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.GELU;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Swish (SiLU) activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with Swish applied.</returns>
    /// <remarks>
    /// <para>
    /// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    /// Also known as SiLU (Sigmoid Linear Unit).
    /// Used in EfficientNet and other modern architectures.
    /// </para>
    /// <para><b>Gradient:</b> d(Swish)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)) = Swish(x) + sigmoid(x) * (1 - Swish(x))</para>
    /// </remarks>
    public static ComputationNode<T> Swish(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.Swish(a.Value);

        // Cache sigmoid for backward pass
        var sigmoidValues = engine.Sigmoid(a.Value);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(Swish)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                //             = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                //             = sigmoid(x) * (1 + x - x * sigmoid(x))
                //             = sigmoid(x) * (1 + x - Swish(x))
                var derivative = a.Value.Transform((x, idx) =>
                {
                    var sig = sigmoidValues[idx];
                    var swishVal = result[idx];
                    // sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    var oneMinusSig = numOps.Subtract(numOps.One, sig);
                    var xTimesSigTimesOneMinusSig = numOps.Multiply(numOps.Multiply(x, sig), oneMinusSig);
                    return numOps.Add(sig, xTimesSigTimesOneMinusSig);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Swish;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Mish activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with Mish applied.</returns>
    /// <remarks>
    /// <para>
    /// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    /// Mish is a smooth, self-regularizing activation function.
    /// </para>
    /// <para><b>Gradient:</b> d(Mish)/dx = sech²(softplus(x)) * sigmoid(x) + tanh(softplus(x))</para>
    /// </remarks>
    public static ComputationNode<T> Mish(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.Mish(a.Value);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Mish(x) = x * tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))
                // d(Mish)/dx = tanh(sp) + x * sech²(sp) * sigmoid(x)
                //            = tanh(sp) + x * (1 - tanh²(sp)) * sigmoid(x)
                var derivative = a.Value.Transform((x, idx) =>
                {
                    // softplus = ln(1 + exp(x)), using stable version
                    T softplus;
                    var expX = numOps.Exp(x);
                    var onePlusExpX = numOps.Add(numOps.One, expX);
                    softplus = numOps.Log(onePlusExpX);

                    var tanhSp = MathHelper.Tanh(softplus);
                    var sigmoid = numOps.Divide(numOps.One, onePlusExpX); // 1/(1+exp(-x)) = exp(x)/(1+exp(x)) when computed this way
                    sigmoid = numOps.Divide(expX, onePlusExpX); // correct sigmoid
                    var sech2Sp = numOps.Subtract(numOps.One, numOps.Multiply(tanhSp, tanhSp));

                    // tanh(sp) + x * sech²(sp) * sigmoid(x)
                    return numOps.Add(tanhSp, numOps.Multiply(numOps.Multiply(x, sech2Sp), sigmoid));
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Mish;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the SoftPlus activation function element-wise: f(x) = ln(1 + e^x).
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with SoftPlus applied.</returns>
    /// <remarks>
    /// <para>
    /// SoftPlus is a smooth approximation of ReLU. The gradient is the sigmoid function:
    /// d(SoftPlus)/dx = sigmoid(x) = 1 / (1 + e^(-x))
    /// </para>
    /// <para><b>For Beginners:</b> SoftPlus smoothly approaches 0 for negative inputs and
    /// approaches the input value for large positive inputs, similar to ReLU but without
    /// the sharp corner at x=0.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SoftPlus(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Forward pass: numerically stable softplus
        // softplus(x) = max(0, x) + ln(1 + exp(-|x|))
        // For large positive x, this avoids exp(x) overflow
        // For large negative x, exp(-|x|) approaches 0, so result ≈ 0
        var result = a.Value.Transform((x, idx) =>
        {
            // Compute |x|: if x >= 0, absX = x, else absX = -x
            var absX = numOps.GreaterThanOrEquals(x, numOps.Zero) ? x : numOps.Negate(x);
            var negAbsX = numOps.Negate(absX);
            var expNegAbsX = numOps.Exp(negAbsX);
            var log1pExpNegAbsX = numOps.Log(numOps.Add(numOps.One, expNegAbsX));
            var maxZeroX = numOps.GreaterThan(x, numOps.Zero) ? x : numOps.Zero;
            return numOps.Add(maxZeroX, log1pExpNegAbsX);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(SoftPlus)/dx = sigmoid(x) = 1 / (1 + e^(-x))
                var derivative = a.Value.Transform((x, idx) =>
                {
                    var negX = numOps.Negate(x);
                    var expNegX = numOps.Exp(negX);
                    var onePlusExpNegX = numOps.Add(numOps.One, expNegX);
                    return numOps.Divide(numOps.One, onePlusExpNegX);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.SoftPlus;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the SELU (Scaled Exponential Linear Unit) activation function element-wise.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with SELU applied.</returns>
    /// <remarks>
    /// <para>
    /// SELU is defined as: λ * x if x > 0, otherwise λ * α * (e^x - 1)
    /// where λ ≈ 1.0507 and α ≈ 1.6733 are fixed constants for self-normalization.
    /// The gradient is: λ if x > 0, otherwise λ * α * e^x
    /// </para>
    /// <para><b>For Beginners:</b> SELU enables self-normalizing neural networks where
    /// activations converge to zero mean and unit variance, reducing the need for
    /// batch normalization.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SELU(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // SELU constants for self-normalization
        var lambda = numOps.FromDouble(1.0507009873554804934193349852946);
        var alpha = numOps.FromDouble(1.6732632423543772848170429916717);
        var lambdaAlpha = numOps.Multiply(lambda, alpha);

        // Forward pass
        var result = a.Value.Transform((x, idx) =>
        {
            if (numOps.GreaterThanOrEquals(x, numOps.Zero))
            {
                return numOps.Multiply(lambda, x);
            }
            else
            {
                var expTerm = numOps.Subtract(numOps.Exp(x), numOps.One);
                return numOps.Multiply(lambdaAlpha, expTerm);
            }
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(SELU)/dx = λ if x >= 0, else λ * α * e^x
                var derivative = a.Value.Transform((x, idx) =>
                {
                    if (numOps.GreaterThanOrEquals(x, numOps.Zero))
                    {
                        return lambda;
                    }
                    else
                    {
                        return numOps.Multiply(lambdaAlpha, numOps.Exp(x));
                    }
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.SELU;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Hard Sigmoid activation function element-wise: f(x) = clip((x + 1) / 2, 0, 1).
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with HardSigmoid applied.</returns>
    /// <remarks>
    /// <para>
    /// HardSigmoid is a piecewise linear approximation of sigmoid that is computationally efficient.
    /// The gradient is 0.5 when -1 &lt; x &lt; 1, and 0 otherwise.
    /// </para>
    /// <para><b>For Beginners:</b> HardSigmoid uses straight lines instead of curves,
    /// making it faster to compute while still mapping inputs to the [0, 1] range.
    /// It's commonly used in mobile and embedded neural networks.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> HardSigmoid(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var half = numOps.FromDouble(0.5);
        var minusOne = numOps.FromDouble(-1.0);

        // Forward pass: clip((x + 1) / 2, 0, 1)
        var result = a.Value.Transform((x, idx) =>
        {
            var shifted = numOps.Add(x, numOps.One);
            var scaled = numOps.Multiply(shifted, half);
            // Clamp to [0, 1]
            if (numOps.LessThan(scaled, numOps.Zero))
                return numOps.Zero;
            if (numOps.GreaterThan(scaled, numOps.One))
                return numOps.One;
            return scaled;
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(HardSigmoid)/dx = 0.5 if -1 < x < 1, else 0
                var derivative = a.Value.Transform((x, idx) =>
                {
                    if (numOps.GreaterThan(x, minusOne) && numOps.LessThan(x, numOps.One))
                    {
                        return half;
                    }
                    return numOps.Zero;
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.HardSigmoid;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Hard Tanh activation function element-wise: f(x) = clip(x, -1, 1).
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with HardTanh applied.</returns>
    /// <remarks>
    /// <para>
    /// HardTanh is a piecewise linear approximation of tanh that is computationally efficient.
    /// The gradient is 1 when -1 &lt; x &lt; 1, and 0 otherwise.
    /// </para>
    /// <para><b>For Beginners:</b> HardTanh clips values to the range [-1, 1], passing
    /// through values in the middle range unchanged. It's faster than regular tanh
    /// and useful when you need bounded outputs.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> HardTanh(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var minusOne = numOps.FromDouble(-1.0);

        // Forward pass: clip(x, -1, 1)
        var result = a.Value.Transform((x, idx) =>
        {
            if (numOps.LessThan(x, minusOne))
                return minusOne;
            if (numOps.GreaterThan(x, numOps.One))
                return numOps.One;
            return x;
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(HardTanh)/dx = 1 if -1 < x < 1, else 0
                var derivative = a.Value.Transform((x, idx) =>
                {
                    if (numOps.GreaterThan(x, minusOne) && numOps.LessThan(x, numOps.One))
                    {
                        return numOps.One;
                    }
                    return numOps.Zero;
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.HardTanh;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the SoftSign activation function element-wise: f(x) = x / (1 + |x|).
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with SoftSign applied.</returns>
    /// <remarks>
    /// <para>
    /// SoftSign is an alternative to tanh with polynomial tails that approach ±1 more slowly.
    /// The gradient is: d(SoftSign)/dx = 1 / (1 + |x|)²
    /// </para>
    /// <para><b>For Beginners:</b> SoftSign maps inputs to (-1, 1) like tanh, but with
    /// a different shape. The slower saturation can help prevent vanishing gradients
    /// in deep networks.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SoftSign(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Forward pass: x / (1 + |x|)
        var result = a.Value.Transform((x, idx) =>
        {
            var absX = numOps.Abs(x);
            var denominator = numOps.Add(numOps.One, absX);
            return numOps.Divide(x, denominator);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(SoftSign)/dx = 1 / (1 + |x|)²
                var derivative = a.Value.Transform((x, idx) =>
                {
                    var absX = numOps.Abs(x);
                    var denominator = numOps.Add(numOps.One, absX);
                    var denominatorSquared = numOps.Multiply(denominator, denominator);
                    return numOps.Divide(numOps.One, denominatorSquared);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.SoftSign;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the CELU (Continuously Differentiable ELU) activation function element-wise.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="alpha">The alpha parameter controlling negative saturation. Default is 1.0.</param>
    /// <returns>A new computation node with CELU applied.</returns>
    /// <remarks>
    /// <para>
    /// CELU is defined as: max(0, x) + min(0, α * (exp(x/α) - 1))
    /// The gradient is: 1 if x >= 0, otherwise exp(x/α)
    /// </para>
    /// <para><b>For Beginners:</b> CELU is an improved version of ELU that is continuously
    /// differentiable everywhere, which can help with optimization and training stability.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> CELU(ComputationNode<T> a, double alpha = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var alphaT = numOps.FromDouble(alpha);

        // Forward pass: max(0, x) + min(0, α * (exp(x/α) - 1))
        var result = a.Value.Transform((x, idx) =>
        {
            var positivePart = numOps.GreaterThanOrEquals(x, numOps.Zero) ? x : numOps.Zero;
            var expTerm = numOps.Subtract(numOps.Exp(numOps.Divide(x, alphaT)), numOps.One);
            var negativePart = numOps.Multiply(alphaT, expTerm);
            var negativeClipped = numOps.LessThan(negativePart, numOps.Zero) ? negativePart : numOps.Zero;
            return numOps.Add(positivePart, negativeClipped);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(CELU)/dx = 1 if x >= 0, else exp(x/α)
                var derivative = a.Value.Transform((x, idx) =>
                {
                    if (numOps.GreaterThanOrEquals(x, numOps.Zero))
                    {
                        return numOps.One;
                    }
                    else
                    {
                        return numOps.Exp(numOps.Divide(x, alphaT));
                    }
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.CELU;
        node.OperationParams = new Dictionary<string, object> { { "alpha", alpha } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the LiSHT (Linearly Scaled Hyperbolic Tangent) activation function element-wise.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with LiSHT applied.</returns>
    /// <remarks>
    /// <para>
    /// LiSHT is defined as: f(x) = x * tanh(x)
    /// The gradient is: tanh(x) + x * (1 - tanh²(x))
    /// </para>
    /// <para><b>For Beginners:</b> LiSHT combines the input with its tanh, creating a smooth
    /// activation that preserves sign and helps prevent vanishing gradients.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> LiSHT(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Forward pass: x * tanh(x)
        var result = a.Value.Transform((x, idx) =>
        {
            var tanhX = MathHelper.Tanh(x);
            return numOps.Multiply(x, tanhX);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(LiSHT)/dx = tanh(x) + x * (1 - tanh²(x))
                var derivative = a.Value.Transform((x, idx) =>
                {
                    var tanhX = MathHelper.Tanh(x);
                    var tanhSquared = numOps.Multiply(tanhX, tanhX);
                    var sech2 = numOps.Subtract(numOps.One, tanhSquared);
                    return numOps.Add(tanhX, numOps.Multiply(x, sech2));
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.LiSHT;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Bent Identity activation function element-wise.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with BentIdentity applied.</returns>
    /// <remarks>
    /// <para>
    /// BentIdentity is defined as: f(x) = (sqrt(x² + 1) - 1) / 2 + x
    /// The gradient is: x / (2 * sqrt(x² + 1)) + 1
    /// </para>
    /// <para><b>For Beginners:</b> BentIdentity is a smooth alternative to ReLU with
    /// non-zero gradient everywhere, preventing dead neurons during training.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> BentIdentity(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var half = numOps.FromDouble(0.5);
        var two = numOps.FromDouble(2.0);

        // Forward pass: (sqrt(x² + 1) - 1) / 2 + x
        var result = a.Value.Transform((x, idx) =>
        {
            var xSquared = numOps.Multiply(x, x);
            var sqrtTerm = numOps.Sqrt(numOps.Add(xSquared, numOps.One));
            var firstPart = numOps.Multiply(half, numOps.Subtract(sqrtTerm, numOps.One));
            return numOps.Add(firstPart, x);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(BentIdentity)/dx = x / (2 * sqrt(x² + 1)) + 1
                var derivative = a.Value.Transform((x, idx) =>
                {
                    var xSquared = numOps.Multiply(x, x);
                    var sqrtTerm = numOps.Sqrt(numOps.Add(xSquared, numOps.One));
                    var firstPart = numOps.Divide(x, numOps.Multiply(two, sqrtTerm));
                    return numOps.Add(firstPart, numOps.One);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.BentIdentity;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Gaussian activation function element-wise: f(x) = exp(-x²).
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <returns>A new computation node with Gaussian applied.</returns>
    /// <remarks>
    /// <para>
    /// Gaussian is defined as: f(x) = exp(-x²)
    /// The gradient is: -2x * exp(-x²)
    /// </para>
    /// <para><b>For Beginners:</b> Gaussian creates a bell-shaped response curve that is
    /// maximum at zero and approaches zero for large inputs in either direction.
    /// Useful for RBF networks and pattern recognition.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Gaussian(ComputationNode<T> a)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var negTwo = numOps.FromDouble(-2.0);

        // Forward pass: exp(-x²)
        var result = a.Value.Transform((x, idx) =>
        {
            var negXSquared = numOps.Negate(numOps.Multiply(x, x));
            return numOps.Exp(negXSquared);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(Gaussian)/dx = -2x * exp(-x²)
                var derivative = a.Value.Transform((x, idx) =>
                {
                    var negXSquared = numOps.Negate(numOps.Multiply(x, x));
                    var expTerm = numOps.Exp(negXSquared);
                    return numOps.Multiply(numOps.Multiply(negTwo, x), expTerm);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.Gaussian;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Scaled Tanh activation function element-wise.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="beta">The steepness parameter. Default is 1.0.</param>
    /// <returns>A new computation node with ScaledTanh applied.</returns>
    /// <remarks>
    /// <para>
    /// ScaledTanh is defined as: f(x) = (1 - exp(-βx)) / (1 + exp(-βx))
    /// The gradient is: β * (1 - f(x)²)
    /// When β = 2, this equals standard tanh.
    /// </para>
    /// <para><b>For Beginners:</b> ScaledTanh allows you to control the steepness of the
    /// tanh curve, which can be useful for tuning network behavior.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ScaledTanh(ComputationNode<T> a, double beta = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var betaT = numOps.FromDouble(beta);

        // Forward pass: (1 - exp(-βx)) / (1 + exp(-βx))
        var result = a.Value.Transform((x, idx) =>
        {
            var negBetaX = numOps.Negate(numOps.Multiply(betaT, x));
            var expTerm = numOps.Exp(negBetaX);
            var numerator = numOps.Subtract(numOps.One, expTerm);
            var denominator = numOps.Add(numOps.One, expTerm);
            return numOps.Divide(numerator, denominator);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(ScaledTanh)/dx = β * (1 - f(x)²)
                var derivative = result.Transform((fx, idx) =>
                {
                    var fxSquared = numOps.Multiply(fx, fx);
                    var oneMinusFxSquared = numOps.Subtract(numOps.One, fxSquared);
                    return numOps.Multiply(betaT, oneMinusFxSquared);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.ScaledTanh;
        node.OperationParams = new Dictionary<string, object> { { "beta", beta } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
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

        var engine = AiDotNetEngine.Current;
        var firstShape = nodes[0].Value.Shape;

        // Normalize axis
        int normalizedAxis = axis < 0 ? firstShape.Length + axis : axis;

        // Use IEngine for GPU-accelerated forward pass
        var tensors = nodes.Select(n => n.Value).ToList();
        var result = engine.Concat(tensors, normalizedAxis);

        // Store sizes and shapes for gradient splitting
        var sizes = nodes.Select(n => n.Value.Shape[normalizedAxis]).ToList();
        var shapes = nodes.Select(n => n.Value.Shape).ToList();
        int capturedAxis = normalizedAxis;

        void BackwardFunction(Tensor<T> gradient)
        {
            // Split gradient along concat axis and distribute to inputs
            var numOps = MathHelper.GetNumericOperations<T>();
            var gradShape = gradient.Shape;
            var strides = ComputeStridesStatic(gradShape);
            var gradData = gradient.ToArray();

            int axisOffset = 0;
            for (int i = 0; i < nodes.Count; i++)
            {
                if (!nodes[i].RequiresGradient)
                {
                    axisOffset += sizes[i];
                    continue;
                }

                var nodeShape = shapes[i];
                var gradPart = ExtractSlice(gradData, gradShape, strides, capturedAxis, axisOffset, sizes[i], nodeShape);

                var existingGrad = nodes[i].Gradient;
                nodes[i].Gradient = existingGrad == null ? gradPart : engine.TensorAdd(existingGrad, gradPart);
                axisOffset += sizes[i];
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: nodes.Any(n => n.RequiresGradient),
            parents: nodes,
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Concat;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axis", normalizedAxis }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    // Helper method to extract a slice from a tensor along a given axis
    private static Tensor<T> ExtractSlice(T[] data, int[] shape, int[] strides, int axis, int start, int length, int[] outputShape)
    {
        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];
        var outputStrides = ComputeStridesStatic(outputShape);

        for (int i = 0; i < outputSize; i++)
        {
            var multiIndex = FlatToMultiIndexStatic(i, outputShape, outputStrides);
            multiIndex[axis] += start;
            int sourceIdx = MultiToFlatIndexStatic(multiIndex, shape, strides);
            outputData[i] = data[sourceIdx];
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    private static int[] ComputeStridesStatic(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int[] FlatToMultiIndexStatic(int flatIndex, int[] shape, int[] strides)
    {
        var multiIndex = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            multiIndex[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }
        return multiIndex;
    }

    private static int MultiToFlatIndexStatic(int[] multiIndex, int[] shape, int[] strides)
    {
        int flatIndex = 0;
        for (int i = 0; i < multiIndex.Length; i++)
        {
            flatIndex += multiIndex[i] * strides[i];
        }
        return flatIndex;
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
                result.SetFlat(i, padValue);
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
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }
            }
            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient,
                parents: new List<ComputationNode<T>> { a },
                backwardFunction: BackwardFunction,
                name: null);

            // Set JIT compiler metadata
            node.OperationType = OperationType.Pad;
            node.OperationParams = new Dictionary<string, object>
            {
                { "PadWidth", padWidth },
                { "Value", value! }
            };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
        }
        else
        {
            // General N-dimensional case
            var result = new Tensor<T>(outputShape);

            // Initialize with pad value
            for (int i = 0; i < result.Length; i++)
            {
                result.SetFlat(i, padValue);
            }

            // Copy input data to appropriate location
            // Use multi-dimensional index iteration
            var inputIndices = new int[shape.Length];
            var outputIndices = new int[outputShape.Length];

            void CopyRecursive(int dim)
            {
                if (dim == shape.Length)
                {
                    // Copy single element
                    var inputFlatIdx = ComputeFlatIndex(inputIndices, shape);
                    var outputFlatIdx = ComputeFlatIndex(outputIndices, outputShape);
                    result[outputFlatIdx] = a.Value[inputFlatIdx];
                }
                else
                {
                    for (int i = 0; i < shape[dim]; i++)
                    {
                        inputIndices[dim] = i;
                        outputIndices[dim] = i + padWidth[dim, 0]; // Add before padding
                        CopyRecursive(dim + 1);
                    }
                }
            }

            CopyRecursive(0);

            // Create backward function for N-dimensional case
            var capturedShape = (int[])shape.Clone();
            var capturedOutputShape = (int[])outputShape.Clone();
            var capturedPadWidth = (int[,])padWidth.Clone();

            void BackwardFunction(Tensor<T> gradient)
            {
                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(capturedShape);
                    var gradInputIndices = new int[capturedShape.Length];
                    var gradOutputIndices = new int[capturedOutputShape.Length];

                    void ExtractGradientRecursive(int dim)
                    {
                        if (dim == capturedShape.Length)
                        {
                            var inputFlatIdx = ComputeFlatIndex(gradInputIndices, capturedShape);
                            var outputFlatIdx = ComputeFlatIndex(gradOutputIndices, capturedOutputShape);
                            gradA[inputFlatIdx] = gradient[outputFlatIdx];
                        }
                        else
                        {
                            for (int i = 0; i < capturedShape[dim]; i++)
                            {
                                gradInputIndices[dim] = i;
                                gradOutputIndices[dim] = i + capturedPadWidth[dim, 0];
                                ExtractGradientRecursive(dim + 1);
                            }
                        }
                    }

                    ExtractGradientRecursive(0);

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

            node.OperationType = OperationType.Pad;
            node.OperationParams = new Dictionary<string, object>
            {
                { "PadWidth", padWidth },
                { "Value", value! }
            };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
        }
    }

    /// <summary>
    /// Computes flat index from multi-dimensional indices for N-dimensional tensors.
    /// </summary>
    private static int ComputeFlatIndex(int[] indices, int[] shape)
    {
        int flatIdx = 0;
        int multiplier = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            flatIdx += indices[d] * multiplier;
            multiplier *= shape[d];
        }
        return flatIdx;
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
        var shape = a.Value.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("MaxPool2D requires 4D input [batch, channels, height, width]");

        strides ??= poolSize;

        // Use IEngine for GPU/CPU acceleration
        var engine = AiDotNetEngine.Current;
        var result = engine.MaxPool2DWithIndices(a.Value, poolSize, strides, out var maxIndices);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Use IEngine for backward pass
                var gradA = engine.MaxPool2DBackward(gradient, maxIndices, shape, poolSize, strides);

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.MaxPool2D;
        node.OperationParams = new Dictionary<string, object>
        {
            { "KernelSize", poolSize },
            { "Stride", strides },
            { "Padding", new int[] { 0, 0 } }
        };

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
        var shape = a.Value.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("AvgPool2D requires 4D input [batch, channels, height, width]");

        strides ??= poolSize;

        // Use IEngine for GPU/CPU acceleration
        var engine = AiDotNetEngine.Current;
        var result = engine.AvgPool2D(a.Value, poolSize, strides);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Use IEngine for backward pass
                var gradA = engine.AvgPool2DBackward(gradient, shape, poolSize, strides);

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.AvgPool2D;
        node.OperationParams = new Dictionary<string, object>
        {
            { "KernelSize", poolSize },
            { "Stride", strides },
            { "Padding", new int[] { 0, 0 } }
        };

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
            // Create non-nullable locals to satisfy compiler flow analysis
            var gammaNode = gamma;
            var betaNode = beta;
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
                        numOps.Multiply(norm, gammaNode.Value[f]),
                        betaNode.Value[f]);
                }
            }
            void BackwardFunction(Tensor<T> gradient)
            {
                // Gradients for gamma and beta
                if (gammaNode.RequiresGradient)
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
                    var existingGrad = gammaNode.Gradient;
                    gammaNode.Gradient = existingGrad == null ? gradGamma : existingGrad.Add(gradGamma);
                }
                if (betaNode.RequiresGradient)
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
                    var existingGrad = betaNode.Gradient;
                    betaNode.Gradient = existingGrad == null ? gradBeta : existingGrad.Add(gradBeta);
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
                            var gradNorm = numOps.Multiply(gradient[b, f], gammaNode.Value[f]);
                            gradNormSum = numOps.Add(gradNormSum, gradNorm);
                            gradNormDotNorm = numOps.Add(gradNormDotNorm,
                                numOps.Multiply(gradNorm, normalized[b, f]));
                        }
                        // Apply gradient formula
                        var featuresT = numOps.FromDouble(features);
                        for (int f = 0; f < features; f++)
                        {
                            var gradNorm = numOps.Multiply(gradient[b, f], gammaNode.Value[f]);
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
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }
            }
            var parents = new List<ComputationNode<T>> { a };
            parents.Add(gammaNode);
            parents.Add(betaNode);
            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || gammaNode.RequiresGradient || betaNode.RequiresGradient,
                parents: parents,
                backwardFunction: BackwardFunction,
                name: null);

            // Set JIT compiler metadata
            node.OperationType = OperationType.LayerNorm;
            node.OperationParams = new Dictionary<string, object>
            {
                { "Epsilon", epsilon }
            };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
        }
        else
        {
            // General N-dimensional LayerNorm implementation
            // Normalize over the last len(normalizedShape) dimensions

            int numNormDims = normalizedShape.Length;
            int numBatchDims = shape.Length - numNormDims;

            if (numBatchDims < 0)
                throw new ArgumentException("normalizedShape has more dimensions than input tensor");

            // Verify normalized dimensions match
            for (int i = 0; i < numNormDims; i++)
            {
                if (shape[numBatchDims + i] != normalizedShape[i])
                    throw new ArgumentException($"Dimension mismatch at position {i}: expected {normalizedShape[i]}, got {shape[numBatchDims + i]}");
            }

            // Compute number of elements in batch dimensions and normalized dimensions
            int batchElements = 1;
            for (int i = 0; i < numBatchDims; i++)
                batchElements *= shape[i];

            int normalizedElements = 1;
            for (int i = 0; i < numNormDims; i++)
                normalizedElements *= normalizedShape[i];

            // Create gamma and beta nodes if not provided
            var gammaNode = gamma ?? new ComputationNode<T>(
                Tensor<T>.CreateDefault(normalizedShape, numOps.One), requiresGradient: false);
            var betaNode = beta ?? new ComputationNode<T>(
                Tensor<T>.CreateDefault(normalizedShape, numOps.Zero), requiresGradient: false);

            var result = new Tensor<T>(shape);
            var normalized = new Tensor<T>(shape);
            var means = new T[batchElements];
            var variances = new T[batchElements];
            var epsInner = numOps.FromDouble(epsilon);

            // Compute mean and variance for each batch element
            for (int b = 0; b < batchElements; b++)
            {
                int batchOffset = b * normalizedElements;

                // Compute mean
                var sum = numOps.Zero;
                for (int n = 0; n < normalizedElements; n++)
                {
                    sum = numOps.Add(sum, a.Value[batchOffset + n]);
                }
                means[b] = numOps.Divide(sum, numOps.FromDouble(normalizedElements));

                // Compute variance
                var varSum = numOps.Zero;
                for (int n = 0; n < normalizedElements; n++)
                {
                    var diff = numOps.Subtract(a.Value[batchOffset + n], means[b]);
                    varSum = numOps.Add(varSum, numOps.Multiply(diff, diff));
                }
                variances[b] = numOps.Divide(varSum, numOps.FromDouble(normalizedElements));

                // Normalize and apply gamma/beta
                var std = numOps.Sqrt(numOps.Add(variances[b], epsInner));
                for (int n = 0; n < normalizedElements; n++)
                {
                    var norm = numOps.Divide(numOps.Subtract(a.Value[batchOffset + n], means[b]), std);
                    normalized[batchOffset + n] = norm;
                    result[batchOffset + n] = numOps.Add(
                        numOps.Multiply(norm, gammaNode.Value[n]),
                        betaNode.Value[n]);
                }
            }

            // Capture variables for backward function
            var capturedShape = (int[])shape.Clone();
            var capturedNumBatchDims = numBatchDims;
            var capturedBatchElements = batchElements;
            var capturedNormalizedElements = normalizedElements;

            void BackwardFunction(Tensor<T> gradient)
            {
                // Gradient for gamma
                if (gammaNode.RequiresGradient)
                {
                    var gradGamma = new Tensor<T>(normalizedShape);
                    for (int n = 0; n < capturedNormalizedElements; n++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < capturedBatchElements; b++)
                        {
                            sum = numOps.Add(sum,
                                numOps.Multiply(gradient[b * capturedNormalizedElements + n], normalized[b * capturedNormalizedElements + n]));
                        }
                        gradGamma[n] = sum;
                    }
                    var existingGrad = gammaNode.Gradient;
                    gammaNode.Gradient = existingGrad == null ? gradGamma : existingGrad.Add(gradGamma);
                }

                // Gradient for beta
                if (betaNode.RequiresGradient)
                {
                    var gradBeta = new Tensor<T>(normalizedShape);
                    for (int n = 0; n < capturedNormalizedElements; n++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < capturedBatchElements; b++)
                        {
                            sum = numOps.Add(sum, gradient[b * capturedNormalizedElements + n]);
                        }
                        gradBeta[n] = sum;
                    }
                    var existingGrad = betaNode.Gradient;
                    betaNode.Gradient = existingGrad == null ? gradBeta : existingGrad.Add(gradBeta);
                }

                // Gradient for input
                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(capturedShape);
                    var featuresT = numOps.FromDouble(capturedNormalizedElements);

                    for (int b = 0; b < capturedBatchElements; b++)
                    {
                        int batchOffset = b * capturedNormalizedElements;
                        var std = numOps.Sqrt(numOps.Add(variances[b], epsInner));
                        var invStd = numOps.Divide(numOps.One, std);

                        // Compute gradient sums
                        var gradNormSum = numOps.Zero;
                        var gradNormDotNorm = numOps.Zero;
                        for (int n = 0; n < capturedNormalizedElements; n++)
                        {
                            var gradNorm = numOps.Multiply(gradient[batchOffset + n], gammaNode.Value[n]);
                            gradNormSum = numOps.Add(gradNormSum, gradNorm);
                            gradNormDotNorm = numOps.Add(gradNormDotNorm,
                                numOps.Multiply(gradNorm, normalized[batchOffset + n]));
                        }

                        // Apply gradient formula
                        for (int n = 0; n < capturedNormalizedElements; n++)
                        {
                            var gradNorm = numOps.Multiply(gradient[batchOffset + n], gammaNode.Value[n]);
                            var term1 = gradNorm;
                            var term2 = numOps.Divide(gradNormSum, featuresT);
                            var term3 = numOps.Divide(
                                numOps.Multiply(normalized[batchOffset + n], gradNormDotNorm), featuresT);
                            gradA[batchOffset + n] = numOps.Multiply(
                                numOps.Subtract(numOps.Subtract(term1, term2), term3), invStd);
                        }
                    }
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }
            }

            var parents = new List<ComputationNode<T>> { a, gammaNode, betaNode };
            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || gammaNode.RequiresGradient || betaNode.RequiresGradient,
                parents: parents,
                backwardFunction: BackwardFunction,
                name: null);

            node.OperationType = OperationType.LayerNorm;
            node.OperationParams = new Dictionary<string, object> { { "Epsilon", epsilon } };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
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
            // Create non-nullable locals to satisfy compiler flow analysis
            var gammaNode = gamma;
            var betaNode = beta;
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
                        numOps.Multiply(norm, gammaNode.Value[f]),
                        betaNode.Value[f]);
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
                                    numOps.Multiply(gradient[b, f], gammaNode.Value[f]),
                                    invStd);
                            }
                        }
                        var existingGrad = a.Gradient;
                        a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                    }
                    return;
                }
                // Training mode: full gradient computation
                // Gradients for gamma and beta
                if (gammaNode.RequiresGradient)
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
                    var existingGrad = gammaNode.Gradient;
                    gammaNode.Gradient = existingGrad == null ? gradGamma : existingGrad.Add(gradGamma);
                }
                if (betaNode.RequiresGradient)
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
                    var existingGrad = betaNode.Gradient;
                    betaNode.Gradient = existingGrad == null ? gradBeta : existingGrad.Add(gradBeta);
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
                            var grad = numOps.Multiply(gradient[b, f], gammaNode.Value[f]);
                            gradSum = numOps.Add(gradSum, grad);
                            gradNormSum = numOps.Add(gradNormSum,
                                numOps.Multiply(grad, normalized[b, f]));
                        }
                        // Apply gradient formula
                        for (int b = 0; b < batchSize; b++)
                        {
                            var grad = numOps.Multiply(gradient[b, f], gammaNode.Value[f]);
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
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }
            }
            var parents = new List<ComputationNode<T>> { a };
            parents.Add(gammaNode);
            parents.Add(betaNode);
            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || gammaNode.RequiresGradient || betaNode.RequiresGradient,
                parents: parents,
                backwardFunction: BackwardFunction,
                name: null);

            // Set JIT compiler metadata
            node.OperationType = OperationType.BatchNorm;
            node.OperationParams = new Dictionary<string, object>
            {
                { "Epsilon", epsilon }
            };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
        }
        else if (shape.Length == 4)
        {
            // 4D tensor BatchNorm: [batch, channels, height, width]
            // Normalize per channel across batch and spatial dimensions
            int batchSize = shape[0];
            int channels = shape[1];
            int height = shape[2];
            int width = shape[3];
            int spatialSize = height * width;

            // Create gamma and beta nodes if not provided
            var gammaNode = gamma ?? new ComputationNode<T>(
                Tensor<T>.CreateDefault(new int[] { channels }, numOps.One), requiresGradient: false);
            var betaNode = beta ?? new ComputationNode<T>(
                Tensor<T>.CreateDefault(new int[] { channels }, numOps.Zero), requiresGradient: false);

            var result = new Tensor<T>(shape);
            var normalized = new Tensor<T>(shape);
            var batchMean = new T[channels];
            var batchVar = new T[channels];
            var eps4D = numOps.FromDouble(epsilon);
            var totalElements = batchSize * spatialSize;
            var totalElementsT = numOps.FromDouble(totalElements);

            // Compute per-channel mean and variance
            for (int c = 0; c < channels; c++)
            {
                // Compute mean
                var sum = numOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            sum = numOps.Add(sum, a.Value[b, c, h, w]);
                        }
                    }
                }
                batchMean[c] = numOps.Divide(sum, totalElementsT);

                // Compute variance
                var varSum = numOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            var diff = numOps.Subtract(a.Value[b, c, h, w], batchMean[c]);
                            varSum = numOps.Add(varSum, numOps.Multiply(diff, diff));
                        }
                    }
                }
                batchVar[c] = numOps.Divide(varSum, totalElementsT);

                // Normalize and apply gamma/beta
                var std = numOps.Sqrt(numOps.Add(batchVar[c], eps4D));
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            var norm = numOps.Divide(numOps.Subtract(a.Value[b, c, h, w], batchMean[c]), std);
                            normalized[b, c, h, w] = norm;
                            result[b, c, h, w] = numOps.Add(
                                numOps.Multiply(norm, gammaNode.Value[c]),
                                betaNode.Value[c]);
                        }
                    }
                }
            }

            void BackwardFunction(Tensor<T> gradient)
            {
                // Gradient for gamma
                if (gammaNode.RequiresGradient)
                {
                    var gradGamma = new Tensor<T>(new int[] { channels });
                    for (int c = 0; c < channels; c++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int h = 0; h < height; h++)
                            {
                                for (int w = 0; w < width; w++)
                                {
                                    sum = numOps.Add(sum, numOps.Multiply(gradient[b, c, h, w], normalized[b, c, h, w]));
                                }
                            }
                        }
                        gradGamma[c] = sum;
                    }
                    var existingGrad = gammaNode.Gradient;
                    gammaNode.Gradient = existingGrad == null ? gradGamma : existingGrad.Add(gradGamma);
                }

                // Gradient for beta
                if (betaNode.RequiresGradient)
                {
                    var gradBeta = new Tensor<T>(new int[] { channels });
                    for (int c = 0; c < channels; c++)
                    {
                        var sum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int h = 0; h < height; h++)
                            {
                                for (int w = 0; w < width; w++)
                                {
                                    sum = numOps.Add(sum, gradient[b, c, h, w]);
                                }
                            }
                        }
                        gradBeta[c] = sum;
                    }
                    var existingGrad = betaNode.Gradient;
                    betaNode.Gradient = existingGrad == null ? gradBeta : existingGrad.Add(gradBeta);
                }

                // Gradient for input
                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(shape);
                    for (int c = 0; c < channels; c++)
                    {
                        var std = numOps.Sqrt(numOps.Add(batchVar[c], eps4D));
                        var invStd = numOps.Divide(numOps.One, std);

                        var gradSum = numOps.Zero;
                        var gradNormSum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int h = 0; h < height; h++)
                            {
                                for (int w = 0; w < width; w++)
                                {
                                    var grad = numOps.Multiply(gradient[b, c, h, w], gammaNode.Value[c]);
                                    gradSum = numOps.Add(gradSum, grad);
                                    gradNormSum = numOps.Add(gradNormSum, numOps.Multiply(grad, normalized[b, c, h, w]));
                                }
                            }
                        }

                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int h = 0; h < height; h++)
                            {
                                for (int w = 0; w < width; w++)
                                {
                                    var grad = numOps.Multiply(gradient[b, c, h, w], gammaNode.Value[c]);
                                    var term1 = grad;
                                    var term2 = numOps.Divide(gradSum, totalElementsT);
                                    var term3 = numOps.Divide(numOps.Multiply(normalized[b, c, h, w], gradNormSum), totalElementsT);
                                    gradA[b, c, h, w] = numOps.Multiply(numOps.Subtract(numOps.Subtract(term1, term2), term3), invStd);
                                }
                            }
                        }
                    }
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }
            }

            var parents = new List<ComputationNode<T>> { a, gammaNode, betaNode };
            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || gammaNode.RequiresGradient || betaNode.RequiresGradient,
                parents: parents,
                backwardFunction: BackwardFunction,
                name: null);

            node.OperationType = OperationType.BatchNorm;
            node.OperationParams = new Dictionary<string, object> { { "Epsilon", epsilon } };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
        }
        else
        {
            // Generic N-dimensional BatchNorm: normalize over all dimensions except axis 1 (channels)
            int channels = shape[1];
            int totalElements = a.Value.Length / channels;
            var totalElementsT = numOps.FromDouble(totalElements);

            var gammaNode = gamma ?? new ComputationNode<T>(
                Tensor<T>.CreateDefault(new int[] { channels }, numOps.One), requiresGradient: false);
            var betaNode = beta ?? new ComputationNode<T>(
                Tensor<T>.CreateDefault(new int[] { channels }, numOps.Zero), requiresGradient: false);

            var result = new Tensor<T>(shape);
            var normalized = new Tensor<T>(shape);
            var batchMean = new T[channels];
            var batchVar = new T[channels];
            var epsND = numOps.FromDouble(epsilon);

            // Compute per-channel statistics
            int elementsPerChannel = a.Value.Length / channels;
            int channelStride = 1;
            for (int i = 2; i < shape.Length; i++)
                channelStride *= shape[i];

            for (int c = 0; c < channels; c++)
            {
                var sum = numOps.Zero;
                int count = 0;
                for (int i = 0; i < a.Value.Length; i++)
                {
                    int channelIdx = (i / channelStride) % channels;
                    if (channelIdx == c)
                    {
                        sum = numOps.Add(sum, a.Value[i]);
                        count++;
                    }
                }
                batchMean[c] = numOps.Divide(sum, numOps.FromDouble(count));

                var varSum = numOps.Zero;
                for (int i = 0; i < a.Value.Length; i++)
                {
                    int channelIdx = (i / channelStride) % channels;
                    if (channelIdx == c)
                    {
                        var diff = numOps.Subtract(a.Value[i], batchMean[c]);
                        varSum = numOps.Add(varSum, numOps.Multiply(diff, diff));
                    }
                }
                batchVar[c] = numOps.Divide(varSum, numOps.FromDouble(count));

                var std = numOps.Sqrt(numOps.Add(batchVar[c], epsND));
                for (int i = 0; i < a.Value.Length; i++)
                {
                    int channelIdx = (i / channelStride) % channels;
                    if (channelIdx == c)
                    {
                        var norm = numOps.Divide(numOps.Subtract(a.Value[i], batchMean[c]), std);
                        normalized[i] = norm;
                        result[i] = numOps.Add(numOps.Multiply(norm, gammaNode.Value[c]), betaNode.Value[c]);
                    }
                }
            }

            void BackwardFunction(Tensor<T> gradient)
            {
                // Simplified backward for N-dimensional case
                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(shape);
                    for (int c = 0; c < channels; c++)
                    {
                        var std = numOps.Sqrt(numOps.Add(batchVar[c], epsND));
                        var invStd = numOps.Divide(numOps.One, std);

                        var gradSum = numOps.Zero;
                        var gradNormSum = numOps.Zero;
                        int count = 0;

                        for (int i = 0; i < gradient.Length; i++)
                        {
                            int channelIdx = (i / channelStride) % channels;
                            if (channelIdx == c)
                            {
                                var grad = numOps.Multiply(gradient[i], gammaNode.Value[c]);
                                gradSum = numOps.Add(gradSum, grad);
                                gradNormSum = numOps.Add(gradNormSum, numOps.Multiply(grad, normalized[i]));
                                count++;
                            }
                        }

                        var countT = numOps.FromDouble(count);
                        for (int i = 0; i < gradient.Length; i++)
                        {
                            int channelIdx = (i / channelStride) % channels;
                            if (channelIdx == c)
                            {
                                var grad = numOps.Multiply(gradient[i], gammaNode.Value[c]);
                                var term1 = grad;
                                var term2 = numOps.Divide(gradSum, countT);
                                var term3 = numOps.Divide(numOps.Multiply(normalized[i], gradNormSum), countT);
                                gradA[i] = numOps.Multiply(numOps.Subtract(numOps.Subtract(term1, term2), term3), invStd);
                            }
                        }
                    }
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }
            }

            var parents = new List<ComputationNode<T>> { a, gammaNode, betaNode };
            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || gammaNode.RequiresGradient || betaNode.RequiresGradient,
                parents: parents,
                backwardFunction: BackwardFunction,
                name: null);

            node.OperationType = OperationType.BatchNorm;
            node.OperationParams = new Dictionary<string, object> { { "Epsilon", epsilon } };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
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
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var kernelShape = kernel.Value.Shape;
        if (inputShape.Length != 4)
            throw new ArgumentException("Conv2D requires 4D input [batch, inChannels, height, width]");
        if (kernelShape.Length != 4)
            throw new ArgumentException("Conv2D requires 4D kernel [outChannels, inChannels, kernelH, kernelW]");
        stride ??= new int[] { 1, 1 };
        padding ??= new int[] { 0, 0 };
        var dilation = new int[] { 1, 1 };
        int outChannels = kernelShape[0];

        // Forward pass: Use engine for GPU-accelerated convolution
        var result = engine.Conv2D(input.Value, kernel.Value, stride, padding, dilation);

        // Add bias if provided
        if (bias != null)
        {
            int batch = result.Shape[0];
            int outH = result.Shape[2];
            int outW = result.Shape[3];
            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    var biasVal = bias.Value[oc];
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            result[b, oc, oh, ow] = numOps.Add(result[b, oc, oh, ow], biasVal);
                        }
                    }
                }
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient w.r.t. input using engine
            if (input.RequiresGradient)
            {
                var gradInput = engine.Conv2DBackwardInput(gradient, kernel.Value, inputShape, stride, padding, dilation);
                var existingGrad = input.Gradient;
                input.Gradient = existingGrad == null ? gradInput : engine.TensorAdd(existingGrad, gradInput);
            }
            // Gradient w.r.t. kernel using engine
            if (kernel.RequiresGradient)
            {
                var gradKernel = engine.Conv2DBackwardKernel(gradient, input.Value, kernelShape, stride, padding, dilation);
                var existingGrad = kernel.Gradient;
                kernel.Gradient = existingGrad == null ? gradKernel : engine.TensorAdd(existingGrad, gradKernel);
            }
            // Gradient w.r.t. bias
            if (bias != null && bias.RequiresGradient)
            {
                var gradBias = new Tensor<T>(new int[] { outChannels });
                int batch = gradient.Shape[0];
                int outH = gradient.Shape[2];
                int outW = gradient.Shape[3];
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
                var existingGrad = bias.Gradient;
                bias.Gradient = existingGrad == null ? gradBias : existingGrad.Add(gradBias);
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

        // Set JIT compiler metadata
        node.OperationType = OperationType.Conv2D;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Stride", stride },
            { "Padding", padding }
        };

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

        int inChannels = inputShape[1];
        int kernelInChannels = kernelShape[0];
        int outChannels = kernelShape[1];

        if (inChannels != kernelInChannels)
            throw new ArgumentException($"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels})");

        // Use IEngine for GPU/CPU acceleration
        var engine = AiDotNetEngine.Current;
        var result = engine.ConvTranspose2D(input.Value, kernel.Value, stride, padding, outputPadding);

        // Add bias if provided
        if (bias != null)
        {
            int batch = result.Shape[0];
            int outH = result.Shape[2];
            int outW = result.Shape[3];
            for (int b = 0; b < batch; b++)
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
            // Gradient w.r.t. input
            if (input.RequiresGradient)
            {
                var gradInput = engine.ConvTranspose2DBackwardInput(gradient, kernel.Value, inputShape, stride, padding);

                if (input.Gradient == null)
                {
                    input.Gradient = gradInput;
                }
                else
                {
                    input.Gradient = input.Gradient.Add(gradInput);
                }
            }

            // Gradient w.r.t. kernel
            if (kernel.RequiresGradient)
            {
                var gradKernel = engine.ConvTranspose2DBackwardKernel(gradient, input.Value, kernelShape, stride, padding);

                if (kernel.Gradient == null)
                {
                    kernel.Gradient = gradKernel;
                }
                else
                {
                    kernel.Gradient = kernel.Gradient.Add(gradKernel);
                }
            }

            // Gradient w.r.t. bias
            if (bias != null && bias.RequiresGradient)
            {
                int batch = gradient.Shape[0];
                int outH = gradient.Shape[2];
                int outW = gradient.Shape[3];

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
                {
                    bias.Gradient = gradBias;
                }
                else
                {
                    bias.Gradient = bias.Gradient.Add(gradBias);
                }
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

        // Set JIT compiler metadata
        node.OperationType = OperationType.ConvTranspose2D;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Stride", stride },
            { "Padding", padding },
            { "OutputPadding", outputPadding }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Reduces a tensor by computing the maximum value along specified axes.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="axes">The axes along which to compute the maximum. If null, reduces over all axes.</param>
    /// <param name="keepDims">Whether to keep the reduced dimensions with size 1.</param>
    /// <returns>A computation node representing the result of the reduce max operation.</returns>
    public static ComputationNode<T> ReduceMax(ComputationNode<T> a, int[]? axes = null, bool keepDims = false)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = a.Value.Shape;
        // If axes is null, reduce all dimensions
        if (axes == null)
        {
            axes = Enumerable.Range(0, inputShape.Length).ToArray();
        }
        // Compute output shape
        var outputShape = new List<int>();
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (!axes.Contains(i))
            {
                outputShape.Add(inputShape[i]);
            }
            else if (keepDims)
            {
                outputShape.Add(1);
            }
        }
        if (outputShape.Count == 0)
            outputShape.Add(1);
        var result = new Tensor<T>(outputShape.ToArray());
        // Store max indices for gradient routing
        var maxIndices = new Dictionary<string, int[]>();
        // Compute forward pass: find max values
        void ComputeMax(int[] currentIndices, int dim, int[] outputIndices, int outDim)
        {
            if (dim == inputShape.Length)
            {
                // Reached a leaf, update result
                var value = a.Value[currentIndices];
                var outKey = string.Join(",", outputIndices.Take(outputShape.Count));
                if (!maxIndices.ContainsKey(outKey))
                {
                    result[outputIndices] = value;
                    maxIndices[outKey] = (int[])currentIndices.Clone();
                }
                else
                {
                    if (numOps.GreaterThan(value, result[outputIndices]))
                    {
                        result[outputIndices] = value;
                        maxIndices[outKey] = (int[])currentIndices.Clone();
                    }
                }
                return;
            }
            if (axes.Contains(dim))
            {
                // Reduce along this dimension
                for (int i = 0; i < inputShape[dim]; i++)
                {
                    currentIndices[dim] = i;
                    ComputeMax(currentIndices, dim + 1, outputIndices, outDim);
                }
            }
            else
            {
                // Keep this dimension
                for (int i = 0; i < inputShape[dim]; i++)
                {
                    currentIndices[dim] = i;
                    outputIndices[outDim] = i;
                    ComputeMax(currentIndices, dim + 1, outputIndices, outDim + 1);
                }
            }
        }
        ComputeMax(new int[inputShape.Length], 0, new int[outputShape.Count], 0);
        // Backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            if (!a.RequiresGradient) return;
            var gradInput = new Tensor<T>(inputShape);
            // Route gradients only to max positions
            foreach (var kvp in maxIndices)
            {
                var outIndices = kvp.Key.Split(',').Select(int.Parse).ToArray();
                var inIndices = kvp.Value;
                gradInput[inIndices] = numOps.Add(gradInput[inIndices], gradient[outIndices]);
            }
            var existingGrad = a.Gradient;
            a.Gradient = existingGrad == null ? gradInput : existingGrad.Add(gradInput);
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.ReduceMax;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axes", axes! },
            { "KeepDims", keepDims }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Reduces a tensor by computing the mean value along specified axes.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="axes">The axes along which to compute the mean. If null, reduces over all axes.</param>
    /// <param name="keepDims">Whether to keep the reduced dimensions with size 1.</param>
    /// <returns>A computation node representing the result of the reduce mean operation.</returns>
    public static ComputationNode<T> ReduceMean(ComputationNode<T> a, int[]? axes = null, bool keepDims = false)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = a.Value.Shape;
        // If axes is null, reduce all dimensions
        if (axes == null)
        {
            axes = Enumerable.Range(0, inputShape.Length).ToArray();
        }
        // Compute output shape and count for averaging
        var outputShape = new List<int>();
        int reduceCount = 1;
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (!axes.Contains(i))
            {
                outputShape.Add(inputShape[i]);
            }
            else
            {
                reduceCount *= inputShape[i];
                if (keepDims)
                {
                    outputShape.Add(1);
                }
            }
        }
        if (outputShape.Count == 0)
            outputShape.Add(1);
        var result = new Tensor<T>(outputShape.ToArray());
        var divisor = numOps.FromDouble((double)reduceCount);
        // Compute forward pass: sum and then divide
        void ComputeSum(int[] currentIndices, int dim, int[] outputIndices)
        {
            if (dim == inputShape.Length)
            {
                var value = a.Value[currentIndices];
                result[outputIndices] = numOps.Add(result[outputIndices], value);
                return;
            }
            if (axes.Contains(dim))
            {
                for (int i = 0; i < inputShape[dim]; i++)
                {
                    currentIndices[dim] = i;
                    ComputeSum(currentIndices, dim + 1, outputIndices);
                }
            }
            else
            {
                int outIdx = Array.IndexOf(outputShape.ToArray(), inputShape[dim]);
                for (int i = 0; i < inputShape[dim]; i++)
                {
                    currentIndices[dim] = i;
                    outputIndices[outIdx] = i;
                    ComputeSum(currentIndices, dim + 1, outputIndices);
                }
            }
        }
        ComputeSum(new int[inputShape.Length], 0, new int[outputShape.Count]);
        // Divide by count to get mean
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = numOps.Divide(result[i], divisor);
        }
        // Backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            if (!a.RequiresGradient) return;
            var gradInput = new Tensor<T>(inputShape);
            var gradScale = numOps.Divide(numOps.One, divisor);
            // Broadcast gradient back to input shape
            void BroadcastGrad(int[] currentIndices, int dim, int[] outputIndices)
            {
                if (dim == inputShape.Length)
                {
                    gradInput[currentIndices] = numOps.Multiply(gradient[outputIndices], gradScale);
                    return;
                }
                if (axes.Contains(dim))
                {
                    for (int i = 0; i < inputShape[dim]; i++)
                    {
                        currentIndices[dim] = i;
                        BroadcastGrad(currentIndices, dim + 1, outputIndices);
                    }
                }
                else
                {
                    int outIdx = Array.IndexOf(outputShape.ToArray(), inputShape[dim]);
                    for (int i = 0; i < inputShape[dim]; i++)
                    {
                        currentIndices[dim] = i;
                        outputIndices[outIdx] = i;
                        BroadcastGrad(currentIndices, dim + 1, outputIndices);
                    }
                }
            }
            BroadcastGrad(new int[inputShape.Length], 0, new int[outputShape.Count]);
            var existingGrad = a.Gradient;
            a.Gradient = existingGrad == null ? gradInput : existingGrad.Add(gradInput);
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.ReduceMean;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axes", axes! },
            { "KeepDims", keepDims }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Splits a tensor along a specified axis into multiple tensors.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="numSplits">The number of splits to create.</param>
    /// <param name="axis">The axis along which to split.</param>
    /// <returns>A list of computation nodes representing the split tensors.</returns>
    public static List<ComputationNode<T>> Split(ComputationNode<T> a, int numSplits, int axis = 0)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = a.Value.Shape;
        if (axis < 0 || axis >= inputShape.Length)
            throw new ArgumentException($"Axis {axis} is out of bounds for tensor with {inputShape.Length} dimensions.");
        if (inputShape[axis] % numSplits != 0)
            throw new ArgumentException($"Dimension size {inputShape[axis]} is not evenly divisible by {numSplits}.");
        int splitSize = inputShape[axis] / numSplits;
        var results = new List<ComputationNode<T>>();
        // Create output shapes
        var outputShape = (int[])inputShape.Clone();
        outputShape[axis] = splitSize;
        // Forward pass: split the tensor
        var splitTensors = new List<Tensor<T>>();
        for (int split = 0; split < numSplits; split++)
        {
            var splitTensor = new Tensor<T>(outputShape);
            splitTensors.Add(splitTensor);
        }
        // Copy data to split tensors
        void CopySplit(int[] currentIndices, int dim)
        {
            if (dim == inputShape.Length)
            {
                var value = a.Value[currentIndices];
                int splitIdx = currentIndices[axis] / splitSize;
                var localIndices = (int[])currentIndices.Clone();
                localIndices[axis] = currentIndices[axis] % splitSize;
                splitTensors[splitIdx][localIndices] = value;
                return;
            }
            for (int i = 0; i < inputShape[dim]; i++)
            {
                currentIndices[dim] = i;
                CopySplit(currentIndices, dim + 1);
            }
        }
        CopySplit(new int[inputShape.Length], 0);
        // Create nodes for each split
        for (int split = 0; split < numSplits; split++)
        {
            var splitIndex = split;
            void BackwardFunction(Tensor<T> gradient)
            {
                if (!a.RequiresGradient) return;
                if (a.Gradient == null)
                    a.Gradient = new Tensor<T>(inputShape);
                // Accumulate gradient back to input
                void AccumulateGrad(int[] currentIndices, int dim)
                {
                    if (dim == outputShape.Length)
                    {
                        var inputIndices = (int[])currentIndices.Clone();
                        inputIndices[axis] = currentIndices[axis] + splitIndex * splitSize;
                        a.Gradient[inputIndices] = numOps.Add(a.Gradient[inputIndices], gradient[currentIndices]);
                        return;
                    }
                    for (int i = 0; i < outputShape[dim]; i++)
                    {
                        currentIndices[dim] = i;
                        AccumulateGrad(currentIndices, dim + 1);
                    }
                }
                AccumulateGrad(new int[outputShape.Length], 0);
            }
            var node = new ComputationNode<T>(
                value: splitTensors[split],
                requiresGradient: a.RequiresGradient,
                parents: new List<ComputationNode<T>> { a },
                backwardFunction: BackwardFunction,
                name: null);

            // Set JIT compiler metadata
            node.OperationType = OperationType.Split;
            node.OperationParams = new Dictionary<string, object>
            {
                { "Axis", axis },
                { "NumSplits", numSplits },
                { "SplitIndex", split }
            };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            results.Add(node);
        }
        return results;
    }
    /// <summary>
    /// Crops a tensor by removing elements from the edges.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="cropping">Array of [top, bottom, left, right] cropping amounts for 4D tensors.</param>
    /// <returns>A computation node representing the cropped tensor.</returns>
    public static ComputationNode<T> Crop(ComputationNode<T> a, int[] cropping)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = a.Value.Shape;
        if (inputShape.Length == 4 && cropping.Length == 4)
        {
            // 4D tensor: [batch, channels, height, width]
            int top = cropping[0];
            int bottom = cropping[1];
            int left = cropping[2];
            int right = cropping[3];
            int outH = inputShape[2] - top - bottom;
            int outW = inputShape[3] - left - right;
            if (outH <= 0 || outW <= 0)
                throw new ArgumentException("Cropping results in non-positive dimensions.");
            var outputShape = new int[] { inputShape[0], inputShape[1], outH, outW };
            var result = new Tensor<T>(outputShape);
            // Forward: copy cropped region
            for (int b = 0; b < outputShape[0]; b++)
            {
                for (int c = 0; c < outputShape[1]; c++)
                {
                    for (int h = 0; h < outH; h++)
                    {
                        for (int w = 0; w < outW; w++)
                        {
                            result[b, c, h, w] = a.Value[b, c, h + top, w + left];
                        }
                    }
                }
            }
            void BackwardFunction(Tensor<T> gradient)
            {
                if (!a.RequiresGradient) return;
                if (a.Gradient == null)
                    a.Gradient = new Tensor<T>(inputShape);
                // Backward: place gradient in cropped region
                for (int b = 0; b < outputShape[0]; b++)
                {
                    for (int c = 0; c < outputShape[1]; c++)
                    {
                        for (int h = 0; h < outH; h++)
                        {
                            for (int w = 0; w < outW; w++)
                            {
                                a.Gradient[b, c, h + top, w + left] = numOps.Add(
                                    a.Gradient[b, c, h + top, w + left],
                                    gradient[b, c, h, w]);
                            }
                        }
                    }
                }
            }
            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient,
                parents: new List<ComputationNode<T>> { a },
                backwardFunction: BackwardFunction,
                name: null);

            // Set JIT compiler metadata
            node.OperationType = OperationType.Crop;
            node.OperationParams = new Dictionary<string, object>
            {
                { "Cropping", cropping }
            };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);
            return node;
        }
        else
        {
            throw new NotSupportedException($"Crop operation not supported for shape {string.Join("x", inputShape)} with cropping {string.Join(",", cropping)}");
        }
    }
    /// <summary>
    /// Upsamples a tensor using nearest neighbor interpolation.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="scale">The upsampling scale factor.</param>
    /// <returns>A computation node representing the upsampled tensor.</returns>
    public static ComputationNode<T> Upsample(ComputationNode<T> a, int scale)
    {
        var engine = AiDotNetEngine.Current;
        var inputShape = a.Value.Shape;

        if (inputShape.Length != 4)
            throw new ArgumentException("Upsample expects 4D input [batch, channels, height, width]");

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.Upsample(a.Value, scale, scale);

        // Capture for backward pass
        int[] capturedInputShape = inputShape;
        int capturedScale = scale;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (!a.RequiresGradient) return;

            // Use IEngine for GPU-accelerated backward pass
            var gradA = engine.UpsampleBackward(gradient, capturedInputShape, capturedScale, capturedScale);

            var existingGrad = a.Gradient;

            a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Upsample;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Scale", scale }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Performs pixel shuffle (depth-to-space) operation for sub-pixel convolution.
    /// </summary>
    /// <param name="a">The input computation node with shape [batch, channels, height, width].</param>
    /// <param name="upscaleFactor">The upscaling factor (r). Channels must be divisible by r².</param>
    /// <returns>A computation node with shape [batch, channels/(r²), height*r, width*r].</returns>
    public static ComputationNode<T> PixelShuffle(ComputationNode<T> a, int upscaleFactor)
    {
        var engine = AiDotNetEngine.Current;
        var inputShape = a.Value.Shape;

        if (inputShape.Length != 4)
            throw new ArgumentException("PixelShuffle expects 4D input [batch, channels, height, width]");

        int r2 = upscaleFactor * upscaleFactor;
        if (inputShape[1] % r2 != 0)
            throw new ArgumentException($"Channels {inputShape[1]} must be divisible by upscale_factor² ({r2})");

        // Use IEngine for GPU-accelerated forward pass
        var result = engine.PixelShuffle(a.Value, upscaleFactor);

        // Capture for backward pass
        int[] capturedInputShape = inputShape;
        int capturedUpscaleFactor = upscaleFactor;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (!a.RequiresGradient) return;

            // Use IEngine for GPU-accelerated backward pass
            var gradA = engine.PixelShuffleBackward(gradient, capturedInputShape, capturedUpscaleFactor);

            var existingGrad = a.Gradient;

            a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.PixelShuffle;
        node.OperationParams = new Dictionary<string, object>
        {
            { "UpscaleFactor", upscaleFactor }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Performs dilated (atrous) 2D convolution operation.
    /// </summary>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <param name="kernel">The convolution kernel with shape [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="bias">Optional bias tensor with shape [out_channels].</param>
    /// <param name="stride">The stride for the convolution. Defaults to [1, 1].</param>
    /// <param name="padding">The padding for the convolution. Defaults to [0, 0].</param>
    /// <param name="dilation">The dilation rate for the convolution. Defaults to [1, 1].</param>
    /// <returns>A computation node representing the dilated convolution result.</returns>
    public static ComputationNode<T> DilatedConv2D(
        ComputationNode<T> input,
        ComputationNode<T> kernel,
        ComputationNode<T>? bias = null,
        int[]? stride = null,
        int[]? padding = null,
        int[]? dilation = null)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var kernelShape = kernel.Value.Shape;
        if (inputShape.Length != 4 || kernelShape.Length != 4)
            throw new ArgumentException("DilatedConv2D expects 4D tensors [batch, channels, height, width]");
        stride ??= new int[] { 1, 1 };
        padding ??= new int[] { 0, 0 };
        dilation ??= new int[] { 1, 1 };
        int outChannels = kernelShape[0];

        // Forward pass: Use engine for GPU-accelerated dilated convolution
        var result = engine.Conv2D(input.Value, kernel.Value, stride, padding, dilation);

        // Add bias if provided
        if (bias != null)
        {
            int batch = result.Shape[0];
            int outH = result.Shape[2];
            int outW = result.Shape[3];
            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    var biasVal = bias.Value[oc];
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            result[b, oc, oh, ow] = numOps.Add(result[b, oc, oh, ow], biasVal);
                        }
                    }
                }
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient w.r.t. input using engine
            if (input.RequiresGradient)
            {
                var gradInput = engine.Conv2DBackwardInput(gradient, kernel.Value, inputShape, stride, padding, dilation);
                var existingGrad = input.Gradient;
                input.Gradient = existingGrad == null ? gradInput : engine.TensorAdd(existingGrad, gradInput);
            }
            // Gradient w.r.t. kernel using engine
            if (kernel.RequiresGradient)
            {
                var gradKernel = engine.Conv2DBackwardKernel(gradient, input.Value, kernelShape, stride, padding, dilation);
                var existingGrad = kernel.Gradient;
                kernel.Gradient = existingGrad == null ? gradKernel : engine.TensorAdd(existingGrad, gradKernel);
            }
            // Gradient w.r.t. bias
            if (bias != null && bias.RequiresGradient)
            {
                var gradBias = new Tensor<T>(new int[] { outChannels });
                int batch = gradient.Shape[0];
                int outH = gradient.Shape[2];
                int outW = gradient.Shape[3];
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
                var existingGrad = bias.Gradient;
                bias.Gradient = existingGrad == null ? gradBias : existingGrad.Add(gradBias);
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

        // Set JIT compiler metadata
        node.OperationType = OperationType.DilatedConv2D;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Stride", stride },
            { "Padding", padding },
            { "Dilation", dilation }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Performs depthwise 2D convolution where each input channel is convolved with its own set of filters.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, in_channels, height, width]</param>
    /// <param name="kernel">Kernel tensor of shape [in_channels, multiplier, kernel_height, kernel_width]</param>
    /// <param name="bias">Optional bias tensor of shape [in_channels * multiplier]</param>
    /// <param name="stride">Stride for the convolution, defaults to [1, 1]</param>
    /// <param name="padding">Padding for the convolution, defaults to [0, 0]</param>
    /// <returns>Output tensor of shape [batch, in_channels * multiplier, out_height, out_width]</returns>
    /// <remarks>
    /// <para>
    /// Depthwise convolution applies a separate filter to each input channel independently, with no mixing
    /// across channels. This is in contrast to standard convolution which mixes all input channels.
    /// Each input channel gets 'multiplier' filters applied to it, producing 'multiplier' output channels.
    /// The total output channels is in_channels * multiplier.
    /// </para>
    /// <para>
    /// This operation is commonly used in MobileNets and other efficient architectures, often followed
    /// by a pointwise (1x1) convolution to mix channels. The combination dramatically reduces
    /// computational cost compared to standard convolution.
    /// </para>
    /// <para>
    /// Forward pass computes the depthwise convolution by applying each filter only to its corresponding
    /// input channel. Backward pass computes gradients with respect to input, kernel, and bias.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> DepthwiseConv2D(
        ComputationNode<T> input,
        ComputationNode<T> kernel,
        ComputationNode<T>? bias = null,
        int[]? stride = null,
        int[]? padding = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var kernelShape = kernel.Value.Shape;

        // Validate input shape (must be 4D: [batch, in_channels, height, width])
        if (inputShape.Length != 4)
            throw new ArgumentException("Input must be 4D tensor [batch, in_channels, height, width]");
        // Validate kernel shape (must be 4D: [in_channels, multiplier, kernel_height, kernel_width])
        if (kernelShape.Length != 4)
            throw new ArgumentException("Kernel must be 4D tensor [in_channels, multiplier, kernel_height, kernel_width]");
        if (inputShape[1] != kernelShape[0])
            throw new ArgumentException($"Input channels ({inputShape[1]}) must match kernel input channels ({kernelShape[0]})");

        // Default stride and padding
        stride ??= new int[] { 1, 1 };
        padding ??= new int[] { 0, 0 };
        if (stride.Length != 2 || padding.Length != 2)
            throw new ArgumentException("Stride and padding must be 2D arrays [height, width]");

        int multiplier = kernelShape[1];
        int outChannels = inputShape[1] * multiplier;

        // Validate bias if provided
        if (bias != null)
        {
            var biasShape = bias.Value.Shape;
            if (biasShape.Length != 1 || biasShape[0] != outChannels)
                throw new ArgumentException($"Bias must be 1D tensor of length {outChannels}");
        }

        // Use IEngine for GPU/CPU acceleration
        var engine = AiDotNetEngine.Current;
        var result = engine.DepthwiseConv2D(input.Value, kernel.Value, stride, padding);

        // Add bias if provided
        if (bias != null)
        {
            int batch = result.Shape[0];
            int outH = result.Shape[2];
            int outW = result.Shape[3];
            for (int b = 0; b < batch; b++)
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
            // Gradient w.r.t. input
            if (input.RequiresGradient)
            {
                var gradInput = engine.DepthwiseConv2DBackwardInput(gradient, kernel.Value, inputShape, stride, padding);

                if (input.Gradient == null)
                {
                    input.Gradient = gradInput;
                }
                else
                {
                    input.Gradient = input.Gradient.Add(gradInput);
                }
            }

            // Gradient w.r.t. kernel
            if (kernel.RequiresGradient)
            {
                var gradKernel = engine.DepthwiseConv2DBackwardKernel(gradient, input.Value, kernelShape, stride, padding);

                if (kernel.Gradient == null)
                {
                    kernel.Gradient = gradKernel;
                }
                else
                {
                    kernel.Gradient = kernel.Gradient.Add(gradKernel);
                }
            }

            // Gradient w.r.t. bias
            if (bias != null && bias.RequiresGradient)
            {
                if (bias.Gradient == null)
                    bias.Gradient = new Tensor<T>(new int[] { outChannels });

                int batch = gradient.Shape[0];
                int outH = gradient.Shape[2];
                int outW = gradient.Shape[3];
                for (int b = 0; b < batch; b++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            for (int ow = 0; ow < outW; ow++)
                            {
                                bias.Gradient[oc] = numOps.Add(bias.Gradient[oc], gradient[b, oc, oh, ow]);
                            }
                        }
                    }
                }
            }
        }

        var parents = bias != null
            ? new List<ComputationNode<T>> { input, kernel, bias }
            : new List<ComputationNode<T>> { input, kernel };

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient || kernel.RequiresGradient || (bias?.RequiresGradient ?? false),
            parents: parents,
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.DepthwiseConv2D;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Stride", stride },
            { "Padding", padding }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Performs locally connected 2D convolution where weights are NOT shared across spatial locations.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, in_channels, height, width]</param>
    /// <param name="weights">Weight tensor of shape [out_h, out_w, out_channels, in_channels, kernel_h, kernel_w]</param>
    /// <param name="bias">Optional bias tensor of shape [out_channels]</param>
    /// <param name="stride">Stride for the convolution, defaults to [1, 1]</param>
    /// <returns>Output tensor of shape [batch, out_channels, out_h, out_w]</returns>
    /// <remarks>
    /// <para>
    /// Locally connected convolution is like regular convolution but uses different weights for each
    /// spatial output location. This increases parameters but allows position-specific feature detection.
    /// </para>
    /// <para>
    /// Unlike Conv2D where weights are shared across all positions, LocallyConnectedConv2D uses
    /// unique weights for each (h,w) output position. This is useful when different regions have
    /// fundamentally different characteristics (e.g., face recognition where eyes/nose/mouth are
    /// at specific locations).
    /// </para>
    /// <para>
    /// Forward pass applies position-specific filters at each output location.
    /// Backward pass computes gradients with respect to input, position-specific weights, and bias.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> LocallyConnectedConv2D(
        ComputationNode<T> input,
        ComputationNode<T> weights,
        ComputationNode<T>? bias = null,
        int[]? stride = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var weightsShape = weights.Value.Shape;
        // Validate input shape (must be 4D: [batch, in_channels, height, width])
        if (inputShape.Length != 4)
            throw new ArgumentException("Input must be 4D tensor [batch, in_channels, height, width]");
        // Validate weights shape (must be 6D: [out_h, out_w, out_channels, in_channels, kernel_h, kernel_w])
        if (weightsShape.Length != 6)
            throw new ArgumentException("Weights must be 6D tensor [out_h, out_w, out_channels, in_channels, kernel_h, kernel_w]");
        // Default stride
        stride ??= new int[] { 1, 1 };
        if (stride.Length != 2)
            throw new ArgumentException("Stride must be 2D array [height, width]");
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        int outHeight = weightsShape[0];
        int outWidth = weightsShape[1];
        int outChannels = weightsShape[2];
        int kernelHeight = weightsShape[4];
        int kernelWidth = weightsShape[5];
        int strideH = stride[0];
        int strideW = stride[1];
        // Validate weight dimensions match input
        if (weightsShape[3] != inChannels)
            throw new ArgumentException($"Weight in_channels ({weightsShape[3]}) must match input in_channels ({inChannels})");
        // Validate bias if provided
        if (bias != null)
        {
            var biasShape = bias.Value.Shape;
            if (biasShape.Length != 1 || biasShape[0] != outChannels)
                throw new ArgumentException($"Bias must be 1D tensor of length {outChannels}");
        }
        var outputShape = new int[] { batch, outChannels, outHeight, outWidth };
        var result = new Tensor<T>(outputShape);
        // Forward pass: Locally connected convolution
        for (int b = 0; b < batch; b++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        T sum = numOps.Zero;
                        // Apply position-specific filter
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh * strideH + kh;
                                    int iw = ow * strideW + kw;
                                    // Check bounds
                                    if (ih < inHeight && iw < inWidth)
                                    {
                                        T inputVal = input.Value[b, ic, ih, iw];
                                        T weightVal = weights.Value[oh, ow, oc, ic, kh, kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, weightVal));
                                    }
                                }
                            }
                        }
                        // Add bias if provided
                        if (bias != null)
                            sum = numOps.Add(sum, bias.Value[oc]);
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
                if (input.Gradient == null)
                    input.Gradient = new Tensor<T>(inputShape);
                for (int b = 0; b < batch; b++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                T grad = gradient[b, oc, oh, ow];
                                for (int ic = 0; ic < inChannels; ic++)
                                {
                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        for (int kw = 0; kw < kernelWidth; kw++)
                                        {
                                            int ih = oh * strideH + kh;
                                            int iw = ow * strideW + kw;
                                            if (ih < inHeight && iw < inWidth)
                                            {
                                                T weightVal = weights.Value[oh, ow, oc, ic, kh, kw];
                                                T delta = numOps.Multiply(grad, weightVal);
                                                input.Gradient[b, ic, ih, iw] = numOps.Add(
                                                    input.Gradient[b, ic, ih, iw], delta);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Gradient w.r.t. weights
            if (weights.RequiresGradient)
            {
                if (weights.Gradient == null)
                    weights.Gradient = new Tensor<T>(weightsShape);
                for (int b = 0; b < batch; b++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                T grad = gradient[b, oc, oh, ow];
                                for (int ic = 0; ic < inChannels; ic++)
                                {
                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        for (int kw = 0; kw < kernelWidth; kw++)
                                        {
                                            int ih = oh * strideH + kh;
                                            int iw = ow * strideW + kw;
                                            if (ih < inHeight && iw < inWidth)
                                            {
                                                T inputVal = input.Value[b, ic, ih, iw];
                                                T delta = numOps.Multiply(grad, inputVal);
                                                weights.Gradient[oh, ow, oc, ic, kh, kw] = numOps.Add(
                                                    weights.Gradient[oh, ow, oc, ic, kh, kw], delta);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Gradient w.r.t. bias
            if (bias != null && bias.RequiresGradient)
            {
                if (bias.Gradient == null)
                    bias.Gradient = new Tensor<T>(new int[] { outChannels });
                for (int b = 0; b < batch; b++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int oh = 0; oh < outHeight; oh++)
                        {
                            for (int ow = 0; ow < outWidth; ow++)
                            {
                                bias.Gradient[oc] = numOps.Add(bias.Gradient[oc], gradient[b, oc, oh, ow]);
                            }
                        }
                    }
                }
            }
        }
        var parents = bias != null
            ? new List<ComputationNode<T>> { input, weights, bias }
            : new List<ComputationNode<T>> { input, weights };
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient || weights.RequiresGradient || (bias?.RequiresGradient ?? false),
            parents: parents,
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.LocallyConnectedConv2D;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Stride", stride }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Computes the natural logarithm of variance along the specified axis.
    /// </summary>
    /// <param name="input">Input tensor of any shape</param>
    /// <param name="axis">The axis along which to compute variance (must be specified)</param>
    /// <param name="epsilon">Small constant for numerical stability (default: 1e-8)</param>
    /// <returns>Tensor with reduced shape containing log-variance values</returns>
    /// <remarks>
    /// <para>
    /// This operation computes log(variance + epsilon) along the specified axis. The output shape
    /// has the specified axis dimension removed from the input shape.
    /// </para>
    /// <para>
    /// Forward pass: log(variance + epsilon) where variance = mean((x - mean(x))^2)
    /// </para>
    /// <para>
    /// Backward pass uses chain rule:
    /// ∂L/∂x_i = ∂L/∂log_var * (1/variance) * (2/N) * (x_i - mean)
    /// where N is the size of the reduction axis.
    /// </para>
    /// <para><b>For Beginners:</b> This operation measures how spread out values are along an axis,
    /// then takes the logarithm. Commonly used in variational autoencoders and uncertainty estimation.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ReduceLogVariance(
        ComputationNode<T> input,
        int axis,
        double epsilon = 1e-8)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        if (axis < 0 || axis >= inputShape.Length)
            throw new ArgumentException($"Axis {axis} is out of range for tensor of rank {inputShape.Length}");
        // Compute output shape (remove the reduction axis)
        var outputShape = new int[inputShape.Length - 1];
        int outIdx = 0;
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (i != axis)
                outputShape[outIdx++] = inputShape[i];
        }
        if (outputShape.Length == 0)
            outputShape = new int[] { 1 };
        var result = new Tensor<T>(outputShape);
        var meanValues = new Tensor<T>(outputShape);
        int axisSize = inputShape[axis];
        T axisScale = numOps.FromDouble(1.0 / axisSize);
        T eps = numOps.FromDouble(epsilon);
        // Helper to iterate over all positions except the reduction axis
        void IterateOverDimensions(Action<int[], int[]> action)
        {
            void Recurse(int[] inputIndices, int[] outputIndices, int dim)
            {
                if (dim == inputShape.Length)
                {
                    action(inputIndices, outputIndices);
                    return;
                }
                if (dim == axis)
                {
                    Recurse(inputIndices, outputIndices, dim + 1);
                }
                else
                {
                    int outDim = dim < axis ? dim : dim - 1;
                    for (int i = 0; i < inputShape[dim]; i++)
                    {
                        inputIndices[dim] = i;
                        outputIndices[outDim] = i;
                        Recurse(inputIndices, outputIndices, dim + 1);
                    }
                }
            }
            Recurse(new int[inputShape.Length], new int[outputShape.Length], 0);
        }
        // Forward pass: compute mean
        IterateOverDimensions((inputIndices, outputIndices) =>
        {
            T sum = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                inputIndices[axis] = i;
                sum = numOps.Add(sum, input.Value[inputIndices]);
            }
            meanValues[outputIndices] = numOps.Multiply(sum, axisScale);
        });
        // Forward pass: compute log variance
        IterateOverDimensions((inputIndices, outputIndices) =>
        {
            T sumSquaredDiff = numOps.Zero;
            T mean = meanValues[outputIndices];
            for (int i = 0; i < axisSize; i++)
            {
                inputIndices[axis] = i;
                T diff = numOps.Subtract(input.Value[inputIndices], mean);
                sumSquaredDiff = numOps.Add(sumSquaredDiff, numOps.Square(diff));
            }
            T variance = numOps.Multiply(sumSquaredDiff, axisScale);
            result[outputIndices] = numOps.Log(numOps.Add(variance, eps));
        });
        // Backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            if (!input.RequiresGradient) return;
            var inputGradient = new Tensor<T>(inputShape);
            T two = numOps.FromDouble(2.0);
            T twoOverN = numOps.FromDouble(2.0 / axisSize);
            // Compute gradients: ∂L/∂x_i = ∂L/∂log_var * (1/variance) * (2/N) * (x_i - mean)
            IterateOverDimensions((inputIndices, outputIndices) =>
            {
                T mean = meanValues[outputIndices];
                T logVar = result[outputIndices];
                T variance = numOps.Exp(logVar);  // Recover variance from log_variance
                T grad = gradient[outputIndices];
                T gradScale = numOps.Divide(grad, variance);
                for (int i = 0; i < axisSize; i++)
                {
                    inputIndices[axis] = i;
                    T diff = numOps.Subtract(input.Value[inputIndices], mean);
                    T inputGrad = numOps.Multiply(
                        numOps.Multiply(diff, gradScale),
                        twoOverN);
                    inputGradient[inputIndices] = inputGrad;
                }
            });
            var existingInputGrad = input.Gradient;
            input.Gradient = existingInputGrad == null ? inputGradient : existingInputGrad.Add(inputGradient);
        }
        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient,
            parents: new List<ComputationNode<T>> { input },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.ReduceLogVariance;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axis", axis },
            { "Epsilon", epsilon }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Computes Gaussian Radial Basis Function (RBF) kernel activations.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, inputSize]</param>
    /// <param name="centers">Center points tensor of shape [numCenters, inputSize]</param>
    /// <param name="epsilons">Width parameters tensor of shape [numCenters]</param>
    /// <returns>Output tensor of shape [batch, numCenters] containing RBF activations</returns>
    /// <remarks>
    /// <para>
    /// This operation implements the Gaussian RBF: f(r) = exp(-epsilon * r²)
    /// where r is the Euclidean distance between input and center.
    /// </para>
    /// <para>
    /// Forward pass: For each input and center pair, computes:
    /// 1. distance = sqrt(sum((input - center)²))
    /// 2. output = exp(-epsilon * distance²)
    /// </para>
    /// <para>
    /// Backward pass gradients:
    /// - ∂L/∂input = ∂L/∂output * (-2 * epsilon * distance) * (input - center) / distance
    /// - ∂L/∂centers = -∂L/∂input (opposite direction)
    /// - ∂L/∂epsilon = ∂L/∂output * (-distance²) * output
    /// </para>
    /// <para><b>For Beginners:</b> This operation creates "similarity scores" between inputs and centers.
    /// Each RBF neuron responds strongly (value near 1) when input is close to its center,
    /// and weakly (value near 0) when far away. The epsilon parameter controls how quickly
    /// the response decreases with distance.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> RBFKernel(
        ComputationNode<T> input,
        ComputationNode<T> centers,
        ComputationNode<T> epsilons)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var centersShape = centers.Value.Shape;
        var epsilonsShape = epsilons.Value.Shape;
        // Validate shapes
        if (inputShape.Length != 2)
            throw new ArgumentException("Input must be 2D tensor [batch, inputSize]");
        if (centersShape.Length != 2)
            throw new ArgumentException("Centers must be 2D tensor [numCenters, inputSize]");
        if (epsilonsShape.Length != 1)
            throw new ArgumentException("Epsilons must be 1D tensor [numCenters]");
        if (inputShape[1] != centersShape[1])
            throw new ArgumentException($"Input size {inputShape[1]} must match centers input size {centersShape[1]}");
        if (epsilonsShape[0] != centersShape[0])
            throw new ArgumentException($"Number of epsilons {epsilonsShape[0]} must match number of centers {centersShape[0]}");
        int batchSize = inputShape[0];
        int inputSize = inputShape[1];
        int numCenters = centersShape[0];
        var output = new Tensor<T>([batchSize, numCenters]);
        var distances = new Tensor<T>([batchSize, numCenters]);
        // Forward pass: compute RBF activations
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numCenters; c++)
            {
                // Compute Euclidean distance
                T sumSquaredDiff = numOps.Zero;
                for (int i = 0; i < inputSize; i++)
                {
                    T diff = numOps.Subtract(input.Value[b, i], centers.Value[c, i]);
                    sumSquaredDiff = numOps.Add(sumSquaredDiff, numOps.Multiply(diff, diff));
                }
                T distance = numOps.Sqrt(sumSquaredDiff);
                distances[b, c] = distance;
                // Compute Gaussian RBF: exp(-epsilon * distance²)
                T distanceSquared = numOps.Multiply(distance, distance);
                T epsilon = epsilons.Value[c];
                T exponent = numOps.Negate(numOps.Multiply(epsilon, distanceSquared));
                output[b, c] = numOps.Exp(exponent);
            }
        }
        // Backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            T two = numOps.FromDouble(2.0);
            T minusTwo = numOps.FromDouble(-2.0);
            // Gradients w.r.t. input
            if (input.RequiresGradient)
            {
                var inputGradient = new Tensor<T>(inputShape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int c = 0; c < numCenters; c++)
                    {
                        T distance = distances[b, c];
                        T epsilon = epsilons.Value[c];
                        T outputVal = output[b, c];
                        T grad = gradient[b, c];
                        // Derivative: -2 * epsilon * r * exp(-epsilon * r²) = -2 * epsilon * r * output
                        T gradScale = numOps.Multiply(
                            numOps.Multiply(minusTwo, epsilon),
                            numOps.Multiply(distance, outputVal));
                        gradScale = numOps.Multiply(gradScale, grad);
                        // Scale by (input - center) / distance to get gradient direction
                        T invDistance = numOps.Equals(distance, numOps.Zero) ? numOps.Zero : numOps.Divide(numOps.One, distance);
                        for (int i = 0; i < inputSize; i++)
                        {
                            T diff = numOps.Subtract(input.Value[b, i], centers.Value[c, i]);
                            T inputGrad = numOps.Multiply(gradScale, numOps.Multiply(diff, invDistance));
                            inputGradient[b, i] = numOps.Add(inputGradient[b, i], inputGrad);
                        }
                    }
                }
                var existingInputGrad = input.Gradient;
                input.Gradient = existingInputGrad == null ? inputGradient : existingInputGrad.Add(inputGradient);
            }
            // Gradients w.r.t. centers
            if (centers.RequiresGradient)
            {
                var centersGradient = new Tensor<T>(centersShape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int c = 0; c < numCenters; c++)
                    {
                        T distance = distances[b, c];
                        T epsilon = epsilons.Value[c];
                        T outputVal = output[b, c];
                        T grad = gradient[b, c];
                        // Same as input gradient but opposite sign
                        T gradScale = numOps.Multiply(
                            numOps.Multiply(two, epsilon),
                            numOps.Multiply(distance, outputVal));
                        gradScale = numOps.Multiply(gradScale, grad);
                        T invDistance = numOps.Equals(distance, numOps.Zero) ? numOps.Zero : numOps.Divide(numOps.One, distance);
                        for (int i = 0; i < inputSize; i++)
                        {
                            T diff = numOps.Subtract(input.Value[b, i], centers.Value[c, i]);
                            T centerGrad = numOps.Multiply(gradScale, numOps.Multiply(diff, invDistance));
                            centersGradient[c, i] = numOps.Add(centersGradient[c, i], centerGrad);
                        }
                    }
                }
                var existingCentersGrad = centers.Gradient;
                centers.Gradient = existingCentersGrad == null ? centersGradient : existingCentersGrad.Add(centersGradient);
            }
            // Gradients w.r.t. epsilons
            if (epsilons.RequiresGradient)
            {
                var epsilonsGradient = new Tensor<T>(epsilonsShape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int c = 0; c < numCenters; c++)
                    {
                        T distance = distances[b, c];
                        T distanceSquared = numOps.Multiply(distance, distance);
                        T outputVal = output[b, c];
                        T grad = gradient[b, c];
                        // Derivative w.r.t. epsilon: -r² * exp(-epsilon * r²) = -r² * output
                        T epsilonGrad = numOps.Multiply(
                            numOps.Negate(distanceSquared),
                            numOps.Multiply(outputVal, grad));
                        epsilonsGradient[c] = numOps.Add(epsilonsGradient[c], epsilonGrad);
                    }
                }
                var existingEpsilonsGrad = epsilons.Gradient;
                epsilons.Gradient = existingEpsilonsGrad == null ? epsilonsGradient : existingEpsilonsGrad.Add(epsilonsGradient);
            }
        }
        var node = new ComputationNode<T>(
            value: output,
            requiresGradient: input.RequiresGradient || centers.RequiresGradient || epsilons.RequiresGradient,
            parents: new List<ComputationNode<T>> { input, centers, epsilons },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.RBFKernel;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Generates a sampling grid for spatial transformer networks using affine transformation matrices.
    /// </summary>
    /// <param name="theta">Affine transformation matrices of shape [batch, 2, 3]</param>
    /// <param name="outputHeight">Height of the output grid</param>
    /// <param name="outputWidth">Width of the output grid</param>
    /// <returns>Sampling grid of shape [batch, outputHeight, outputWidth, 2] with (x, y) coordinates</returns>
    /// <remarks>
    /// <para>
    /// This operation generates a grid of sampling coordinates for spatial transformations.
    /// The output grid starts as a regular grid in normalized coordinates [-1, 1], then
    /// each point is transformed using the affine matrix.
    /// </para>
    /// <para>
    /// Forward pass:
    /// 1. Generate base grid in [-1, 1] normalized space
    /// 2. For each point (x_out, y_out) in output space:
    ///    x_in = theta[0,0]*x_out + theta[0,1]*y_out + theta[0,2]
    ///    y_in = theta[1,0]*x_out + theta[1,1]*y_out + theta[1,2]
    /// </para>
    /// <para>
    /// Backward pass:
    /// - ∂L/∂theta[i,j] = sum over all grid points of (∂L/∂grid * ∂grid/∂theta)
    /// </para>
    /// <para><b>For Beginners:</b> This creates a map showing where each output pixel should sample from.
    /// The affine matrix controls rotation, scaling, translation, and shearing of the grid.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> AffineGrid(
        ComputationNode<T> theta,
        int outputHeight,
        int outputWidth)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var thetaShape = theta.Value.Shape;
        // Validate shapes
        if (thetaShape.Length != 3 || thetaShape[1] != 2 || thetaShape[2] != 3)
            throw new ArgumentException("Theta must be of shape [batch, 2, 3]");
        int batchSize = thetaShape[0];
        var grid = new Tensor<T>([batchSize, outputHeight, outputWidth, 2]);
        // Generate base grid coordinates in [-1, 1] range
        T[,] baseGrid = new T[outputHeight * outputWidth, 3];
        int idx = 0;
        for (int h = 0; h < outputHeight; h++)
        {
            for (int w = 0; w < outputWidth; w++)
            {
                // Normalized coordinates [-1, 1]
                T x = numOps.FromDouble((double)w / Math.Max(outputWidth - 1, 1) * 2.0 - 1.0);
                T y = numOps.FromDouble((double)h / Math.Max(outputHeight - 1, 1) * 2.0 - 1.0);
                baseGrid[idx, 0] = x;
                baseGrid[idx, 1] = y;
                baseGrid[idx, 2] = numOps.One;  // Homogeneous coordinate
                idx++;
            }
        }
        // Forward pass: apply affine transformation to each grid point
        for (int b = 0; b < batchSize; b++)
        {
            idx = 0;
            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    T x = baseGrid[idx, 0];
                    T y = baseGrid[idx, 1];
                    // Apply affine transformation: [x_in, y_in]^T = theta * [x_out, y_out, 1]^T
                    T xTransformed = numOps.Add(
                        numOps.Add(
                            numOps.Multiply(theta.Value[b, 0, 0], x),
                            numOps.Multiply(theta.Value[b, 0, 1], y)),
                        theta.Value[b, 0, 2]);
                    T yTransformed = numOps.Add(
                        numOps.Add(
                            numOps.Multiply(theta.Value[b, 1, 0], x),
                            numOps.Multiply(theta.Value[b, 1, 1], y)),
                        theta.Value[b, 1, 2]);
                    grid[b, h, w, 0] = xTransformed;
                    grid[b, h, w, 1] = yTransformed;
                    idx++;
                }
            }
        }
        // Backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            if (!theta.RequiresGradient) return;
            var thetaGradient = new Tensor<T>(thetaShape);
            // Compute gradients w.r.t. theta
            for (int b = 0; b < batchSize; b++)
            {
                idx = 0;
                for (int h = 0; h < outputHeight; h++)
                {
                    for (int w = 0; w < outputWidth; w++)
                    {
                        T x = baseGrid[idx, 0];
                        T y = baseGrid[idx, 1];
                        T gradX = gradient[b, h, w, 0];
                        T gradY = gradient[b, h, w, 1];
                        // Gradient for theta[b, 0, :] from x_transformed
                        thetaGradient[b, 0, 0] = numOps.Add(thetaGradient[b, 0, 0], numOps.Multiply(gradX, x));
                        thetaGradient[b, 0, 1] = numOps.Add(thetaGradient[b, 0, 1], numOps.Multiply(gradX, y));
                        thetaGradient[b, 0, 2] = numOps.Add(thetaGradient[b, 0, 2], gradX);
                        // Gradient for theta[b, 1, :] from y_transformed
                        thetaGradient[b, 1, 0] = numOps.Add(thetaGradient[b, 1, 0], numOps.Multiply(gradY, x));
                        thetaGradient[b, 1, 1] = numOps.Add(thetaGradient[b, 1, 1], numOps.Multiply(gradY, y));
                        thetaGradient[b, 1, 2] = numOps.Add(thetaGradient[b, 1, 2], gradY);
                        idx++;
                    }
                }
            }
            var existingThetaGrad = theta.Gradient;
            theta.Gradient = existingThetaGrad == null ? thetaGradient : existingThetaGrad.Add(thetaGradient);
        }
        var node = new ComputationNode<T>(
            value: grid,
            requiresGradient: theta.RequiresGradient,
            parents: new List<ComputationNode<T>> { theta },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.AffineGrid;
        node.OperationParams = new Dictionary<string, object>
        {
            { "OutputSize", new int[] { outputHeight, outputWidth } }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Samples input using bilinear interpolation at grid locations for spatial transformer networks.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, height, width, channels]</param>
    /// <param name="grid">Sampling grid of shape [batch, out_height, out_width, 2] with normalized coordinates in [-1, 1]</param>
    /// <returns>Sampled output of shape [batch, out_height, out_width, channels]</returns>
    /// <remarks>
    /// <para>
    /// This operation performs differentiable bilinear sampling from the input tensor
    /// using coordinates specified in the grid. Grid coordinates are in normalized [-1, 1] space
    /// where (-1, -1) is top-left and (1, 1) is bottom-right.
    /// </para>
    /// <para>
    /// Forward pass:
    /// 1. Convert normalized grid coordinates to input pixel coordinates
    /// 2. For each sampling point, find the 4 nearest pixels
    /// 3. Compute bilinear interpolation weights
    /// 4. Interpolate: out = w00*v00 + w01*v01 + w10*v10 + w11*v11
    /// </para>
    /// <para>
    /// Backward pass:
    /// - ∂L/∂input: Distribute gradients back to the 4 nearest pixels using same weights
    /// - ∂L/∂grid: Compute how grid coordinates affect the sampling result
    /// </para>
    /// <para><b>For Beginners:</b> This samples from an image using smooth interpolation.
    /// Instead of reading exact pixels, it can sample from positions between pixels
    /// by blending nearby pixel values. This enables smooth transformations like rotation.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> GridSample(
        ComputationNode<T> input,
        ComputationNode<T> grid)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var gridShape = grid.Value.Shape;
        // Validate shapes
        if (inputShape.Length != 4)
            throw new ArgumentException("Input must be 4D tensor [batch, height, width, channels]");
        if (gridShape.Length != 4 || gridShape[3] != 2)
            throw new ArgumentException("Grid must be 4D tensor [batch, out_height, out_width, 2]");
        if (inputShape[0] != gridShape[0])
            throw new ArgumentException($"Batch size mismatch: input {inputShape[0]} vs grid {gridShape[0]}");
        int batchSize = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        int channels = inputShape[3];
        int outHeight = gridShape[1];
        int outWidth = gridShape[2];
        var output = new Tensor<T>([batchSize, outHeight, outWidth, channels]);
        // Cache for backward pass
        var interpolationWeights = new Tensor<T>([batchSize, outHeight, outWidth, 4]);  // w00, w01, w10, w11
        var pixelCoords = new int[batchSize, outHeight, outWidth, 4];  // x0, x1, y0, y1
        T half = numOps.FromDouble(0.5);
        T heightScale = numOps.FromDouble((inputHeight - 1) / 2.0);
        T widthScale = numOps.FromDouble((inputWidth - 1) / 2.0);
        // Forward pass: bilinear sampling
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < outHeight; h++)
            {
                for (int w = 0; w < outWidth; w++)
                {
                    // Convert normalized grid coordinates [-1, 1] to pixel coordinates
                    T gridX = grid.Value[b, h, w, 0];
                    T gridY = grid.Value[b, h, w, 1];
                    // Map from [-1, 1] to [0, width-1] and [0, height-1]
                    T srcX = numOps.Multiply(numOps.Add(gridX, numOps.One), widthScale);
                    T srcY = numOps.Multiply(numOps.Add(gridY, numOps.One), heightScale);
                    // Compute nearest neighbor coordinates
                    double srcXDouble = Convert.ToDouble(srcX);
                    double srcYDouble = Convert.ToDouble(srcY);
                    int x0 = Math.Max(0, Math.Min((int)Math.Floor(srcXDouble), inputWidth - 1));
                    int x1 = Math.Max(0, Math.Min(x0 + 1, inputWidth - 1));
                    int y0 = Math.Max(0, Math.Min((int)Math.Floor(srcYDouble), inputHeight - 1));
                    int y1 = Math.Max(0, Math.Min(y0 + 1, inputHeight - 1));
                    // Store for backward pass
                    pixelCoords[b, h, w, 0] = x0;
                    pixelCoords[b, h, w, 1] = x1;
                    pixelCoords[b, h, w, 2] = y0;
                    pixelCoords[b, h, w, 3] = y1;
                    // Compute interpolation weights
                    T wx1 = numOps.Subtract(srcX, numOps.FromDouble(x0));
                    T wx0 = numOps.Subtract(numOps.One, wx1);
                    T wy1 = numOps.Subtract(srcY, numOps.FromDouble(y0));
                    T wy0 = numOps.Subtract(numOps.One, wy1);
                    // Clamp weights to [0, 1]
                    wx0 = numOps.LessThan(wx0, numOps.Zero) ? numOps.Zero : wx0;
                    wx1 = numOps.LessThan(wx1, numOps.Zero) ? numOps.Zero : wx1;
                    wy0 = numOps.LessThan(wy0, numOps.Zero) ? numOps.Zero : wy0;
                    wy1 = numOps.LessThan(wy1, numOps.Zero) ? numOps.Zero : wy1;
                    T w00 = numOps.Multiply(wx0, wy0);
                    T w01 = numOps.Multiply(wx1, wy0);
                    T w10 = numOps.Multiply(wx0, wy1);
                    T w11 = numOps.Multiply(wx1, wy1);
                    // Store weights for backward pass
                    interpolationWeights[b, h, w, 0] = w00;
                    interpolationWeights[b, h, w, 1] = w01;
                    interpolationWeights[b, h, w, 2] = w10;
                    interpolationWeights[b, h, w, 3] = w11;
                    // Perform bilinear interpolation for each channel
                    for (int c = 0; c < channels; c++)
                    {
                        T v00 = input.Value[b, y0, x0, c];
                        T v01 = input.Value[b, y0, x1, c];
                        T v10 = input.Value[b, y1, x0, c];
                        T v11 = input.Value[b, y1, x1, c];
                        T interpolated = numOps.Add(
                            numOps.Add(
                                numOps.Multiply(v00, w00),
                                numOps.Multiply(v01, w01)),
                            numOps.Add(
                                numOps.Multiply(v10, w10),
                                numOps.Multiply(v11, w11)));
                        output[b, h, w, c] = interpolated;
                    }
                }
            }
        }
        // Backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient w.r.t. input
            if (input.RequiresGradient)
            {
                var inputGradient = new Tensor<T>(inputShape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < outHeight; h++)
                    {
                        for (int w = 0; w < outWidth; w++)
                        {
                            int x0 = pixelCoords[b, h, w, 0];
                            int x1 = pixelCoords[b, h, w, 1];
                            int y0 = pixelCoords[b, h, w, 2];
                            int y1 = pixelCoords[b, h, w, 3];
                            T w00 = interpolationWeights[b, h, w, 0];
                            T w01 = interpolationWeights[b, h, w, 1];
                            T w10 = interpolationWeights[b, h, w, 2];
                            T w11 = interpolationWeights[b, h, w, 3];
                            for (int c = 0; c < channels; c++)
                            {
                                T grad = gradient[b, h, w, c];
                                // Distribute gradient to the 4 nearest pixels
                                inputGradient[b, y0, x0, c] = numOps.Add(inputGradient[b, y0, x0, c], numOps.Multiply(grad, w00));
                                inputGradient[b, y0, x1, c] = numOps.Add(inputGradient[b, y0, x1, c], numOps.Multiply(grad, w01));
                                inputGradient[b, y1, x0, c] = numOps.Add(inputGradient[b, y1, x0, c], numOps.Multiply(grad, w10));
                                inputGradient[b, y1, x1, c] = numOps.Add(inputGradient[b, y1, x1, c], numOps.Multiply(grad, w11));
                            }
                        }
                    }
                }
                var existingInputGrad = input.Gradient;
                input.Gradient = existingInputGrad == null ? inputGradient : existingInputGrad.Add(inputGradient);
            }
            // Gradient w.r.t. grid
            if (grid.RequiresGradient)
            {
                var gridGradient = new Tensor<T>(gridShape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < outHeight; h++)
                    {
                        for (int w = 0; w < outWidth; w++)
                        {
                            int x0 = pixelCoords[b, h, w, 0];
                            int x1 = pixelCoords[b, h, w, 1];
                            int y0 = pixelCoords[b, h, w, 2];
                            int y1 = pixelCoords[b, h, w, 3];
                            T w00 = interpolationWeights[b, h, w, 0];
                            T w01 = interpolationWeights[b, h, w, 1];
                            T w10 = interpolationWeights[b, h, w, 2];
                            T w11 = interpolationWeights[b, h, w, 3];
                            T gradX = numOps.Zero;
                            T gradY = numOps.Zero;
                            for (int c = 0; c < channels; c++)
                            {
                                T grad = gradient[b, h, w, c];
                                T v00 = input.Value[b, y0, x0, c];
                                T v01 = input.Value[b, y0, x1, c];
                                T v10 = input.Value[b, y1, x0, c];
                                T v11 = input.Value[b, y1, x1, c];
                                // Gradient w.r.t. srcX
                                T dOutDSrcX = numOps.Subtract(
                                    numOps.Add(numOps.Multiply(v01, w01), numOps.Multiply(v11, w11)),
                                    numOps.Add(numOps.Multiply(v00, w00), numOps.Multiply(v10, w10)));
                                // Gradient w.r.t. srcY
                                T dOutDSrcY = numOps.Subtract(
                                    numOps.Add(numOps.Multiply(v10, w10), numOps.Multiply(v11, w11)),
                                    numOps.Add(numOps.Multiply(v00, w00), numOps.Multiply(v01, w01)));
                                gradX = numOps.Add(gradX, numOps.Multiply(grad, dOutDSrcX));
                                gradY = numOps.Add(gradY, numOps.Multiply(grad, dOutDSrcY));
                            }
                            // Chain rule: dL/dgrid = dL/dout * dout/dsrc * dsrc/dgrid
                            gridGradient[b, h, w, 0] = numOps.Multiply(gradX, widthScale);
                            gridGradient[b, h, w, 1] = numOps.Multiply(gradY, heightScale);
                        }
                    }
                }
                var existingGridGrad = grid.Gradient;
                grid.Gradient = existingGridGrad == null ? gridGradient : existingGridGrad.Add(gridGradient);
            }
        }
        var node = new ComputationNode<T>(
            value: output,
            requiresGradient: input.RequiresGradient || grid.RequiresGradient,
            parents: new List<ComputationNode<T>> { input, grid },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.GridSample;
        node.OperationParams = new Dictionary<string, object>();

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }
    /// <summary>
    /// Performs graph convolution operation for graph neural networks.
    /// </summary>
    /// <param name="input">Input node features of shape [batch, numNodes, inputFeatures]</param>
    /// <param name="adjacency">Adjacency matrix of shape [batch, numNodes, numNodes]</param>
    /// <param name="weights">Weight matrix of shape [inputFeatures, outputFeatures]</param>
    /// <param name="bias">Optional bias vector of shape [outputFeatures]</param>
    /// <returns>Output node features of shape [batch, numNodes, outputFeatures]</returns>
    /// <remarks>
    /// <para>
    /// This operation implements graph convolution: output = adjacency @ (input @ weights) + bias.
    /// It aggregates features from neighboring nodes according to the graph structure defined by the adjacency matrix.
    /// </para>
    /// <para>
    /// Forward pass:
    /// 1. Transform node features: X' = X @ W
    /// 2. Aggregate via graph structure: output = A @ X'
    /// 3. Add bias: output = output + b
    /// </para>
    /// <para>
    /// Backward pass gradients:
    /// - ∂L/∂X = A^T @ (∂L/∂out) @ W^T
    /// - ∂L/∂W = X^T @ A^T @ (∂L/∂out)
    /// - ∂L/∂b = sum(∂L/∂out) across batch and nodes
    /// - ∂L/∂A = (∂L/∂out) @ (X @ W)^T
    /// </para>
    /// <para><b>For Beginners:</b> This operation helps neural networks learn from graph-structured data.
    ///
    /// Think of it like spreading information through a social network:
    /// - Each person (node) has certain features
    /// - The adjacency matrix shows who is connected to whom
    /// - This operation lets each person's features be influenced by their connections
    /// - The weights control how features are transformed during this process
    /// </para>
    /// </remarks>
    public static ComputationNode<T> GraphConv(
        ComputationNode<T> input,
        ComputationNode<T> adjacency,
        ComputationNode<T> weights,
        ComputationNode<T>? bias = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var adjShape = adjacency.Value.Shape;
        var weightsShape = weights.Value.Shape;
        // Validate shapes
        if (inputShape.Length != 3)
            throw new ArgumentException("Input must be 3D tensor [batch, numNodes, inputFeatures]");
        if (adjShape.Length != 3 || adjShape[1] != adjShape[2])
            throw new ArgumentException("Adjacency must be 3D tensor [batch, numNodes, numNodes]");
        if (weightsShape.Length != 2)
            throw new ArgumentException("Weights must be 2D tensor [inputFeatures, outputFeatures]");
        if (inputShape[0] != adjShape[0])
            throw new ArgumentException($"Batch size mismatch: input {inputShape[0]} vs adjacency {adjShape[0]}");
        if (inputShape[1] != adjShape[1])
            throw new ArgumentException($"Number of nodes mismatch: input {inputShape[1]} vs adjacency {adjShape[1]}");
        if (inputShape[2] != weightsShape[0])
            throw new ArgumentException($"Feature size mismatch: input features {inputShape[2]} vs weights {weightsShape[0]}");
        if (bias != null && (bias.Value.Shape.Length != 1 || bias.Value.Shape[0] != weightsShape[1]))
            throw new ArgumentException($"Bias must be 1D tensor with {weightsShape[1]} elements");
        int batchSize = inputShape[0];
        int numNodes = inputShape[1];
        int inputFeatures = inputShape[2];
        int outputFeatures = weightsShape[1];
        var output = new Tensor<T>([batchSize, numNodes, outputFeatures]);
        // Forward pass: A @ (X @ W) + b
        // Step 1: X @ W
        var xw = new Tensor<T>([batchSize, numNodes, outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < outputFeatures; outF++)
                {
                    T sum = numOps.Zero;
                    for (int inF = 0; inF < inputFeatures; inF++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(
                            input.Value[b, n, inF],
                            weights.Value[inF, outF]));
                    }
                    xw[b, n, outF] = sum;
                }
            }
        }
        // Step 2: A @ (X @ W)
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int outF = 0; outF < outputFeatures; outF++)
                {
                    T sum = numOps.Zero;
                    for (int j = 0; j < numNodes; j++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(
                            adjacency.Value[b, i, j],
                            xw[b, j, outF]));
                    }
                    output[b, i, outF] = sum;
                }
            }
        }
        // Step 3: Add bias
        if (bias != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int outF = 0; outF < outputFeatures; outF++)
                    {
                        output[b, n, outF] = numOps.Add(output[b, n, outF], bias.Value[outF]);
                    }
                }
            }
        }
        // Backward function
        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient w.r.t. input: A^T @ grad @ W^T
            if (input.RequiresGradient)
            {
                var inputGradient = new Tensor<T>(inputShape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < numNodes; i++)
                    {
                        for (int inF = 0; inF < inputFeatures; inF++)
                        {
                            T sum = numOps.Zero;
                            for (int j = 0; j < numNodes; j++)
                            {
                                for (int outF = 0; outF < outputFeatures; outF++)
                                {
                                    // A^T[i,j] = A[j,i]
                                    sum = numOps.Add(sum, numOps.Multiply(
                                        numOps.Multiply(adjacency.Value[b, j, i], gradient[b, j, outF]),
                                        weights.Value[inF, outF]));
                                }
                            }
                            inputGradient[b, i, inF] = sum;
                        }
                    }
                }
                var existingInputGrad = input.Gradient;
                input.Gradient = existingInputGrad == null ? inputGradient : existingInputGrad.Add(inputGradient);
            }
            // Gradient w.r.t. weights: X^T @ A^T @ grad
            if (weights.RequiresGradient)
            {
                var weightsGradient = new Tensor<T>(weightsShape);
                for (int inF = 0; inF < inputFeatures; inF++)
                {
                    for (int outF = 0; outF < outputFeatures; outF++)
                    {
                        T sum = numOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int i = 0; i < numNodes; i++)
                            {
                                for (int j = 0; j < numNodes; j++)
                                {
                                    // A^T[j,i] = A[i,j]
                                    sum = numOps.Add(sum, numOps.Multiply(
                                        numOps.Multiply(input.Value[b, j, inF], adjacency.Value[b, i, j]),
                                        gradient[b, i, outF]));
                                }
                            }
                        }
                        weightsGradient[inF, outF] = sum;
                    }
                }
                var existingWeightsGrad = weights.Gradient;
                weights.Gradient = existingWeightsGrad == null ? weightsGradient : existingWeightsGrad.Add(weightsGradient);
            }
            // Gradient w.r.t. bias: sum across batch and nodes
            if (bias != null && bias.RequiresGradient)
            {
                var biasGradient = new Tensor<T>([outputFeatures]);
                for (int outF = 0; outF < outputFeatures; outF++)
                {
                    T sum = numOps.Zero;
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int n = 0; n < numNodes; n++)
                        {
                            sum = numOps.Add(sum, gradient[b, n, outF]);
                        }
                    }
                    biasGradient[outF] = sum;
                }
                var existingBiasGrad = bias.Gradient;
                bias.Gradient = existingBiasGrad == null ? biasGradient : existingBiasGrad.Add(biasGradient);
            }
            // Gradient w.r.t. adjacency: grad @ (X @ W)^T
            if (adjacency.RequiresGradient)
            {
                var adjGradient = new Tensor<T>(adjShape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < numNodes; i++)
                    {
                        for (int j = 0; j < numNodes; j++)
                        {
                            T sum = numOps.Zero;
                            for (int outF = 0; outF < outputFeatures; outF++)
                            {
                                sum = numOps.Add(sum, numOps.Multiply(
                                    gradient[b, i, outF],
                                    xw[b, j, outF]));
                            }
                            adjGradient[b, i, j] = sum;
                        }
                    }
                }
                var existingAdjacencyGrad = adjacency.Gradient;
                adjacency.Gradient = existingAdjacencyGrad == null ? adjGradient : existingAdjacencyGrad.Add(adjGradient);
            }
        }
        var parents = new List<ComputationNode<T>> { input, adjacency, weights };
        if (bias != null) parents.Add(bias);
        var node = new ComputationNode<T>(
            value: output,
            requiresGradient: input.RequiresGradient || adjacency.RequiresGradient || weights.RequiresGradient || (bias?.RequiresGradient ?? false),
            parents: parents,
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.GraphConv;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Pads a tensor with zeros along specified dimensions.
    /// </summary>
    /// <param name="a">The input computation node to pad.</param>
    /// <param name="padding">Array specifying padding amount for each dimension (applied symmetrically on both sides).</param>
    /// <returns>A new computation node containing the padded tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method pads the input tensor by adding zeros around each dimension.
    /// The padding array specifies how many zeros to add on BOTH sides of each dimension.
    /// For example, padding[1] = 2 means add 2 zeros on the left AND 2 zeros on the right of dimension 1.
    /// </para>
    /// <para>
    /// The backward function for padding simply extracts the non-padded region from the output gradient,
    /// since ∂(pad(x))/∂x is an extraction operation that removes the padded regions.
    /// </para>
    /// <para><b>For Beginners:</b> Padding adds a border of zeros around your data.
    ///
    /// For padding (output = pad(input, [p0, p1, ...])):
    /// - The forward pass creates a larger tensor and copies input to the center
    /// - Padding p on dimension d means: add p zeros on left, p zeros on right
    /// - The backward pass extracts the center region from the gradient (removes the padding)
    ///
    /// This is commonly used in convolutional neural networks to preserve spatial dimensions.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Pad(ComputationNode<T> a, int[] padding)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = a.Value.Shape;

        if (padding.Length != inputShape.Length)
            throw new ArgumentException($"Padding array length ({padding.Length}) must match input rank ({inputShape.Length})");

        // Calculate output shape: each dimension grows by 2 * padding[i]
        var outputShape = new int[inputShape.Length];
        for (int i = 0; i < inputShape.Length; i++)
        {
            outputShape[i] = inputShape[i] + 2 * padding[i];
        }

        // Forward pass: Create padded tensor and copy input data to center
        var result = new Tensor<T>(outputShape);
        // result is already zero-initialized, so we only need to copy the input data

        // For 4D tensors (typical in CNNs): [batch, height, width, channels]
        if (inputShape.Length == 4)
        {
            int batchSize = inputShape[0];
            int inputHeight = inputShape[1];
            int inputWidth = inputShape[2];
            int channels = inputShape[3];

            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < inputHeight; h++)
                {
                    for (int w = 0; w < inputWidth; w++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            result[b + padding[0], h + padding[1], w + padding[2], c + padding[3]] =
                                a.Value[b, h, w, c];
                        }
                    }
                }
            }
        }
        else
        {
            // General N-dimensional padding (slower but works for any rank)
            CopyPaddedDataRecursive(a.Value, result, padding, new int[inputShape.Length], new int[outputShape.Length], 0);
        }

        // Backward function: Extract the non-padded region from the output gradient
        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // The gradient for the input is just the center region of the output gradient
                // (removing the padded borders)
                var gradA = new Tensor<T>(inputShape);

                if (inputShape.Length == 4)
                {
                    int batchSize = inputShape[0];
                    int inputHeight = inputShape[1];
                    int inputWidth = inputShape[2];
                    int channels = inputShape[3];

                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int h = 0; h < inputHeight; h++)
                        {
                            for (int w = 0; w < inputWidth; w++)
                            {
                                for (int c = 0; c < channels; c++)
                                {
                                    gradA[b, h, w, c] = gradient[b + padding[0], h + padding[1], w + padding[2], c + padding[3]];
                                }
                            }
                        }
                    }
                }
                else
                {
                    // General N-dimensional unpadding
                    ExtractPaddedDataRecursive(gradient, gradA, padding, new int[inputShape.Length], new int[outputShape.Length], 0);
                }

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Pad;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Padding", padding }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Helper method to recursively copy data from source to padded destination tensor.
    /// </summary>
    private static void CopyPaddedDataRecursive(Tensor<T> source, Tensor<T> dest, int[] padding,
        int[] sourceIndices, int[] destIndices, int dimension)
    {
        if (dimension == source.Shape.Length)
        {
            // Base case: copy the value
            dest[destIndices] = source[sourceIndices];
            return;
        }

        for (int i = 0; i < source.Shape[dimension]; i++)
        {
            sourceIndices[dimension] = i;
            destIndices[dimension] = i + padding[dimension];
            CopyPaddedDataRecursive(source, dest, padding, sourceIndices, destIndices, dimension + 1);
        }
    }

    /// <summary>
    /// Helper method to recursively extract data from padded source to unpadded destination tensor.
    /// </summary>
    private static void ExtractPaddedDataRecursive(Tensor<T> source, Tensor<T> dest, int[] padding,
        int[] destIndices, int[] sourceIndices, int dimension)
    {
        if (dimension == dest.Shape.Length)
        {
            // Base case: copy the value
            dest[destIndices] = source[sourceIndices];
            return;
        }

        for (int i = 0; i < dest.Shape[dimension]; i++)
        {
            destIndices[dimension] = i;
            sourceIndices[dimension] = i + padding[dimension];
            ExtractPaddedDataRecursive(source, dest, padding, destIndices, sourceIndices, dimension + 1);
        }
    }

    /// <summary>
    /// Applies a generic activation function (scalar or element-wise) with automatic differentiation.
    /// </summary>
    /// <param name="input">The input computation node.</param>
    /// <param name="activation">The activation function to apply.</param>
    /// <returns>A new computation node with the activation applied.</returns>
    /// <remarks>
    /// <para>
    /// This method provides generic autodiff support for ANY activation function that implements
    /// IActivationFunction{T}. It works by applying the activation function element-wise during
    /// the forward pass, then using the activation's ComputeDerivative method during backpropagation.
    /// </para>
    /// <para>
    /// This means ALL 39 built-in activation functions automatically work with autodiff,
    /// and only truly custom user-defined activations (that don't inherit from ActivationFunctionBase)
    /// would fail.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ApplyActivation(
        ComputationNode<T> input,
        Interfaces.IActivationFunction<T> activation)
    {
        if (activation == null)
            throw new ArgumentNullException(nameof(activation));

        // Forward pass: apply activation element-wise
        var result = input.Value.Transform((x, _) => activation.Activate(x));

        // Backward function: use activation's derivative
        void BackwardFunction(Tensor<T> gradient)
        {
            if (input.RequiresGradient)
            {
                // Compute derivative at each point: grad_in = grad_out * f'(input)
                var gradA = new Tensor<T>(gradient.Shape);
                var numOps = MathHelper.GetNumericOperations<T>();
                for (int i = 0; i < gradient.Length; i++)
                {
                    var derivative = activation.Derivative(input.Value.GetFlat(i));
                    gradA.SetFlat(i, numOps.Multiply(gradient.GetFlat(i), derivative));
                }

                var existingGrad = input.Gradient;

                input.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient,
            parents: new List<ComputationNode<T>> { input },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Activation;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Performs embedding lookup operation.
    /// </summary>
    /// <param name="embeddings">The embedding matrix [vocab_size, embedding_dim].</param>
    /// <param name="indices">The indices to lookup [batch_size, sequence_length].</param>
    /// <returns>The looked up embeddings [batch_size, sequence_length, embedding_dim].</returns>
    public static ComputationNode<T> EmbeddingLookup(ComputationNode<T> embeddings, ComputationNode<T> indices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var embeddingMatrix = embeddings.Value;
        var indexTensor = indices.Value;

        var batchSize = indexTensor.Shape[0];
        var seqLength = indexTensor.Shape.Length > 1 ? indexTensor.Shape[1] : 1;
        var embeddingDim = embeddingMatrix.Shape[1];

        var resultShape = seqLength > 1 ? new int[] { batchSize, seqLength, embeddingDim } : new int[] { batchSize, embeddingDim };
        var resultData = new T[batchSize * seqLength * embeddingDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLength; s++)
            {
                var idx = (int)Convert.ToDouble(seqLength > 1 ? indexTensor[b, s] : indexTensor[b, 0]);
                for (int e = 0; e < embeddingDim; e++)
                {
                    resultData[(b * seqLength + s) * embeddingDim + e] = embeddingMatrix[idx, e];
                }
            }
        }

        var result = new Tensor<T>(resultShape, new Vector<T>(resultData));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (embeddings.RequiresGradient)
            {
                var embeddingGrad = new Tensor<T>(embeddingMatrix.Shape);

                for (int b = 0; b < batchSize; b++)
                {
                    for (int s = 0; s < seqLength; s++)
                    {
                        var idx = (int)Convert.ToDouble(seqLength > 1 ? indexTensor[b, s] : indexTensor[b, 0]);
                        for (int e = 0; e < embeddingDim; e++)
                        {
                            var gradVal = seqLength > 1 ? gradient[b, s, e] : gradient[b, e];
                            embeddingGrad[idx, e] = numOps.Add(embeddingGrad[idx, e], gradVal);
                        }
                    }
                }

                if (embeddings.Gradient == null)
                    embeddings.Gradient = embeddingGrad;
                else
                    embeddings.Gradient = embeddings.Gradient.Add(embeddingGrad);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: embeddings.RequiresGradient,
            parents: new List<ComputationNode<T>> { embeddings, indices },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Embedding;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
    /// </summary>
    /// <param name="query">Query tensor [batch, seq_len_q, d_k].</param>
    /// <param name="key">Key tensor [batch, seq_len_k, d_k].</param>
    /// <param name="value">Value tensor [batch, seq_len_k, d_v].</param>
    /// <param name="mask">Optional attention mask.</param>
    /// <returns>Attention output [batch, seq_len_q, d_v].</returns>
    public static ComputationNode<T> ScaledDotProductAttention(
        ComputationNode<T> query,
        ComputationNode<T> key,
        ComputationNode<T> value,
        ComputationNode<T>? mask = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        // Q @ K^T
        var keyTransposed = Transpose(key);
        var scores = MatrixMultiply(query, keyTransposed);

        // Scale by sqrt(d_k)
        var dk = query.Value.Shape[query.Value.Shape.Length - 1];
        var scaleFactor = numOps.FromDouble(1.0 / Math.Sqrt(dk));
        var scaleShape = new int[] { 1 };
        var scaleTensor = new Tensor<T>(scaleShape, new Vector<T>(new T[] { scaleFactor }));
        var scaleNode = Constant(scaleTensor, "scale");
        scores = ElementwiseMultiply(scores, scaleNode);

        // Apply mask if provided
        if (mask != null)
        {
            var largeNegValue = numOps.FromDouble(-1e9);
            var maskShape = new int[] { 1 };
            var maskTensor = new Tensor<T>(maskShape, new Vector<T>(new T[] { largeNegValue }));
            var maskNode = Constant(maskTensor, "mask_value");

            // scores = scores + mask * large_neg_value (simplified masking)
            var maskedScores = ElementwiseMultiply(mask, maskNode);
            scores = Add(scores, maskedScores);
        }

        // Softmax
        var attentionWeights = Softmax(scores);

        // Attention @ V
        var output = MatrixMultiply(attentionWeights, value);

        return output;
    }

    /// <summary>
    /// Applies multi-head attention mechanism.
    /// </summary>
    /// <param name="query">Query tensor.</param>
    /// <param name="key">Key tensor.</param>
    /// <param name="value">Value tensor.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="wQ">Query projection weights.</param>
    /// <param name="wK">Key projection weights.</param>
    /// <param name="wV">Value projection weights.</param>
    /// <param name="wO">Output projection weights.</param>
    /// <returns>Multi-head attention output.</returns>
    public static ComputationNode<T> MultiHeadAttention(
        ComputationNode<T> query,
        ComputationNode<T> key,
        ComputationNode<T> value,
        int numHeads,
        ComputationNode<T> wQ,
        ComputationNode<T> wK,
        ComputationNode<T> wV,
        ComputationNode<T> wO)
    {
        // Project Q, K, V
        var q = MatrixMultiply(query, wQ);
        var k = MatrixMultiply(key, wK);
        var v = MatrixMultiply(value, wV);

        // For simplicity, compute single-head attention (multi-head would require splitting and concatenating)
        var attention = ScaledDotProductAttention(q, k, v);

        // Output projection
        var output = MatrixMultiply(attention, wO);

        return output;
    }

    /// <summary>
    /// LSTM cell forward pass.
    /// </summary>
    /// <param name="input">Input tensor [batch, input_dim].</param>
    /// <param name="hiddenState">Previous hidden state [batch, hidden_dim].</param>
    /// <param name="cellState">Previous cell state [batch, hidden_dim].</param>
    /// <param name="weightIH">Input-to-hidden weights [input_dim, 4*hidden_dim].</param>
    /// <param name="weightHH">Hidden-to-hidden weights [hidden_dim, 4*hidden_dim].</param>
    /// <param name="bias">Bias terms [4*hidden_dim].</param>
    /// <returns>Tuple of (new hidden state, new cell state).</returns>
    public static (ComputationNode<T>, ComputationNode<T>) LSTMCell(
        ComputationNode<T> input,
        ComputationNode<T> hiddenState,
        ComputationNode<T> cellState,
        ComputationNode<T> weightIH,
        ComputationNode<T> weightHH,
        ComputationNode<T> bias)
    {
        // Compute gates: input @ W_ih + hidden @ W_hh + bias
        var inputTransform = MatrixMultiply(input, weightIH);
        var hiddenTransform = MatrixMultiply(hiddenState, weightHH);
        var gates = Add(Add(inputTransform, hiddenTransform), bias);

        // Get hidden dimension from hidden state shape
        var hiddenDim = hiddenState.Value.Shape[hiddenState.Value.Shape.Length - 1];
        var lastAxis = gates.Value.Shape.Length - 1;

        // Validate gates shape: should be [batch, 4*hidden_dim]
        var gatesLastDim = gates.Value.Shape[lastAxis];
        if (gatesLastDim != 4 * hiddenDim)
        {
            throw new ArgumentException(
                $"Gates dimension {gatesLastDim} does not match expected 4*hidden_dim ({4 * hiddenDim}). " +
                $"Ensure weightIH and weightHH have shape [*, 4*hidden_dim].");
        }

        // Split gates into 4 segments along the last axis: [i, f, g, o]
        // Each gate has shape [batch, hidden_dim]
        var inputGateRaw = Slice(gates, 0, hiddenDim, 1, lastAxis);           // i_t
        var forgetGateRaw = Slice(gates, hiddenDim, hiddenDim, 1, lastAxis);  // f_t
        var cellGateRaw = Slice(gates, 2 * hiddenDim, hiddenDim, 1, lastAxis); // g_t
        var outputGateRaw = Slice(gates, 3 * hiddenDim, hiddenDim, 1, lastAxis); // o_t

        // Apply activations: sigmoid for gates, tanh for candidate
        var inputGate = Sigmoid(inputGateRaw);      // i_t = sigmoid(...)
        var forgetGate = Sigmoid(forgetGateRaw);    // f_t = sigmoid(...)
        var candidateCell = Tanh(cellGateRaw);      // g_t = tanh(...)
        var outputGate = Sigmoid(outputGateRaw);    // o_t = sigmoid(...)

        // New cell state: c_t = f_t * c_{t-1} + i_t * g_t
        var forgetPart = ElementwiseMultiply(forgetGate, cellState);
        var inputPart = ElementwiseMultiply(inputGate, candidateCell);
        var newCellState = Add(forgetPart, inputPart);

        // New hidden state: h_t = o_t * tanh(c_t)
        var newCellTanh = Tanh(newCellState);
        var newHiddenState = ElementwiseMultiply(outputGate, newCellTanh);

        return (newHiddenState, newCellState);
    }

    /// <summary>
    /// GRU cell forward pass.
    /// </summary>
    /// <param name="input">Input tensor [batch, input_dim].</param>
    /// <param name="hiddenState">Previous hidden state [batch, hidden_dim].</param>
    /// <param name="weightIH">Input-to-hidden weights [input_dim, 3*hidden_dim].</param>
    /// <param name="weightHH">Hidden-to-hidden weights [hidden_dim, 3*hidden_dim].</param>
    /// <param name="bias">Bias terms [3*hidden_dim].</param>
    /// <returns>New hidden state.</returns>
    public static ComputationNode<T> GRUCell(
        ComputationNode<T> input,
        ComputationNode<T> hiddenState,
        ComputationNode<T> weightIH,
        ComputationNode<T> weightHH,
        ComputationNode<T> bias)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute gates: input @ W_ih + hidden @ W_hh + bias
        var inputTransform = MatrixMultiply(input, weightIH);
        var hiddenTransform = MatrixMultiply(hiddenState, weightHH);
        var gates = Add(Add(inputTransform, hiddenTransform), bias);

        // Get hidden dimension from hidden state shape
        var hiddenDim = hiddenState.Value.Shape[hiddenState.Value.Shape.Length - 1];
        var lastAxis = gates.Value.Shape.Length - 1;

        // Validate gates shape: should be [batch, 3*hidden_dim]
        var gatesLastDim = gates.Value.Shape[lastAxis];
        if (gatesLastDim != 3 * hiddenDim)
        {
            throw new ArgumentException(
                $"Gates dimension {gatesLastDim} does not match expected 3*hidden_dim ({3 * hiddenDim}). " +
                $"Ensure weightIH and weightHH have shape [*, 3*hidden_dim].");
        }

        // Split gates into 3 segments along the last axis: [r, z, n]
        // Each gate has shape [batch, hidden_dim]
        var resetGateRaw = Slice(gates, 0, hiddenDim, 1, lastAxis);           // r_t
        var updateGateRaw = Slice(gates, hiddenDim, hiddenDim, 1, lastAxis);  // z_t
        var newGateRaw = Slice(gates, 2 * hiddenDim, hiddenDim, 1, lastAxis); // n_t (partial)

        // Apply sigmoid to reset and update gates
        var resetGate = Sigmoid(resetGateRaw);   // r_t = sigmoid(...)
        var updateGate = Sigmoid(updateGateRaw); // z_t = sigmoid(...)

        // Candidate hidden state: n_t = tanh(W_in * x + b_in + r_t * (W_hn * h + b_hn))
        // Simplified: we use the newGateRaw and apply reset gate
        var resetHidden = ElementwiseMultiply(resetGate, hiddenState);

        // For the candidate, we need to recompute with reset applied to hidden part
        // Split the input-to-hidden weights contribution for the new gate
        var inputNew = Slice(inputTransform, 2 * hiddenDim, hiddenDim, 1, lastAxis);
        var hiddenNew = Slice(hiddenTransform, 2 * hiddenDim, hiddenDim, 1, lastAxis);
        var biasNew = Slice(bias, 2 * hiddenDim, hiddenDim, 1, lastAxis);

        // Apply reset gate to hidden contribution only
        var resetHiddenNew = ElementwiseMultiply(resetGate, hiddenNew);
        var candidateInput = Add(Add(inputNew, resetHiddenNew), biasNew);
        var candidateHidden = Tanh(candidateInput); // n_t = tanh(...)

        // New hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * n_t
        var onesTensor = new Tensor<T>(updateGate.Value.Shape);
        for (int i = 0; i < onesTensor.Length; i++)
            onesTensor[i] = numOps.FromDouble(1.0);
        var onesNode = Constant(onesTensor, "ones");

        var inverseUpdate = Subtract(onesNode, updateGate);
        var oldPart = ElementwiseMultiply(inverseUpdate, hiddenState);
        var newPart = ElementwiseMultiply(updateGate, candidateHidden);
        var newHiddenState = Add(oldPart, newPart);

        return newHiddenState;
    }

    /// <summary>
    /// Computes the element-wise square of the input (x²).
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <returns>A new computation node containing the squared result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the square of each element (x²) and records the operation.
    /// The backward function uses: ∂(x²)/∂x = 2x.
    /// </para>
    /// <para><b>For Beginners:</b> Square is a common operation in neural networks.
    ///
    /// For square (c = a²):
    /// - The forward pass computes a² for each element
    /// - The backward pass: gradient to 'a' is incoming gradient * 2a
    ///
    /// This is more efficient than using Power(a, 2) and is frequently needed for
    /// operations like computing distances, norms, and variance.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Square(ComputationNode<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = a.Value.Transform((x, _) => numOps.Multiply(x, x));

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // ∂(a²)/∂a = 2a
                var two = numOps.FromDouble(2.0);
                var gradA = new Tensor<T>(gradient.Shape);
                for (int i = 0; i < gradient.Length; i++)
                {
                    var twoTimesA = numOps.Multiply(two, a.Value[i]);
                    gradA[i] = numOps.Multiply(gradient[i], twoTimesA);
                }

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Square;
        node.OperationParams = null;

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the squashing function used in capsule networks: s(x) = ||x||² / (1 + ||x||²) * (x / ||x||).
    /// </summary>
    /// <param name="a">The input node representing capsule vectors.</param>
    /// <param name="epsilon">Small value for numerical stability (default: 1e-7).</param>
    /// <returns>A new computation node containing the squashed result.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the squashing nonlinearity used in capsule networks.
    /// The squashing function ensures that short vectors shrink to near zero length
    /// and long vectors shrink to a length slightly below 1.
    /// </para>
    /// <para><b>For Beginners:</b> Squashing is the activation function for capsule layers.
    ///
    /// The squashing function:
    /// - Keeps the direction of the vector unchanged
    /// - Scales the length to be between 0 and 1
    /// - Short vectors get much shorter (near 0)
    /// - Long vectors approach length 1
    ///
    /// This is crucial for capsule networks where the length represents the probability
    /// that the entity represented by the capsule exists, and the direction represents
    /// its properties.
    ///
    /// Formula: s(v) = ||v||² / (1 + ||v||²) * (v / ||v||)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Squash(ComputationNode<T> a, double epsilon = 1e-7)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = a.Value.Shape;

        // Assume last dimension is the capsule dimension
        int capsuleDim = inputShape[inputShape.Length - 1];
        var result = new Tensor<T>(inputShape);
        var norms = new Tensor<T>(inputShape.Take(inputShape.Length - 1).ToArray());

        // Compute squashed vectors
        void ComputeSquash(int[] indices, int dim)
        {
            if (dim == inputShape.Length - 1)
            {
                // Compute norm for this capsule
                T normSquared = numOps.Zero;
                for (int i = 0; i < capsuleDim; i++)
                {
                    var idx = indices.Take(indices.Length - 1).Concat(new[] { i }).ToArray();
                    T val = a.Value[idx];
                    normSquared = numOps.Add(normSquared, numOps.Multiply(val, val));
                }

                T norm = numOps.Sqrt(numOps.Add(normSquared, numOps.FromDouble(epsilon)));
                var normIdx = indices.Take(indices.Length - 1).ToArray();
                norms[normIdx] = norm;

                // Compute scaling factor: ||v||² / (1 + ||v||²)
                T onePlusNormSquared = numOps.Add(numOps.One, normSquared);
                T scaleFactor = numOps.Divide(normSquared, onePlusNormSquared);

                // Scale each element: scale * v / ||v||
                for (int i = 0; i < capsuleDim; i++)
                {
                    var idx = indices.Take(indices.Length - 1).Concat(new[] { i }).ToArray();
                    T val = a.Value[idx];
                    T normalized = numOps.Divide(val, norm);
                    result[idx] = numOps.Multiply(scaleFactor, normalized);
                }
            }
            else
            {
                for (int i = 0; i < inputShape[dim]; i++)
                {
                    indices[dim] = i;
                    ComputeSquash(indices, dim + 1);
                }
            }
        }

        ComputeSquash(new int[inputShape.Length], 0);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(inputShape);

                // Compute gradient through squashing
                void ComputeGradient(int[] indices, int dim)
                {
                    if (dim == inputShape.Length - 1)
                    {
                        var normIdx = indices.Take(indices.Length - 1).ToArray();
                        T norm = norms[normIdx];
                        T normSquared = numOps.Multiply(norm, norm);
                        T onePlusNormSquared = numOps.Add(numOps.One, normSquared);

                        // Simplified gradient computation
                        // Full derivation requires chain rule through normalization and scaling
                        for (int i = 0; i < capsuleDim; i++)
                        {
                            var idx = indices.Take(indices.Length - 1).Concat(new[] { i }).ToArray();
                            // Approximate gradient (full computation is complex)
                            T scale = numOps.Divide(
                                numOps.FromDouble(2.0),
                                numOps.Multiply(onePlusNormSquared, norm));
                            gradA[idx] = numOps.Multiply(gradient[idx], scale);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < inputShape[dim]; i++)
                        {
                            indices[dim] = i;
                            ComputeGradient(indices, dim + 1);
                        }
                    }
                }

                ComputeGradient(new int[inputShape.Length], 0);

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Squash;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Epsilon", epsilon }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Computes the L2 norm along a specified axis.
    /// </summary>
    /// <param name="a">The input node.</param>
    /// <param name="axis">The axis along which to compute the norm. Default is -1 (last axis).</param>
    /// <param name="keepDims">Whether to keep the reduced dimensions. Default is false.</param>
    /// <param name="epsilon">Small value for numerical stability. Default is 1e-12.</param>
    /// <returns>A new computation node containing the norm along the specified axis.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the L2 (Euclidean) norm: sqrt(sum(x²)) along the specified axis.
    /// The gradient is computed as: ∂||x||/∂x = x / ||x||.
    /// </para>
    /// <para><b>For Beginners:</b> The norm measures the "length" of vectors.
    ///
    /// For example, with axis=-1:
    /// - Input shape: [batch, features]
    /// - Output shape: [batch] (or [batch, 1] with keepDims=True)
    /// - Each output value is sqrt(sum of squares along that row)
    ///
    /// This is commonly used in capsule networks to compute capsule lengths,
    /// and in normalization operations.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Norm(ComputationNode<T> a, int axis = -1, bool keepDims = false, double epsilon = 1e-12)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = a.Value.Shape;

        // Normalize axis to positive index
        if (axis < 0)
            axis = inputShape.Length + axis;

        if (axis < 0 || axis >= inputShape.Length)
            throw new ArgumentException($"Axis {axis} is out of range for tensor with {inputShape.Length} dimensions.");

        // Compute output shape
        var outputShape = keepDims
            ? inputShape.Select((s, i) => i == axis ? 1 : s).ToArray()
            : inputShape.Where((_, i) => i != axis).ToArray();

        var result = new Tensor<T>(outputShape);

        // Compute norms
        void ComputeNorm(int[] indices, int dim)
        {
            if (dim == axis)
            {
                // Compute norm along this axis
                T sumSquares = numOps.Zero;
                for (int i = 0; i < inputShape[axis]; i++)
                {
                    indices[axis] = i;
                    T val = a.Value[indices];
                    sumSquares = numOps.Add(sumSquares, numOps.Multiply(val, val));
                }

                T norm = numOps.Sqrt(numOps.Add(sumSquares, numOps.FromDouble(epsilon)));

                // Map to output indices
                var outIndices = keepDims
                    ? indices.Select((idx, i) => i == axis ? 0 : idx).ToArray()
                    : indices.Where((_, i) => i != axis).ToArray();

                result[outIndices] = norm;
            }
            else if (dim < inputShape.Length)
            {
                for (int i = 0; i < inputShape[dim]; i++)
                {
                    indices[dim] = i;
                    ComputeNorm(indices, dim == axis - 1 ? axis : dim + 1);
                }
            }
        }

        var startIndices = new int[inputShape.Length];
        if (axis == 0)
        {
            ComputeNorm(startIndices, 0);
        }
        else
        {
            ComputeNorm(startIndices, 0);
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(inputShape);

                // Gradient: ∂||x||/∂x = x / ||x||
                void ComputeGradient(int[] indices, int dim)
                {
                    if (dim == axis)
                    {
                        var outIndices = keepDims
                            ? indices.Select((idx, i) => i == axis ? 0 : idx).ToArray()
                            : indices.Where((_, i) => i != axis).ToArray();

                        T norm = result[outIndices];
                        T gradNorm = gradient[outIndices];

                        for (int i = 0; i < inputShape[axis]; i++)
                        {
                            indices[axis] = i;
                            T val = a.Value[indices];
                            gradA[indices] = numOps.Multiply(gradNorm, numOps.Divide(val, norm));
                        }
                    }
                    else if (dim < inputShape.Length)
                    {
                        for (int i = 0; i < inputShape[dim]; i++)
                        {
                            indices[dim] = i;
                            ComputeGradient(indices, dim == axis - 1 ? axis : dim + 1);
                        }
                    }
                }

                ComputeGradient(new int[inputShape.Length], axis == 0 ? 0 : 0);

                if (a.Gradient == null)
                {
                    a.Gradient = gradA;
                }
                else
                {
                    a.Gradient = a.Gradient.Add(gradA);
                }
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        // Set JIT compiler metadata
        node.OperationType = OperationType.Norm;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Axis", axis },
            { "KeepDims", keepDims },
            { "Epsilon", epsilon }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Performs complex matrix multiplication on tensors representing complex numbers as [real, imag] pairs.
    /// </summary>
    /// <param name="a">First complex matrix [batch, m, 2*k] where dimensions are [real, imag] interleaved or concatenated.</param>
    /// <param name="b">Second complex matrix [batch, 2*k, n].</param>
    /// <param name="format">Whether complex numbers are "interleaved" ([r,i,r,i,...]) or "split" ([r,r,...,i,i,...]).</param>
    /// <returns>Complex matrix product [batch, m, 2*n].</returns>
    /// <remarks>
    /// <para>
    /// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    /// </para>
    /// <para><b>For Beginners:</b> This multiplies matrices of complex numbers.
    ///
    /// Complex numbers are represented as pairs of real numbers [real_part, imaginary_part].
    /// This operation implements the full complex matrix multiplication formula.
    ///
    /// Used in quantum computing layers where quantum gates are unitary matrices.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ComplexMatMul(ComputationNode<T> a, ComputationNode<T> b, string format = "split")
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shapeA = a.Value.Shape;
        var shapeB = b.Value.Shape;

        // For split format: [batch, m, 2*k] and [batch, 2*k, n]
        // Split into real and imaginary parts
        if (format == "split")
        {
            // a is [batch, m, 2*k] -> split into [batch, m, k] for real and imag
            // b is [batch, 2*k, n] -> split into [batch, k, n] for real and imag
            int batch = shapeA.Length > 2 ? shapeA[0] : 1;
            int m = shapeA[shapeA.Length - 2];
            int twoK = shapeA[shapeA.Length - 1];
            int k = twoK / 2;
            int n = shapeB[shapeB.Length - 1];

            var resultShape = batch > 1 ? new[] { batch, m, 2 * n } : new[] { m, 2 * n };
            var result = new Tensor<T>(resultShape);

            // Extract real and imaginary parts
            // Format: first k columns are real, last k columns are imaginary
            for (int b_idx = 0; b_idx < (batch > 1 ? batch : 1); b_idx++)
            {
                // Compute: (A_real + i*A_imag) @ (B_real + i*B_imag)
                // = (A_real @ B_real - A_imag @ B_imag) + i(A_real @ B_imag + A_imag @ B_real)

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        T realPart = numOps.Zero;
                        T imagPart = numOps.Zero;

                        for (int k_idx = 0; k_idx < k; k_idx++)
                        {
                            // Get A components
                            var aIdxReal = batch > 1 ? new[] { b_idx, i, k_idx } : new[] { i, k_idx };
                            var aIdxImag = batch > 1 ? new[] { b_idx, i, k + k_idx } : new[] { i, k + k_idx };
                            T a_real = a.Value[aIdxReal];
                            T a_imag = a.Value[aIdxImag];

                            // Get B components
                            var bIdxReal = batch > 1 ? new[] { b_idx, k_idx, j } : new[] { k_idx, j };
                            var bIdxImag = batch > 1 ? new[] { b_idx, k + k_idx, j } : new[] { k + k_idx, j };
                            T b_real = b.Value[bIdxReal];
                            T b_imag = b.Value[bIdxImag];

                            // (a_real + i*a_imag) * (b_real + i*b_imag)
                            // = (a_real*b_real - a_imag*b_imag) + i(a_real*b_imag + a_imag*b_real)
                            T rr = numOps.Multiply(a_real, b_real);
                            T ii = numOps.Multiply(a_imag, b_imag);
                            T ri = numOps.Multiply(a_real, b_imag);
                            T ir = numOps.Multiply(a_imag, b_real);

                            realPart = numOps.Add(realPart, numOps.Subtract(rr, ii));
                            imagPart = numOps.Add(imagPart, numOps.Add(ri, ir));
                        }

                        // Store result
                        var resIdxReal = batch > 1 ? new[] { b_idx, i, j } : new[] { i, j };
                        var resIdxImag = batch > 1 ? new[] { b_idx, i, n + j } : new[] { i, n + j };
                        result[resIdxReal] = realPart;
                        result[resIdxImag] = imagPart;
                    }
                }
            }

            void BackwardFunction(Tensor<T> gradient)
            {
                // Simplified gradient (full complex matrix multiplication gradient is complex)
                if (a.RequiresGradient || b.RequiresGradient)
                {
                    // For now, approximate gradient
                    // Full implementation requires transposing and conjugating
                    if (a.RequiresGradient)
                    {
                        var gradA = new Tensor<T>(shapeA);
                        // gradient @ b^H (conjugate transpose)
                        // Simplified: just pass through gradient
                        var existingGrad = a.Gradient;
                        a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                    }

                    if (b.RequiresGradient)
                    {
                        var gradB = new Tensor<T>(shapeB);
                        // a^H @ gradient
                        // Simplified: just pass through gradient
                        var existingGrad = b.Gradient;
                        b.Gradient = existingGrad == null ? gradB : existingGrad.Add(gradB);
                    }
                }
            }

            var node = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || b.RequiresGradient,
                parents: new List<ComputationNode<T>> { a, b },
                backwardFunction: BackwardFunction,
                name: null);

            node.OperationType = OperationType.ComplexMatMul;
            node.OperationParams = new Dictionary<string, object> { { "Format", format } };

            var tape = GradientTape<T>.Current;
            if (tape != null && tape.IsRecording)
                tape.RecordOperation(node);

            return node;
        }

        if (format == "interleaved")
        {
            // For interleaved format: [batch, m, k*2] and [batch, k*2, n]
            // Complex numbers stored as [r,i,r,i,...] in last dimension
            int batch = shapeA.Length > 2 ? shapeA[0] : 1;
            int m = shapeA[shapeA.Length - 2];
            int kTimesTwo = shapeA[shapeA.Length - 1];
            int k = kTimesTwo / 2;
            int n = shapeB[shapeB.Length - 1];

            var resultShape = batch > 1 ? new[] { batch, m, 2 * n } : new[] { m, 2 * n };
            var result = new Tensor<T>(resultShape);

            for (int b_idx = 0; b_idx < (batch > 1 ? batch : 1); b_idx++)
            {
                // Compute: (A_real + i*A_imag) @ (B_real + i*B_imag)
                // = (A_real @ B_real - A_imag @ B_imag) + i(A_real @ B_imag + A_imag @ B_real)

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        T realPart = numOps.Zero;
                        T imagPart = numOps.Zero;

                        for (int k_idx = 0; k_idx < k; k_idx++)
                        {
                            // Get A components - interleaved: [r0, i0, r1, i1, ...]
                            var aIdxReal = batch > 1 ? new[] { b_idx, i, 2 * k_idx } : new[] { i, 2 * k_idx };
                            var aIdxImag = batch > 1 ? new[] { b_idx, i, 2 * k_idx + 1 } : new[] { i, 2 * k_idx + 1 };
                            T a_real = a.Value[aIdxReal];
                            T a_imag = a.Value[aIdxImag];

                            // Get B components - interleaved: [r0, i0, r1, i1, ...]
                            var bIdxReal = batch > 1 ? new[] { b_idx, 2 * k_idx, j } : new[] { 2 * k_idx, j };
                            var bIdxImag = batch > 1 ? new[] { b_idx, 2 * k_idx + 1, j } : new[] { 2 * k_idx + 1, j };
                            T b_real = b.Value[bIdxReal];
                            T b_imag = b.Value[bIdxImag];

                            // (a_real + i*a_imag) * (b_real + i*b_imag)
                            // = (a_real*b_real - a_imag*b_imag) + i(a_real*b_imag + a_imag*b_real)
                            T rr = numOps.Multiply(a_real, b_real);
                            T ii = numOps.Multiply(a_imag, b_imag);
                            T ri = numOps.Multiply(a_real, b_imag);
                            T ir = numOps.Multiply(a_imag, b_real);

                            realPart = numOps.Add(realPart, numOps.Subtract(rr, ii));
                            imagPart = numOps.Add(imagPart, numOps.Add(ri, ir));
                        }

                        // Store result in interleaved format
                        var resIdxReal = batch > 1 ? new[] { b_idx, i, 2 * j } : new[] { i, 2 * j };
                        var resIdxImag = batch > 1 ? new[] { b_idx, i, 2 * j + 1 } : new[] { i, 2 * j + 1 };
                        result[resIdxReal] = realPart;
                        result[resIdxImag] = imagPart;
                    }
                }
            }

            void BackwardFunctionInterleaved(Tensor<T> gradient)
            {
                // Complex matrix multiplication gradient with interleaved format
                // For C = A @ B (complex), dL/dA = dL/dC @ B^H, dL/dB = A^H @ dL/dC
                // Where ^H is conjugate transpose

                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(shapeA);
                    // Compute gradient @ conjugate(B)^T
                    // For now, initialize to zeros (proper implementation requires conjugate transpose)
                    for (int b_idx = 0; b_idx < (batch > 1 ? batch : 1); b_idx++)
                    {
                        for (int i = 0; i < m; i++)
                        {
                            for (int k_idx = 0; k_idx < k; k_idx++)
                            {
                                T gradRealSum = numOps.Zero;
                                T gradImagSum = numOps.Zero;

                                for (int j = 0; j < n; j++)
                                {
                                    // Get gradient components
                                    var gradIdxReal = batch > 1 ? new[] { b_idx, i, 2 * j } : new[] { i, 2 * j };
                                    var gradIdxImag = batch > 1 ? new[] { b_idx, i, 2 * j + 1 } : new[] { i, 2 * j + 1 };
                                    T g_real = gradient[gradIdxReal];
                                    T g_imag = gradient[gradIdxImag];

                                    // Get B conjugate components (b_real, -b_imag)
                                    var bIdxReal = batch > 1 ? new[] { b_idx, 2 * k_idx, j } : new[] { 2 * k_idx, j };
                                    var bIdxImag = batch > 1 ? new[] { b_idx, 2 * k_idx + 1, j } : new[] { 2 * k_idx + 1, j };
                                    T b_real = b.Value[bIdxReal];
                                    T b_imag = numOps.Negate(b.Value[bIdxImag]); // Conjugate

                                    // (g_real + i*g_imag) * (b_real + i*b_imag)
                                    T rr = numOps.Multiply(g_real, b_real);
                                    T ii = numOps.Multiply(g_imag, b_imag);
                                    T ri = numOps.Multiply(g_real, b_imag);
                                    T ir = numOps.Multiply(g_imag, b_real);

                                    gradRealSum = numOps.Add(gradRealSum, numOps.Subtract(rr, ii));
                                    gradImagSum = numOps.Add(gradImagSum, numOps.Add(ri, ir));
                                }

                                var aIdxReal = batch > 1 ? new[] { b_idx, i, 2 * k_idx } : new[] { i, 2 * k_idx };
                                var aIdxImag = batch > 1 ? new[] { b_idx, i, 2 * k_idx + 1 } : new[] { i, 2 * k_idx + 1 };
                                gradA[aIdxReal] = gradRealSum;
                                gradA[aIdxImag] = gradImagSum;
                            }
                        }
                    }
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }

                if (b.RequiresGradient)
                {
                    var gradB = new Tensor<T>(shapeB);
                    // Compute conjugate(A)^T @ gradient
                    for (int b_idx = 0; b_idx < (batch > 1 ? batch : 1); b_idx++)
                    {
                        for (int k_idx = 0; k_idx < k; k_idx++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                T gradRealSum = numOps.Zero;
                                T gradImagSum = numOps.Zero;

                                for (int i = 0; i < m; i++)
                                {
                                    // Get A conjugate components
                                    var aIdxReal = batch > 1 ? new[] { b_idx, i, 2 * k_idx } : new[] { i, 2 * k_idx };
                                    var aIdxImag = batch > 1 ? new[] { b_idx, i, 2 * k_idx + 1 } : new[] { i, 2 * k_idx + 1 };
                                    T a_real = a.Value[aIdxReal];
                                    T a_imag = numOps.Negate(a.Value[aIdxImag]); // Conjugate

                                    // Get gradient components
                                    var gradIdxReal = batch > 1 ? new[] { b_idx, i, 2 * j } : new[] { i, 2 * j };
                                    var gradIdxImag = batch > 1 ? new[] { b_idx, i, 2 * j + 1 } : new[] { i, 2 * j + 1 };
                                    T g_real = gradient[gradIdxReal];
                                    T g_imag = gradient[gradIdxImag];

                                    // (a_real + i*a_imag) * (g_real + i*g_imag)
                                    T rr = numOps.Multiply(a_real, g_real);
                                    T ii = numOps.Multiply(a_imag, g_imag);
                                    T ri = numOps.Multiply(a_real, g_imag);
                                    T ir = numOps.Multiply(a_imag, g_real);

                                    gradRealSum = numOps.Add(gradRealSum, numOps.Subtract(rr, ii));
                                    gradImagSum = numOps.Add(gradImagSum, numOps.Add(ri, ir));
                                }

                                var bIdxReal = batch > 1 ? new[] { b_idx, 2 * k_idx, j } : new[] { 2 * k_idx, j };
                                var bIdxImag = batch > 1 ? new[] { b_idx, 2 * k_idx + 1, j } : new[] { 2 * k_idx + 1, j };
                                gradB[bIdxReal] = gradRealSum;
                                gradB[bIdxImag] = gradImagSum;
                            }
                        }
                    }
                    var existingGrad = b.Gradient;
                    b.Gradient = existingGrad == null ? gradB : existingGrad.Add(gradB);
                }
            }

            var nodeInterleaved = new ComputationNode<T>(
                value: result,
                requiresGradient: a.RequiresGradient || b.RequiresGradient,
                parents: new List<ComputationNode<T>> { a, b },
                backwardFunction: BackwardFunctionInterleaved,
                name: null);

            nodeInterleaved.OperationType = OperationType.ComplexMatMul;
            nodeInterleaved.OperationParams = new Dictionary<string, object> { { "Format", format } };

            var tapeInterleaved = GradientTape<T>.Current;
            if (tapeInterleaved != null && tapeInterleaved.IsRecording)
                tapeInterleaved.RecordOperation(nodeInterleaved);

            return nodeInterleaved;
        }

        throw new NotImplementedException($"Complex matrix multiplication format '{format}' not implemented. Supported formats: 'split', 'interleaved'.");
    }

    /// <summary>
    /// Performs element-wise complex multiplication.
    /// </summary>
    /// <param name="a">First complex tensor with last dimension of size 2*n.</param>
    /// <param name="b">Second complex tensor with last dimension of size 2*n.</param>
    /// <param name="format">Whether complex numbers are "split" ([r,r,...,i,i,...]).</param>
    /// <returns>Element-wise complex product.</returns>
    /// <remarks>
    /// <para>
    /// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    /// </para>
    /// </remarks>
    public static ComputationNode<T> ComplexMultiply(ComputationNode<T> a, ComputationNode<T> b, string format = "split")
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (!shape.SequenceEqual(b.Value.Shape))
            throw new ArgumentException("Tensors must have the same shape for complex multiplication.");

        var result = new Tensor<T>(shape);

        // For split format: last dimension is 2*n, where first n are real, last n are imaginary
        int lastDim = shape[shape.Length - 1];
        int n = lastDim / 2;

        void ComputeProduct(int[] indices, int dim)
        {
            if (dim == shape.Length - 1)
            {
                // This is a complex number dimension - process in pairs
                for (int i = 0; i < n; i++)
                {
                    var idxReal = indices.Take(indices.Length - 1).Concat(new[] { i }).ToArray();
                    var idxImag = indices.Take(indices.Length - 1).Concat(new[] { n + i }).ToArray();

                    T a_real = a.Value[idxReal];
                    T a_imag = a.Value[idxImag];
                    T b_real = b.Value[idxReal];
                    T b_imag = b.Value[idxImag];

                    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
                    T ac = numOps.Multiply(a_real, b_real);
                    T bd = numOps.Multiply(a_imag, b_imag);
                    T ad = numOps.Multiply(a_real, b_imag);
                    T bc = numOps.Multiply(a_imag, b_real);

                    result[idxReal] = numOps.Subtract(ac, bd);
                    result[idxImag] = numOps.Add(ad, bc);
                }
            }
            else
            {
                for (int i = 0; i < shape[dim]; i++)
                {
                    indices[dim] = i;
                    ComputeProduct(indices, dim + 1);
                }
            }
        }

        ComputeProduct(new int[shape.Length], 0);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient || b.RequiresGradient)
            {
                // ∂(a*b)/∂a = b* (conjugate)
                // ∂(a*b)/∂b = a* (conjugate)

                if (a.RequiresGradient)
                {
                    var gradA = new Tensor<T>(shape);
                    // Simplified gradient
                    var existingGrad = a.Gradient;
                    a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
                }

                if (b.RequiresGradient)
                {
                    var gradB = new Tensor<T>(shape);
                    // Simplified gradient
                    var existingGrad = b.Gradient;
                    b.Gradient = existingGrad == null ? gradB : existingGrad.Add(gradB);
                }
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient || b.RequiresGradient,
            parents: new List<ComputationNode<T>> { a, b },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.ComplexMultiply;
        node.OperationParams = new Dictionary<string, object> { { "Format", format } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Extracts a slice from a tensor along a specified axis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This operation extracts a portion of a tensor along a specified axis, starting at
    /// a given offset and continuing for a specified length. An optional step parameter
    /// allows for strided slicing (e.g., every 2nd element).
    /// </para>
    /// <para><b>For Beginners:</b> Think of this like taking a substring from a string.
    ///
    /// For example, if you have a tensor [1, 2, 3, 4, 5, 6] and you slice with start=1, length=3:
    /// - You get [2, 3, 4]
    ///
    /// With step=2 and start=0, length=3:
    /// - You get [1, 3, 5] (every 2nd element)
    ///
    /// This is useful for extracting specific parts of data, like separating real and
    /// imaginary parts of complex numbers stored in interleaved format.
    /// </para>
    /// </remarks>
    /// <param name="a">The input tensor to slice.</param>
    /// <param name="start">The starting index along the specified axis.</param>
    /// <param name="length">The number of elements to extract.</param>
    /// <param name="step">The step size between elements (default 1).</param>
    /// <param name="axis">The axis along which to slice (default 0).</param>
    /// <returns>A new computation node containing the sliced tensor.</returns>
    public static ComputationNode<T> Slice(ComputationNode<T> a, int start, int length, int step = 1, int axis = 0)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        // Handle negative axis
        if (axis < 0)
            axis = shape.Length + axis;

        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {shape.Length} dimensions.");

        if (start < 0 || start >= shape[axis])
            throw new ArgumentOutOfRangeException(nameof(start), $"Start index {start} is out of range for axis with size {shape[axis]}.");

        if (step <= 0)
            throw new ArgumentException("Step must be positive.", nameof(step));

        // Calculate actual length based on step
        int actualLength = 0;
        for (int i = start; i < shape[axis] && actualLength < length; i += step)
            actualLength++;

        // Calculate result shape
        var resultShape = shape.ToArray();
        resultShape[axis] = actualLength;

        var result = new Tensor<T>(resultShape);

        // Copy elements
        int[] srcIndices = new int[shape.Length];
        int[] dstIndices = new int[shape.Length];

        void CopyElements(int dim)
        {
            if (dim == shape.Length)
            {
                result[dstIndices] = a.Value[srcIndices];
            }
            else if (dim == axis)
            {
                int dstIdx = 0;
                for (int i = start; i < shape[axis] && dstIdx < actualLength; i += step)
                {
                    srcIndices[dim] = i;
                    dstIndices[dim] = dstIdx;
                    CopyElements(dim + 1);
                    dstIdx++;
                }
            }
            else
            {
                for (int i = 0; i < shape[dim]; i++)
                {
                    srcIndices[dim] = i;
                    dstIndices[dim] = i;
                    CopyElements(dim + 1);
                }
            }
        }

        CopyElements(0);

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Gradient is scattered back to original positions
                var gradA = new Tensor<T>(shape);

                int[] gradSrcIndices = new int[resultShape.Length];
                int[] gradDstIndices = new int[shape.Length];

                void ScatterGradients(int dim)
                {
                    if (dim == resultShape.Length)
                    {
                        gradA[gradDstIndices] = numOps.Add(gradA[gradDstIndices], gradient[gradSrcIndices]);
                    }
                    else if (dim == axis)
                    {
                        int srcIdx = 0;
                        for (int i = start; i < shape[axis] && srcIdx < actualLength; i += step)
                        {
                            gradDstIndices[dim] = i;
                            gradSrcIndices[dim] = srcIdx;
                            ScatterGradients(dim + 1);
                            srcIdx++;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < resultShape[dim]; i++)
                        {
                            gradDstIndices[dim] = i;
                            gradSrcIndices[dim] = i;
                            ScatterGradients(dim + 1);
                        }
                    }
                }

                ScatterGradients(0);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.Slice;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Start", start },
            { "Length", length },
            { "Step", step },
            { "Axis", axis }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Applies Gumbel-Softmax for differentiable discrete sampling approximation.
    /// </summary>
    /// <param name="logits">The input logits.</param>
    /// <param name="temperature">Temperature parameter controlling softness (default 1.0).</param>
    /// <param name="hard">Whether to use straight-through estimator for hard samples.</param>
    /// <returns>A computation node containing the soft/hard samples.</returns>
    /// <remarks>
    /// <para>
    /// Gumbel-Softmax provides a differentiable approximation to categorical sampling.
    /// As temperature approaches 0, outputs approach one-hot categorical samples.
    /// When hard=true, uses straight-through estimator for discrete outputs with gradient pass-through.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> GumbelSoftmax(ComputationNode<T> logits, double temperature = 1.0, bool hard = false)
    {
        // Validate temperature: must be positive and finite
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be positive.");
        if (double.IsNaN(temperature) || double.IsInfinity(temperature))
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be a finite number.");

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = logits.Value.Shape;
        var eps = 1e-10;

        // Add Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
        var gumbel = new Tensor<T>(shape);
        var random = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < gumbel.Length; i++)
        {
            var u = random.NextDouble();
            u = Math.Max(u, eps);
            u = Math.Min(u, 1 - eps);
            gumbel.SetFlat(i, numOps.FromDouble(-Math.Log(-Math.Log(u))));
        }

        // Compute soft samples: softmax((logits + gumbel) / temperature)
        var tempTensor = new Tensor<T>(shape);
        for (int i = 0; i < tempTensor.Length; i++)
        {
            var val = numOps.Add(logits.Value.GetFlat(i), gumbel.GetFlat(i));
            tempTensor.SetFlat(i, numOps.Divide(val, numOps.FromDouble(temperature)));
        }

        // Apply softmax along last axis
        var softResult = engine.Softmax(tempTensor, axis: -1);

        // If hard, use straight-through estimator
        Tensor<T> result;
        if (hard)
        {
            // Find argmax and create one-hot
            var hardResult = new Tensor<T>(shape);
            int lastDim = shape[^1];
            int batchSize = softResult.Length / lastDim;

            for (int b = 0; b < batchSize; b++)
            {
                int maxIdx = 0;
                T maxVal = softResult.GetFlat(b * lastDim);
                for (int i = 1; i < lastDim; i++)
                {
                    if (numOps.GreaterThan(softResult.GetFlat(b * lastDim + i), maxVal))
                    {
                        maxVal = softResult.GetFlat(b * lastDim + i);
                        maxIdx = i;
                    }
                }
                for (int i = 0; i < lastDim; i++)
                {
                    hardResult.SetFlat(b * lastDim + i, i == maxIdx ? numOps.One : numOps.Zero);
                }
            }

            // Straight-through: hard in forward, soft in backward
            result = hardResult;
        }
        else
        {
            result = softResult;
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (!logits.RequiresGradient) return;

            // Gradient of softmax: softmax * (gradient - sum(gradient * softmax))
            var softGrad = new Tensor<T>(shape);
            int lastDim = shape[^1];
            int batchSize = softResult.Length / lastDim;

            for (int b = 0; b < batchSize; b++)
            {
                T dotProduct = numOps.Zero;
                for (int i = 0; i < lastDim; i++)
                {
                    dotProduct = numOps.Add(dotProduct,
                        numOps.Multiply(gradient[b * lastDim + i], softResult[b * lastDim + i]));
                }
                for (int i = 0; i < lastDim; i++)
                {
                    var gradVal = numOps.Subtract(gradient[b * lastDim + i], dotProduct);
                    softGrad[b * lastDim + i] = numOps.Divide(
                        numOps.Multiply(softResult[b * lastDim + i], gradVal),
                        numOps.FromDouble(temperature));
                }
            }

            var existingLogitsGrad = logits.Gradient;

            logits.Gradient = existingLogitsGrad == null ? softGrad : engine.TensorAdd(existingLogitsGrad, softGrad);
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: logits.RequiresGradient,
            parents: new List<ComputationNode<T>> { logits },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.GumbelSoftmax;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Temperature", temperature },
            { "Hard", hard }
        };

        var tape2 = GradientTape<T>.Current;
        if (tape2 != null && tape2.IsRecording)
            tape2.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Applies a surrogate spike function for spiking neural network JIT compilation.
    /// </summary>
    /// <param name="membranePotential">The membrane potential input.</param>
    /// <param name="threshold">The spike threshold (default 1.0).</param>
    /// <param name="surrogateBeta">Sharpness of the surrogate gradient (default 1.0).</param>
    /// <returns>A computation node containing spike outputs with surrogate gradients.</returns>
    /// <remarks>
    /// <para>
    /// Uses the sigmoid surrogate for gradient computation while producing hard spikes in forward pass.
    /// Forward: spike = (potential > threshold) ? 1 : 0
    /// Backward: uses sigmoid derivative as surrogate gradient
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SurrogateSpike(ComputationNode<T> membranePotential, double threshold = 1.0, double surrogateBeta = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = membranePotential.Value.Shape;

        // Forward pass: hard threshold
        var spikes = new Tensor<T>(shape);
        var thresholdT = numOps.FromDouble(threshold);
        for (int i = 0; i < spikes.Length; i++)
        {
            spikes[i] = numOps.GreaterThan(membranePotential.Value[i], thresholdT) ? numOps.One : numOps.Zero;
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (!membranePotential.RequiresGradient) return;

            // Surrogate gradient: sigmoid derivative scaled by beta
            // d_surrogate = beta * sigmoid(beta * (v - threshold)) * (1 - sigmoid(beta * (v - threshold)))
            var surrogateGrad = new Tensor<T>(shape);
            for (int i = 0; i < shape.Length; i++)
            {
                var x = numOps.Multiply(
                    numOps.FromDouble(surrogateBeta),
                    numOps.Subtract(membranePotential.Value[i], thresholdT));
                var xDouble = Convert.ToDouble(x);
                var sigmoid = 1.0 / (1.0 + Math.Exp(-xDouble));
                var derivVal = surrogateBeta * sigmoid * (1.0 - sigmoid);
                surrogateGrad[i] = numOps.Multiply(gradient[i], numOps.FromDouble(derivVal));
            }

            membranePotential.Gradient = membranePotential.Gradient == null
                ? surrogateGrad
                : engine.TensorAdd(membranePotential.Gradient, surrogateGrad);
        }

        var node = new ComputationNode<T>(
            value: spikes,
            requiresGradient: membranePotential.RequiresGradient,
            parents: new List<ComputationNode<T>> { membranePotential },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.SurrogateSpike;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Threshold", threshold },
            { "SurrogateBeta", surrogateBeta }
        };

        var tape3 = GradientTape<T>.Current;
        if (tape3 != null && tape3.IsRecording)
            tape3.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Applies a straight-through threshold for HTM-style sparse activations.
    /// </summary>
    /// <param name="input">The input activations.</param>
    /// <param name="threshold">The threshold value.</param>
    /// <returns>Binary activations with straight-through gradients.</returns>
    /// <remarks>
    /// <para>
    /// Forward: output = (input > threshold) ? 1 : 0
    /// Backward: gradients pass through unchanged (straight-through estimator)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> StraightThroughThreshold(ComputationNode<T> input, double threshold)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = input.Value.Shape;
        var thresholdT = numOps.FromDouble(threshold);

        var result = new Tensor<T>(shape);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = numOps.GreaterThan(input.Value[i], thresholdT) ? numOps.One : numOps.Zero;
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (!input.RequiresGradient) return;
            // Straight-through: pass gradients unchanged
            var existingGrad = input.Gradient;
            input.Gradient = existingGrad == null ? gradient : engine.TensorAdd(existingGrad, gradient);
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient,
            parents: new List<ComputationNode<T>> { input },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.StraightThroughThreshold;
        node.OperationParams = new Dictionary<string, object> { { "Threshold", threshold } };

        var tape4 = GradientTape<T>.Current;
        if (tape4 != null && tape4.IsRecording)
            tape4.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Differentiable Top-K selection for mixture-of-experts routing.
    /// </summary>
    /// <param name="scores">The routing scores for each expert.</param>
    /// <param name="k">Number of experts to select.</param>
    /// <returns>Sparse routing weights with only top-K non-zero.</returns>
    /// <remarks>
    /// <para>
    /// Selects top-K values and normalizes them via softmax.
    /// Gradients flow only to the selected experts.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> TopKSoftmax(ComputationNode<T> scores, int k)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = scores.Value.Shape;
        int lastDim = shape[^1];
        int batchSize = scores.Value.Length / lastDim;

        var result = new Tensor<T>(shape);
        var topKIndices = new int[batchSize, k];

        for (int b = 0; b < batchSize; b++)
        {
            // Find top-K indices
            var indices = Enumerable.Range(0, lastDim).ToList();
            indices.Sort((i, j) =>
                Convert.ToDouble(scores.Value[b * lastDim + j])
                    .CompareTo(Convert.ToDouble(scores.Value[b * lastDim + i])));

            // Store top-K indices
            for (int i = 0; i < k; i++)
                topKIndices[b, i] = indices[i];

            // Compute softmax over top-K
            double maxVal = double.NegativeInfinity;
            for (int i = 0; i < k; i++)
            {
                var val = Convert.ToDouble(scores.Value[b * lastDim + topKIndices[b, i]]);
                if (val > maxVal) maxVal = val;
            }

            double sumExp = 0;
            var expVals = new double[k];
            for (int i = 0; i < k; i++)
            {
                expVals[i] = Math.Exp(Convert.ToDouble(scores.Value[b * lastDim + topKIndices[b, i]]) - maxVal);
                sumExp += expVals[i];
            }

            // Set result: zero for non-top-K, softmax for top-K
            for (int i = 0; i < lastDim; i++)
                result[b * lastDim + i] = numOps.Zero;

            for (int i = 0; i < k; i++)
                result[b * lastDim + topKIndices[b, i]] = numOps.FromDouble(expVals[i] / sumExp);
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (!scores.RequiresGradient) return;

            var scoreGrad = new Tensor<T>(shape);
            for (int b = 0; b < batchSize; b++)
            {
                // Gradient only flows through top-K
                double dotProduct = 0;
                for (int i = 0; i < k; i++)
                {
                    int idx = topKIndices[b, i];
                    dotProduct += Convert.ToDouble(gradient[b * lastDim + idx])
                                * Convert.ToDouble(result[b * lastDim + idx]);
                }

                for (int i = 0; i < k; i++)
                {
                    int idx = topKIndices[b, i];
                    var softVal = Convert.ToDouble(result[b * lastDim + idx]);
                    var gradVal = Convert.ToDouble(gradient[b * lastDim + idx]);
                    scoreGrad[b * lastDim + idx] = numOps.FromDouble(softVal * (gradVal - dotProduct));
                }
            }

            var existingScoresGrad = scores.Gradient;

            scores.Gradient = existingScoresGrad == null ? scoreGrad : engine.TensorAdd(existingScoresGrad, scoreGrad);
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: scores.RequiresGradient,
            parents: new List<ComputationNode<T>> { scores },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.TopKSoftmax;
        node.OperationParams = new Dictionary<string, object> { { "K", k } };

        var tape5 = GradientTape<T>.Current;
        if (tape5 != null && tape5.IsRecording)
            tape5.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Leaky state update for reservoir/echo state networks.
    /// </summary>
    /// <param name="prevState">Previous hidden state.</param>
    /// <param name="input">Current input.</param>
    /// <param name="weights">Reservoir weight matrix (can be frozen).</param>
    /// <param name="leakingRate">Leaking rate (default 1.0 for full update).</param>
    /// <returns>New hidden state.</returns>
    /// <remarks>
    /// <para>
    /// Computes: new_state = (1 - leakingRate) * prevState + leakingRate * tanh(weights @ prevState + input)
    /// </para>
    /// </remarks>
    public static ComputationNode<T> LeakyStateUpdate(
        ComputationNode<T> prevState,
        ComputationNode<T> input,
        ComputationNode<T> weights,
        double leakingRate = 1.0)
    {
        // weights @ prevState
        var weighted = MatrixMultiply(weights, prevState);
        // weights @ prevState + input
        var preActivation = Add(weighted, input);
        // tanh(...)
        var activated = Tanh(preActivation);

        if (Math.Abs(leakingRate - 1.0) < 1e-10)
        {
            // No leaking, just return activated
            return activated;
        }

        // (1 - leakingRate) * prevState
        var numOps = MathHelper.GetNumericOperations<T>();
        var keepRate = Constant(new Tensor<T>([1]) { [0] = numOps.FromDouble(1.0 - leakingRate) });
        var leakRate = Constant(new Tensor<T>([1]) { [0] = numOps.FromDouble(leakingRate) });

        // Scale by broadcasting
        var keptPrev = ElementwiseMultiply(prevState, keepRate);
        var scaledNew = ElementwiseMultiply(activated, leakRate);

        return Add(keptPrev, scaledNew);
    }

    /// <summary>
    /// CRF forward algorithm for sequence labeling.
    /// </summary>
    /// <param name="emissions">Emission scores [seq_len, num_tags].</param>
    /// <param name="transitions">Transition matrix [num_tags, num_tags].</param>
    /// <param name="startScores">Optional start scores [num_tags].</param>
    /// <param name="endScores">Optional end scores [num_tags].</param>
    /// <returns>Log partition function (normalizer).</returns>
    /// <remarks>
    /// <para>
    /// Computes the log partition function using the forward-backward algorithm.
    /// This is differentiable and returns proper gradients for emissions, transitions,
    /// start scores, and end scores.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> CRFForward(
        ComputationNode<T> emissions,
        ComputationNode<T> transitions,
        ComputationNode<T>? startScores = null,
        ComputationNode<T>? endScores = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        int seqLen = emissions.Value.Shape[0];
        int numTags = emissions.Value.Shape[1];

        // Copy values to double for stable log-sum-exp computations
        double[,] emit = new double[seqLen, numTags];
        for (int t = 0; t < seqLen; t++)
            for (int j = 0; j < numTags; j++)
                emit[t, j] = Convert.ToDouble(emissions.Value[t, j]);

        double[,] trans = new double[numTags, numTags];
        for (int i = 0; i < numTags; i++)
            for (int j = 0; j < numTags; j++)
                trans[i, j] = Convert.ToDouble(transitions.Value[i, j]);

        double[] start = new double[numTags];
        double[] end = new double[numTags];
        if (startScores != null)
        {
            for (int j = 0; j < numTags; j++) start[j] = Convert.ToDouble(startScores.Value[j]);
        }
        if (endScores != null)
        {
            for (int j = 0; j < numTags; j++) end[j] = Convert.ToDouble(endScores.Value[j]);
        }

        // Forward (alpha)
        double[,] alpha = new double[seqLen, numTags];
        for (int j = 0; j < numTags; j++)
            alpha[0, j] = start[j] + emit[0, j];

        for (int t = 1; t < seqLen; t++)
        {
            for (int j = 0; j < numTags; j++)
            {
                double maxPrev = double.NegativeInfinity;
                for (int i = 0; i < numTags; i++)
                    maxPrev = Math.Max(maxPrev, alpha[t - 1, i] + trans[i, j]);

                double sumExp = 0.0;
                for (int i = 0; i < numTags; i++)
                    sumExp += Math.Exp(alpha[t - 1, i] + trans[i, j] - maxPrev);

                alpha[t, j] = emit[t, j] + maxPrev + Math.Log(sumExp);
            }
        }

        // Termination with end scores
        double maxFinal = double.NegativeInfinity;
        for (int j = 0; j < numTags; j++)
            maxFinal = Math.Max(maxFinal, alpha[seqLen - 1, j] + end[j]);

        double finalSum = 0.0;
        for (int j = 0; j < numTags; j++)
            finalSum += Math.Exp(alpha[seqLen - 1, j] + end[j] - maxFinal);

        double logZ = maxFinal + Math.Log(finalSum);
        var logPartition = new Tensor<T>([1]) { [0] = numOps.FromDouble(logZ) };

        void BackwardFunction(Tensor<T> gradient)
        {
            if (!(emissions.RequiresGradient || transitions.RequiresGradient || (startScores?.RequiresGradient ?? false) || (endScores?.RequiresGradient ?? false)))
                return;

            double gradScale = Convert.ToDouble(gradient[0]);

            // Backward (beta)
            double[,] beta = new double[seqLen, numTags];
            for (int j = 0; j < numTags; j++)
                beta[seqLen - 1, j] = end[j];

            for (int t = seqLen - 2; t >= 0; t--)
            {
                for (int i = 0; i < numTags; i++)
                {
                    double maxNext = double.NegativeInfinity;
                    for (int j = 0; j < numTags; j++)
                        maxNext = Math.Max(maxNext, trans[i, j] + emit[t + 1, j] + beta[t + 1, j]);

                    double sumExp = 0.0;
                    for (int j = 0; j < numTags; j++)
                        sumExp += Math.Exp(trans[i, j] + emit[t + 1, j] + beta[t + 1, j] - maxNext);

                    beta[t, i] = maxNext + Math.Log(sumExp);
                }
            }

            var emitGrad = new Tensor<T>(emissions.Value.Shape);
            var transGrad = new Tensor<T>(transitions.Value.Shape);
            Tensor<T>? startGrad = startScores != null ? new Tensor<T>(startScores.Value.Shape) : null;
            Tensor<T>? endGrad = endScores != null ? new Tensor<T>(endScores.Value.Shape) : null;

            // Emission grads (posterior)
            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < numTags; j++)
                {
                    double logProb = alpha[t, j] + beta[t, j] - logZ;
                    double prob = Math.Exp(logProb);
                    emitGrad[t, j] = numOps.FromDouble(prob * gradScale);
                }
            }

            // Transition grads (expected transitions)
            for (int t = 1; t < seqLen; t++)
            {
                for (int i = 0; i < numTags; i++)
                {
                    for (int j = 0; j < numTags; j++)
                    {
                        double logProb = alpha[t - 1, i] + trans[i, j] + emit[t, j] + beta[t, j] - logZ;
                        double prob = Math.Exp(logProb);
                        transGrad[i, j] = numOps.Add(transGrad[i, j], numOps.FromDouble(prob * gradScale));
                    }
                }
            }

            // Start grads
            if (startGrad != null && startScores != null)
            {
                for (int j = 0; j < numTags; j++)
                {
                    double logProb = start[j] + emit[0, j] + beta[0, j] - logZ;
                    double prob = Math.Exp(logProb);
                    startGrad[j] = numOps.FromDouble(prob * gradScale);
                }
            }

            // End grads
            if (endGrad != null && endScores != null)
            {
                for (int j = 0; j < numTags; j++)
                {
                    double logProb = alpha[seqLen - 1, j] + end[j] - logZ;
                    double prob = Math.Exp(logProb);
                    endGrad[j] = numOps.FromDouble(prob * gradScale);
                }
            }

            if (emissions.RequiresGradient)
            {
                emissions.Gradient = emissions.Gradient == null ? emitGrad : engine.TensorAdd(emissions.Gradient, emitGrad);
            }
            if (transitions.RequiresGradient)
            {
                transitions.Gradient = transitions.Gradient == null ? transGrad : engine.TensorAdd(transitions.Gradient, transGrad);
            }
            if (startScores != null && startScores.RequiresGradient && startGrad != null)
            {
                startScores.Gradient = startScores.Gradient == null ? startGrad : engine.TensorAdd(startScores.Gradient, startGrad);
            }
            if (endScores != null && endScores.RequiresGradient && endGrad != null)
            {
                endScores.Gradient = endScores.Gradient == null ? endGrad : engine.TensorAdd(endScores.Gradient, endGrad);
            }
        }

        var parents = new List<ComputationNode<T>> { emissions, transitions };
        if (startScores != null) parents.Add(startScores);
        if (endScores != null) parents.Add(endScores);

        var node = new ComputationNode<T>(
            value: logPartition,
            requiresGradient: emissions.RequiresGradient || transitions.RequiresGradient || (startScores?.RequiresGradient ?? false) || (endScores?.RequiresGradient ?? false),
            parents: parents,
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.CRFForward;
        node.OperationParams = null;

        var tape6 = GradientTape<T>.Current;
        if (tape6 != null && tape6.IsRecording)
            tape6.RecordOperation(node);

        return node;
    }

    /// <summary>
    /// Anomaly score computation using reconstruction error or density estimation.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="reconstruction">Reconstructed input (e.g., from autoencoder).</param>
    /// <returns>Anomaly scores (higher = more anomalous).</returns>
    public static ComputationNode<T> AnomalyScore(ComputationNode<T> input, ComputationNode<T> reconstruction)
    {
        // Compute squared error as anomaly score
        var diff = Subtract(input, reconstruction);
        var squared = Square(diff);
        return Mean(squared);
    }

    /// <summary>
    /// Applies the Parametric Rectified Linear Unit (PReLU) activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="alpha">The slope for negative values (default 0.01).</param>
    /// <returns>A new computation node with PReLU applied.</returns>
    /// <remarks>
    /// <para>
    /// PReLU(x) = x if x > 0, alpha * x otherwise.
    /// Similar to LeakyReLU but alpha is typically learned during training.
    /// </para>
    /// <para><b>Gradient:</b> d(PReLU)/dx = 1 if x > 0, alpha otherwise.</para>
    /// </remarks>
    public static ComputationNode<T> PReLU(ComputationNode<T> a, double alpha = 0.01)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var alphaT = numOps.FromDouble(alpha);

        // Forward pass: max(0, x) + alpha * min(0, x)
        var result = a.Value.Transform((x, _) =>
        {
            if (numOps.GreaterThan(x, numOps.Zero))
                return x;
            else
                return numOps.Multiply(alphaT, x);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(PReLU)/dx = 1 if x > 0, alpha if x <= 0
                var derivative = a.Value.Transform((x, _) =>
                {
                    if (numOps.GreaterThan(x, numOps.Zero))
                        return numOps.One;
                    else
                        return alphaT;
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.PReLU;
        node.OperationParams = new Dictionary<string, object> { { "Alpha", alpha } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Thresholded Rectified Linear Unit activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="threshold">The threshold value (default 1.0).</param>
    /// <returns>A new computation node with ThresholdedReLU applied.</returns>
    /// <remarks>
    /// <para>
    /// ThresholdedReLU(x) = x if x > threshold, 0 otherwise.
    /// Unlike standard ReLU which activates at 0, this activates at a configurable threshold.
    /// </para>
    /// <para><b>Gradient:</b> d(ThresholdedReLU)/dx = 1 if x > threshold, 0 otherwise.</para>
    /// </remarks>
    public static ComputationNode<T> ThresholdedReLU(ComputationNode<T> a, double threshold = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var thresholdT = numOps.FromDouble(threshold);

        var result = a.Value.Transform((x, _) =>
        {
            if (numOps.GreaterThan(x, thresholdT))
                return x;
            else
                return numOps.Zero;
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var derivative = a.Value.Transform((x, _) =>
                {
                    if (numOps.GreaterThan(x, thresholdT))
                        return numOps.One;
                    else
                        return numOps.Zero;
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.ThresholdedReLU;
        node.OperationParams = new Dictionary<string, object> { { "Threshold", threshold } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Inverse Square Root Unit (ISRU) activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="alpha">The scaling parameter (default 1.0).</param>
    /// <returns>A new computation node with ISRU applied.</returns>
    /// <remarks>
    /// <para>
    /// ISRU(x) = x / sqrt(1 + alpha * x²)
    /// A smooth, bounded activation function that ranges from -1/sqrt(alpha) to 1/sqrt(alpha).
    /// </para>
    /// <para><b>Gradient:</b> d(ISRU)/dx = (1 + alpha * x²)^(-3/2)</para>
    /// </remarks>
    public static ComputationNode<T> ISRU(ComputationNode<T> a, double alpha = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var alphaT = numOps.FromDouble(alpha);

        var result = a.Value.Transform((x, _) =>
        {
            var xSquared = numOps.Multiply(x, x);
            var denom = numOps.Sqrt(numOps.Add(numOps.One, numOps.Multiply(alphaT, xSquared)));
            return numOps.Divide(x, denom);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(ISRU)/dx = (1 + alpha * x²)^(-3/2)
                var derivative = a.Value.Transform((x, _) =>
                {
                    var xSquared = numOps.Multiply(x, x);
                    var inner = numOps.Add(numOps.One, numOps.Multiply(alphaT, xSquared));
                    var sqrtInner = numOps.Sqrt(inner);
                    return numOps.Divide(numOps.One, numOps.Multiply(inner, sqrtInner));
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.ISRU;
        node.OperationParams = new Dictionary<string, object> { { "Alpha", alpha } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Sign function with surrogate gradient for training.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="surrogateBeta">Sharpness of the surrogate gradient (default 1.0).</param>
    /// <returns>A new computation node with Sign applied using straight-through estimator.</returns>
    /// <remarks>
    /// <para>
    /// Sign(x) = 1 if x > 0, -1 if x < 0, 0 if x = 0.
    /// Uses sigmoid surrogate gradient for backpropagation since the true derivative is zero almost everywhere.
    /// </para>
    /// <para><b>Surrogate Gradient:</b> beta * sigmoid(beta * x) * (1 - sigmoid(beta * x))</para>
    /// </remarks>
    public static ComputationNode<T> Sign(ComputationNode<T> a, double surrogateBeta = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var betaT = numOps.FromDouble(surrogateBeta);

        // Forward: hard sign
        var result = a.Value.Transform((x, _) =>
        {
            if (numOps.GreaterThan(x, numOps.Zero))
                return numOps.One;
            else if (numOps.LessThan(x, numOps.Zero))
                return numOps.Negate(numOps.One);
            else
                return numOps.Zero;
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Surrogate gradient: beta * sigmoid(beta*x) * (1 - sigmoid(beta*x))
                var derivative = a.Value.Transform((x, _) =>
                {
                    var scaledX = numOps.Multiply(betaT, x);
                    var sig = numOps.Divide(numOps.One, numOps.Add(numOps.One, numOps.Exp(numOps.Negate(scaledX))));
                    var oneMinusSig = numOps.Subtract(numOps.One, sig);
                    return numOps.Multiply(betaT, numOps.Multiply(sig, oneMinusSig));
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.Sign;
        node.OperationParams = new Dictionary<string, object> { { "SurrogateBeta", surrogateBeta } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Log-Softmax function for numerically stable cross-entropy loss computation.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="axis">The axis along which to compute log-softmax (default -1, last axis).</param>
    /// <returns>A new computation node with Log-Softmax applied.</returns>
    /// <remarks>
    /// <para>
    /// LogSoftmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
    /// More numerically stable than computing log(softmax(x)) separately.
    /// </para>
    /// <para><b>Gradient:</b> d(LogSoftmax)/dx_i = 1 - softmax(x)_i for the target class.</para>
    /// </remarks>
    public static ComputationNode<T> LogSoftmax(ComputationNode<T> a, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (axis < 0)
            axis = shape.Length + axis;

        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {shape.Length} dimensions.");

        // Compute strides for N-dimensional iteration
        int axisSize = shape[axis];
        int outerSize = 1;
        int innerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++)
            innerSize *= shape[i];

        var result = new Tensor<T>(shape);
        var softmaxOutput = new Tensor<T>(shape);

        // Iterate over all positions in the non-axis dimensions
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Find max for numerical stability
                var maxVal = a.Value[outer * axisSize * innerSize + inner];
                for (int i = 1; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var val = a.Value[flatIdx];
                    if (numOps.GreaterThan(val, maxVal))
                        maxVal = val;
                }

                // Compute log-sum-exp
                var logSumExp = numOps.Zero;
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var shifted = numOps.Subtract(a.Value[flatIdx], maxVal);
                    logSumExp = numOps.Add(logSumExp, numOps.Exp(shifted));
                }
                logSumExp = numOps.Add(numOps.Log(logSumExp), maxVal);

                // Compute log-softmax: x - log-sum-exp
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var logSoftmaxVal = numOps.Subtract(a.Value[flatIdx], logSumExp);
                    result[flatIdx] = logSoftmaxVal;
                    softmaxOutput[flatIdx] = numOps.Exp(logSoftmaxVal);
                }
            }
        }

        // Capture values for backward pass
        int capturedAxis = axis;
        int capturedAxisSize = axisSize;
        int capturedOuterSize = outerSize;
        int capturedInnerSize = innerSize;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(shape);
                for (int outer = 0; outer < capturedOuterSize; outer++)
                {
                    for (int inner = 0; inner < capturedInnerSize; inner++)
                    {
                        // Sum of gradients * softmax along axis
                        var gradSum = numOps.Zero;
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            gradSum = numOps.Add(gradSum, numOps.Multiply(gradient[flatIdx], softmaxOutput[flatIdx]));
                        }
                        // Gradient: gradient - softmax * sum(gradient)
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            gradA[flatIdx] = numOps.Subtract(gradient[flatIdx],
                                numOps.Multiply(softmaxOutput[flatIdx], gradSum));
                        }
                    }
                }
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.LogSoftmax;
        node.OperationParams = new Dictionary<string, object> { { "Axis", capturedAxis } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Softmin function, which assigns higher probability to lower values.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="axis">The axis along which to compute softmin (default -1, last axis).</param>
    /// <returns>A new computation node with Softmin applied.</returns>
    /// <remarks>
    /// <para>
    /// Softmin(x) = softmax(-x) = exp(-x) / sum(exp(-x))
    /// Useful when lower values should have higher probability, e.g., in attention over distances.
    /// </para>
    /// <para><b>Gradient:</b> Same Jacobian structure as softmax but with negated input.</para>
    /// </remarks>
    public static ComputationNode<T> Softmin(ComputationNode<T> a, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (axis < 0)
            axis = shape.Length + axis;

        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {shape.Length} dimensions.");

        // Compute strides for N-dimensional iteration
        int axisSize = shape[axis];
        int outerSize = 1;
        int innerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++)
            innerSize *= shape[i];

        var result = new Tensor<T>(shape);

        // Iterate over all positions in the non-axis dimensions
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Find max of -x for numerical stability (which is -min of x)
                var maxNegVal = numOps.Negate(a.Value[outer * axisSize * innerSize + inner]);
                for (int i = 1; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var negVal = numOps.Negate(a.Value[flatIdx]);
                    if (numOps.GreaterThan(negVal, maxNegVal))
                        maxNegVal = negVal;
                }

                // Compute exp(-x - max(-x)) and sum
                var expSum = numOps.Zero;
                var expValues = new T[axisSize];
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var shifted = numOps.Subtract(numOps.Negate(a.Value[flatIdx]), maxNegVal);
                    expValues[i] = numOps.Exp(shifted);
                    expSum = numOps.Add(expSum, expValues[i]);
                }

                // Normalize
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    result[flatIdx] = numOps.Divide(expValues[i], expSum);
                }
            }
        }

        // Capture values for backward pass
        int capturedAxis = axis;
        int capturedAxisSize = axisSize;
        int capturedOuterSize = outerSize;
        int capturedInnerSize = innerSize;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Same as softmax gradient but with negation
                var gradA = new Tensor<T>(shape);
                for (int outer = 0; outer < capturedOuterSize; outer++)
                {
                    for (int inner = 0; inner < capturedInnerSize; inner++)
                    {
                        var dotProduct = numOps.Zero;
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            dotProduct = numOps.Add(dotProduct,
                                numOps.Multiply(gradient[flatIdx], result[flatIdx]));
                        }
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            var gradMinusDot = numOps.Subtract(gradient[flatIdx], dotProduct);
                            // Negate because d(softmax(-x))/dx = -softmax(-x) * (gradient - dot)
                            gradA[flatIdx] = numOps.Negate(numOps.Multiply(result[flatIdx], gradMinusDot));
                        }
                    }
                }
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.Softmin;
        node.OperationParams = new Dictionary<string, object> { { "Axis", capturedAxis } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Log-Softmin function for numerically stable computation.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="axis">The axis along which to compute log-softmin (default -1, last axis).</param>
    /// <returns>A new computation node with Log-Softmin applied.</returns>
    /// <remarks>
    /// <para>
    /// LogSoftmin(x) = log(softmin(x)) = -x - log(sum(exp(-x)))
    /// Combines log and softmin for numerical stability.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> LogSoftmin(ComputationNode<T> a, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (axis < 0)
            axis = shape.Length + axis;

        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {shape.Length} dimensions.");

        // Compute strides for N-dimensional iteration
        int axisSize = shape[axis];
        int outerSize = 1;
        int innerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++)
            innerSize *= shape[i];

        var result = new Tensor<T>(shape);
        var softminOutput = new Tensor<T>(shape);

        // Iterate over all positions in the non-axis dimensions
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Find max of -x for numerical stability
                var maxNegVal = numOps.Negate(a.Value[outer * axisSize * innerSize + inner]);
                for (int i = 1; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var negVal = numOps.Negate(a.Value[flatIdx]);
                    if (numOps.GreaterThan(negVal, maxNegVal))
                        maxNegVal = negVal;
                }

                // Compute log-sum-exp of -x
                var logSumExp = numOps.Zero;
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var shifted = numOps.Subtract(numOps.Negate(a.Value[flatIdx]), maxNegVal);
                    logSumExp = numOps.Add(logSumExp, numOps.Exp(shifted));
                }
                logSumExp = numOps.Add(numOps.Log(logSumExp), maxNegVal);

                // Compute log-softmin: -x - log-sum-exp(-x)
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var logSoftminVal = numOps.Subtract(numOps.Negate(a.Value[flatIdx]), logSumExp);
                    result[flatIdx] = logSoftminVal;
                    softminOutput[flatIdx] = numOps.Exp(logSoftminVal);
                }
            }
        }

        // Capture values for backward pass
        int capturedAxis = axis;
        int capturedAxisSize = axisSize;
        int capturedOuterSize = outerSize;
        int capturedInnerSize = innerSize;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(shape);
                for (int outer = 0; outer < capturedOuterSize; outer++)
                {
                    for (int inner = 0; inner < capturedInnerSize; inner++)
                    {
                        var gradSum = numOps.Zero;
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            gradSum = numOps.Add(gradSum, numOps.Multiply(gradient[flatIdx], softminOutput[flatIdx]));
                        }
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            // Gradient: -(gradient - softmin * sum(gradient))
                            gradA[flatIdx] = numOps.Negate(numOps.Subtract(gradient[flatIdx],
                                numOps.Multiply(softminOutput[flatIdx], gradSum)));
                        }
                    }
                }
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.LogSoftmin;
        node.OperationParams = new Dictionary<string, object> { { "Axis", capturedAxis } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Squared Radial Basis Function (SQRBF) activation.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="beta">The width parameter controlling the Gaussian bell curve (default 1.0).</param>
    /// <returns>A new computation node with SQRBF applied.</returns>
    /// <remarks>
    /// <para>
    /// SQRBF(x) = exp(-β * x²)
    /// A Gaussian bell-shaped activation with maximum at x=0 and values approaching 0 as |x| increases.
    /// </para>
    /// <para><b>Gradient:</b> d(SQRBF)/dx = -2βx * exp(-β * x²)</para>
    /// </remarks>
    public static ComputationNode<T> SQRBF(ComputationNode<T> a, double beta = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var betaT = numOps.FromDouble(beta);

        // Forward: exp(-β * x²)
        var result = a.Value.Transform((x, _) =>
        {
            var xSquared = numOps.Multiply(x, x);
            var negBetaSquared = numOps.Negate(numOps.Multiply(betaT, xSquared));
            return numOps.Exp(negBetaSquared);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // d(SQRBF)/dx = -2βx * exp(-β * x²) = -2βx * SQRBF(x)
                var derivative = a.Value.Transform((x, idx) =>
                {
                    var xSquared = numOps.Multiply(x, x);
                    var negBetaSquared = numOps.Negate(numOps.Multiply(betaT, xSquared));
                    var activation = numOps.Exp(negBetaSquared);
                    var negTwoBeta = numOps.Negate(numOps.Multiply(numOps.FromDouble(2.0), betaT));
                    return numOps.Multiply(numOps.Multiply(negTwoBeta, x), activation);
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.SQRBF;
        node.OperationParams = new Dictionary<string, object> { { "Beta", beta } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Maxout activation function which takes maximum over groups of inputs.
    /// </summary>
    /// <param name="a">The input computation node (2D: batch × features).</param>
    /// <param name="numPieces">Number of inputs per group (default 2).</param>
    /// <returns>A new computation node with Maxout applied.</returns>
    /// <remarks>
    /// <para>
    /// Maxout groups consecutive features and outputs the maximum from each group.
    /// Input features must be divisible by numPieces.
    /// Output shape: [batch, features / numPieces].
    /// </para>
    /// <para><b>Gradient:</b> Flows only to the maximum element in each group (sparse gradient).</para>
    /// </remarks>
    public static ComputationNode<T> Maxout(ComputationNode<T> a, int numPieces = 2)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (shape.Length != 2)
            throw new ArgumentException($"Maxout requires 2D input [batch, features], got {shape.Length}D");

        int batchSize = shape[0];
        int features = shape[1];

        if (features % numPieces != 0)
            throw new ArgumentException($"Features ({features}) must be divisible by numPieces ({numPieces})");

        int outputFeatures = features / numPieces;
        var resultShape = new int[] { batchSize, outputFeatures };
        var result = new Tensor<T>(resultShape);
        var maxIndices = new int[batchSize, outputFeatures]; // Track which input was max

        // Forward: find max in each group
        for (int b = 0; b < batchSize; b++)
        {
            for (int g = 0; g < outputFeatures; g++)
            {
                int startIdx = g * numPieces;
                var maxVal = a.Value[b, startIdx];
                int maxIdx = 0;

                for (int p = 1; p < numPieces; p++)
                {
                    var val = a.Value[b, startIdx + p];
                    if (numOps.GreaterThan(val, maxVal))
                    {
                        maxVal = val;
                        maxIdx = p;
                    }
                }

                result[b, g] = maxVal;
                maxIndices[b, g] = startIdx + maxIdx;
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                // Gradient flows only to max elements
                var gradA = new Tensor<T>(shape);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int g = 0; g < outputFeatures; g++)
                    {
                        int maxIdx = maxIndices[b, g];
                        gradA[b, maxIdx] = numOps.Add(gradA[b, maxIdx], gradient[b, g]);
                    }
                }

                if (a.Gradient == null)
                {
                    a.Gradient = gradA;
                }
                else
                {
                    a.Gradient = a.Gradient.Add(gradA);
                }
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.Maxout;
        node.OperationParams = new Dictionary<string, object> { { "NumPieces", numPieces } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Randomized Leaky ReLU (RReLU) activation function.
    /// </summary>
    /// <param name="a">The input computation node.</param>
    /// <param name="lower">Lower bound for alpha (default 1/8).</param>
    /// <param name="upper">Upper bound for alpha (default 1/3).</param>
    /// <param name="isTraining">If true, samples random alpha; if false, uses average (default false for JIT).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A new computation node with RReLU applied.</returns>
    /// <remarks>
    /// <para>
    /// RReLU(x) = x if x >= 0, alpha * x otherwise.
    /// During training, alpha is sampled uniformly from [lower, upper].
    /// During inference (JIT default), alpha = (lower + upper) / 2.
    /// </para>
    /// <para><b>Gradient:</b> 1 for x >= 0, alpha for x &lt; 0.</para>
    /// </remarks>
    public static ComputationNode<T> RReLU(ComputationNode<T> a, double lower = 0.125, double upper = 0.333, bool isTraining = false, int? seed = null)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // For JIT, we use a fixed alpha (inference mode) or sample once per forward pass
        double alpha;
        if (isTraining)
        {
            var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
            alpha = lower + rng.NextDouble() * (upper - lower);
        }
        else
        {
            alpha = (lower + upper) / 2.0;
        }

        var alphaT = numOps.FromDouble(alpha);

        // Forward pass
        var result = a.Value.Transform((x, _) =>
        {
            if (numOps.GreaterThanOrEquals(x, numOps.Zero))
                return x;
            else
                return numOps.Multiply(alphaT, x);
        });

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var derivative = a.Value.Transform((x, _) =>
                {
                    if (numOps.GreaterThanOrEquals(x, numOps.Zero))
                        return numOps.One;
                    else
                        return alphaT;
                });
                var gradA = engine.TensorMultiply(gradient, derivative);
                var existingGrad = a.Gradient;
                a.Gradient = existingGrad == null ? gradA : engine.TensorAdd(existingGrad, gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.RReLU;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Lower", lower },
            { "Upper", upper },
            { "Alpha", alpha },
            { "IsTraining", isTraining }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Spherical Softmax activation function.
    /// </summary>
    /// <param name="a">The input computation node (2D: batch × features).</param>
    /// <param name="axis">Axis along which to apply (default -1, last axis).</param>
    /// <returns>A new computation node with SphericalSoftmax applied.</returns>
    /// <remarks>
    /// <para>
    /// SphericalSoftmax = softmax(x / ||x||₂)
    /// First L2-normalizes the input, then applies softmax.
    /// This improves numerical stability for inputs with varying magnitudes.
    /// </para>
    /// <para><b>Gradient:</b> Chain rule through L2 normalization and softmax.</para>
    /// </remarks>
    public static ComputationNode<T> SphericalSoftmax(ComputationNode<T> a, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (axis < 0)
            axis = shape.Length + axis;

        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {shape.Length} dimensions.");

        // Compute strides for N-dimensional iteration
        int axisSize = shape[axis];
        int outerSize = 1;
        int innerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++)
            innerSize *= shape[i];

        var result = new Tensor<T>(shape);
        var norms = new T[outerSize * innerSize];
        var normalized = new Tensor<T>(shape);

        // Iterate over all positions in the non-axis dimensions
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                int normIdx = outer * innerSize + inner;

                // Compute L2 norm
                var sumSquares = numOps.Zero;
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var val = a.Value[flatIdx];
                    sumSquares = numOps.Add(sumSquares, numOps.Multiply(val, val));
                }
                norms[normIdx] = numOps.Sqrt(sumSquares);

                // Prevent division by zero
                var norm = numOps.GreaterThan(norms[normIdx], numOps.FromDouble(1e-12))
                    ? norms[normIdx]
                    : numOps.FromDouble(1e-12);

                // L2 normalize
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    normalized[flatIdx] = numOps.Divide(a.Value[flatIdx], norm);
                }

                // Apply softmax to normalized values
                var maxVal = normalized[outer * axisSize * innerSize + inner];
                for (int i = 1; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var val = normalized[flatIdx];
                    if (numOps.GreaterThan(val, maxVal))
                        maxVal = val;
                }

                var expSum = numOps.Zero;
                var expValues = new T[axisSize];
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var shifted = numOps.Subtract(normalized[flatIdx], maxVal);
                    expValues[i] = numOps.Exp(shifted);
                    expSum = numOps.Add(expSum, expValues[i]);
                }

                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    result[flatIdx] = numOps.Divide(expValues[i], expSum);
                }
            }
        }

        // Capture values for backward pass
        int capturedAxis = axis;
        int capturedAxisSize = axisSize;
        int capturedOuterSize = outerSize;
        int capturedInnerSize = innerSize;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(shape);

                for (int outer = 0; outer < capturedOuterSize; outer++)
                {
                    for (int inner = 0; inner < capturedInnerSize; inner++)
                    {
                        int normIdx = outer * capturedInnerSize + inner;
                        var norm = numOps.GreaterThan(norms[normIdx], numOps.FromDouble(1e-12))
                            ? norms[normIdx]
                            : numOps.FromDouble(1e-12);
                        var normCubed = numOps.Multiply(norm, numOps.Multiply(norm, norm));

                        // Softmax Jacobian-vector product
                        var dotProduct = numOps.Zero;
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            dotProduct = numOps.Add(dotProduct, numOps.Multiply(gradient[flatIdx], result[flatIdx]));
                        }

                        // Gradient through softmax
                        var softmaxGrad = new T[capturedAxisSize];
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            softmaxGrad[i] = numOps.Multiply(result[flatIdx],
                                numOps.Subtract(gradient[flatIdx], dotProduct));
                        }

                        // Gradient through L2 normalization
                        var dotNorm = numOps.Zero;
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            dotNorm = numOps.Add(dotNorm,
                                numOps.Multiply(softmaxGrad[i], a.Value[flatIdx]));
                        }

                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            var term1 = numOps.Divide(softmaxGrad[i], norm);
                            var term2 = numOps.Divide(
                                numOps.Multiply(a.Value[flatIdx], dotNorm),
                                normCubed);
                            gradA[flatIdx] = numOps.Subtract(term1, term2);
                        }
                    }
                }

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.SphericalSoftmax;
        node.OperationParams = new Dictionary<string, object> { { "Axis", capturedAxis } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Taylor Softmax activation function using Taylor series approximation.
    /// </summary>
    /// <param name="a">The input computation node (2D: batch × features).</param>
    /// <param name="order">Order of Taylor series expansion (default 2).</param>
    /// <param name="axis">Axis along which to apply (default -1, last axis).</param>
    /// <returns>A new computation node with TaylorSoftmax applied.</returns>
    /// <remarks>
    /// <para>
    /// TaylorSoftmax uses Taylor series approximation of exp(x):
    /// exp(x) ≈ 1 + x + x²/2! + x³/3! + ... + xⁿ/n!
    /// Then normalizes like standard softmax.
    /// More computationally efficient than standard softmax for some hardware.
    /// </para>
    /// <para><b>Gradient:</b> Similar to softmax but using polynomial derivatives.</para>
    /// </remarks>
    public static ComputationNode<T> TaylorSoftmax(ComputationNode<T> a, int order = 2, int axis = -1)
    {
        if (order < 1)
            throw new ArgumentOutOfRangeException(nameof(order), order, "Order must be at least 1.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (axis < 0)
            axis = shape.Length + axis;

        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {shape.Length} dimensions.");

        // Compute strides for N-dimensional iteration
        int axisSize = shape[axis];
        int outerSize = 1;
        int innerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++)
            innerSize *= shape[i];

        var result = new Tensor<T>(shape);
        var taylorExpValues = new Tensor<T>(shape);

        // Precompute factorials
        var factorials = new double[order + 1];
        factorials[0] = 1;
        for (int i = 1; i <= order; i++)
            factorials[i] = factorials[i - 1] * i;

        // Iterate over all positions in the non-axis dimensions
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Compute Taylor approximation of exp for each position along axis
                var expSum = numOps.Zero;
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var x = a.Value.GetFlat(flatIdx);
                    var taylorExp = numOps.One; // Start with 1
                    var xPower = numOps.One;

                    for (int n = 1; n <= order; n++)
                    {
                        xPower = numOps.Multiply(xPower, x);
                        var term = numOps.Divide(xPower, numOps.FromDouble(factorials[n]));
                        taylorExp = numOps.Add(taylorExp, term);
                    }

                    // Ensure non-negative (Taylor can go negative for large negative x)
                    taylorExp = numOps.GreaterThan(taylorExp, numOps.Zero)
                        ? taylorExp
                        : numOps.FromDouble(1e-10);

                    taylorExpValues.SetFlat(flatIdx, taylorExp);
                    expSum = numOps.Add(expSum, taylorExp);
                }

                // Normalize
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    result.SetFlat(flatIdx, numOps.Divide(taylorExpValues.GetFlat(flatIdx), expSum));
                }
            }
        }

        // Capture values for backward pass
        int capturedAxis = axis;
        int capturedOrder = order;
        int capturedAxisSize = axisSize;
        int capturedOuterSize = outerSize;
        int capturedInnerSize = innerSize;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(shape);

                for (int outer = 0; outer < capturedOuterSize; outer++)
                {
                    for (int inner = 0; inner < capturedInnerSize; inner++)
                    {
                        // Compute sum for normalization denominator
                        var expSum = numOps.Zero;
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            expSum = numOps.Add(expSum, taylorExpValues.GetFlat(flatIdx));
                        }

                        // Softmax-style Jacobian: s_i * (δ_ij - s_j)
                        var dotProduct = numOps.Zero;
                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            dotProduct = numOps.Add(dotProduct,
                                numOps.Multiply(gradient.GetFlat(flatIdx), result.GetFlat(flatIdx)));
                        }

                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            // Softmax gradient part: s_i * (grad_i - dot(grad, s))
                            var softmaxGrad = numOps.Multiply(result.GetFlat(flatIdx),
                                numOps.Subtract(gradient.GetFlat(flatIdx), dotProduct));

                            // Taylor exp derivative: d/dx[1 + x + x²/2! + ... + x^n/n!] = 1 + x + ... + x^(n-1)/(n-1)!
                            // This is Taylor_{n-1}(x) for exp
                            var x = a.Value.GetFlat(flatIdx);
                            var taylorExpDeriv = numOps.One;
                            var xPower = numOps.One;
                            for (int n = 1; n < capturedOrder; n++)
                            {
                                xPower = numOps.Multiply(xPower, x);
                                var term = numOps.Divide(xPower, numOps.FromDouble(factorials[n]));
                                taylorExpDeriv = numOps.Add(taylorExpDeriv, term);
                            }

                            // For y_i = g(x_i) / sum_j(g(x_j)), the chain rule requires:
                            // grad_x_i = softmaxGrad * g'(x_i) / g(x_i)
                            // where g is the Taylor approximation of exp
                            var gVal = taylorExpValues.GetFlat(flatIdx);
                            var gPrimeOverG = numOps.Divide(taylorExpDeriv, gVal);
                            gradA.SetFlat(flatIdx, numOps.Multiply(softmaxGrad, gPrimeOverG));
                        }
                    }
                }

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.TaylorSoftmax;
        node.OperationParams = new Dictionary<string, object> { { "Order", capturedOrder }, { "Axis", capturedAxis } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Sparsemax activation function which projects onto the probability simplex.
    /// </summary>
    /// <param name="a">The input computation node (2D: batch × features).</param>
    /// <param name="axis">Axis along which to apply (default -1, last axis).</param>
    /// <returns>A new computation node with Sparsemax applied.</returns>
    /// <remarks>
    /// <para>
    /// Sparsemax produces sparse probability distributions where some outputs are exactly zero.
    /// Unlike softmax which always gives positive probabilities to all classes, sparsemax
    /// can assign exactly zero to low-scoring classes.
    /// </para>
    /// <para><b>Gradient:</b> For support set S (non-zero outputs): grad = upstream - mean(upstream[S])</para>
    /// </remarks>
    public static ComputationNode<T> Sparsemax(ComputationNode<T> a, int axis = -1)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = a.Value.Shape;

        if (axis < 0)
            axis = shape.Length + axis;

        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {shape.Length} dimensions.");

        // Compute strides for N-dimensional iteration
        int axisSize = shape[axis];
        int outerSize = 1;
        int innerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++)
            innerSize *= shape[i];

        var result = new Tensor<T>(shape);
        var supportMasks = new bool[outerSize * innerSize * axisSize]; // Track support set for backward

        // Iterate over all positions in the non-axis dimensions
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Extract values along axis and sort indices by value (descending)
                var indexed = new List<(T value, int idx)>();
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    indexed.Add((a.Value[flatIdx], i));
                }

                // Sort by value descending
                indexed.Sort((x, y) =>
                {
                    if (numOps.GreaterThan(x.value, y.value)) return -1;
                    if (numOps.LessThan(x.value, y.value)) return 1;
                    return 0;
                });

                // Find k (support size) and threshold tau using standard sparsemax algorithm
                // Standard algorithm: find k* = max{k : 1 + k * z_k > sum_{j<=k} z_j}
                // Then tau = (sum_{j<=k*} z_j - 1) / k*
                var cumSum = numOps.Zero;
                int k = 0;
                var tau = numOps.Zero;

                for (int i = 0; i < axisSize; i++)
                {
                    cumSum = numOps.Add(cumSum, indexed[i].value);
                    int kCandidate = i + 1;

                    // t_k = 1 + k * z_k - sum_{j<=k} z_j
                    var t = numOps.Subtract(
                        numOps.Add(
                            numOps.One,
                            numOps.Multiply(numOps.FromDouble(kCandidate), indexed[i].value)),
                        cumSum);

                    if (numOps.GreaterThan(t, numOps.Zero))
                    {
                        k = kCandidate;
                        tau = numOps.Divide(
                            numOps.Subtract(cumSum, numOps.One),
                            numOps.FromDouble(k));
                    }
                }

                // Compute output and support mask
                for (int i = 0; i < axisSize; i++)
                {
                    int flatIdx = outer * axisSize * innerSize + i * innerSize + inner;
                    var diff = numOps.Subtract(a.Value[flatIdx], tau);
                    if (numOps.GreaterThan(diff, numOps.Zero))
                    {
                        result[flatIdx] = diff;
                        supportMasks[flatIdx] = true;
                    }
                    else
                    {
                        result[flatIdx] = numOps.Zero;
                        supportMasks[flatIdx] = false;
                    }
                }
            }
        }

        // Capture values for backward pass
        int capturedAxis = axis;
        int capturedAxisSize = axisSize;
        int capturedOuterSize = outerSize;
        int capturedInnerSize = innerSize;

        void BackwardFunction(Tensor<T> gradient)
        {
            if (a.RequiresGradient)
            {
                var gradA = new Tensor<T>(shape);

                for (int outer = 0; outer < capturedOuterSize; outer++)
                {
                    for (int inner = 0; inner < capturedInnerSize; inner++)
                    {
                        // Count support size and compute mean of gradients on support
                        int supportSize = 0;
                        var gradSum = numOps.Zero;

                        for (int i = 0; i < capturedAxisSize; i++)
                        {
                            int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                            if (supportMasks[flatIdx])
                            {
                                supportSize++;
                                gradSum = numOps.Add(gradSum, gradient[flatIdx]);
                            }
                        }

                        // Compute gradient: for support elements, subtract mean
                        if (supportSize > 0)
                        {
                            var gradMean = numOps.Divide(gradSum, numOps.FromDouble(supportSize));

                            for (int i = 0; i < capturedAxisSize; i++)
                            {
                                int flatIdx = outer * capturedAxisSize * capturedInnerSize + i * capturedInnerSize + inner;
                                gradA[flatIdx] = supportMasks[flatIdx]
                                    ? numOps.Subtract(gradient[flatIdx], gradMean)
                                    : numOps.Zero;
                            }
                        }
                    }
                }

                var existingGrad = a.Gradient;

                a.Gradient = existingGrad == null ? gradA : existingGrad.Add(gradA);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: a.RequiresGradient,
            parents: new List<ComputationNode<T>> { a },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.Sparsemax;
        node.OperationParams = new Dictionary<string, object> { { "Axis", capturedAxis } };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Applies the Hierarchical Softmax activation function for efficient large-vocabulary classification.
    /// </summary>
    /// <param name="input">The input computation node (2D: batch × inputDim).</param>
    /// <param name="nodeWeights">The tree node weights (2D: treeDepth × inputDim).</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <returns>A new computation node with HierarchicalSoftmax applied.</returns>
    /// <remarks>
    /// <para>
    /// Hierarchical Softmax organizes classes in a binary tree structure.
    /// Each node makes a binary decision using sigmoid, and the final probability
    /// is the product of probabilities along the path to each class.
    /// </para>
    /// <para>
    /// Computational complexity is O(log N) instead of O(N) for standard softmax.
    /// </para>
    /// <para><b>Gradient:</b> Flows through sigmoid derivatives at each tree node.</para>
    /// </remarks>
    public static ComputationNode<T> HierarchicalSoftmax(
        ComputationNode<T> input,
        ComputationNode<T> nodeWeights,
        int numClasses)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Value.Shape;
        var weightsShape = nodeWeights.Value.Shape;

        if (inputShape.Length != 2)
            throw new ArgumentException($"Input must be 2D [batch, inputDim], got {inputShape.Length}D");

        if (weightsShape.Length != 2)
            throw new ArgumentException($"NodeWeights must be 2D [treeDepth, inputDim], got {weightsShape.Length}D");

        int batchSize = inputShape[0];
        int inputDim = inputShape[1];
        int treeDepth = weightsShape[0];

        if (weightsShape[1] != inputDim)
            throw new ArgumentException($"NodeWeights inputDim ({weightsShape[1]}) must match input inputDim ({inputDim})");

        var resultShape = new int[] { batchSize, numClasses };
        var result = new Tensor<T>(resultShape);

        // Store intermediate sigmoid outputs for backward pass
        var sigmoidOutputs = new T[batchSize, treeDepth];
        var pathDirections = new bool[numClasses, treeDepth]; // Pre-compute paths

        // Pre-compute path directions for each class
        for (int c = 0; c < numClasses; c++)
        {
            for (int d = 0; d < treeDepth; d++)
            {
                pathDirections[c, d] = (c & (1 << (treeDepth - d - 1))) != 0;
            }
        }

        // Forward pass: compute class probabilities
        for (int b = 0; b < batchSize; b++)
        {
            // Compute sigmoid at each depth
            for (int d = 0; d < treeDepth; d++)
            {
                var dotProduct = numOps.Zero;
                for (int i = 0; i < inputDim; i++)
                {
                    dotProduct = numOps.Add(dotProduct,
                        numOps.Multiply(input.Value[b, i], nodeWeights.Value[d, i]));
                }
                // Sigmoid: 1 / (1 + exp(-x))
                var negDot = numOps.Negate(dotProduct);
                var expNegDot = numOps.Exp(negDot);
                sigmoidOutputs[b, d] = numOps.Divide(numOps.One, numOps.Add(numOps.One, expNegDot));
            }

            // Compute probability for each class
            for (int c = 0; c < numClasses; c++)
            {
                var prob = numOps.One;
                for (int d = 0; d < treeDepth; d++)
                {
                    var sigOut = sigmoidOutputs[b, d];
                    if (pathDirections[c, d])
                    {
                        prob = numOps.Multiply(prob, sigOut);
                    }
                    else
                    {
                        prob = numOps.Multiply(prob, numOps.Subtract(numOps.One, sigOut));
                    }

                    // Early termination if probability becomes negligible
                    if (numOps.LessThan(prob, numOps.FromDouble(1e-10)))
                    {
                        prob = numOps.FromDouble(1e-10);
                        break;
                    }
                }
                result[b, c] = prob;
            }
        }

        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient w.r.t. input
            if (input.RequiresGradient)
            {
                var gradInput = new Tensor<T>(inputShape);

                for (int b = 0; b < batchSize; b++)
                {
                    for (int c = 0; c < numClasses; c++)
                    {
                        var classGrad = gradient[b, c];

                        // Gradient flows through each node in the path
                        for (int d = 0; d < treeDepth; d++)
                        {
                            var sigOut = sigmoidOutputs[b, d];
                            var sigDeriv = numOps.Multiply(sigOut, numOps.Subtract(numOps.One, sigOut));

                            // Compute the probability contribution excluding this node
                            var otherProb = numOps.One;
                            for (int d2 = 0; d2 < treeDepth; d2++)
                            {
                                if (d2 != d)
                                {
                                    var sig = sigmoidOutputs[b, d2];
                                    otherProb = numOps.Multiply(otherProb,
                                        pathDirections[c, d2] ? sig : numOps.Subtract(numOps.One, sig));
                                }
                            }

                            // Gradient factor depends on path direction
                            var factor = pathDirections[c, d]
                                ? numOps.Multiply(classGrad, numOps.Multiply(sigDeriv, otherProb))
                                : numOps.Negate(numOps.Multiply(classGrad, numOps.Multiply(sigDeriv, otherProb)));

                            // Accumulate gradient w.r.t. input
                            for (int i = 0; i < inputDim; i++)
                            {
                                gradInput[b, i] = numOps.Add(gradInput[b, i],
                                    numOps.Multiply(factor, nodeWeights.Value[d, i]));
                            }
                        }
                    }
                }

                if (input.Gradient == null)
                {
                    input.Gradient = gradInput;
                }
                else
                {
                    input.Gradient = input.Gradient.Add(gradInput);
                }
            }

            // Gradient w.r.t. weights
            if (nodeWeights.RequiresGradient)
            {
                var gradWeights = new Tensor<T>(weightsShape);

                for (int b = 0; b < batchSize; b++)
                {
                    for (int c = 0; c < numClasses; c++)
                    {
                        var classGrad = gradient[b, c];

                        for (int d = 0; d < treeDepth; d++)
                        {
                            var sigOut = sigmoidOutputs[b, d];
                            var sigDeriv = numOps.Multiply(sigOut, numOps.Subtract(numOps.One, sigOut));

                            var otherProb = numOps.One;
                            for (int d2 = 0; d2 < treeDepth; d2++)
                            {
                                if (d2 != d)
                                {
                                    var sig = sigmoidOutputs[b, d2];
                                    otherProb = numOps.Multiply(otherProb,
                                        pathDirections[c, d2] ? sig : numOps.Subtract(numOps.One, sig));
                                }
                            }

                            var factor = pathDirections[c, d]
                                ? numOps.Multiply(classGrad, numOps.Multiply(sigDeriv, otherProb))
                                : numOps.Negate(numOps.Multiply(classGrad, numOps.Multiply(sigDeriv, otherProb)));

                            // Accumulate gradient w.r.t. weights
                            for (int i = 0; i < inputDim; i++)
                            {
                                gradWeights[d, i] = numOps.Add(gradWeights[d, i],
                                    numOps.Multiply(factor, input.Value[b, i]));
                            }
                        }
                    }
                }

                if (nodeWeights.Gradient == null)
                {
                    nodeWeights.Gradient = gradWeights;
                }
                else
                {
                    nodeWeights.Gradient = nodeWeights.Gradient.Add(gradWeights);
                }
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient || nodeWeights.RequiresGradient,
            parents: new List<ComputationNode<T>> { input, nodeWeights },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.HierarchicalSoftmax;
        node.OperationParams = new Dictionary<string, object>
        {
            { "NumClasses", numClasses },
            { "TreeDepth", treeDepth }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    // ============================================================================
    // Differentiable Approximation Operations
    // These operations enable JIT compilation for traditionally non-differentiable
    // models like decision trees, KNN, and locally-weighted regression.
    // ============================================================================

    /// <summary>
    /// Performs a soft split operation for differentiable decision trees.
    /// </summary>
    /// <param name="input">The input features tensor.</param>
    /// <param name="leftValue">The value to return if going left.</param>
    /// <param name="rightValue">The value to return if going right.</param>
    /// <param name="featureIndex">The index of the feature to split on.</param>
    /// <param name="threshold">The threshold value for the split.</param>
    /// <param name="temperature">Temperature parameter controlling split sharpness (default: 1.0).</param>
    /// <returns>A weighted combination of left and right values based on soft split.</returns>
    /// <remarks>
    /// <para>
    /// Computes: p_left = σ((threshold - x[featureIndex]) / temperature)
    ///           output = p_left * leftValue + (1 - p_left) * rightValue
    /// </para>
    /// <para><b>For Beginners:</b> This makes decision tree splits differentiable by using
    /// a smooth sigmoid function instead of a hard if-then-else. Lower temperature makes
    /// the split sharper (more like a hard decision), while higher temperature makes it softer.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SoftSplit(
        ComputationNode<T> input,
        ComputationNode<T> leftValue,
        ComputationNode<T> rightValue,
        int featureIndex,
        T threshold,
        T? temperature = default)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var temp = temperature ?? numOps.FromDouble(1.0);

        // Extract the feature value at featureIndex
        // Compute p_left = σ((threshold - x[featureIndex]) / temperature)
        var inputData = input.Value.ToVector();
        var featureValue = featureIndex < inputData.Length ? inputData[featureIndex] : numOps.Zero;
        var diff = numOps.Subtract(threshold, featureValue);
        var scaled = numOps.Divide(diff, temp);
        var pLeft = numOps.Divide(numOps.One, numOps.Add(numOps.One, numOps.Exp(numOps.Negate(scaled))));
        var pRight = numOps.Subtract(numOps.One, pLeft);

        // output = p_left * leftValue + p_right * rightValue
        var leftScaled = leftValue.Value.Transform((x, _) => numOps.Multiply(x, pLeft));
        var rightScaled = rightValue.Value.Transform((x, _) => numOps.Multiply(x, pRight));
        var result = leftScaled.Add(rightScaled);

        // Store values needed for backward pass
        var storedPLeft = pLeft;
        var storedDiff = diff;
        var storedTemp = temp;

        void BackwardFunction(Tensor<T> gradient)
        {
            // ∂output/∂leftValue = p_left
            // ∂output/∂rightValue = (1 - p_left) = p_right
            // ∂output/∂input[featureIndex] = (rightValue - leftValue) * p_left * (1 - p_left) / temperature

            if (leftValue.RequiresGradient)
            {
                var gradLeft = gradient.Transform((g, _) => numOps.Multiply(g, storedPLeft));
                if (leftValue.Gradient == null)
                    leftValue.Gradient = gradLeft;
                else
                    leftValue.Gradient = leftValue.Gradient.Add(gradLeft);
            }

            if (rightValue.RequiresGradient)
            {
                var pR = numOps.Subtract(numOps.One, storedPLeft);
                var gradRight = gradient.Transform((g, _) => numOps.Multiply(g, pR));
                if (rightValue.Gradient == null)
                    rightValue.Gradient = gradRight;
                else
                    rightValue.Gradient = rightValue.Gradient.Add(gradRight);
            }

            if (input.RequiresGradient)
            {
                // Gradient w.r.t. input feature
                // ∂σ(z)/∂z = σ(z) * (1 - σ(z)) where z = (threshold - x[feature]) / temp
                // ∂z/∂x[feature] = -1/temp
                // ∂output/∂x[feature] = (rightValue - leftValue) * σ(z) * (1 - σ(z)) * (-1/temp)
                var pR = numOps.Subtract(numOps.One, storedPLeft);
                var sigmoidGrad = numOps.Multiply(storedPLeft, pR);
                var tempFactor = numOps.Negate(numOps.Divide(numOps.One, storedTemp));

                var valueDiff = rightValue.Value.Subtract(leftValue.Value);
                var gradScale = numOps.Multiply(sigmoidGrad, tempFactor);

                // Sum over output dimensions to get scalar gradient for the feature
                var gradSum = numOps.Zero;
                for (int i = 0; i < gradient.Length && i < valueDiff.Length; i++)
                {
                    gradSum = numOps.Add(gradSum, numOps.Multiply(gradient.GetFlatIndexValue(i),
                        numOps.Multiply(valueDiff.GetFlatIndexValue(i), gradScale)));
                }

                // Create gradient tensor with gradient at featureIndex
                var inputGrad = new T[input.Value.Length];
                for (int i = 0; i < inputGrad.Length; i++)
                    inputGrad[i] = numOps.Zero;
                if (featureIndex < inputGrad.Length)
                    inputGrad[featureIndex] = gradSum;

                var gradInput = new Tensor<T>(input.Value.Shape, new Vector<T>(inputGrad));
                if (input.Gradient == null)
                    input.Gradient = gradInput;
                else
                    input.Gradient = input.Gradient.Add(gradInput);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient || leftValue.RequiresGradient || rightValue.RequiresGradient,
            parents: new List<ComputationNode<T>> { input, leftValue, rightValue },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.SoftSplit;
        node.OperationParams = new Dictionary<string, object>
        {
            { "FeatureIndex", featureIndex },
            { "Threshold", threshold! },
            { "Temperature", temp! }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Performs a soft K-Nearest Neighbors operation for differentiable instance-based learning.
    /// </summary>
    /// <param name="input">The query input tensor.</param>
    /// <param name="supportVectors">Matrix of support vectors (training points) [n_samples, n_features].</param>
    /// <param name="labels">Labels for each support vector [n_samples] or [n_samples, n_outputs].</param>
    /// <param name="temperature">Temperature for softmax attention (default: 1.0).</param>
    /// <returns>Attention-weighted sum of labels.</returns>
    /// <remarks>
    /// <para>
    /// Computes: distances[i] = ||input - supportVectors[i]||²
    ///           weights = softmax(-distances / temperature)
    ///           output = Σ weights[i] * labels[i]
    /// </para>
    /// <para><b>For Beginners:</b> Instead of finding exactly k nearest neighbors, this
    /// computes attention weights for ALL neighbors based on distance. Closer neighbors
    /// get higher attention. This makes KNN differentiable and JIT-compilable.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SoftKNN(
        ComputationNode<T> input,
        ComputationNode<T> supportVectors,
        ComputationNode<T> labels,
        T? temperature = default)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var temp = temperature ?? numOps.FromDouble(1.0);

        var inputData = input.Value.ToVector();
        var svData = supportVectors.Value.ToVector();
        var labelData = labels.Value.ToVector();

        // Determine number of support vectors and features
        var svShape = supportVectors.Value.Shape;
        var nSamples = svShape.Length > 0 ? svShape[0] : svData.Length;
        var nFeatures = svShape.Length > 1 ? svShape[1] : 1;

        // Compute squared distances to each support vector
        var distances = new T[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            var dist = numOps.Zero;
            for (int j = 0; j < nFeatures && j < inputData.Length; j++)
            {
                var svIdx = i * nFeatures + j;
                if (svIdx < svData.Length)
                {
                    var diff = numOps.Subtract(inputData[j], svData[svIdx]);
                    dist = numOps.Add(dist, numOps.Multiply(diff, diff));
                }
            }
            distances[i] = dist;
        }

        // Compute softmax attention weights: softmax(-distances / temperature)
        var scaledDists = distances.Select(d => numOps.Negate(numOps.Divide(d, temp))).ToArray();
        var maxScaled = scaledDists.Aggregate(scaledDists[0], (a, b) => numOps.GreaterThan(a, b) ? a : b);
        var expDists = scaledDists.Select(d => numOps.Exp(numOps.Subtract(d, maxScaled))).ToArray();
        var sumExp = expDists.Aggregate(numOps.Zero, (a, b) => numOps.Add(a, b));
        var weights = expDists.Select(e => numOps.Divide(e, sumExp)).ToArray();

        // Compute weighted sum of labels
        var outputSize = labelData.Length / nSamples;
        if (outputSize < 1) outputSize = 1;
        var output = new T[outputSize];
        for (int i = 0; i < outputSize; i++)
            output[i] = numOps.Zero;

        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                var labelIdx = i * outputSize + j;
                if (labelIdx < labelData.Length)
                {
                    output[j] = numOps.Add(output[j], numOps.Multiply(weights[i], labelData[labelIdx]));
                }
            }
        }

        var resultTensor = new Tensor<T>(new[] { outputSize }, new Vector<T>(output));

        // Store for backward pass
        var storedWeights = weights;
        var storedDistances = distances;
        var storedNSamples = nSamples;
        var storedNFeatures = nFeatures;
        var storedOutputSize = outputSize;

        void BackwardFunction(Tensor<T> gradient)
        {
            // Gradient computation for SoftKNN
            // This is complex - involves gradients through softmax and distance computation

            if (labels.RequiresGradient)
            {
                // ∂output/∂labels[i] = weights[i]
                var gradLabels = new T[labelData.Length];
                for (int i = 0; i < storedNSamples; i++)
                {
                    for (int j = 0; j < storedOutputSize; j++)
                    {
                        var idx = i * storedOutputSize + j;
                        if (idx < gradLabels.Length && j < gradient.Length)
                        {
                            gradLabels[idx] = numOps.Multiply(storedWeights[i], gradient.GetFlatIndexValue(j));
                        }
                    }
                }
                var gradLabelsTensor = new Tensor<T>(labels.Value.Shape, new Vector<T>(gradLabels));
                if (labels.Gradient == null)
                    labels.Gradient = gradLabelsTensor;
                else
                    labels.Gradient = labels.Gradient.Add(gradLabelsTensor);
            }

            if (input.RequiresGradient)
            {
                // ∂output/∂input involves softmax Jacobian and distance gradients
                // Simplified: gradient flows through distance computation
                var gradInput = new T[inputData.Length];
                for (int i = 0; i < gradInput.Length; i++)
                    gradInput[i] = numOps.Zero;

                // For each output dimension and each support vector
                for (int j = 0; j < storedOutputSize && j < gradient.Length; j++)
                {
                    for (int i = 0; i < storedNSamples; i++)
                    {
                        // Softmax Jacobian contribution
                        var labelIdx = i * storedOutputSize + j;
                        var labelVal = labelIdx < labelData.Length ? labelData[labelIdx] : numOps.Zero;

                        for (int i2 = 0; i2 < storedNSamples; i2++)
                        {
                            var labelIdx2 = i2 * storedOutputSize + j;
                            var labelVal2 = labelIdx2 < labelData.Length ? labelData[labelIdx2] : numOps.Zero;

                            var jacobian = i == i2
                                ? numOps.Multiply(storedWeights[i], numOps.Subtract(numOps.One, storedWeights[i]))
                                : numOps.Negate(numOps.Multiply(storedWeights[i], storedWeights[i2]));

                            // Distance gradient: ∂dist/∂input = 2 * (input - sv)
                            for (int f = 0; f < storedNFeatures && f < gradInput.Length; f++)
                            {
                                var svIdx = i2 * storedNFeatures + f;
                                var svVal = svIdx < svData.Length ? svData[svIdx] : numOps.Zero;
                                var inputVal = f < inputData.Length ? inputData[f] : numOps.Zero;
                                var distGrad = numOps.Multiply(numOps.FromDouble(2.0), numOps.Subtract(inputVal, svVal));

                                var scaleFactor = numOps.Negate(numOps.Divide(numOps.One, temp));
                                var contrib = numOps.Multiply(gradient.GetFlatIndexValue(j),
                                    numOps.Multiply(labelVal2,
                                        numOps.Multiply(jacobian,
                                            numOps.Multiply(scaleFactor, distGrad))));
                                gradInput[f] = numOps.Add(gradInput[f], contrib);
                            }
                        }
                    }
                }

                var gradInputTensor = new Tensor<T>(input.Value.Shape, new Vector<T>(gradInput));
                if (input.Gradient == null)
                    input.Gradient = gradInputTensor;
                else
                    input.Gradient = input.Gradient.Add(gradInputTensor);
            }
        }

        var node = new ComputationNode<T>(
            value: resultTensor,
            requiresGradient: input.RequiresGradient || labels.RequiresGradient,
            parents: new List<ComputationNode<T>> { input, supportVectors, labels },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.SoftKNN;
        node.OperationParams = new Dictionary<string, object>
        {
            { "Temperature", temp! }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Performs soft locally-weighted regression for differentiable instance-based learning.
    /// </summary>
    /// <param name="input">The query input tensor.</param>
    /// <param name="xTrain">Training feature matrix [n_samples, n_features].</param>
    /// <param name="yTrain">Training target values [n_samples] or [n_samples, n_outputs].</param>
    /// <param name="bandwidth">Bandwidth parameter controlling locality (default: 1.0).</param>
    /// <returns>Attention-weighted prediction.</returns>
    /// <remarks>
    /// <para>
    /// Computes: distances[i] = ||input - xTrain[i]||²
    ///           weights = softmax(-distances / bandwidth)
    ///           output = Σ weights[i] * yTrain[i]
    /// </para>
    /// <para><b>For Beginners:</b> This is similar to SoftKNN but specifically designed for
    /// regression with a bandwidth parameter that controls how local the weighting is.
    /// Smaller bandwidth = more local predictions.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SoftLocallyWeighted(
        ComputationNode<T> input,
        ComputationNode<T> xTrain,
        ComputationNode<T> yTrain,
        T? bandwidth = default)
    {
        // This is essentially the same as SoftKNN with bandwidth instead of temperature
        return SoftKNN(input, xTrain, yTrain, bandwidth);
    }

    /// <summary>
    /// Performs fake quantization with Straight-Through Estimator (STE) for differentiable quantization.
    /// </summary>
    /// <param name="input">The input tensor to quantize.</param>
    /// <param name="numBits">Number of quantization bits (default: 8).</param>
    /// <param name="scale">Scale factor (if null, computed from input range).</param>
    /// <param name="zeroPoint">Zero point for asymmetric quantization (default: 0).</param>
    /// <param name="symmetric">Whether to use symmetric quantization (default: true).</param>
    /// <returns>Fake-quantized tensor (quantized forward, STE backward).</returns>
    /// <remarks>
    /// <para>
    /// Forward: output = round(input / scale) * scale (clipped to valid range)
    /// Backward: gradient passes through unchanged (Straight-Through Estimator)
    /// </para>
    /// <para><b>For Beginners:</b> This simulates quantization during training while allowing
    /// gradients to flow back for optimization. The forward pass applies real quantization,
    /// but the backward pass pretends it didn't happen - this trick (STE) lets us train
    /// models that will be quantized for deployment.
    /// </para>
    /// </remarks>
    public static ComputationNode<T> FakeQuantize(
        ComputationNode<T> input,
        int numBits = 8,
        T? scale = default,
        T? zeroPoint = default,
        bool symmetric = true)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.Value.ToVector();

        // Compute quantization parameters
        var qMin = symmetric ? numOps.FromDouble(-(1 << (numBits - 1))) : numOps.Zero;
        var qMax = symmetric ? numOps.FromDouble((1 << (numBits - 1)) - 1) : numOps.FromDouble((1 << numBits) - 1);

        // Compute scale from data if not provided
        T actualScale;
        if (scale != null && !numOps.Equals(scale, numOps.Zero))
        {
            actualScale = scale;
        }
        else
        {
            // Find min/max of input
            var minVal = inputData.Aggregate(inputData[0], (a, b) => numOps.LessThan(a, b) ? a : b);
            var maxVal = inputData.Aggregate(inputData[0], (a, b) => numOps.GreaterThan(a, b) ? a : b);

            if (symmetric)
            {
                var absMax = numOps.GreaterThan(numOps.Abs(minVal), numOps.Abs(maxVal))
                    ? numOps.Abs(minVal) : numOps.Abs(maxVal);
                actualScale = numOps.Divide(absMax, qMax);
            }
            else
            {
                actualScale = numOps.Divide(numOps.Subtract(maxVal, minVal),
                    numOps.Subtract(qMax, qMin));
            }

            // Avoid division by zero
            if (numOps.Equals(actualScale, numOps.Zero))
                actualScale = numOps.One;
        }

        var actualZeroPoint = zeroPoint ?? numOps.Zero;

        // Apply fake quantization
        var outputData = new T[inputData.Length];
        for (int i = 0; i < inputData.Length; i++)
        {
            // Quantize: q = round(x / scale) + zeroPoint
            var scaled = numOps.Divide(inputData[i], actualScale);
            var rounded = numOps.FromDouble(Math.Round(numOps.ToDouble(scaled)));
            var shifted = numOps.Add(rounded, actualZeroPoint);

            // Clamp to valid range
            if (numOps.LessThan(shifted, qMin)) shifted = qMin;
            if (numOps.GreaterThan(shifted, qMax)) shifted = qMax;

            // Dequantize: x' = (q - zeroPoint) * scale
            var unshifted = numOps.Subtract(shifted, actualZeroPoint);
            outputData[i] = numOps.Multiply(unshifted, actualScale);
        }

        var result = new Tensor<T>(input.Value.Shape, new Vector<T>(outputData));

        void BackwardFunction(Tensor<T> gradient)
        {
            // Straight-Through Estimator: gradient passes through unchanged
            if (input.RequiresGradient)
            {
                if (input.Gradient == null)
                    input.Gradient = gradient;
                else
                    input.Gradient = input.Gradient.Add(gradient);
            }
        }

        var node = new ComputationNode<T>(
            value: result,
            requiresGradient: input.RequiresGradient,
            parents: new List<ComputationNode<T>> { input },
            backwardFunction: BackwardFunction,
            name: null);

        node.OperationType = OperationType.FakeQuantization;
        node.OperationParams = new Dictionary<string, object>
        {
            { "NumBits", numBits },
            { "Scale", actualScale! },
            { "ZeroPoint", actualZeroPoint! },
            { "Symmetric", symmetric }
        };

        var tape = GradientTape<T>.Current;
        if (tape != null && tape.IsRecording)
            tape.RecordOperation(node);
        return node;
    }

    /// <summary>
    /// Performs broadcasting addition of two tensors with different shapes.
    /// </summary>
    private static Tensor<T> BroadcastAdd(Tensor<T> a, Tensor<T> b, INumericOperations<T> numOps)
    {
        // Determine which tensor is smaller and needs broadcasting
        return (a.Rank < b.Rank || (a.Rank == b.Rank && a.Length < b.Length))
            ? BroadcastAddHelper(b, a, numOps)
            : BroadcastAddHelper(a, b, numOps);
    }

    /// <summary>
    /// Helper method that broadcasts the smaller tensor to match the larger one.
    /// </summary>
    private static Tensor<T> BroadcastAddHelper(Tensor<T> larger, Tensor<T> smaller, INumericOperations<T> numOps)
    {
        var result = larger.Clone();

        // Handle simple case: adding a 1D tensor to a 2D tensor along the last dimension
        // E.g., [batchSize, features] + [features]
        if (larger.Rank == 2 && smaller.Rank == 1 && larger.Shape[1] == smaller.Shape[0])
        {
            int batchSize = larger.Shape[0];
            int features = larger.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    result[b, f] = numOps.Add(larger[b, f], smaller[f]);
                }
            }
        }
        else
        {
            // General broadcasting: add element-wise where shapes match
            throw new NotSupportedException($"Broadcasting from shape [{string.Join(", ", smaller.Shape)}] to [{string.Join(", ", larger.Shape)}] is not yet implemented for this shape combination.");
        }

        return result;
    }

    /// <summary>
    /// Performs broadcasting multiplication of two tensors with different shapes.
    /// </summary>
    private static Tensor<T> BroadcastMultiply(Tensor<T> a, Tensor<T> b, INumericOperations<T> numOps)
    {
        // Scalar broadcast: [1] * X or X * [1]
        if (a.Length == 1)
        {
            return b.Multiply(a[0]);
        }

        if (b.Length == 1)
        {
            return a.Multiply(b[0]);
        }

        // Handle common case: [batchSize, features] * [features]
        if (a.Rank == 2 && b.Rank == 1 && a.Shape[1] == b.Shape[0])
        {
            int batchSize = a.Shape[0];
            int features = a.Shape[1];
            var result = new Tensor<T>(a.Shape);

            for (int batch = 0; batch < batchSize; batch++)
            {
                for (int f = 0; f < features; f++)
                {
                    result[batch, f] = numOps.Multiply(a[batch, f], b[f]);
                }
            }

            return result;
        }

        // Handle common case: [features] * [batchSize, features]
        if (b.Rank == 2 && a.Rank == 1 && b.Shape[1] == a.Shape[0])
        {
            int batchSize = b.Shape[0];
            int features = b.Shape[1];
            var result = new Tensor<T>(b.Shape);

            for (int batch = 0; batch < batchSize; batch++)
            {
                for (int f = 0; f < features; f++)
                {
                    result[batch, f] = numOps.Multiply(a[f], b[batch, f]);
                }
            }

            return result;
        }

        throw new NotSupportedException(
            $"Broadcasting multiplication from shape [{string.Join(", ", a.Shape)}] and [{string.Join(", ", b.Shape)}] is not yet implemented for this shape combination.");
    }

    /// <summary>
    /// Reduces gradient to match the original shape by summing across broadcasted dimensions.
    /// </summary>
    private static Tensor<T> ReduceGradient(Tensor<T> gradient, int[] originalShape)
    {
        // If shapes already match, no reduction needed
        if (gradient.Shape.SequenceEqual(originalShape))
        {
            return gradient;
        }

        // Scalar reduction: broadcasted scalar receives the sum of all gradient contributions.
        if (originalShape.Length == 1 && originalShape[0] == 1)
        {
            return gradient.Sum();
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // Handle simple case: reducing [batchSize, features] to [features]
        if (gradient.Rank == 2 && originalShape.Length == 1 && gradient.Shape[1] == originalShape[0])
        {
            int batchSize = gradient.Shape[0];
            int features = gradient.Shape[1];
            var result = new Tensor<T>(originalShape);

            for (int f = 0; f < features; f++)
            {
                T sum = numOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    sum = numOps.Add(sum, gradient[b, f]);
                }
                result[f] = sum;
            }

            return result;
        }
        else
        {
            // General reduction: sum across dimensions that were broadcasted
            throw new NotSupportedException($"Gradient reduction from shape [{string.Join(", ", gradient.Shape)}] to [{string.Join(", ", originalShape)}] is not yet implemented for this shape combination.");
        }
    }
}


