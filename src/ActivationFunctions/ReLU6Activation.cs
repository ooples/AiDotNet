using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the ReLU6 (Rectified Linear Unit capped at 6) activation function.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ReLU6 is a variant of the ReLU activation function that clips the output at 6.
/// It is defined as: f(x) = min(max(0, x), 6)
/// </para>
/// <para>
/// <b>For Beginners:</b> ReLU6 works like regular ReLU but adds an upper limit of 6.
///
/// While ReLU allows any positive value to pass through unchanged, ReLU6 caps the output at 6:
/// - Negative inputs become 0 (same as ReLU)
/// - Values between 0 and 6 pass through unchanged
/// - Values above 6 are capped at 6
///
/// This is particularly useful in mobile neural networks (like MobileNet) because:
/// - It prevents activations from becoming too large, improving numerical stability
/// - It works well with low-precision arithmetic (like 8-bit integers) on mobile devices
/// - It helps the network learn more robust features by limiting extreme activations
/// </para>
/// </remarks>
public class ReLU6Activation<T> : ActivationFunctionBase<T>
{
    private readonly T _six;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReLU6Activation{T}"/> class.
    /// </summary>
    public ReLU6Activation()
    {
        _six = NumOps.FromDouble(6.0);
    }

    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as ReLU6 supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the ReLU6 activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>min(max(0, input), 6)</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a single number and:
    /// - Returns 0 if it's negative
    /// - Returns the number unchanged if it's between 0 and 6
    /// - Returns 6 if it's greater than 6
    /// </remarks>
    public override T Activate(T input)
    {
        return MathHelper.Min(_six, MathHelper.Max(NumOps.Zero, input));
    }

    /// <summary>
    /// Calculates the derivative of the ReLU6 function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>1 if 0 &lt; input &lt; 6, otherwise 0.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// For ReLU6:
    /// - The derivative is 1 for inputs between 0 and 6 (output changes at the same rate as input)
    /// - The derivative is 0 for inputs outside this range (output doesn't change)
    /// </remarks>
    public override T Derivative(T input)
    {
        bool inRange = NumOps.GreaterThan(input, NumOps.Zero) && NumOps.LessThan(input, _six);
        return inRange ? NumOps.One : NumOps.Zero;
    }

    /// <summary>
    /// Applies the ReLU6 activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The input vector to activate.</param>
    /// <returns>A new vector with the ReLU6 function applied to each element.</returns>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => MathHelper.Min(_six, MathHelper.Max(NumOps.Zero, x)));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the ReLU6 function for a vector input.
    /// </summary>
    /// <param name="input">The input vector to calculate the derivative for.</param>
    /// <returns>A diagonal matrix where each diagonal element is the derivative of ReLU6 for the corresponding input element.</returns>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            bool inRange = NumOps.GreaterThan(input[i], NumOps.Zero) && NumOps.LessThan(input[i], _six);
            jacobian[i, i] = inRange ? NumOps.One : NumOps.Zero;
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the ReLU6 activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>A new tensor with the ReLU6 function applied to each element.</returns>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        return input.Transform((x, _) => MathHelper.Min(_six, MathHelper.Max(NumOps.Zero, x)));
    }

    /// <summary>
    /// Calculates the derivative of the ReLU6 function for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor to calculate the derivative for.</param>
    /// <returns>A new tensor containing the derivatives of the ReLU6 function for each input element.</returns>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        return input.Transform((x, _) =>
        {
            bool inRange = NumOps.GreaterThan(x, NumOps.Zero) && NumOps.LessThan(x, _six);
            return inRange ? NumOps.One : NumOps.Zero;
        });
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>False because TensorOperations.Minimum is not yet implemented.</value>
    /// <remarks>
    /// ReLU6 would be implemented as min(max(0, x), 6) using the ReLU and Minimum operations.
    /// Currently disabled until TensorOperations.Minimum is available.
    /// </remarks>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with ReLU6 activation applied.</returns>
    /// <exception cref="NotSupportedException">Always thrown as JIT compilation is not supported.</exception>
    /// <remarks>
    /// <para>
    /// ReLU6 would be implemented as min(max(0, x), 6) = min(ReLU(x), 6).
    /// Currently not supported until TensorOperations.Minimum is implemented.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        throw new NotSupportedException(
            "ReLU6Activation does not support JIT compilation. Use the standard Activate method instead.");
    }
}
