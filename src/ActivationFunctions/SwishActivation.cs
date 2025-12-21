using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Swish activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Swish is a newer activation function developed by researchers at Google.
/// Its formula is x * sigmoid(x), which means it multiplies the input by the sigmoid of the input.
/// 
/// Key characteristics of Swish include:
/// - It's smooth everywhere, unlike ReLU which has a sharp corner at x=0
/// - It allows some negative values through, which can help with learning
/// - It behaves somewhat like ReLU for positive values, but has a smoother transition
/// - It has been shown to outperform ReLU in some deep neural networks
/// 
/// Swish combines some of the best properties of ReLU and sigmoid functions,
/// making it effective for many deep learning applications.
/// </para>
/// </remarks>
public class SwishActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True, as Swish can be applied to individual scalar values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns true because Swish can process one number at a time.
    /// Unlike functions like Softmax that need to look at all values together,
    /// Swish transforms each value independently.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Swish activation function to a scalar input value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>x * sigmoid(x)</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Swish function multiplies the input by the sigmoid of the input.
    /// The sigmoid function transforms values to be between 0 and 1, so Swish:
    /// - For large positive values: returns approximately the value itself (like ReLU)
    /// - For large negative values: returns values close to zero
    /// - For values near zero: has a smooth curve that allows some negative values
    /// 
    /// This behavior helps neural networks learn more complex patterns while avoiding
    /// some of the training problems that can occur with other activation functions.
    /// </para>
    /// </remarks>
    public override T Activate(T x)
    {
        return NumOps.Multiply(x, Sigmoid(x));
    }

    /// <summary>
    /// Applies the Swish activation function to each element of an input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector with the Swish activation applied to each element.</returns>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => Activate(x));
    }

    /// <summary>
    /// Calculates the derivative of the Swish activation function for a scalar input value.
    /// </summary>
    /// <param name="x">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative of Swish at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative of Swish is more complex than many other activation functions.
    /// It combines the sigmoid function with its derivative in a specific way.
    /// 
    /// The formula is: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    /// 
    /// This derivative is used during backpropagation to determine how to adjust the weights
    /// of the neural network based on the error of the predictions.
    /// </para>
    /// </remarks>
    public override T Derivative(T x)
    {
        T sigX = Sigmoid(x);
        return NumOps.Add(
            sigX,
            NumOps.Multiply(
                x,
                NumOps.Multiply(
                    sigX,
                    NumOps.Subtract(NumOps.One, sigX)
                )
            )
        );
    }

    /// <summary>
    /// Calculates the derivative of the Swish activation function for each element of an input vector.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A diagonal matrix containing the derivatives for each input element.</returns>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            // Set diagonal elements to the derivative at that point
            jacobian[i, i] = Derivative(input[i]);

            // Off-diagonal elements are zero for element-wise activation functions
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    jacobian[i, j] = NumOps.Zero;
                }
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Calculates the sigmoid function for a scalar value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The sigmoid of x, a value between 0 and 1.</returns>
    private T Sigmoid(T x)
    {
        return NumOps.Divide(
            NumOps.One,
            NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(x)))
        );
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.Swish.</value>
    /// <remarks>
    /// <para>
    /// Swish supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The operation uses IEngine for GPU acceleration
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Swish activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the Swish activation to TensorOperations&lt;T&gt;.Swish(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// Swish (also known as SiLU) is used in EfficientNet and other modern architectures.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Swish(input);
    }
}
