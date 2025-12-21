using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Exponential Linear Unit (ELU) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ELU is an activation function that, like ReLU, returns the input directly for positive values.
/// For negative inputs, it returns a smooth curve that approaches -alpha as the input becomes more negative.
/// 
/// Key advantages of ELU include:
/// - It helps prevent "dying neurons" (a problem with ReLU) by allowing negative values
/// - It has a smooth curve for negative inputs, which can help with gradient-based learning
/// - The parameter alpha controls how negative the curve can go
/// - It centers the activations closer to zero, which can speed up learning
/// 
/// ELU is often used in deep neural networks where ReLU might cause training issues.
/// </para>
/// </remarks>
public class ELUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The alpha parameter that controls the saturation value for negative inputs.
    /// </summary>
    private readonly T _alpha;

    /// <summary>
    /// Gets the alpha parameter that controls the saturation value for negative inputs.
    /// </summary>
    public T Alpha => _alpha;

    /// <summary>
    /// Initializes a new instance of the ELUActivation class.
    /// </summary>
    /// <param name="alpha">The alpha parameter that controls the negative saturation value. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The alpha parameter controls how negative the function can get.
    /// A larger alpha means that the function can reach more negative values for very negative inputs.
    /// The default value of 1.0 works well for most applications.
    /// </para>
    /// </remarks>
    public ELUActivation(double alpha = 1.0)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True, as ELU can be applied to individual scalar values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns true because ELU can process one number at a time.
    /// Unlike functions like Softmax that need to look at all values together,
    /// ELU transforms each value independently.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the ELU activation function to a scalar input value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>x if x > 0, otherwise alpha * (e^x - 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When given a positive number, ELU simply returns that number unchanged.
    /// When given a negative number, it returns a value on a smooth curve that approaches -alpha as 
    /// the input becomes more negative. This helps the neural network learn from negative signals
    /// while avoiding the "dying neuron" problem that can happen with ReLU.
    /// </para>
    /// </remarks>
    public override T Activate(T x)
    {
        if (NumOps.GreaterThan(x, NumOps.Zero))
        {
            return x;
        }
        else
        {
            return NumOps.Multiply(_alpha, NumOps.Subtract(NumOps.Exp(x), NumOps.One));
        }
    }

    /// <summary>
    /// Applies the ELU activation function to each element of an input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector with the ELU activation applied to each element.</returns>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => Activate(x));
    }

    /// <summary>
    /// Calculates the derivative of the ELU activation function for a scalar input value.
    /// </summary>
    /// <param name="x">The input value at which to calculate the derivative.</param>
    /// <returns>1 if x > 0, otherwise ELU(x, alpha) + alpha.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the ELU function changes when its input changes slightly.
    /// For positive inputs, the derivative is 1, meaning the function increases at the same rate as the input.
    /// For negative inputs, the derivative is related to the function's value at that point.
    /// 
    /// This information is essential for backpropagation during neural network training,
    /// as it helps determine how to adjust the weights to improve the network's performance.
    /// </para>
    /// </remarks>
    public override T Derivative(T x)
    {
        if (NumOps.GreaterThan(x, NumOps.Zero))
        {
            return NumOps.One;
        }
        else
        {
            // ELU(x, alpha) + alpha
            return NumOps.Add(Activate(x), _alpha);
        }
    }

    /// <summary>
    /// Calculates the derivative of the ELU activation function for each element of an input vector.
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
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.ELU.</value>
    /// <remarks>
    /// <para>
    /// ELU supports JIT compilation because:
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
    /// <returns>A new computation node with ELU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the ELU activation to TensorOperations&lt;T&gt;.ELU(input, alpha),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // Convert alpha to double for TensorOperations
        double alphaDouble = Convert.ToDouble(_alpha);
        return TensorOperations<T>.ELU(input, alphaDouble);
    }
}
