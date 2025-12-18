using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Inverse Square Root Unit (ISRU) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The ISRU (Inverse Square Root Unit) activation function is designed to:
/// 
/// - Keep the output values bounded (they never grow too large)
/// - Preserve the sign of the input (positive inputs give positive outputs, negative inputs give negative outputs)
/// - Allow gradients to flow more easily during training compared to some other functions
/// 
/// The function looks like an "S" shape that's been stretched horizontally, similar to tanh but with 
/// different mathematical properties. It approaches +1 for large positive inputs and -1 for large negative inputs,
/// but never quite reaches these values.
/// 
/// The a (alpha) parameter controls how quickly the function "saturates" (flattens out):
/// - Smaller a values make the function change more gradually
/// - Larger a values make the function change more abruptly
/// 
/// ISRU is useful in neural networks where you want bounded outputs but need to avoid the vanishing 
/// gradient problem that affects some other activation functions.
/// </para>
/// </remarks>
public class ISRUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The alpha parameter that controls the shape of the ISRU function.
    /// </summary>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the ISRU activation function with the specified alpha parameter.
    /// </summary>
    /// <param name="alpha">
    /// Controls how quickly the function saturates (approaches its asymptotic values).
    /// Default value is 1.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The alpha parameter determines the "steepness" of the ISRU function:
    /// 
    /// - With alpha = 1.0 (default), you get the standard ISRU behavior
    /// - With smaller alpha values (e.g., 0.5), the function changes more gradually
    /// - With larger alpha values (e.g., 2.0), the function changes more abruptly
    /// 
    /// You can think of alpha as controlling how "aggressive" the function is in squashing values.
    /// In most cases, the default value works well, but you might adjust it if you find your
    /// neural network is learning too slowly or too erratically.
    /// </para>
    /// </remarks>
    public ISRUActivation(double alpha = 1.0)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as the ISRU function can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the ISRU activation function to a single value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The transformed value after applying the ISRU function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input value using the formula:
    /// 
    /// f(x) = x / sqrt(1 + a·x²)
    /// 
    /// This creates a smooth curve that:
    /// - For small inputs, behaves almost like the identity function (output ˜ input)
    /// - For large positive inputs, approaches but never exceeds +1
    /// - For large negative inputs, approaches but never exceeds -1
    /// 
    /// For example, with the default a = 1:
    /// - Input of 0 ? Output of 0
    /// - Input of 1 ? Output of about 0.707
    /// - Input of 10 ? Output of about 0.995
    /// - Input of -5 ? Output of about -0.981
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // f(x) = x / sqrt(1 + ax^2)
        T squaredInput = NumOps.Multiply(input, input);
        T alphaSquaredInput = NumOps.Multiply(_alpha, squaredInput);
        T denominator = NumOps.Sqrt(NumOps.Add(NumOps.One, alphaSquaredInput));

        return NumOps.Divide(input, denominator);
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the ISRU function for a single value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value at the specified input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// This information is crucial during neural network training.
    /// 
    /// For the ISRU function, the derivative is calculated using:
    /// 
    /// f'(x) = (1 + a·x²)^(-3/2)
    /// 
    /// Key properties of this derivative:
    /// - It's always positive (meaning the function always increases as input increases)
    /// - It's highest at x = 0 (equals 1.0)
    /// - It gradually decreases as |x| increases (approaches 0 for very large or very small inputs)
    /// 
    /// This behavior helps prevent the "vanishing gradient problem" that can occur with some
    /// other activation functions, making ISRU useful for deep neural networks.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // f'(x) = (1 + ax^2)^(-3/2)
        T squaredInput = NumOps.Multiply(input, input);
        T alphaSquaredInput = NumOps.Multiply(_alpha, squaredInput);
        T baseValue = NumOps.Add(NumOps.One, alphaSquaredInput);
        T exponent = NumOps.FromDouble(-1.5);

        return NumOps.Power(baseValue, exponent);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.ISRU provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// ISRU supports JIT compilation with full gradient computation.
    /// The backward pass correctly computes gradients: (1 + alpha * x²)^(-3/2).
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with ISRU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.ISRU(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double alpha = Convert.ToDouble(_alpha);
        return TensorOperations<T>.ISRU(input, alpha);
    }
}
