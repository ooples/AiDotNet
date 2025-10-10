namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Continuously Differentiable Exponential Linear Unit (CELU) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The CELU activation function is an improved version of the popular ReLU function.
/// While ReLU simply turns negative values to zero (which can cause "dead neurons"), CELU replaces
/// negative values with a smooth exponential curve that approaches a negative limit.
/// 
/// Key benefits of CELU:
/// - For positive inputs, it behaves exactly like ReLU (returns the input value)
/// - For negative inputs, it returns a negative value that smoothly approaches -α
/// - This smooth transition helps prevent "dead neurons" during training
/// - The α parameter controls how quickly the function approaches its negative limit
/// 
/// CELU is particularly useful in deep neural networks where maintaining gradient flow
/// through all neurons is important for effective learning.
/// </para>
/// </remarks>
public class CELUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The alpha parameter that controls the negative saturation value of the function.
    /// </summary>
    private readonly T _alpha = default!;

    /// <summary>
    /// Initializes a new instance of the CELUActivation class with the specified alpha parameter.
    /// </summary>
    /// <param name="alpha">The alpha parameter that controls the negative saturation value. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The alpha parameter determines how steeply the function curves for negative inputs
    /// and what negative value it will approach as inputs become more negative.
    /// 
    /// - A larger alpha (e.g., 2.0) means the function can reach more negative values
    /// - A smaller alpha (e.g., 0.5) limits the function to less negative values
    /// 
    /// The default value of 1.0 works well for most applications, but you might adjust it if:
    /// - Your network is learning too slowly (try increasing alpha)
    /// - Your network is becoming unstable during training (try decreasing alpha)
    /// </para>
    /// </remarks>
    public CELUActivation(double alpha = 1.0)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Indicates that this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as CELU can be applied to scalar values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the CELU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value using the CELU function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms an input value using the formula:
    /// f(x) = max(0, x) + min(0, α * (exp(x/α) - 1))
    /// 
    /// In simpler terms:
    /// - For positive inputs (x ≥ 0): the output is just x (like ReLU)
    /// - For negative inputs (x &lt; 0): the output follows a smooth curve that approaches -α
    /// 
    /// This combination gives CELU the benefits of ReLU for positive values while avoiding
    /// the "dead neuron" problem for negative values.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // CELU: max(0, x) + min(0, α * (exp(x/α) - 1))
        T expTerm = NumOps.Subtract(NumOps.Exp(NumOps.Divide(input, _alpha)), NumOps.One);
        T negativepart = NumOps.Multiply(_alpha, expTerm);
        
        return NumOps.Add(
            MathHelper.Max(NumOps.Zero, input),
            MathHelper.Min(NumOps.Zero, negativepart)
        );
    }

    /// <summary>
    /// Calculates the derivative of the CELU function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the CELU function's output changes
    /// when its input changes slightly. This is used during neural network training to determine
    /// how to adjust weights.
    /// 
    /// The derivative of CELU has these properties:
    /// - For positive inputs (x ≥ 0): the derivative is 1 (constant slope)
    /// - For negative inputs (x &lt; 0): the derivative is exp(x/α) (gradually decreasing)
    /// 
    /// Unlike ReLU, the derivative is never exactly zero, which helps prevent neurons from
    /// becoming completely inactive ("dead") during training.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // Derivative of CELU:
        // 1 if x >= 0
        // exp(x/α) if x < 0
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return NumOps.One;
        }
        else
        {
            return NumOps.Exp(NumOps.Divide(input, _alpha));
        }
    }
}