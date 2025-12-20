using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Parametric Rectified Linear Unit (PReLU) activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> PReLU is an improved version of the popular ReLU activation function.
/// 
/// While ReLU completely blocks negative values (turning them to zero), PReLU allows
/// a small portion of negative values to pass through, controlled by a parameter called "alpha".
/// 
/// How PReLU works:
/// - For positive inputs (x > 0): PReLU returns the input unchanged (just like ReLU)
/// - For negative inputs (x = 0): PReLU returns alpha * x (a scaled-down version of the input)
/// 
/// The alpha parameter is typically a small positive number (default 0.01). This "leakiness"
/// helps prevent a problem called "dying ReLU" where neurons can get stuck and stop learning.
/// 
/// PReLU can be thought of as a sloped line for negative inputs rather than a flat line at zero.
/// </para>
/// </remarks>
public class PReLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The scaling factor applied to negative inputs.
    /// </summary>
    private T _alpha;

    /// <summary>
    /// Initializes a new instance of the PReLU activation function with the specified alpha value.
    /// </summary>
    /// <param name="alpha">The scaling factor for negative inputs. Default is 0.01.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The alpha parameter determines how much of negative inputs should "leak through".
    /// 
    /// - A value of 0 makes PReLU behave exactly like regular ReLU (no leakage)
    /// - A small value like 0.01 (default) allows a small portion of negative values to pass
    /// - A value of 1 would allow negative values to pass unchanged
    /// 
    /// Smaller alpha values (0.01-0.1) are most common in practice.
    /// </para>
    /// </remarks>
    public PReLUActivation(double alpha = 0.01)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True, as PReLU can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the PReLU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value using the PReLU function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a single number using the PReLU formula:
    /// 
    /// - If the input is positive (> 0): the output is the same as the input
    /// - If the input is negative (= 0): the output is alpha * input
    /// 
    /// For example, with the default alpha = 0.01:
    /// - Input of 5 ? Output of 5
    /// - Input of -5 ? Output of -0.05 (5 * 0.01)
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // PReLU: max(0, x) + alpha * min(0, x)
        T positivepart = MathHelper.Max(NumOps.Zero, input);
        T negativepart = NumOps.Multiply(_alpha, MathHelper.Min(NumOps.Zero, input));

        return NumOps.Add(positivepart, negativepart);
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the PReLU function for a single input value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the PReLU function's output changes when we
    /// slightly change the input. This is essential for training neural networks.
    /// 
    /// For PReLU, the derivative is very simple:
    /// - If the input is positive (> 0): the derivative is 1
    /// - If the input is negative (= 0): the derivative is alpha
    /// 
    /// A derivative of 1 means the output changes at the same rate as the input.
    /// A derivative of alpha means the output changes at alpha times the rate of the input.
    /// 
    /// During training, this derivative helps determine how much to adjust each neuron's weights
    /// based on the errors the network makes.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // Derivative of PReLU:
        // 1 if x > 0
        // alpha if x <= 0
        if (NumOps.GreaterThan(input, NumOps.Zero))
        {
            return NumOps.One;
        }

        return _alpha;
    }

    /// <summary>
    /// Updates the alpha parameter of the PReLU function.
    /// </summary>
    /// <param name="newAlpha">The new alpha value to use.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows you to change the alpha parameter after creating the PReLU function.
    /// 
    /// This can be useful when:
    /// - You want to experiment with different alpha values to see which works best
    /// - You're implementing a learning algorithm that adjusts alpha automatically during training
    /// - You need to fine-tune the network's behavior for specific inputs
    /// </para>
    /// </remarks>
    public void UpdateAlpha(T newAlpha)
    {
        _alpha = newAlpha;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.PReLU provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// PReLU supports JIT compilation with full gradient computation.
    /// The backward pass correctly computes gradients: 1 for positive inputs, alpha for negative inputs.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with PReLU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.PReLU(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double alpha = Convert.ToDouble(_alpha);
        return TensorOperations<T>.PReLU(input, alpha);
    }
}
