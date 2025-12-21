using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Hard Sigmoid activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Hard Sigmoid is a simplified version of the standard Sigmoid function.
/// 
/// While the regular Sigmoid creates an S-shaped curve that smoothly transitions from 0 to 1,
/// the Hard Sigmoid uses straight lines to approximate this curve, making it:
/// 
/// 1. Computationally faster (uses simple math operations instead of exponentials)
/// 2. Less smooth but still useful for many neural network applications
/// 
/// The function works like this:
/// - If input = -1: output = 0
/// - If input = 1: output = 1
/// - If -1 &lt; input &lt; 1: output = (input + 1) / 2
/// 
/// This creates a straight line between (-1, 0) and (1, 1), with values clamped to the range [0, 1].
/// 
/// Hard Sigmoid is often used in mobile or embedded applications where computational
/// efficiency is important, or in certain types of recurrent neural networks.
/// </para>
/// </remarks>
public class HardSigmoidActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as Hard Sigmoid can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Hard Sigmoid activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to transform.</param>
    /// <returns>The transformed value, constrained between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms any input number into a value between 0 and 1:
    /// 
    /// 1. It adds 1 to the input
    /// 2. It multiplies the result by 0.5
    /// 3. It ensures the final value is between 0 and 1
    /// 
    /// For example:
    /// - If input = -2, output = 0
    /// - If input = 0, output = 0.5
    /// - If input = 2, output = 1
    /// 
    /// This creates a function that "squashes" any input into the range [0, 1],
    /// which is useful for representing probabilities or normalized values.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // Hard Sigmoid: max(0, min(1, (x + 1) / 2))
        T shifted = NumOps.Add(input, NumOps.One);
        T scaled = NumOps.Multiply(shifted, NumOps.FromDouble(0.5));
        T clamped = MathHelper.Max(NumOps.Zero, MathHelper.Min(NumOps.One, scaled));

        return clamped;
    }

    /// <summary>
    /// Calculates the derivative of the Hard Sigmoid function for a given input value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value (0.5 if input is between -1 and 1, otherwise 0).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when the input changes slightly.
    /// 
    /// For Hard Sigmoid:
    /// - If input is between -1 and 1: derivative = 0.5 (the slope of the line)
    /// - Otherwise: derivative = 0 (flat regions have no change)
    /// 
    /// This information is crucial during neural network training, as it determines:
    /// - How much to adjust weights during backpropagation
    /// - Which neurons are actively learning (those with non-zero derivatives)
    /// 
    /// The Hard Sigmoid's derivative is simpler than regular Sigmoid's, making
    /// calculations faster during training.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // Derivative of Hard Sigmoid:
        // 0.5 if -1 < x < 1
        // 0 otherwise
        T minBound = NumOps.FromDouble(-1);
        T maxBound = NumOps.One;

        if (NumOps.GreaterThan(input, minBound) && NumOps.LessThan(input, maxBound))
        {
            return NumOps.FromDouble(0.5);
        }

        return NumOps.Zero;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because gradient computation is fully implemented in TensorOperations.HardSigmoid.</value>
    /// <remarks>
    /// <para>
    /// HardSigmoid supports JIT compilation because:
    /// - The gradient computation (backward pass) is fully implemented in TensorOperations
    /// - The gradient is 0.5 when -1 &lt; x &lt; 1, and 0 otherwise
    /// - It's computationally efficient and commonly used in mobile/embedded applications
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with HardSigmoid activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps the HardSigmoid activation to TensorOperations&lt;T&gt;.HardSigmoid(input),
    /// which handles both forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.HardSigmoid(input);
    }
}
