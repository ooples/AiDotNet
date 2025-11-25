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
    /// <value>False because gradient computation is not yet implemented.</value>
    /// <remarks>
    /// <para>
    /// This activation does not yet support JIT compilation because the gradient
    /// computation (backward pass) has not been implemented in TensorOperations.HardSigmoid.
    /// </para>
    /// <para>
    /// To enable JIT support:
    /// 1. Implement the backward pass in TensorOperations.HardSigmoid
    /// 2. Test the gradient computation
    /// 3. Change SupportsJitCompilation to return true
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with HardSigmoid activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <exception cref="NotSupportedException">Thrown because gradient is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// This method would map the activation to TensorOperations&lt;T&gt;.HardSigmoid(input)
    /// once the gradient computation is implemented.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        throw new NotSupportedException(
            $"HardSigmoidActivation does not support JIT compilation yet. " +
            $"The gradient computation (backward pass) has not been implemented in TensorOperations.HardSigmoid. " +
            $"Once gradients are implemented, this activation can be used in JIT-compiled computation graphs.");
    }
}