using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Identity activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Identity activation function is the simplest activation function - it returns exactly what you give it.
/// 
/// When you pass a value through this function:
/// - Input of 2 ? Output of 2
/// - Input of -3.5 ? Output of -3.5
/// - And so on...
/// 
/// Think of it like a straight line on a graph where y = x.
/// 
/// While most neural networks use non-linear activation functions (like ReLU or Sigmoid) to model complex patterns,
/// the Identity function can be useful in:
/// - The output layer of regression problems (when predicting continuous values)
/// - Testing or debugging neural networks
/// - Certain network architectures where you want values to pass through unchanged
/// </para>
/// </remarks>
public class IdentityActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Applies the Identity activation function to a single value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The same value that was input (unchanged).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method simply returns the exact same value you give it.
    /// </remarks>
    public override T Activate(T input)
    {
        return input;
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Identity function for a single value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>Always returns 1, as the derivative of the Identity function is constant.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// 
    /// For the Identity function (f(x) = x), the derivative is always 1, meaning:
    /// - When you increase the input by a small amount
    /// - The output increases by exactly the same amount
    /// 
    /// This constant derivative of 1 means the function has a constant slope (it's a straight line).
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        return NumOps.One;
    }

    /// <summary>
    /// Applies the Identity activation function to a vector of values.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>The same vector that was input (unchanged).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method works on a collection of values at once (a vector),
    /// returning each value unchanged.
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input;
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Identity function for a vector of values.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A diagonal matrix with all 1s on the diagonal.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how the output vector changes when we slightly change each input value.
    /// 
    /// For the Identity function, this creates a special matrix called a "diagonal matrix" where:
    /// - All values on the main diagonal (top-left to bottom-right) are 1
    /// - All other values are 0
    /// 
    /// This matrix structure indicates that each output is affected only by its corresponding input,
    /// and the relationship is one-to-one (a change of x in the input causes a change of x in the output).
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        return Vector<T>.CreateDefault(input.Length, NumOps.One).ToDiagonalMatrix();
    }

    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as the Identity function can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because Identity activation requires no computation and is trivially differentiable.</value>
    /// <remarks>
    /// <para>
    /// Identity supports JIT compilation because:
    /// - It's a no-op (returns input unchanged)
    /// - The gradient is constant (always 1)
    /// - It can be represented as a static computation graph node
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>The same computation node (Identity is a no-op).</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method returns the input node unchanged, as Identity activation does nothing.
    /// No TensorOperations call is needed.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // Identity is a no-op, just return the input
        return input;
    }
}
