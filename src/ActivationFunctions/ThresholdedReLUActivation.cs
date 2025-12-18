using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Thresholded ReLU activation function, a variant of the standard ReLU function with an adjustable threshold.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Thresholded ReLU (Rectified Linear Unit) is a variation of the standard ReLU activation function.
/// 
/// While a standard ReLU outputs the input value when it's positive and zero when it's negative (f(x) = max(0, x)),
/// the Thresholded ReLU adds an additional parameter called "theta" (?) that acts as a threshold.
/// 
/// The Thresholded ReLU only activates (returns the input value) when the input exceeds this threshold.
/// Otherwise, it returns zero. The formula is:
/// 
/// f(x) = x if x > ?, otherwise f(x) = 0
/// 
/// This allows the neural network to ignore small positive activations that might be noise, potentially
/// creating more robust models. By adjusting the threshold value, you can control how sensitive the
/// activation function is to input signals.
/// </para>
/// </remarks>
public class ThresholdedReLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The threshold value that determines when the activation function returns the input value versus zero.
    /// </summary>
    /// <remarks>
    /// Inputs greater than this threshold will pass through unchanged; inputs less than or equal to this threshold will be set to zero.
    /// </remarks>
    private T _theta;

    /// <summary>
    /// Initializes a new instance of the ThresholdedReLUActivation class with the specified threshold value.
    /// </summary>
    /// <param name="theta">The threshold value. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The theta parameter controls how "picky" the activation function is.
    /// 
    /// - With a higher theta value (e.g., 2.0), the function will only activate for stronger input signals,
    ///   ignoring weaker ones.
    /// - With a lower theta value (e.g., 0.1), the function will activate for most positive inputs,
    ///   behaving more like a standard ReLU.
    /// 
    /// The default value of 1.0 provides a moderate threshold that works well for many applications.
    /// You can adjust this value during experimentation to see what works best for your specific problem.
    /// </para>
    /// </remarks>
    public ThresholdedReLUActivation(double theta = 1.0)
    {
        _theta = NumOps.FromDouble(theta);
    }

    /// <summary>
    /// Indicates whether this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as Thresholded ReLU can operate on individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Thresholded ReLU activation function to an input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The input value if it's greater than the threshold, otherwise zero.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method implements the core functionality of the Thresholded ReLU:
    /// 
    /// - If the input is greater than the threshold (theta), it returns the input unchanged
    /// - If the input is less than or equal to the threshold, it returns zero
    /// 
    /// For example, with the default threshold of 1.0:
    /// - An input of 2.5 would return 2.5
    /// - An input of 0.8 would return 0
    /// - An input of -3.0 would return 0
    /// 
    /// This creates a "step" in the activation function at the threshold value.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // ThresholdedReLU: x if x > theta, else 0
        return NumOps.GreaterThan(input, _theta) ? input : NumOps.Zero;
    }

    /// <summary>
    /// Calculates the derivative of the Thresholded ReLU function for a given input.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>1 if the input is greater than the threshold, otherwise 0.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how the output changes when the input changes slightly.
    /// This is crucial for training neural networks through backpropagation.
    /// 
    /// For Thresholded ReLU:
    /// - If the input is greater than the threshold, the derivative is 1, meaning the function passes
    ///   changes in the input directly to the output.
    /// - If the input is less than or equal to the threshold, the derivative is 0, meaning small changes
    ///   in the input have no effect on the output.
    /// 
    /// Note that technically, the derivative is undefined exactly at the threshold point, but for
    /// practical purposes in neural networks, we define it as 0 at that point.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // Derivative of ThresholdedReLU:
        // 1 if x > theta
        // 0 otherwise
        return NumOps.GreaterThan(input, _theta) ? NumOps.One : NumOps.Zero;
    }

    /// <summary>
    /// Updates the threshold value used by the activation function.
    /// </summary>
    /// <param name="newTheta">The new threshold value to use.</param>
    /// <remarks>
    /// <para>
    /// This method allows you to change the threshold parameter after the activation function has been created.
    /// This can be useful for experimentation or for implementing adaptive activation functions that change
    /// their behavior during training.
    /// </para>
    /// </remarks>
    public void UpdateTheta(T newTheta)
    {
        _theta = newTheta;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.ThresholdedReLU provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// ThresholdedReLU supports JIT compilation with full gradient computation.
    /// The backward pass correctly computes gradients: 1 for inputs above threshold, 0 otherwise.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with ThresholdedReLU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.ThresholdedReLU(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double theta = Convert.ToDouble(_theta);
        return TensorOperations<T>.ThresholdedReLU(input, theta);
    }
}
