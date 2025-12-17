using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Squared Radial Basis Function (SQRBF) activation function.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SQRBF activation function is defined as f(x) = exp(-ß * x²), where ß is a parameter that controls
/// the width of the Gaussian bell curve. This function outputs values between 0 and 1, with the maximum value
/// of 1 occurring when the input is 0, and values approaching 0 as the input moves away from 0 in either direction.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Squared Radial Basis Function (SQRBF) is an activation function that produces a bell-shaped curve.
/// Unlike functions like ReLU or Sigmoid that are used in standard neural networks, SQRBF is commonly used in 
/// Radial Basis Function Networks (RBFNs).
/// 
/// Think of SQRBF like a "proximity detector" - it gives its highest output (1.0) when the input is exactly 0,
/// and progressively smaller outputs as the input moves away from 0 in either direction (positive or negative).
/// The ß parameter controls how quickly the output drops off as you move away from 0:
/// - A larger ß makes the bell curve narrower (drops off quickly)
/// - A smaller ß makes the bell curve wider (drops off slowly)
/// 
/// This is useful in machine learning when you want to measure how close an input is to a specific reference point.
/// </para>
/// </remarks>
public class SQRBFActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The width parameter that controls the shape of the Gaussian bell curve.
    /// </summary>
    /// <remarks>
    /// A larger value of beta results in a narrower bell curve, while a smaller value results in a wider bell curve.
    /// </remarks>
    private readonly T _beta;

    /// <summary>
    /// Initializes a new instance of the <see cref="SQRBFActivation{T}"/> class with the specified beta parameter.
    /// </summary>
    /// <param name="beta">The width parameter for the Gaussian bell curve. Default value is 1.0.</param>
    /// <remarks>
    /// <para>
    /// The beta parameter controls the width of the Gaussian bell curve. A larger beta value results in a narrower curve,
    /// while a smaller beta value results in a wider curve.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of beta as a "sensitivity knob" for the function:
    /// - Higher beta values (e.g., 5.0) make the function more sensitive to changes in input, creating a narrow peak
    /// - Lower beta values (e.g., 0.1) make the function less sensitive, creating a wide, gentle curve
    /// 
    /// Choosing the right beta value depends on your specific application and the range of your input data.
    /// </para>
    /// </remarks>
    public SQRBFActivation(double beta = 1.0)
    {
        _beta = NumOps.FromDouble(beta);
    }

    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as SQRBF can be applied to individual scalar values.</returns>
    /// <remarks>
    /// Unlike some activation functions that require vector inputs, SQRBF can be applied independently to each value.
    /// </remarks>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the SQRBF activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The result of applying the SQRBF function to the input.</returns>
    /// <remarks>
    /// <para>
    /// The SQRBF function is calculated as f(x) = exp(-ß * x²), where ß is the width parameter.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes an input value and returns a value between 0 and 1:
    /// - When the input is 0, the output is exactly 1
    /// - As the input moves away from 0 (in either direction), the output gets closer to 0
    /// - The rate at which the output decreases is controlled by the beta parameter
    /// 
    /// For example, with the default beta=1.0:
    /// - Input of 0 gives output of 1.0
    /// - Input of 1 gives output of about 0.368
    /// - Input of 2 gives output of about 0.018
    /// - Input of -1 gives output of about 0.368 (it's symmetric around 0)
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // f(x) = exp(-ß * x^2)
        T square = NumOps.Multiply(input, input);
        T negBetaSquare = NumOps.Negate(NumOps.Multiply(_beta, square));

        return NumOps.Exp(negBetaSquare);
    }

    /// <summary>
    /// Calculates the derivative of the SQRBF function for a given input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative of the SQRBF function at the input value.</returns>
    /// <remarks>
    /// <para>
    /// The derivative of the SQRBF function is calculated as f'(x) = -2ßx * exp(-ß * x²).
    /// This derivative is used during the backpropagation step of neural network training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how the output of the SQRBF function changes when we make a small change to the input.
    /// 
    /// Some key properties of this derivative:
    /// - At x=0, the derivative is 0 (the function is at its peak and momentarily flat)
    /// - For positive inputs, the derivative is negative (the function is decreasing)
    /// - For negative inputs, the derivative is positive (the function is increasing)
    /// - The derivative approaches 0 as the input gets very large or very small
    /// 
    /// During neural network training, this derivative helps determine how to adjust the weights to minimize errors.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // f'(x) = -2ßx * exp(-ß * x^2)
        T activationValue = Activate(input);
        T negTwoBeta = NumOps.Negate(NumOps.Multiply(NumOps.FromDouble(2), _beta));

        return NumOps.Multiply(NumOps.Multiply(negTwoBeta, input), activationValue);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.SQRBF provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// SQRBF supports JIT compilation with full gradient computation.
    /// The backward pass correctly computes gradients using the derivative: -2βx * exp(-β * x²).
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with SQRBF activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.SQRBF(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double beta = Convert.ToDouble(_beta);
        return TensorOperations<T>.SQRBF(input, beta);
    }
}
