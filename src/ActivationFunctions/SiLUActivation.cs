namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the SiLU (Sigmoid Linear Unit) activation function, also known as Swish.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SiLU function is defined as f(x) = x * sigmoid(x), where sigmoid(x) = 1/(1+e^(-x)).
/// It was introduced in 2017 and has shown strong performance in deep neural networks.
/// </para>
/// <para>
/// For Beginners: SiLU (or Swish) is a relatively new activation function that has become 
/// popular in modern neural networks. Unlike simpler functions like ReLU that either pass 
/// a value through or block it, SiLU smoothly scales inputs based on their value. It keeps 
/// most positive values, reduces small positive values, and allows some negative values to 
/// pass through (but reduced in magnitude). This smooth behavior helps neural networks learn 
/// more complex patterns. SiLU is used in many state-of-the-art models, especially in deep 
/// learning applications like computer vision and natural language processing.
/// </para>
/// </remarks>
public class SiLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as SiLU supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the SiLU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>The result of x * sigmoid(x).</returns>
    /// <remarks>
    /// <para>
    /// The SiLU function multiplies the input by the sigmoid of the input: f(x) = x * sigmoid(x).
    /// </para>
    /// <para>
    /// For Beginners: This method takes a number and transforms it using the SiLU formula.
    /// First, it calculates the sigmoid of the input (a value between 0 and 1), then multiplies
    /// the original input by this sigmoid value. This creates a smooth curve that:
    /// - For large positive values: outputs approximately the same value (since sigmoid approaches 1)
    /// - For values near zero: outputs a smaller value (since sigmoid is around 0.5)
    /// - For negative values: can output small negative values (unlike ReLU which outputs 0)
    /// 
    /// This behavior helps neural networks learn more effectively in many situations.
    /// </para>
    /// </remarks>
    public override T Activate(T input)
    {
        // SiLU: x * sigmoid(x)
        T sigmoid = MathHelper.Sigmoid(input);
        return NumOps.Multiply(input, sigmoid);
    }

    /// <summary>
    /// Calculates the derivative of the SiLU function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>The derivative of SiLU at the given input.</returns>
    /// <remarks>
    /// <para>
    /// The derivative of SiLU is: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
    /// This derivative is used during the backpropagation phase of neural network training.
    /// </para>
    /// <para>
    /// For Beginners: The derivative tells us how much the output changes when we slightly change the input.
    /// This information is crucial for training neural networks because it guides how the network's weights
    /// should be adjusted. The SiLU derivative has a nice property: it's non-zero for most input values,
    /// which helps prevent the "dying neuron" problem that can occur with simpler activation functions like ReLU.
    /// This means SiLU neurons can continue to learn even if they receive negative inputs.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        T sigmoid = MathHelper.Sigmoid(input);
        T sigmoidDerivative = NumOps.Multiply(sigmoid, NumOps.Subtract(NumOps.One, sigmoid));
        T xSigmoidDerivative = NumOps.Multiply(input, sigmoidDerivative);

        return NumOps.Add(sigmoid, xSigmoidDerivative);
    }
}