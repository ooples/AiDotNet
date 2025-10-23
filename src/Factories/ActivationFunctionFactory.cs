namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates activation functions for neural networks.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Activation functions are mathematical operations that determine whether a neuron 
/// in a neural network should be activated (fired) based on its input. They introduce non-linearity into 
/// the network, allowing it to learn complex patterns.
/// </para>
/// <para>
/// This factory class helps you create different types of activation functions without needing to know 
/// their internal implementation details. Think of it like selecting a specific tool from a toolbox - 
/// you just specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public static class ActivationFunctionFactory<T>
{
    /// <summary>
    /// Creates a single-value activation function of the specified type.
    /// </summary>
    /// <param name="activationFunction">The type of activation function to create.</param>
    /// <returns>An implementation of IActivationFunction<T> for the specified activation function type.</returns>
    /// <exception cref="NotSupportedException">Thrown when trying to create a Softmax activation function, which requires vector input.</exception>
    /// <exception cref="ArgumentException">Thrown when the requested activation function type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Single-value activation functions process one number at a time. They're typically 
    /// used in the hidden layers of neural networks to introduce non-linearity.
    /// </para>
    /// <para>
    /// Supported scalar activation functions:
    /// <list type="bullet">
    /// <item><description>ReLU (Rectified Linear Unit): Outputs the input if it's positive, otherwise outputs zero.</description></item>
    /// <item><description>Sigmoid: Maps inputs to values between 0 and 1.</description></item>
    /// <item><description>Tanh: Maps inputs to values between -1 and 1.</description></item>
    /// <item><description>Linear/Identity: Returns the input value unchanged.</description></item>
    /// <item><description>LeakyReLU: Similar to ReLU but allows small negative values.</description></item>
    /// <item><description>ELU: Exponential Linear Unit with smooth negative values.</description></item>
    /// <item><description>SELU: Scaled Exponential Linear Unit for self-normalizing networks.</description></item>
    /// <item><description>Softplus: Smooth approximation of ReLU.</description></item>
    /// <item><description>SoftSign: Maps inputs to values between -1 and 1 with smooth asymptotes.</description></item>
    /// <item><description>Swish: Self-gated activation function (x * sigmoid(x)).</description></item>
    /// <item><description>GELU: Gaussian Error Linear Unit used in transformers.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Note that Softmax cannot be used as a single-value activation function because it needs to process 
    /// multiple values together to calculate probabilities.
    /// </para>
    /// </remarks>
    public static IActivationFunction<T> CreateActivationFunction(ActivationFunction activationFunction)
    {
        return activationFunction switch
        {
            ActivationFunction.ReLU => new ReLUActivation<T>(),
            ActivationFunction.Sigmoid => new SigmoidActivation<T>(),
            ActivationFunction.Tanh => new TanhActivation<T>(),
            ActivationFunction.Linear or ActivationFunction.Identity => new IdentityActivation<T>(),
            ActivationFunction.LeakyReLU => new LeakyReLUActivation<T>(),
            ActivationFunction.ELU => new ELUActivation<T>(),
            ActivationFunction.SELU => new SELUActivation<T>(),
            ActivationFunction.Softmax => throw new NotSupportedException("Softmax is not applicable to single values. Use CreateVectorActivationFunction for Softmax."),
            ActivationFunction.Softplus => new SoftPlusActivation<T>(),
            ActivationFunction.SoftSign => new SoftSignActivation<T>(),
            ActivationFunction.Swish => new SwishActivation<T>(),
            ActivationFunction.GELU => new GELUActivation<T>(),
            _ => throw new ArgumentException($"Unsupported activation function: {activationFunction}", nameof(activationFunction))
        };
    }

    /// <summary>
    /// Creates a vector activation function of the specified type.
    /// </summary>
    /// <param name="activationFunction">The type of activation function to create.</param>
    /// <returns>An implementation of IVectorActivationFunction<T> for the specified activation function type.</returns>
    /// <exception cref="ArgumentException">Thrown when the requested vector activation function type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vector activation functions process multiple numbers together as a group. 
    /// They're typically used in the output layer of neural networks for classification tasks.
    /// </para>
    /// <para>
    /// Supported vector activation functions (all scalar functions can also be applied element-wise to vectors):
    /// <list type="bullet">
    /// <item><description>Softmax: Converts a vector into a probability distribution (values sum to 1).</description></item>
    /// <item><description>ReLU: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>Sigmoid: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>Tanh: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>Linear/Identity: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>LeakyReLU: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>ELU: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>SELU: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>Softplus: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>SoftSign: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>Swish: Applied element-wise to each value in the vector.</description></item>
    /// <item><description>GELU: Applied element-wise to each value in the vector.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IVectorActivationFunction<T> CreateVectorActivationFunction(ActivationFunction activationFunction)
    {
        return activationFunction switch
        {
            ActivationFunction.Softmax => new SoftmaxActivation<T>(),
            ActivationFunction.ReLU => new ReLUActivation<T>(),
            ActivationFunction.Sigmoid => new SigmoidActivation<T>(),
            ActivationFunction.Tanh => new TanhActivation<T>(),
            ActivationFunction.Linear or ActivationFunction.Identity => new IdentityActivation<T>(),
            ActivationFunction.LeakyReLU => new LeakyReLUActivation<T>(),
            ActivationFunction.ELU => new ELUActivation<T>(),
            ActivationFunction.SELU => new SELUActivation<T>(),
            ActivationFunction.Softplus => new SoftPlusActivation<T>(),
            ActivationFunction.SoftSign => new SoftSignActivation<T>(),
            ActivationFunction.Swish => new SwishActivation<T>(),
            ActivationFunction.GELU => new GELUActivation<T>(),
            _ => throw new ArgumentException($"Unsupported vector activation function: {activationFunction}", nameof(activationFunction))
        };
    }
}
