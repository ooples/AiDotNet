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
    /// <exception cref="NotImplementedException">Thrown when the requested activation function type is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Single-value activation functions process one number at a time. They're typically 
    /// used in the hidden layers of neural networks to introduce non-linearity.
    /// </para>
    /// <para>
    /// Currently supported activation functions:
    /// <list type="bullet">
    /// <item><description>ReLU (Rectified Linear Unit): Outputs the input if it's positive, otherwise outputs zero. 
    /// It's like a gate that only lets positive values through.</description></item>
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
            ActivationFunction.Softmax => throw new NotSupportedException("Softmax is not applicable to single values. Use CreateVectorActivationFunction for Softmax."),
            _ => throw new NotImplementedException($"Activation function {activationFunction} not implemented.")
        };
    }

    /// <summary>
    /// Creates a vector activation function of the specified type.
    /// </summary>
    /// <param name="activationFunction">The type of activation function to create.</param>
    /// <returns>An implementation of IVectorActivationFunction<T> for the specified activation function type.</returns>
    /// <exception cref="NotImplementedException">Thrown when the requested vector activation function type is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vector<double> activation functions process multiple numbers together as a group. 
    /// They're typically used in the output layer of neural networks for classification tasks.
    /// </para>
    /// <para>
    /// Currently supported vector activation functions:
    /// <list type="bullet">
    /// <item><description>Softmax: Converts a vector of numbers into a probability distribution (values between 0 and 1 
    /// that sum to 1). It's commonly used in the output layer of classification networks to represent the probability 
    /// of each possible class.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IVectorActivationFunction<T> CreateVectorActivationFunction(ActivationFunction activationFunction)
    {
        return activationFunction switch
        {
            ActivationFunction.Softmax => new SoftmaxActivation<T>(),
            _ => throw new NotImplementedException($"Vector<double> activation function {activationFunction} not implemented.")
        };
    }
}