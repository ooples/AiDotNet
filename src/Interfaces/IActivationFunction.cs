namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for activation functions used in neural networks and other machine learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An activation function is like a decision-maker in a neural network.
/// 
/// Imagine each neuron (node) in a neural network receives a number as input. The activation 
/// function decides how strongly that neuron should "fire" or activate based on that input.
/// 
/// For example:
/// - If the input is very negative, the neuron might not activate at all (output = 0)
/// - If the input is very positive, the neuron might activate fully (output = 1)
/// - If the input is around zero, the neuron might activate partially
/// 
/// Different activation functions create different patterns of activation, which helps
/// neural networks learn different types of patterns in data. Common activation functions
/// include Sigmoid, ReLU (Rectified Linear Unit), and Tanh (Hyperbolic Tangent).
/// 
/// This interface defines the standard methods that all activation functions must implement.
/// </remarks>
public interface IActivationFunction<T>
{
    /// <summary>
    /// Applies the activation function to the input value.
    /// </summary>
    /// <param name="input">The input value to the activation function.</param>
    /// <returns>The activated output value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a number (which could be positive, negative, or zero)
    /// and transforms it according to the specific activation function's rule.
    /// 
    /// For example, with the ReLU activation function:
    /// - If input is negative, output is 0
    /// - If input is positive, output is the same as the input
    /// 
    /// This transformation helps neural networks model complex, non-linear relationships
    /// in data, which is essential for tasks like image recognition or language processing.
    /// </remarks>
    T Activate(T input);

    /// <summary>
    /// Calculates the derivative (slope) of the activation function at the given input value.
    /// </summary>
    /// <param name="input">The input value at which to calculate the derivative.</param>
    /// <returns>The derivative value of the activation function at the input point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The derivative tells us how quickly the activation function's output
    /// changes when we make a small change to the input.
    /// 
    /// Think of it as the "slope" or "steepness" at a particular point on the activation function's curve.
    /// 
    /// This is crucial for training neural networks because:
    /// - It helps determine how much to adjust the network's weights during learning
    /// - A higher derivative means a stronger signal for learning
    /// - A derivative of zero means no learning signal (which can be a problem known as "vanishing gradient")
    /// 
    /// During training, the neural network uses this derivative to figure out how to adjust
    /// its internal parameters to improve its predictions.
    /// </remarks>
    T Derivative(T input);
}