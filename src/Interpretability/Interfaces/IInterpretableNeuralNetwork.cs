namespace AiDotNet.Interpretability.Interfaces;

/// <summary>
/// Extends neural network capabilities with methods needed for advanced interpretability explainers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This interface defines additional methods that neural networks can
/// implement to enable advanced explanation techniques. Not all neural networks will support
/// all of these methods, but implementing them enables deeper interpretability.</para>
///
/// <para>Methods include:
/// - Gradient computation for gradient-based attributions (Integrated Gradients, Saliency)
/// - Activation extraction for DeepLIFT
/// - Layer-wise information for LRP
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
/// <typeparam name="TInput">The input type to the model.</typeparam>
/// <typeparam name="TOutput">The output type from the model.</typeparam>
public interface IInterpretableNeuralNetwork<T, TInput, TOutput>
{
    /// <summary>
    /// Computes the gradient of the output with respect to the input.
    /// </summary>
    /// <param name="input">The input tensor to compute gradients for.</param>
    /// <returns>A tensor containing gradients with respect to each input element.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients tell us how much the output would change if we
    /// slightly changed each input feature. This is the foundation for many explanation
    /// methods like Saliency Maps and Integrated Gradients.</para>
    ///
    /// <para>Think of it like asking "if I made this pixel slightly brighter, how much
    /// would it affect whether the model thinks this is a cat?"</para>
    /// </remarks>
    Tensor<T> ComputeGradient(Tensor<T> input);

    /// <summary>
    /// Performs a forward pass and returns both the output and intermediate layer activations.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tuple containing the output and an array of activation tensors per layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> As data flows through a neural network, each layer
    /// transforms it. The "activations" are the intermediate values at each layer.
    /// DeepLIFT needs these to understand how the network processes the input.</para>
    ///
    /// <para>Imagine watching a factory assembly line - activations are like snapshots
    /// at each station showing how the product is being transformed.</para>
    /// </remarks>
    (Tensor<T> output, Tensor<T>[] activations) ForwardWithActivations(Tensor<T> input);

    /// <summary>
    /// Performs a forward pass and returns output, activations, and layer weight matrices.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tuple containing output, layer activations, and weight matrices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> LRP (Layer-wise Relevance Propagation) needs to know
    /// not just what values flow through the network (activations), but also how the
    /// layers are connected (weights). This enables tracing importance backwards
    /// from the output to the input.</para>
    ///
    /// <para>Weights represent the "wiring" of the network - how strongly each neuron
    /// in one layer connects to neurons in the next layer.</para>
    /// </remarks>
    (Tensor<T> output, Tensor<T>[] activations, Matrix<T>[] weights) ForwardWithNetworkInfo(Tensor<T> input);
}
