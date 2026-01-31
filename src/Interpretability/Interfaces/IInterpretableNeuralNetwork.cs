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

/// <summary>
/// Interface for convolutional neural networks that support Grad-CAM explanation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Grad-CAM creates visual explanations showing which parts of
/// an image the CNN focused on. This interface provides the methods needed to extract
/// feature maps (what the CNN "sees") and their gradients (what matters for the prediction).</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public interface IConvolutionalNetwork<T, TInput, TOutput>
{
    /// <summary>
    /// Gets feature maps and their gradients from the last convolutional layer.
    /// </summary>
    /// <param name="input">The input tensor (typically an image).</param>
    /// <param name="targetClass">The class to explain (which prediction to analyze).</param>
    /// <returns>A tuple containing feature maps and their gradients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feature maps are what CNN layers detect - edges, textures,
    /// patterns, etc. The gradients tell us which of these detections mattered most for
    /// predicting the target class.</para>
    ///
    /// <para>Grad-CAM combines these to create a heatmap showing which image regions
    /// were most important for the prediction.</para>
    /// </remarks>
    (Tensor<T> featureMaps, Tensor<T> gradients) GetFeatureMapsAndGradients(Tensor<T> input, int targetClass);
}

/// <summary>
/// Interface for transformer networks that support attention visualization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Transformers use "attention" to focus on different parts
/// of the input when making predictions. This interface provides methods to extract
/// these attention patterns for visualization and analysis.</para>
///
/// <para>For example, in a text classifier, attention might show which words the model
/// focused on most when making its classification decision.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public interface ITransformerNetwork<T, TInput, TOutput>
{
    /// <summary>
    /// Gets attention weights from all transformer layers.
    /// </summary>
    /// <param name="input">The input tensor (typically token embeddings).</param>
    /// <returns>A list of attention weight tensors, one per layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each attention tensor shows how much each position
    /// in the input attends to every other position. Shape is typically
    /// [num_heads, sequence_length, sequence_length].</para>
    ///
    /// <para>High attention values mean the model considered those position pairs
    /// to be related or important. Attention rollout aggregates these across layers
    /// to show overall importance.</para>
    /// </remarks>
    List<Tensor<T>> GetAttentionWeights(Tensor<T> input);
}
