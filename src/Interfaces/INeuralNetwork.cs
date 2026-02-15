namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality for neural network models in the AiDotNet library.
/// </summary>
/// <remarks>
/// This interface provides methods for making predictions, updating model parameters,
/// saving and loading models, and controlling training behavior.
/// 
/// <b>For Beginners:</b> A neural network is a type of machine learning model inspired by the human brain.
/// 
/// Think of a neural network as a system that learns patterns:
/// - It's made up of interconnected "neurons" (small computing units)
/// - These neurons are organized in layers (input layer, hidden layers, output layer)
/// - Each connection between neurons has a "weight" (importance)
/// - The network learns by adjusting these weights based on examples it sees
/// 
/// For example, in an image recognition neural network:
/// - The input layer receives pixel values from an image
/// - Hidden layers detect patterns like edges, shapes, and textures
/// - The output layer determines what the image contains (e.g., "cat" or "dog")
/// 
/// Neural networks are powerful because they can:
/// - Learn complex patterns from data
/// - Make predictions on new, unseen data
/// - Improve their accuracy with more training
/// 
/// This interface provides the essential methods needed to work with neural networks in AiDotNet.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("NeuralNetwork")]
public interface INeuralNetwork<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Updates the internal parameters (weights and biases) of the neural network.
    /// </summary>
    /// <remarks>
    /// This method replaces the current parameters of the neural network with new ones,
    /// typically used during training or when fine-tuning a model.
    /// 
    /// <b>For Beginners:</b> This is like updating the neural network's knowledge.
    /// 
    /// Neural networks learn by adjusting their parameters (weights and biases):
    /// - Parameters determine how the network processes information
    /// - During training, these parameters are gradually adjusted to improve predictions
    /// - This method allows you to directly set new parameter values
    /// 
    /// For example:
    /// - An optimization algorithm might calculate better parameter values
    /// - You call this method to update the network with these improved values
    /// - The network will now make different (hopefully better) predictions
    /// 
    /// This method is primarily used:
    /// - During the training process
    /// - When implementing custom training algorithms
    /// - When fine-tuning a pre-trained model
    /// </remarks>
    /// <param name="parameters">A vector containing the new parameter values for the neural network.</param>
    void UpdateParameters(Vector<T> parameters);

    /// <summary>
    /// Sets whether the neural network is in training mode or inference (prediction) mode.
    /// </summary>
    /// <remarks>
    /// This method toggles the neural network's internal state between training and inference modes,
    /// which can affect how certain layers behave.
    /// 
    /// <b>For Beginners:</b> This is like switching the neural network between "learning mode" and "working mode".
    /// 
    /// Some neural network components behave differently during training versus prediction:
    /// - Dropout layers: randomly deactivate neurons during training to prevent overfitting,
    ///   but use all neurons during prediction
    /// - Batch normalization: uses different statistics during training versus prediction
    /// - Other regularization techniques: may only apply during training
    /// 
    /// For example:
    /// - Before training: call SetTrainingMode(true)
    /// - Before making predictions on new data: call SetTrainingMode(false)
    /// 
    /// This ensures that:
    /// - During training, the network uses techniques that help it learn better
    /// - During prediction, the network uses all its knowledge for the best possible results
    /// 
    /// Forgetting to set the correct mode can lead to unexpected or poor results.
    /// </remarks>
    /// <param name="isTrainingMode">True to set the network to training mode; false to set it to inference mode.</param>
    void SetTrainingMode(bool isTrainingMode);

    /// <summary>
    /// Performs a forward pass while storing intermediate activations for backpropagation.
    /// </summary>
    /// <remarks>
    /// This method processes input through the network while caching layer activations,
    /// enabling gradient computation during backpropagation.
    ///
    /// <b>For Beginners:</b> This is like the regular forward pass, but it remembers
    /// what happened at each step so the network can learn from its mistakes.
    ///
    /// During training:
    /// 1. Input flows forward through layers (this method)
    /// 2. Each layer's output is saved in memory
    /// 3. After seeing the error, we go backwards (Backpropagate)
    /// 4. The saved outputs help calculate how to improve each layer
    /// </remarks>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor from the network.</returns>
    Tensor<T> ForwardWithMemory(Tensor<T> input);

    /// <summary>
    /// Performs backpropagation to compute gradients for all parameters.
    /// </summary>
    /// <remarks>
    /// This method propagates error gradients backward through the network,
    /// computing how much each parameter contributed to the error.
    ///
    /// <b>For Beginners:</b> This is how the network learns from its mistakes.
    ///
    /// After making a prediction:
    /// 1. We calculate the error (how wrong was the prediction?)
    /// 2. Backpropagate sends this error backwards through layers
    /// 3. Each layer calculates "how much did I contribute to this error?"
    /// 4. These calculations (gradients) tell us how to adjust each weight
    ///
    /// This must be called after ForwardWithMemory() to have activations available.
    /// </remarks>
    /// <param name="outputGradients">Gradients of the loss with respect to network outputs.</param>
    /// <returns>Gradients with respect to the input (for chaining networks).</returns>
    Tensor<T> Backpropagate(Tensor<T> outputGradients);

    /// <summary>
    /// Gets the gradients computed during the most recent backpropagation.
    /// </summary>
    /// <remarks>
    /// This method returns the accumulated gradients for all trainable parameters
    /// after a backpropagation pass.
    ///
    /// <b>For Beginners:</b> After backpropagation figures out how to improve,
    /// this method retrieves those improvement instructions.
    ///
    /// The returned gradients tell the optimizer:
    /// - Which direction to adjust each weight
    /// - How strongly to adjust it
    ///
    /// The optimizer then uses these gradients to update the parameters.
    /// </remarks>
    /// <returns>A vector containing gradients for all trainable parameters.</returns>
    Vector<T> GetParameterGradients();
}
