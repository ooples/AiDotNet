using AiDotNet.NeuralNetworks;

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
public interface INeuralNetworkModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
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
    /// Gets the intermediate activations from each layer when processing the given input.
    /// </summary>
    /// <remarks>
    /// This method returns the output values from each layer in the neural network when processing
    /// the provided input. This is useful for understanding what the network is learning and for
    /// techniques like model quantization and pruning.
    /// 
    /// <b>For Beginners:</b> This is like looking at what happens at each step as data flows through the network.
    /// 
    /// When a neural network processes data, it goes through multiple layers:
    /// - The input layer receives the raw data
    /// - Each hidden layer transforms the data in some way
    /// - The output layer produces the final prediction
    /// 
    /// This method lets you see the transformed data at each layer, which is helpful for:
    /// - Understanding what patterns each layer is detecting
    /// - Debugging when the network isn't working as expected
    /// - Optimizing the network by identifying layers that aren't contributing much
    /// - Techniques like quantization (reducing model size) that need to know value ranges
    /// 
    /// For example, in an image recognition network:
    /// - Early layers might detect edges and simple shapes
    /// - Middle layers might detect more complex features like eyes or wheels
    /// - Later layers might recognize complete objects
    /// 
    /// By examining these activations, you can see how the network gradually builds up
    /// its understanding of the input data.
    /// </remarks>
    /// <param name="input">The input tensor to process through the network.</param>
    /// <returns>A dictionary mapping layer names to their activation tensors.</returns>
    Dictionary<string, Tensor<T>> GetLayerActivations(Tensor<T> input);
    
    /// <summary>
    /// Gets the architectural structure of the neural network.
    /// </summary>
    /// <remarks>
    /// This method returns information about the network's structure, including its layers,
    /// their types, sizes, and connections. This is useful for model analysis, optimization,
    /// and understanding the network's design.
    /// 
    /// <b>For Beginners:</b> This returns the blueprint of your neural network.
    /// 
    /// Just like a building's blueprint shows:
    /// - How many floors it has
    /// - The size and purpose of each room
    /// - How rooms are connected
    /// 
    /// This method returns information about:
    /// - How many layers your network has
    /// - What type each layer is (dense, convolutional, etc.)
    /// - How many neurons are in each layer
    /// - How the layers connect to each other
    /// 
    /// This information is useful for:
    /// - Understanding the complexity of your model
    /// - Optimizing the network (removing unnecessary layers or connections)
    /// - Calculating how much memory the model will use
    /// - Comparing different network designs
    /// 
    /// For example, you might discover that your network has 50 layers with millions
    /// of parameters, helping you understand why it might be slow or require lots of memory.
    /// </remarks>
    /// <returns>The neural network architecture containing layer information and structure.</returns>
    NeuralNetworkArchitecture<T> GetArchitecture();
}