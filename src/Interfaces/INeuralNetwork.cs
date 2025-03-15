namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality for neural network models in the AiDotNet library.
/// </summary>
/// <remarks>
/// This interface provides methods for making predictions, updating model parameters,
/// saving and loading models, and controlling training behavior.
/// 
/// For Beginners: A neural network is a type of machine learning model inspired by the human brain.
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
public interface INeuralNetwork<T>
{
    /// <summary>
    /// Makes a prediction using the neural network for a given input.
    /// </summary>
    /// <remarks>
    /// This method processes the input data through the neural network and returns the predicted output.
    /// 
    /// For Beginners: This is like asking the neural network a question and getting its answer.
    /// 
    /// When you call this method:
    /// - You provide some input data (like an image, text, or measurements)
    /// - The neural network processes this data through its layers
    /// - Each neuron applies mathematical operations to the data
    /// - The network returns its prediction or classification
    /// 
    /// For example:
    /// - If you're using a neural network for house price prediction:
    ///   * Input: house features (size, location, bedrooms, etc.)
    ///   * Output: predicted price
    /// - If you're using a neural network for image recognition:
    ///   * Input: pixel values of an image
    ///   * Output: probabilities for different object categories
    /// 
    /// This is the main method you'll use when applying a trained neural network to new data.
    /// </remarks>
    /// <param name="input">A vector containing the input data for the neural network.</param>
    /// <returns>A vector containing the neural network's prediction or output.</returns>
    Vector<T> Predict(Vector<T> input);

    /// <summary>
    /// Updates the internal parameters (weights and biases) of the neural network.
    /// </summary>
    /// <remarks>
    /// This method replaces the current parameters of the neural network with new ones,
    /// typically used during training or when fine-tuning a model.
    /// 
    /// For Beginners: This is like updating the neural network's knowledge.
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
    /// Saves the neural network's state to a binary stream.
    /// </summary>
    /// <remarks>
    /// This method writes the neural network's architecture and parameters to a binary stream,
    /// allowing the model to be saved to a file or other storage medium.
    /// 
    /// For Beginners: This is like saving your neural network to a file.
    /// 
    /// When you call this method:
    /// - The neural network's structure and all its learned knowledge are converted to binary data
    /// - This data is written to the provided BinaryWriter
    /// - You can then save this data to a file, database, or other storage
    /// 
    /// For example:
    /// - After spending hours or days training a neural network
    /// - You can serialize it to save all its learned knowledge
    /// - Later, you can load this saved network and use it without retraining
    /// 
    /// This is useful when:
    /// - You want to save a trained model for later use
    /// - You want to share your model with others
    /// - You want to deploy your model to a production environment
    /// </remarks>
    /// <param name="writer">The BinaryWriter to which the neural network will be serialized.</param>
    void Serialize(BinaryWriter writer);

    /// <summary>
    /// Loads a neural network's state from a binary stream.
    /// </summary>
    /// <remarks>
    /// This method reads the neural network's architecture and parameters from a binary stream,
    /// restoring a previously saved model.
    /// 
    /// For Beginners: This is like loading a saved neural network from a file.
    /// 
    /// When you call this method:
    /// - The neural network reads its structure and learned knowledge from binary data
    /// - The network rebuilds itself using this data
    /// - After deserializing, the network is exactly as it was when serialized
    /// 
    /// For example:
    /// - You download a pre-trained neural network for language translation
    /// - You deserialize this network into your application
    /// - Immediately, your application can translate text without any training
    /// 
    /// This is particularly useful when:
    /// - You want to use a model that took a long time to train
    /// - You need to deploy the same model across multiple devices or applications
    /// - You're creating an application that non-technical users will use
    /// </remarks>
    /// <param name="reader">The BinaryReader from which the neural network will be deserialized.</param>
    void Deserialize(BinaryReader reader);

    /// <summary>
    /// Retrieves the current parameters (weights and biases) of the neural network.
    /// </summary>
    /// <remarks>
    /// This method returns a vector containing all the parameters that define the neural network's behavior.
    /// 
    /// For Beginners: This is like looking at the neural network's current knowledge.
    /// 
    /// Neural networks store their knowledge in parameters (weights and biases):
    /// - These parameters determine how the network processes information
    /// - They represent what the network has learned from training data
    /// - This method lets you see the current values of these parameters
    /// 
    /// For example:
    /// - You might want to analyze how the parameters have changed during training
    /// - You could save these parameters to use in another model
    /// - You might need to modify specific parameters for experimentation
    /// 
    /// This method is useful for:
    /// - Debugging neural network behavior
    /// - Implementing custom training algorithms
    /// - Analyzing what the network has learned
    /// - Transferring knowledge between models
    /// </remarks>
    /// <returns>A vector containing all the neural network's current parameter values.</returns>
    Vector<T> GetParameters();

    /// <summary>
    /// Sets whether the neural network is in training mode or inference (prediction) mode.
    /// </summary>
    /// <remarks>
    /// This method toggles the neural network's internal state between training and inference modes,
    /// which can affect how certain layers behave.
    /// 
    /// For Beginners: This is like switching the neural network between "learning mode" and "working mode".
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
}