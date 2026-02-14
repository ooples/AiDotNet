namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for neural network models with advanced architectural introspection capabilities.
/// </summary>
/// <remarks>
/// This interface extends the basic neural network functionality with methods for accessing
/// the internal architecture and layer-wise activations of neural networks.
///
/// <b>For Beginners:</b> This interface represents a neural network that can tell you about its structure.
///
/// Think of INeuralNetworkModel as a neural network with "x-ray vision" into its own structure:
/// - It can show you what each layer produces (activations)
/// - It can describe its own architecture (how it's built)
/// - It combines all the basic neural network abilities with introspection capabilities
///
/// For example, if you're debugging or analyzing a neural network:
/// - You can see what each layer outputs for a given input
/// - You can examine the network's structure (number of layers, layer types, connections)
/// - You can understand how information flows through the network
///
/// This is particularly useful for:
/// - Debugging neural networks (seeing where things go wrong)
/// - Understanding what the network has learned
/// - Visualizing how the network processes information
/// - Implementing advanced techniques like transfer learning or feature extraction
///
/// This interface is typically implemented by neural network base classes that provide
/// comprehensive access to their internal structure and computations.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("NeuralNetworkModel")]
public interface INeuralNetworkModel<T> : INeuralNetwork<T>
{
    /// <summary>
    /// Gets the intermediate activations from each layer when processing the given input with named keys.
    /// </summary>
    /// <param name="input">The input tensor to process through the network.</param>
    /// <returns>A dictionary mapping layer names to their activation tensors.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input through all layers of the network and returns a dictionary
    /// where each key is a descriptive layer name and each value is the output (activation) of that layer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This shows you what each layer "sees" or produces when given an input.
    ///
    /// Think of a neural network as a series of transformations:
    /// - The input enters the first layer
    /// - Each layer transforms the data and passes it to the next layer
    /// - The final layer produces the output
    ///
    /// This method lets you see the intermediate results at each step. For example, in an image
    /// recognition network:
    /// - Layer 1 might detect edges (its activation shows edge patterns)
    /// - Layer 2 might detect simple shapes (its activation shows shape patterns)
    /// - Layer 3 might detect object parts (its activation shows parts like eyes or wheels)
    /// - Final layer produces the classification result
    ///
    /// Each layer's name includes its position and type (e.g., "Layer_0_DenseLayer", "Layer_1_ConvolutionalLayer").
    /// This is useful for:
    /// - Visualizing what the network learned
    /// - Debugging why the network makes certain predictions
    /// - Extracting features from intermediate layers
    /// - Understanding the network's decision-making process
    /// </para>
    /// </remarks>
    Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input);

    /// <summary>
    /// Gets the architectural structure of the neural network.
    /// </summary>
    /// <returns>The architecture object describing the network's structure.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an object that describes the complete structure of the neural network,
    /// including all layers, their configurations, and how they connect to each other.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This gives you the "blueprint" of how the neural network is built.
    ///
    /// Just like a building has an architectural blueprint showing:
    /// - How many floors it has
    /// - The layout of each floor
    /// - How rooms connect to each other
    ///
    /// A neural network architecture describes:
    /// - How many layers the network has
    /// - What type each layer is (dense, convolutional, etc.)
    /// - How layers are connected
    /// - The size and shape of each layer
    ///
    /// This information is useful for:
    /// - Understanding how the network is structured
    /// - Documenting your model
    /// - Recreating the same architecture with different parameters
    /// - Comparing different network designs
    /// - Implementing model serialization and deserialization
    ///
    /// The architecture is typically set when the network is created and remains constant
    /// throughout the network's lifetime (though the parameters/weights change during training).
    /// </para>
    /// </remarks>
    NeuralNetworkArchitecture<T> GetArchitecture();
}
