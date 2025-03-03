namespace AiDotNet.NeuralNetworks;

/// <summary>
/// A neural network implementation that processes data through multiple layers to make predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double)</typeparam>
/// <remarks>
/// Neural networks are computing systems inspired by the human brain. They consist of multiple layers
/// of interconnected nodes (neurons) that process input data to produce predictions. This class provides
/// a straightforward implementation that can be used for various machine learning tasks.
/// </remarks>
public class NeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Creates a new neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure and configuration of the neural network</param>
    /// <remarks>
    /// The architecture determines important aspects of the neural network such as:
    /// - The number and types of layers
    /// - The number of neurons in each layer
    /// - The activation functions used
    /// - Other configuration parameters
    /// 
    /// After creating the neural network, it automatically initializes the layers based on the provided architecture.
    /// </remarks>
    public NeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the architecture.
    /// </summary>
    /// <remarks>
    /// This method sets up the neural network's structure by either:
    /// 1. Using custom layers provided in the architecture, or
    /// 2. Creating default layers if none were specified
    /// 
    /// The layers determine how data flows through the network and how computations are performed.
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the neural network for the given input.
    /// </summary>
    /// <param name="input">The input data as a vector</param>
    /// <returns>The predicted output as a vector</returns>
    /// <remarks>
    /// This method passes the input data through each layer of the neural network in sequence.
    /// Each layer transforms the data according to its parameters and activation function.
    /// 
    /// For example, in a simple classification task:
    /// 1. The input vector might contain features of an object
    /// 2. The neural network processes these features through its layers
    /// 3. The output vector contains the predicted probabilities for each class
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    /// <summary>
    /// Updates the parameters (weights and biases) of the neural network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the entire network</param>
    /// <remarks>
    /// This method distributes the provided parameters to each layer of the neural network.
    /// It's typically used during training when an optimization algorithm has calculated
    /// new parameter values to improve the network's performance.
    /// 
    /// The parameters vector must contain values for all trainable parameters in the network,
    /// arranged in the same order as the layers.
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Saves the neural network's structure and parameters to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the data to</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when serialization encounters an error</exception>
    /// <remarks>
    /// This method saves the complete state of the neural network, including:
    /// - The number and types of layers
    /// - All weights, biases, and other parameters
    /// 
    /// The saved data can later be loaded using the Deserialize method to recreate
    /// the exact same neural network without needing to train it again.
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Loads a neural network's structure and parameters from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the data from</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when deserialization encounters an error</exception>
    /// <remarks>
    /// This method reconstructs a neural network that was previously saved using the Serialize method.
    /// It recreates:
    /// - All layers with their original types
    /// - All weights, biases, and other parameters
    /// 
    /// After deserialization, the neural network is ready to use for making predictions
    /// without needing to be trained again.
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }
    }
}