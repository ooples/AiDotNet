namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Capsule Network, a type of neural network that preserves spatial relationships between features.
/// </summary>
/// <remarks>
/// <para>
/// A Capsule Network is a neural network architecture designed to address limitations of traditional convolutional
/// neural networks. Instead of using scalar-output feature detectors (neurons), Capsule Networks use vector-output
/// capsules. Each capsule's output vector represents the presence of an entity and its instantiation parameters
/// (like position, orientation, and scale). This architecture helps to preserve hierarchical relationships
/// between features, making it particularly effective for tasks requiring understanding of spatial relationships.
/// </para>
/// <para><b>For Beginners:</b> A Capsule Network is like a more advanced version of traditional neural networks.
/// 
/// Think of it this way:
/// - Traditional networks detect features like edges or textures, but lose information about how these features relate to each other
/// - Capsule Networks not only detect features, but also understand their relationships, orientations, and positions
/// - This is like the difference between recognizing individual puzzle pieces versus understanding how they fit together
/// 
/// For example, a traditional network might recognize an eye, a nose, and a mouth separately, but a Capsule Network
/// can better understand that these features need to be in a specific arrangement to make a face. This makes
/// Capsule Networks particularly good at recognizing objects from different angles or when parts are arranged differently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class CapsuleNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CapsuleNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Capsule Network with the specified architecture. The architecture
    /// defines the structure of the network, including the input dimensions, number and types of layers,
    /// and output dimensions. The initialization process sets up the layers based on the provided architecture
    /// or creates default capsule network layers if none are specified.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Capsule Network with your chosen settings.
    /// 
    /// When you create a Capsule Network:
    /// - You provide an "architecture" that defines how the network is structured
    /// - This includes information like how large the input is and what kinds of layers to use
    /// - The constructor sets up the basic structure, but doesn't actually train the network yet
    /// 
    /// Think of it like setting up a blank canvas and easel before you start painting -
    /// you're just getting everything ready to use.
    /// </para>
    /// </remarks>
    public CapsuleNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
    }

    /// <summary>
    /// Initializes the layers of the Capsule Network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Capsule Network. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default capsule network layers are created based on the architecture's
    /// specifications. After adding the layers, the method validates that the custom layers are properly configured.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the actual structure of the network.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers that work well for most cases
    /// - The method also checks that all layers are compatible with each other
    /// 
    /// This is like assembling the different sections of a factory production line -
    /// each layer processes the data and passes it to the next layer.
    /// </para>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCapsuleNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the current state of the Capsule Network.
    /// </summary>
    /// <param name="input">The input vector to make a prediction for.</param>
    /// <returns>The predicted output vector after passing through all layers of the network.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction by passing the input vector through each layer of the Capsule Network
    /// in sequence. Each layer processes the output of the previous layer, transforming the data until it reaches
    /// the final output layer. The result is a vector representing the network's prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the network to make a prediction based on input data.
    /// 
    /// The prediction process works like this:
    /// - The input data enters the first layer of the network
    /// - Each layer processes the data and passes it to the next layer
    /// - The data is transformed as it flows through the network
    /// - The final layer produces the prediction result
    /// 
    /// It's like an assembly line where each station modifies the product until it emerges
    /// as the finished item at the end.
    /// </para>
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
    /// Updates the parameters of all layers in the Capsule Network.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter vector among all the layers in the network.
    /// Each layer receives a portion of the parameter vector corresponding to its number of parameters.
    /// The method keeps track of the starting index for each layer's parameters in the input vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the network's internal values during training.
    /// 
    /// When updating parameters:
    /// - The input is a long list of numbers representing all values in the entire network
    /// - The method divides this list into smaller chunks
    /// - Each layer gets its own chunk of values
    /// - The layers use these values to adjust their internal settings
    /// 
    /// Think of it like giving each department in a company their specific budget allocations
    /// from the overall company budget.
    /// </para>
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
    /// Serializes the Capsule Network to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Capsule Network to a binary stream. It writes the number of layers,
    /// followed by the type name and serialized state of each layer. This allows the Capsule Network
    /// to be saved to disk and later restored with its trained parameters intact.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the network to a file.
    /// 
    /// When saving the Capsule Network:
    /// - First, it saves how many layers the network has
    /// - Then, for each layer, it saves:
    ///   - What type of layer it is (like "Convolution" or "Capsule")
    ///   - All the values and settings for that layer
    /// 
    /// This is like taking a complete snapshot of the network so you can reload it later
    /// without having to train it all over again.
    /// </para>
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
    /// Deserializes the Capsule Network from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer type information is invalid or instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Capsule Network from a binary stream. It reads the number of layers,
    /// followed by the type name and serialized state of each layer. This allows a previously saved
    /// Capsule Network to be restored from disk with all its trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved network from a file.
    /// 
    /// When loading the Capsule Network:
    /// - First, it reads how many layers the network had
    /// - Then, for each layer, it:
    ///   - Reads what type of layer it was
    ///   - Creates a new layer of that type
    ///   - Loads all the values and settings for that layer
    ///   - Adds the layer to the network
    /// 
    /// This is like restoring a complete snapshot of your network, bringing back
    /// all the patterns it had learned before.
    /// </para>
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