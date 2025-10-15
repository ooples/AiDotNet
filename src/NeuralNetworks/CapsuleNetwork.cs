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
    private ILossFunction<T> _lossFunction { get; set; }

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
    public CapsuleNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null) : 
        base(architecture, lossFunction ?? new MarginLoss<T>())
    {
        _lossFunction = lossFunction ?? new MarginLoss<T>();

        InitializeLayers();
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
    /// Performs a forward pass through the Capsule Network to make a prediction.
    /// </summary>
    /// <param name="input">The input tensor to the network.</param>
    /// <returns>The output tensor (prediction) from the network.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input tensor through each layer of the network in sequence.
    /// Each layer processes the output from the previous layer (or the input for the first layer)
    /// and produces an output that becomes the input for the next layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like passing a piece of information through a series of processing stations.
    /// 
    /// Imagine an assembly line:
    /// - The input is the raw material
    /// - Each layer is a workstation that modifies or processes the material
    /// - The output is the final product after it has passed through all stations
    /// 
    /// In a Capsule Network, this process preserves and processes spatial relationships,
    /// allowing the network to understand complex structures in the input data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        Tensor<T> current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Trains the Capsule Network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration:
    /// 1. It makes a prediction using the current network parameters.
    /// 2. Calculates the loss between the prediction and the expected output.
    /// 3. Computes the gradient of the loss with respect to the network parameters.
    /// 4. Updates the network parameters based on the computed gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a practice session where the network learns from its mistakes.
    /// 
    /// The process is similar to learning a new skill:
    /// 1. You try to perform the task (make a prediction)
    /// 2. You see how far off you were (calculate the loss)
    /// 3. You figure out what you need to change to do better (compute the gradient)
    /// 4. You adjust your approach based on what you learned (update parameters)
    /// 
    /// This process is repeated many times with different inputs to improve the network's performance.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var prediction = Predict(input);

        // Calculate loss
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        // Backward pass
        var gradient = CalculateGradient(loss);

        // Update parameters
        UpdateParameters(gradient);
    }

    /// <summary>
    /// Retrieves metadata about the Capsule Network model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the network's structure and configuration.
    /// It includes details such as the input and output dimensions, the number of layers, and the types of layers used.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a summary or overview of the network's structure.
    /// 
    /// Think of it as a quick reference guide that tells you:
    /// - What kind of network it is (a Capsule Network)
    /// - How big the input and output are
    /// - How many layers the network has
    /// - What types of layers are used
    /// 
    /// This information is useful for understanding the network's capabilities and for saving/loading the network.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.CapsuleNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", Layers[0].GetInputShape()[0] },
                { "OutputDimension", Layers[Layers.Count - 1].GetOutputShape()[0] },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Capsule Network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// This method saves the loss function used by the network, allowing it to be reconstructed when the network is deserialized.
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
    }

    /// <summary>
    /// Deserializes Capsule Network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// This method loads the loss function used by the network. If deserialization fails, it defaults to using a MarginLoss.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _lossFunction = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader) ?? new MarginLoss<T>();
    }

    /// <summary>
    /// Calculates the gradient of the loss with respect to the network parameters.
    /// </summary>
    /// <param name="loss">The scalar loss value.</param>
    /// <returns>A vector containing the gradients for all network parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a backward pass through the network, computing gradients for each layer.
    /// It starts from the output layer and moves backwards, accumulating gradients along the way.
    /// </para>
    /// <para><b>For Beginners:</b> This is like tracing back through the network to see how each part contributed to the final result.
    /// 
    /// Imagine you're trying to improve a recipe:
    /// - You start with how the final dish turned out (the loss)
    /// - You work backwards through each step of the recipe
    /// - At each step, you figure out how changing that step would affect the final result
    /// - You collect all these potential changes (gradients) to know how to improve the recipe
    /// 
    /// In a neural network, this process helps determine how to adjust each parameter to reduce the loss.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateGradient(T loss)
    {
        List<Tensor<T>> gradients = new List<Tensor<T>>();

        // Backward pass through all layers
        Tensor<T> currentGradient = new Tensor<T>([1], new Vector<T>(Enumerable.Repeat(loss, 1)));
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
            gradients.Insert(0, currentGradient);
        }

        // Flatten all gradients into a single vector
        return new Vector<T>([.. gradients.SelectMany(g => g.ToVector())]);
    }

    /// <summary>
    /// Creates a new instance of the capsule network model.
    /// </summary>
    /// <returns>A new instance of the capsule network model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the capsule network model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization processes to create a fresh instance that can be populated
    /// with the serialized data. The new instance will have the same architecture and loss function as the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the network structure without copying the learned data.
    /// 
    /// Think of it like creating a blueprint of the capsule network:
    /// - It copies the same overall design (architecture)
    /// - It uses the same loss function to measure performance
    /// - But it doesn't copy any of the learned values or weights
    /// 
    /// This is primarily used when saving or loading models, creating a framework that the saved parameters
    /// can be loaded into later. It's like creating an empty duplicate of the network's structure
    /// that can later be filled with the knowledge from the original network.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CapsuleNetwork<T>(
            Architecture,
            _lossFunction
        );
    }
}