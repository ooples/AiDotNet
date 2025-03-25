namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Residual Neural Network, which is a type of neural network that uses skip connections to address the vanishing gradient problem in deep networks.
/// </summary>
/// <remarks>
/// <para>
/// A Residual Neural Network (ResNet) is an advanced neural network architecture that introduces "skip connections" or "shortcuts"
/// that allow information to bypass one or more layers. These residual connections help address the vanishing gradient problem
/// that occurs in very deep networks, enabling the training of networks with many more layers than previously possible.
/// ResNets were a breakthrough in deep learning that significantly improved performance on image recognition and other tasks.
/// </para>
/// <para><b>For Beginners:</b> A Residual Neural Network is like a highway system for information in a neural network.
/// 
/// Think of it like this:
/// - In a traditional neural network, information must pass through every layer sequentially
/// - In a ResNet, there are "shortcut paths" or "highways" that let information skip ahead
/// 
/// For example, imagine trying to pass a message through a line of 100 people:
/// - In a regular network, each person must whisper to the next person in line
/// - In a ResNet, some people can also shout directly to someone 5 positions ahead
/// 
/// This design solves a major problem: in very deep networks (many layers), information and learning signals
/// tend to fade away or "vanish" as they travel through many layers. The shortcuts in ResNets help information
/// flow more easily through the network, allowing for much deeper networks (some with over 100 layers!)
/// that can learn more complex patterns.
/// 
/// ResNets revolutionized image recognition and are now used in many AI systems that need to identify
/// complex patterns in data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ResidualNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ResidualNeuralNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the ResNet.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Residual Neural Network with the specified architecture.
    /// It initializes the network layers based on the architecture, or creates default ResNet layers if
    /// no specific layers are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Residual Neural Network with its basic structure.
    /// 
    /// When creating a new ResNet:
    /// - The architecture defines what the network looks like - how many layers it has, how they're connected, etc.
    /// - The constructor prepares the network by either:
    ///   * Using the specific layers provided in the architecture, or
    ///   * Creating default layers designed for ResNets if none are specified
    /// 
    /// The default ResNet layers include special residual blocks that have both:
    /// - A main path where information is processed through multiple layers
    /// - A shortcut path that allows information to skip these layers
    /// 
    /// This combination of paths is what gives ResNets their special ability to train very deep networks.
    /// </para>
    /// </remarks>
    public ResidualNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Residual Neural Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for ResNets
    /// is created. In a typical ResNet, this involves creating residual blocks that combine a main path
    /// with a shortcut path, allowing information to either pass through layers or bypass them.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for ResNets are created automatically
    /// - The system checks that any custom layers will work properly with the ResNet
    /// 
    /// A typical ResNet has specialized building blocks called "residual blocks" that contain:
    /// - Convolutional layers that process the input
    /// - Batch normalization layers that stabilize learning
    /// - Activation layers that introduce non-linearity
    /// - Shortcut connections that allow information to bypass these layers
    /// 
    /// These blocks are then stacked together, often with increasing complexity as you go deeper into the network.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultResNetLayers(Architecture));
        }
    }

    /// <summary>
    /// Processes the input through the residual neural network to produce a prediction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after processing through the network.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Residual Neural Network. It processes the input
    /// through each layer of the network in sequence, transforming it according to the operations defined
    /// in each layer. For a ResNet, this includes both processing through convolutional layers and shortcuts
    /// that allow information to bypass certain layers, with the results being combined.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the ResNet processes information and makes predictions.
    /// 
    /// During the prediction process:
    /// - The input data (like an image) enters the network
    /// - The data flows through each layer in sequence
    /// - In residual blocks, the data travels down both paths:
    ///   * The main path where it's processed through several layers
    ///   * The shortcut path where it bypasses these layers
    /// - The outputs from both paths are then added together
    /// - This combined output continues to the next block
    /// 
    /// The shortcut paths ensure that even in very deep networks, information can flow easily from input to output,
    /// allowing the network to learn more complex patterns without running into the vanishing gradient problem.
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
    /// Updates the parameters of the residual neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the residual neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer. In a ResNet, these parameters typically include weights
    /// for convolutional layers, as well as parameters for batch normalization and other operations within residual blocks.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the ResNet makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For a ResNet, these adjustments might include:
    /// - How each convolutional filter detects patterns
    /// - How the batch normalization layers stabilize learning
    /// - How information should flow through both the main and shortcut paths
    /// 
    /// The residual connections (shortcuts) make it easier for these updates to flow backward through the network
    /// during training, which helps very deep networks learn effectively.
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
    /// Saves the state of the Residual Neural Network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer serialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the Residual Neural Network, including all layers and their
    /// parameters. It writes the number of layers and the type and state of each layer to the provided binary writer.
    /// This allows the network state to be saved and later restored, which is useful for deploying trained models or
    /// continuing training from a checkpoint.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the ResNet to a file.
    /// 
    /// When serializing:
    /// - All the network's layers are saved (their types and internal values)
    /// - The saved file can later be used to restore the exact same network state
    /// 
    /// This is useful for:
    /// - Saving a trained model to use later
    /// - Sharing a model with others
    /// - Creating backups during long training processes
    /// - Pausing and resuming training
    /// 
    /// Think of it like taking a complete snapshot of the network that can be restored later.
    /// This is especially important for ResNets, which can be very large and take a long time to train.
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
    /// Loads the state of the Residual Neural Network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the Residual Neural Network from a binary reader. It reads
    /// the number of layers, recreates each layer based on its type, and deserializes the layer state.
    /// This allows a previously saved network state to be restored, which is essential for deploying trained
    /// models or continuing training from a checkpoint.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved ResNet state from a file.
    /// 
    /// When deserializing:
    /// - The number and types of layers are read from the file
    /// - Each layer is recreated and its state is restored
    /// 
    /// This allows you to:
    /// - Load a previously trained model
    /// - Continue using or training a model from where you left off
    /// - Use models created by others
    /// 
    /// For example, you might download a pre-trained ResNet model that was trained on millions of images,
    /// and then use it directly or fine-tune it for your specific task. This saves enormous amounts of
    /// computation time and resources compared to training from scratch.
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