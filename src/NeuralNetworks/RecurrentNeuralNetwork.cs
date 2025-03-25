namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Recurrent Neural Network, which is a type of neural network designed to process sequential data by maintaining an internal state.
/// </summary>
/// <remarks>
/// <para>
/// A Recurrent Neural Network (RNN) is a class of neural networks that can process sequential data by maintaining
/// an internal state (memory) that captures information about previous inputs. Unlike traditional feedforward
/// neural networks, RNNs have connections that form directed cycles, allowing information to persist from one step
/// to the next. This architectural feature makes RNNs particularly well-suited for tasks involving sequential data,
/// such as time series prediction, natural language processing, and speech recognition.
/// </para>
/// <para><b>For Beginners:</b> A Recurrent Neural Network is a type of neural network that has memory.
/// 
/// Think of it like reading a book:
/// - A standard neural network looks at each word in isolation
/// - An RNN remembers what it read earlier to understand the current word
/// 
/// For example, in the sentence "The clouds were dark, so I took my umbrella," an RNN understands that
/// "I took my umbrella" is related to "The clouds were dark" because it remembers the earlier part of the sentence.
/// 
/// This memory makes RNNs good at:
/// - Processing sequences like sentences, time series, or music
/// - Understanding context and relationships over time
/// - Predicting what comes next in a sequence
/// 
/// Common applications include text generation, translation, speech recognition, and stock price prediction - 
/// all tasks where what happened before affects how you interpret what's happening now.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RecurrentNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RecurrentNeuralNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the RNN.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Recurrent Neural Network with the specified architecture.
    /// It initializes the network layers based on the architecture, or creates default RNN layers if
    /// no specific layers are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Recurrent Neural Network with its basic structure.
    /// 
    /// When creating a new RNN:
    /// - The architecture defines what the network looks like - how many neurons it has, how they're connected, etc.
    /// - The constructor prepares the network by either:
    ///   * Using the specific layers provided in the architecture, or
    ///   * Creating default layers designed for processing sequences if none are specified
    /// 
    /// This is like setting up a specialized calculator before you start using it. The architecture
    /// determines what kind of calculations it can perform and how it will process information.
    /// </para>
    /// </remarks>
    public RecurrentNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Recurrent Neural Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for RNNs
    /// is created. In a typical RNN, this involves creating recurrent layers that maintain state between
    /// processing steps.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for recurrent networks are created automatically
    /// - The system checks that any custom layers will work properly with the RNN
    /// 
    /// A typical RNN has specialized layers that include:
    /// - Input layers that accept sequences
    /// - Recurrent layers that maintain memory of previous inputs
    /// - Output layers that produce predictions
    /// 
    /// These layers work together to process sequential data while maintaining context from previous steps.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultRNNLayers(Architecture));
        }
    }

    /// <summary>
    /// Processes the input through the recurrent neural network to produce a prediction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after processing through the network.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Recurrent Neural Network. It processes the input
    /// through each layer of the network in sequence, transforming it according to the operations defined
    /// in each layer. For an RNN, this typically involves updating the internal state based on both the
    /// current input and the previous state, capturing temporal dependencies in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the RNN processes information and makes predictions.
    /// 
    /// During the prediction process:
    /// - The input data (like a word or time point) enters the network
    /// - The data flows through each layer in sequence
    /// - Each recurrent layer combines the current input with its memory of previous inputs
    /// - The final output represents the network's prediction or answer
    /// 
    /// What makes this special is that when processing sequences (like words in a sentence):
    /// - The first word is processed with no prior context
    /// - The second word is processed while remembering information about the first word
    /// - The third word is processed with memory of both previous words
    /// ...and so on
    /// 
    /// This allows the network to understand relationships and patterns that unfold over time or sequence.
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
    /// Updates the parameters of the recurrent neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the recurrent neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer. In an RNN, these parameters typically include weights
    /// for both the input connections and the recurrent connections that maintain state.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the RNN makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For an RNN, these adjustments might include:
    /// - How much attention to pay to the current input
    /// - How much to rely on memory of previous inputs
    /// - How to combine different pieces of information
    /// 
    /// These parameter updates help the network learn to:
    /// - Recognize important patterns in sequences
    /// - Decide what information is worth remembering
    /// - Make better predictions based on both current and past information
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
    /// Saves the state of the Recurrent Neural Network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer serialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the Recurrent Neural Network, including all layers and their
    /// parameters. It writes the number of layers and the type and state of each layer to the provided binary writer.
    /// This allows the network state to be saved and later restored, which is useful for deploying trained models or
    /// continuing training from a checkpoint.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the RNN to a file.
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
    /// Loads the state of the Recurrent Neural Network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the Recurrent Neural Network from a binary reader. It reads
    /// the number of layers, recreates each layer based on its type, and deserializes the layer state.
    /// This allows a previously saved network state to be restored, which is essential for deploying trained
    /// models or continuing training from a checkpoint.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved RNN state from a file.
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
    /// Think of it like restoring a complete snapshot of the network that was saved earlier.
    /// For example, you might train an RNN to generate text, save it, and then later load it
    /// to generate new text without having to retrain the model.
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