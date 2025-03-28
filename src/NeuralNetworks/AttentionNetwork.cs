namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a neural network that utilizes attention mechanisms for sequence processing.
/// </summary>
/// <remarks>
/// <para>
/// An attention network is a specialized neural network architecture designed for sequence processing tasks.
/// It uses attention mechanisms to dynamically focus on different parts of the input sequence when generating
/// outputs. This allows the network to capture long-range dependencies and relationships between elements in 
/// the sequence, making it particularly effective for tasks like natural language processing, time series analysis,
/// and other sequence-to-sequence problems.
/// </para>
/// <para><b>For Beginners:</b> This network mimics how humans pay attention to different parts of information.
/// 
/// Think of it like reading a complex paragraph:
/// - When you try to understand a sentence, you don't focus equally on all words
/// - You focus more on the important words that carry meaning
/// - You also connect related words even if they're far apart
/// 
/// For example, in the sentence "The cat, which had a white spot on its tail, chased the mouse":
/// - An attention network would connect "cat" with "chased" even though they're separated
/// - It would assign different importance to different words based on context
/// - This helps it understand the overall meaning better than networks that process words in isolation
/// 
/// This ability to selectively focus and connect distant information makes attention networks
/// powerful for language tasks, time series prediction, and many other sequence-based problems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class AttentionNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The maximum length of sequences this network can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the maximum length of sequences that the attention network is configured to process.
    /// It determines the size of attention matrices and positional encodings.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many elements (like words) the network can handle at once.
    /// 
    /// For example:
    /// - For text processing, this might be 512 tokens (roughly equivalent to paragraphs of text)
    /// - For time series data, this might be 100 time steps
    /// - The network cannot process sequences longer than this limit without truncation or splitting
    /// 
    /// This limit exists because attention mechanisms need to compare each element with every other element,
    /// which becomes computationally expensive for very long sequences.
    /// </para>
    /// </remarks>
    private readonly int _sequenceLength;

    /// <summary>
    /// The size of the embeddings used to represent each element in the sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the dimensionality of the embedding vectors used to represent each element in the input sequence.
    /// Higher dimensional embeddings can capture more nuanced representations but require more computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many features or dimensions each element has.
    /// 
    /// Think of it as the "richness" of information for each element in the sequence:
    /// - Larger values (like 768 or 1024) give more capacity to represent complex patterns
    /// - Common values range from 128 for simple tasks to 1024+ for complex language models
    /// - Higher values require more memory and processing power
    /// 
    /// For example, when processing text, each word might be represented by a 512-dimensional
    /// vector that captures various aspects of its meaning and context.
    /// </para>
    /// </remarks>
    private readonly int _embeddingSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="AttentionNetwork{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture specification for the network.</param>
    /// <param name="sequenceLength">The maximum length of sequences this network can process.</param>
    /// <param name="embeddingSize">The size of the embeddings used to represent each element in the sequence.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an attention network with the specified architecture, sequence length, and embedding size.
    /// It initializes the network's layers according to the architecture specification or uses default layers if none are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new attention network with the specified settings.
    /// 
    /// The parameters you provide determine:
    /// - architecture: The overall design of the network (layers, connections, etc.)
    /// - sequenceLength: How many elements (like words) the network can process at once
    /// - embeddingSize: How rich the representation of each element is
    /// 
    /// These settings control the capacity, expressiveness, and computational requirements of the network.
    /// Larger values for sequenceLength and embeddingSize give the network more capacity to handle
    /// complex tasks but require more memory and processing power.
    /// </para>
    /// </remarks>
    public AttentionNetwork(NeuralNetworkArchitecture<T> architecture, int sequenceLength, int embeddingSize) : base(architecture)
    {
        _sequenceLength = sequenceLength;
        _embeddingSize = embeddingSize;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the attention network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the layers of the attention network either by using the layers provided by the user
    /// in the architecture specification or by creating default attention layers if none are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the attention network.
    /// 
    /// It does one of two things:
    /// 1. If you provided specific layers in the architecture, it uses those
    /// 2. If you didn't provide layers, it creates a default set of attention layers
    /// 
    /// The default layers typically include:
    /// - Embedding layers to convert inputs to vector representations
    /// - Attention layers to focus on relevant parts of the sequence
    /// - Feed-forward layers to process the attended information
    /// - Output layers to produce the final results
    /// 
    /// This flexibility allows both beginners and experts to use the network effectively.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAttentionLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the attention network.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through all layers of the attention network to generate a prediction
    /// for the given input. Each layer's output becomes the input to the next layer in the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes input data through the network to make a prediction.
    /// 
    /// The prediction process works like this:
    /// 1. The input data enters the first layer of the network
    /// 2. Each layer processes the data and passes it to the next layer
    /// 3. The attention layers focus on relevant parts of the sequence
    /// 4. The final layer produces the output prediction
    /// 
    /// For example, in a translation task:
    /// - Input might be a sentence in English
    /// - The network processes this through its layers
    /// - The attention mechanisms focus on relevant words
    /// - Output would be the translated sentence in another language
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
    /// Updates the parameters of the attention network.
    /// </summary>
    /// <param name="parameters">The parameters to update the network with.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the network with the provided parameter values.
    /// It distributes the parameters to each layer based on the number of parameters in each layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the network's internal values to improve its performance.
    /// 
    /// During training:
    /// - The learning algorithm calculates how the parameters should change
    /// - This method applies those changes to the actual parameters
    /// - Each layer gets its own portion of the parameter updates
    /// 
    /// Think of it like fine-tuning all the components of the network based on feedback:
    /// - Attention mechanisms learn to focus on more relevant parts
    /// - Embedding layers learn better representations of the input
    /// - Feed-forward layers learn to process the information more effectively
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
    /// Serializes the attention network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when serialization encounters an error.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the attention network by writing the number of layers and then serializing each layer
    /// in sequence. For each layer, it writes the full type name followed by the layer's serialized data.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the network to a file or stream so it can be used later.
    /// 
    /// Serialization is like taking a snapshot of the network:
    /// - It saves the structure of the network (number and types of layers)
    /// - It saves all the learned parameters (weights, biases, etc.)
    /// - It ensures everything can be reconstructed exactly as it was
    /// 
    /// This is useful for:
    /// - Saving a trained model for later use
    /// - Sharing a model with others
    /// - Creating backups during long training processes
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
    /// Deserializes the attention network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when deserialization encounters an error.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the attention network by reading the number of layers and then deserializing each layer
    /// in sequence. For each layer, it reads the full type name, creates an instance of that type, and then deserializes
    /// the layer's data.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved network from a file or stream.
    /// 
    /// Deserialization is like reconstructing the network from a snapshot:
    /// - It reads the structure of the network (number and types of layers)
    /// - It loads all the learned parameters (weights, biases, etc.)
    /// - It recreates the network exactly as it was when saved
    /// 
    /// This allows you to:
    /// - Use a previously trained model without retraining it
    /// - Continue training from where you left off
    /// - Deploy the same model across different systems
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