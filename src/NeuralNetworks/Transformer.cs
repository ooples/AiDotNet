namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Transformer neural network architecture, which is particularly effective for 
/// sequence-based tasks like natural language processing.
/// </summary>
/// <remarks>
/// <para>
/// The Transformer architecture is a type of neural network design that uses self-attention mechanisms
/// instead of recurrence or convolution. This approach allows the model to weigh the importance of 
/// different parts of the input sequence when producing each part of the output sequence.
/// </para>
/// <para>
/// The key components of a Transformer include:
/// - Multi-head attention layers: Allow the model to focus on different parts of the input
/// - Feed-forward networks: Process the attended information
/// - Layer normalization: Stabilize the network during training
/// - Residual connections: Help information flow through the network
/// </para>
/// <para><b>For Beginners:</b> A Transformer is a modern type of neural network that excels at 
/// understanding sequences of data, like sentences or time series.
/// 
/// Think of it like reading a book:
/// - When you read a sentence, some words are more important than others for understanding the meaning
/// - A Transformer can "pay attention" to different words based on their importance
/// - It can look at the entire context at once, rather than reading one word at a time
/// 
/// For example, in the sentence "The animal didn't cross the street because it was too wide",
/// the Transformer can figure out that "it" refers to "the street" by paying attention to the
/// relationship between these words.
/// 
/// Transformers are behind many recent AI advances, including large language models like GPT and BERT.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
public class Transformer<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The configuration settings for this Transformer network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the architecture configuration which defines the structure and properties
    /// of this Transformer network, including settings like embedding size, number of attention 
    /// heads, and feed-forward dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the blueprint for our Transformer.
    /// 
    /// It contains all the important settings that determine how the Transformer works:
    /// - How many attention mechanisms to use
    /// - How large each part of the network should be
    /// - How information flows through the network
    /// 
    /// Just like a house blueprint defines the structure of a house, this architecture
    /// defines the structure of our Transformer neural network.
    /// </para>
    /// </remarks>
    private readonly TransformerArchitecture<T> _transformerArchitecture;

    /// <summary>
    /// Creates a new Transformer neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">
    /// The architecture configuration that defines how this Transformer will be structured.
    /// This includes settings like embedding size, number of attention heads, and feed-forward dimensions.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Transformer neural network with the provided architecture
    /// configuration. It passes the architecture to the base class constructor and also stores it
    /// for use in initializing the Transformer-specific layers.
    /// </para>
    /// <para><b>For Beginners:</b> This is where we create our Transformer network.
    /// 
    /// When you create a new Transformer, you provide a blueprint (the architecture) that specifies:
    /// - How many layers it should have
    /// - How attention works in the network
    /// - How large the various components should be
    /// 
    /// This is similar to how you might specify the size, number of rooms, and layout when building a house.
    /// </para>
    /// </remarks>
    public Transformer(TransformerArchitecture<T> architecture) : base(architecture)
    {
        _transformerArchitecture = architecture;
    }

    /// <summary>
    /// Sets up the layers of the Transformer network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided by the user or creates default Transformer layers.
    /// A typical Transformer consists of attention mechanisms, normalization layers, and feed-forward networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the actual structure of the Transformer.
    /// 
    /// It works in one of two ways:
    /// - If you've provided your own custom layers, it uses those
    /// - Otherwise, it creates a standard set of Transformer layers
    /// 
    /// These layers typically include:
    /// - Attention layers (which let the model focus on relevant parts of the input)
    /// - Normalization layers (which keep the numbers from getting too large or small)
    /// - Feed-forward layers (which process the information)
    /// 
    /// It's like assembling the rooms and sections of a house according to the blueprint.
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
            // Use default transformer layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultTransformerLayers(_transformerArchitecture));
        }
    }

    /// <summary>
    /// Ensures that custom layers provided for the Transformer meet the minimum requirements.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <remarks>
    /// <para>
    /// A valid Transformer must include at least one attention layer and one normalization layer.
    /// Attention layers allow the model to focus on different parts of the input sequence.
    /// Normalization layers help stabilize training by normalizing the activations.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if your custom layers will actually work as a Transformer.
    /// 
    /// For a Transformer to function properly, it needs at minimum:
    /// - An attention layer (which helps the model focus on important parts of the input)
    /// - A normalization layer (which keeps the numbers stable during training)
    /// 
    /// If either of these is missing, it's like trying to build a house without walls or a foundation - it won't work!
    /// 
    /// This method checks for these essential components and raises an error if they're missing.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the custom layers don't include required layer types.
    /// </exception>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
    
        bool hasAttentionLayer = false;
        bool hasLayerNorm = false;
    
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is MultiHeadAttentionLayer<T>)
            {
                hasAttentionLayer = true;
            }
            else if (layers[i] is LayerNormalizationLayer<T>)
            {
                hasLayerNorm = true;
            }
        }
    
        if (!hasAttentionLayer)
        {
            throw new InvalidOperationException("Custom Transformer must include at least one MultiHeadAttentionLayer.");
        }
    
        if (!hasLayerNorm)
        {
            throw new InvalidOperationException("Custom Transformer must include at least one LayerNormalizationLayer.");
        }
    }

    /// <summary>
    /// Processes an input vector through the Transformer network to produce a prediction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after passing through all layers of the Transformer.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input sequentially through each layer of the Transformer.
    /// In a typical language model, this input might represent a tokenized text sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the Transformer actually performs its task.
    /// 
    /// When you give the Transformer some input data:
    /// - The data passes through each layer in sequence
    /// - Each layer transforms the data in some way
    /// - The final output represents the Transformer's answer or prediction
    /// 
    /// For example, if you're using the Transformer for language translation:
    /// - The input might be a sentence in English
    /// - The output would be the same sentence translated to another language
    /// 
    /// This method handles the entire processing pipeline from input to output.
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
    /// Updates the parameters of all layers in the Transformer network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to each layer based on their parameter counts.
    /// It's typically used during training when applying gradient updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the Transformer's internal values during training.
    /// 
    /// Think of parameters as the "settings" of the Transformer:
    /// - Each layer needs a certain number of parameters to function
    /// - During training, these parameters are constantly adjusted to improve performance
    /// - This method takes a big list of new parameter values and gives each layer its share
    /// 
    /// It's like distributing updated parts to each section of a machine so it works better.
    /// Each layer gets exactly the number of parameters it needs.
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
    /// Saves the Transformer network structure and parameters to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to save the network to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the type information and parameters of each layer,
    /// allowing the network to be reconstructed later using the Deserialize method.
    /// Serialization is useful for saving trained models to disk or transferring them between applications.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the Transformer to a file so you can use it later.
    /// 
    /// When saving the Transformer:
    /// - Information about each layer's type is saved
    /// - All the learned parameter values are saved
    /// - The entire structure of the network is preserved
    /// 
    /// This is useful for:
    /// - Saving a trained model after spending time and resources on training
    /// - Sharing your model with others
    /// - Using your model in a different application
    /// 
    /// It's like taking a snapshot of the entire Transformer that can be restored later.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when serialization encounters issues with layer information.</exception>
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
    /// Loads a Transformer network structure and parameters from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to load the network from.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the network by reading the type information and parameters of each layer.
    /// It's used to load previously saved models for inference or continued training.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved Transformer from a file.
    /// 
    /// When loading the Transformer:
    /// - The number and types of layers are read from the file
    /// - Each layer is created with the correct type
    /// - The parameter values are loaded into each layer
    /// 
    /// This allows you to:
    /// - Use a model that was trained earlier
    /// - Continue training a model from where you left off
    /// - Use models created by others
    /// 
    /// It's like reassembling the Transformer from a blueprint and parts list that was saved earlier.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when deserialization encounters issues with layer information.</exception>
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