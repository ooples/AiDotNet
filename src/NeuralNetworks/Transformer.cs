namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Transformer neural network architecture, which is particularly effective for 
/// sequence-based tasks like natural language processing.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
public class Transformer<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The configuration settings for this Transformer network.
    /// </summary>
    private readonly TransformerArchitecture<T> _transformerArchitecture;

    /// <summary>
    /// Creates a new Transformer neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">
    /// The architecture configuration that defines how this Transformer will be structured.
    /// This includes settings like embedding size, number of attention heads, and feed-forward dimensions.
    /// </param>
    public Transformer(TransformerArchitecture<T> architecture) : base(architecture)
    {
        _transformerArchitecture = architecture;
    }

    /// <summary>
    /// Sets up the layers of the Transformer network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// This method either uses custom layers provided by the user or creates default Transformer layers.
    /// A typical Transformer consists of attention mechanisms, normalization layers, and feed-forward networks.
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
    /// A valid Transformer must include at least one attention layer and one normalization layer.
    /// Attention layers allow the model to focus on different parts of the input sequence.
    /// Normalization layers help stabilize training by normalizing the activations.
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
    /// This method passes the input sequentially through each layer of the Transformer.
    /// In a typical language model, this input might represent a tokenized text sequence.
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
    /// This method distributes the parameters to each layer based on their parameter counts.
    /// It's typically used during training when applying gradient updates.
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
    /// This method saves the type information and parameters of each layer,
    /// allowing the network to be reconstructed later using the Deserialize method.
    /// Serialization is useful for saving trained models to disk or transferring them between applications.
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
    /// This method reconstructs the network by reading the type information and parameters of each layer.
    /// It's used to load previously saved models for inference or continued training.
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