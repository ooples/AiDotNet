namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Long Short-Term Memory (LSTM) Neural Network, which is specialized for processing
/// sequential data like text, time series, or audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
public class LSTMNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Creates a new LSTM Neural Network with the specified architecture.
    /// </summary>
    /// <param name="architecture">
    /// The architecture configuration that defines how the network is structured,
    /// including input size, layer configuration, and other settings.
    /// </param>
    public LSTMNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        InitializeLayers();
    }

    /// <summary>
    /// Sets up the layers of the LSTM network based on the provided architecture.
    /// If no layers are specified in the architecture, default LSTM layers will be created.
    /// </summary>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultLSTMNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the LSTM network for the given input data.
    /// </summary>
    /// <param name="input">
    /// The input data as a vector. For time series data, this would typically be a flattened
    /// representation of your sequence features.
    /// </param>
    /// <returns>
    /// A vector containing the network's prediction or output.
    /// </returns>
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
    /// Updates the internal parameters (weights and biases) of the network with new values.
    /// This is typically used after training to apply optimized parameters.
    /// </summary>
    /// <param name="parameters">
    /// A vector containing all parameters for all layers in the network.
    /// The parameters must be in the correct order matching the network's layer structure.
    /// </param>
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
    /// Saves the LSTM network's structure and parameters to a binary stream.
    /// This allows you to save your trained model for later use.
    /// </summary>
    /// <param name="writer">
    /// The binary writer that will be used to write the network data to a stream or file.
    /// </param>
    public override void Serialize(BinaryWriter writer)
    {
        SerializationValidator.ValidateWriter(writer, nameof(LSTMNeuralNetwork<T>));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
            {
                throw new SerializationException(
                    "Cannot serialize a null layer",
                    nameof(LSTMNeuralNetwork<T>),
                    "Serialize");
            }

            string? fullName = layer.GetType().FullName;
            SerializationValidator.ValidateLayerTypeName(fullName, nameof(LSTMNeuralNetwork<T>));

            writer.Write(fullName!);
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Loads an LSTM network's structure and parameters from a binary stream.
    /// This allows you to load a previously trained model.
    /// </summary>
    /// <param name="reader">
    /// The binary reader that will be used to read the network data from a stream or file.
    /// </param>
    public override void Deserialize(BinaryReader reader)
    {
        SerializationValidator.ValidateReader(reader, nameof(LSTMNeuralNetwork<T>));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            SerializationValidator.ValidateLayerTypeName(layerTypeName, nameof(LSTMNeuralNetwork<T>));

            Type? layerType = Type.GetType(layerTypeName);
            SerializationValidator.ValidateLayerTypeExists(layerTypeName, layerType, nameof(LSTMNeuralNetwork<T>));

            ILayer<T> layer = (ILayer<T>)Activator.CreateInstance(layerType!)!;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }
    }
}