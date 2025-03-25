namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Long Short-Term Memory (LSTM) Neural Network, which is specialized for processing
/// sequential data like text, time series, or audio.
/// </summary>
/// <remarks>
/// <para>
/// Long Short-Term Memory networks are a special kind of recurrent neural network designed to overcome
/// the vanishing gradient problem that traditional RNNs face. LSTMs have a complex internal structure 
/// with specialized "gates" that regulate the flow of information, allowing them to remember patterns
/// over long sequences and selectively forget irrelevant information.
/// </para>
/// <para><b>For Beginners:</b> An LSTM Neural Network is a special type of neural network designed for understanding sequences and patterns that unfold over time.
/// 
/// Think of an LSTM like a smart notepad that can:
/// - Remember important information for long periods
/// - Forget irrelevant details
/// - Update its notes with new information
/// - Decide what parts of its memory to use for making predictions
/// 
/// For example, when processing a sentence like "The clouds are in the sky", an LSTM can:
/// - Remember "The clouds" as the subject even after seeing several more words
/// - Understand that "are" should agree with the plural "clouds" 
/// - Predict that "sky" might come after "in the" because clouds are typically in the sky
/// 
/// LSTMs are particularly good at:
/// - Text generation and language modeling
/// - Speech recognition
/// - Time series prediction (like stock prices or weather)
/// - Translation between languages
/// - Any task where the order of data matters and patterns may span across long sequences
/// </para>
/// </remarks>
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
    /// <remarks>
    /// <para>
    /// This constructor initializes an LSTM neural network based on the provided architecture. The architecture
    /// defines important structural aspects such as input size, number of hidden units, number of LSTM layers,
    /// and output size. Once created, the network layers are initialized according to this architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new LSTM network with your desired structure.
    /// 
    /// When you create an LSTM network, you specify its architecture, which is like a blueprint that defines:
    /// - How many inputs the network accepts (like word embeddings or sensor readings)
    /// - How many hidden units to use (the network's "memory capacity")
    /// - How many LSTM layers to stack (deeper networks can learn more complex patterns)
    /// - How many outputs the network produces
    /// 
    /// Think of it like designing a building - you're setting up the basic structure
    /// before you start "training" it (like furnishing the rooms).
    /// 
    /// For example, if you want to predict stock prices based on the past 10 days of data,
    /// you might create an LSTM with 10 input features, 64 hidden units, 2 layers, and 1 output.
    /// </para>
    /// </remarks>
    public LSTMNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        InitializeLayers();
    }

    /// <summary>
    /// Sets up the layers of the LSTM network based on the provided architecture.
    /// If no layers are specified in the architecture, default LSTM layers will be created.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the network layers according to the provided architecture. If the architecture
    /// includes a specific set of layers, those are used directly. Otherwise, the method creates a default
    /// LSTM layer configuration, which typically includes embeddings (for text data), one or more LSTM layers,
    /// and appropriate output layers based on the task type.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your LSTM network.
    /// 
    /// When initializing the network:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates standard LSTM layers automatically
    /// 
    /// The standard LSTM setup typically includes:
    /// 1. Input processing layers (like embedding layers for text)
    /// 2. One or more LSTM layers that process the sequence
    /// 3. Output layers that produce the final prediction
    /// 
    /// This is like assembling all the components of your network before training begins.
    /// Each layer has a specific role in processing your sequential data.
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
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the LSTM network, processing the input data through
    /// each layer sequentially. The input vector is first converted to a tensor, which is the format
    /// required by the internal layers. Each layer transforms the data according to its function and
    /// passes the result to the next layer. The final output is converted back to a vector format.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and passes it through the network to get a prediction.
    /// 
    /// The process works like this:
    /// 1. Your input data (like a sequence of words or time series values) enters the network
    /// 2. Each layer processes the data in sequence:
    ///    - The LSTM layers maintain an internal memory state as they process each element
    ///    - They use their "gates" to control what to remember, forget, and output
    /// 3. The final layer produces the prediction based on what the LSTM has learned
    /// 
    /// For example, if you're using an LSTM for weather prediction:
    /// - Your input might be temperature, humidity, and pressure readings for the past week
    /// - The LSTM processes this sequence, remembering important patterns
    /// - The output might be a prediction of tomorrow's weather
    /// 
    /// This method is how you use the network after it's been trained to make predictions on new data.
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
    /// Updates the internal parameters (weights and biases) of the network with new values.
    /// This is typically used after training to apply optimized parameters.
    /// </summary>
    /// <param name="parameters">
    /// A vector containing all parameters for all layers in the network.
    /// The parameters must be in the correct order matching the network's layer structure.
    /// </param>
    /// <remarks>
    /// <para>
    /// This method distributes a vector of parameters to the appropriate layers in the network. It determines
    /// how many parameters each layer needs, extracts the corresponding segment from the input parameter vector,
    /// and updates each layer with its respective parameters. This is commonly used after optimization algorithms
    /// have calculated improved weights for the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learned values in the network.
    /// 
    /// During training, an LSTM network learns many values (called parameters) that determine
    /// how it processes information. These include:
    /// - Weights that control how inputs affect the network
    /// - Gate parameters that control what information to remember or forget
    /// - Output parameters that determine how predictions are made
    /// 
    /// This method:
    /// 1. Takes a long list of all these parameters
    /// 2. Figures out which parameters belong to which layers
    /// 3. Updates each layer with its corresponding parameters
    /// 
    /// Think of it like updating the settings on different parts of a machine
    /// based on what it has learned through experience.
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
    /// Saves the LSTM network's structure and parameters to a binary stream.
    /// This allows you to save your trained model for later use.
    /// </summary>
    /// <param name="writer">
    /// The binary writer that will be used to write the network data to a stream or file.
    /// </param>
    /// <remarks>
    /// <para>
    /// This method serializes the LSTM network to a binary format for storage. It first validates the writer
    /// and then writes the number of layers. For each layer, it writes the type name and then calls the layer's
    /// own serialization method to save its specific parameters. This enables the network to be reconstructed
    /// later with the exact same structure and parameter values.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained LSTM network to a file.
    /// 
    /// After spending time training an LSTM (which can take hours or days for complex tasks),
    /// you'll want to save it so you can use it later without training again. This method:
    /// 
    /// 1. Counts how many layers your network has
    /// 2. Writes this count to the file
    /// 3. For each layer:
    ///    - Writes what type of layer it is
    ///    - Saves all the learned values for that layer
    /// 
    /// This is like taking a snapshot of the network's brain that you can reload later.
    /// For example, if you've trained an LSTM to generate music in a particular style,
    /// you can save it and then load it whenever you want to create new compositions.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method deserializes an LSTM network from a binary format. It first validates the reader
    /// and then reads the number of layers. For each layer, it reads the type name, creates an instance
    /// of that layer type, and calls the layer's deserialization method to load its specific parameters.
    /// This reconstructs the network with the exact same structure and parameter values it had when serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved LSTM network from a file.
    /// 
    /// When you want to use a network that was trained earlier, this method:
    /// 
    /// 1. Reads how many layers the network should have
    /// 2. Creates a new, empty network
    /// 3. For each layer:
    ///    - Reads what type of layer it should be
    ///    - Creates that type of layer
    ///    - Loads all the learned values for that layer
    /// 
    /// This is like restoring the network's "brain" from a snapshot.
    /// 
    /// For example, if you trained an LSTM to translate English to Spanish,
    /// you could save it after training on millions of sentences, and then
    /// load it years later to translate new text without retraining.
    /// </para>
    /// </remarks>
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