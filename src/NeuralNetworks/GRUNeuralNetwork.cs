namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Gated Recurrent Unit (GRU) Neural Network for processing sequential data.
/// </summary>
/// <remarks>
/// <para>
/// A GRU Neural Network is a type of recurrent neural network designed to effectively model sequential data.
/// GRU networks use gating mechanisms to control information flow through the network, allowing them to
/// capture long-term dependencies in sequence data while avoiding the vanishing gradient problem
/// that affects simple recurrent networks.
/// </para>
/// <para><b>For Beginners:</b> A GRU Neural Network is a special type of neural network that's good at processing data that comes in sequences.
/// 
/// Think of it like reading a book:
/// - A regular neural network would look at each word in isolation
/// - A GRU network remembers what it read earlier and uses that context to understand each new word
/// 
/// GRU networks have special "gates" that control what information to remember and what to forget:
/// - This helps them understand patterns that stretch over long sequences
/// - For example, in a sentence like "John went to the store because he needed milk," a GRU can connect "he" with "John"
/// 
/// GRU networks are useful for:
/// - Text processing and generation
/// - Time series prediction (like stock prices or weather)
/// - Speech recognition
/// - Any task where the order and context of data matters
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GRUNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GRUNeuralNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a GRU neural network with the specified architecture. The architecture defines
    /// important properties like input size, hidden layer sizes, and output size of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GRU neural network with your chosen design.
    /// 
    /// When you create a GRU network, you specify its architecture, which is like a blueprint that defines:
    /// - How many inputs the network accepts
    /// - How many hidden units to use (the network's "memory capacity")
    /// - How many outputs the network produces
    /// - Other structural aspects of the network
    /// 
    /// Think of it like defining the floor plan before building a house - you're setting up the basic structure
    /// that will determine how information flows through your network.
    /// </para>
    /// </remarks>
    public GRUNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the GRU neural network. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for a GRU neural network are created.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your neural network.
    /// 
    /// When initializing layers:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates a standard set of GRU layers automatically
    /// 
    /// This is like assembling all the components of your network before training begins.
    /// The standard GRU layers typically include:
    /// - Input layers to receive your data
    /// - GRU layers that process sequential information
    /// - Output layers that produce the final prediction
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultGRULayers(Architecture));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network to generate a prediction from an input vector.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method processes an input vector through all layers of the network sequentially, transforming
    /// it at each step according to the layer's function, and returns the final output vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and passes it through 
    /// the network to get a prediction.
    /// 
    /// The process works like an assembly line:
    /// - Data enters the first layer
    /// - Each layer transforms the data in some way
    /// - Each GRU layer maintains an internal memory state as it processes the sequence
    /// - The transformed data is passed to the next layer
    /// - The final layer produces the prediction
    /// 
    /// For example, if you're predicting the next word in a sentence, this method would:
    /// 1. Take the words you already have as input
    /// 2. Process them through the GRU layers, which remember important context
    /// 3. Output the most likely next word
    /// 
    /// This is how you use the network after it's been trained to make predictions.
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
    /// Updates the parameters of all layers in the network using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing updated parameters for all layers.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter values to each layer in the network. It extracts
    /// the appropriate segment of the parameter vector for each layer based on the layer's parameter count.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learned values in the network.
    /// 
    /// During training, a neural network adjusts its internal values (parameters) to make
    /// better predictions. This method:
    /// 
    /// 1. Takes a long list of new parameter values
    /// 2. Figures out which values belong to which layers
    /// 3. Updates each layer with its corresponding values
    /// 
    /// Think of it like fine-tuning different parts of a machine based on how well it performed.
    /// GRU networks have several important parameters:
    /// - Update gate parameters: control what information to add from the current input
    /// - Reset gate parameters: control what past information to forget
    /// - Memory parameters: store information across the sequence
    /// 
    /// This method ensures all these parameters get updated correctly during training.
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
    /// Serializes the neural network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized network to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or when a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the neural network's structure and parameters to a binary format that can be stored
    /// and later loaded. It writes the number of layers and then serializes each layer individually.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained neural network to a file.
    /// 
    /// After spending time training a GRU network (which can take hours or days for complex tasks),
    /// you'll want to save it so you can use it later without training again. This method:
    /// 
    /// 1. Counts how many layers your network has
    /// 2. Writes this count to the file
    /// 3. For each layer:
    ///    - Writes the type of layer (what it does)
    ///    - Saves all the learned values (parameters) for that layer
    /// 
    /// This is like saving a document so you can open it again later without redoing all your work.
    /// You can then load this saved network whenever you need to make predictions on new data.
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
    /// Deserializes the neural network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized network from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when an empty layer type name is encountered, when a layer type cannot be found, when a type does not implement the required interface, or when a layer instance cannot be created.</exception>
    /// <remarks>
    /// <para>
    /// This method loads a previously serialized neural network from a binary format. It reads the number of layers
    /// and then deserializes each layer individually, recreating the network's structure and parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved neural network from a file.
    /// 
    /// When you want to use a GRU network that was trained earlier, this method:
    /// 
    /// 1. Reads how many layers the network should have
    /// 2. Creates a new, empty network
    /// 3. For each layer:
    ///    - Reads what type of layer it should be
    ///    - Creates that type of layer
    ///    - Loads all the learned values (parameters) for that layer
    /// 
    /// This is like opening a saved document to continue working where you left off.
    /// 
    /// For example, if you trained a GRU network to predict stock prices,
    /// you could save it after training and then load it weeks later to make
    /// new predictions without having to retrain the model.
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