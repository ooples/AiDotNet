namespace AiDotNet.NeuralNetworks;

/// <summary>
/// A neural network implementation that processes data through multiple layers to make predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double)</typeparam>
/// <remarks>
/// <para>
/// Neural networks are computing systems inspired by the human brain. They consist of multiple layers
/// of interconnected nodes (neurons) that process input data to produce predictions. This class provides
/// a straightforward implementation that can be used for various machine learning tasks.
/// </para>
/// <para><b>For Beginners:</b> A neural network is like a brain-inspired system that learns from examples.
/// 
/// Think of a neural network as an assembly line for information:
/// - Input data enters the "factory" (like features of an image or text)
/// - It passes through several processing stations (layers of neurons)
/// - Each station transforms the information in specific ways
/// - Finally, it produces an output (like a prediction or classification)
/// 
/// For example, if you want to classify images of cats and dogs:
/// - The input would be the pixel values of the image
/// - The neural network processes these values through its layers
/// - Each layer learns to recognize different patterns (edges, shapes, textures, etc.)
/// - The output tells you the probability of the image containing a cat or dog
/// 
/// The network "learns" by adjusting its internal parameters based on examples,
/// gradually improving its predictions through a process called training.
/// </para>
/// </remarks>
public class NeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Creates a new neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure and configuration of the neural network</param>
    /// <remarks>
    /// <para>
    /// The architecture determines important aspects of the neural network such as:
    /// - The number and types of layers
    /// - The number of neurons in each layer
    /// - The activation functions used
    /// - Other configuration parameters
    /// 
    /// After creating the neural network, it automatically initializes the layers based on the provided architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new neural network with your desired structure.
    /// 
    /// When creating a neural network, you need to define its "architecture" - the blueprint that specifies:
    /// - How many inputs it will accept (like the number of features in your data)
    /// - How many layers it has (more layers can learn more complex patterns)
    /// - How many neurons are in each layer (more neurons can capture more information)
    /// - What activation functions to use (these add non-linearity, allowing the network to learn complex patterns)
    /// 
    /// Think of it like designing a building - you're establishing the foundation and framework
    /// before you start "training" it (like furnishing the rooms).
    /// 
    /// For example, a simple network for classifying handwritten digits might have:
    /// - 784 inputs (for a 28×28 pixel image)
    /// - 2 hidden layers with 128 neurons each
    /// - 10 outputs (one for each digit 0-9)
    /// </para>
    /// </remarks>
    public NeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network's structure by either:
    /// 1. Using custom layers provided in the architecture, or
    /// 2. Creating default layers if none were specified
    /// 
    /// The layers determine how data flows through the network and how computations are performed.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your neural network.
    /// 
    /// Think of this as assembling the components of your network:
    /// - If you've specified exactly what layers you want, those are used
    /// - If not, standard layers are created based on your architecture settings
    /// 
    /// Layers are the key processing units in a neural network. Common types include:
    /// - Input Layer: Receives your data
    /// - Hidden Layers: Process the information, extracting patterns
    /// - Output Layer: Produces the final prediction
    /// 
    /// Each layer contains neurons that apply mathematical operations and activation functions
    /// to transform the data as it flows through the network.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the neural network for the given input.
    /// </summary>
    /// <param name="input">The input data as a vector</param>
    /// <returns>The predicted output as a vector</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input data through each layer of the neural network in sequence.
    /// Each layer transforms the data according to its parameters and activation function.
    /// 
    /// For example, in a simple classification task:
    /// 1. The input vector might contain features of an object
    /// 2. The neural network processes these features through its layers
    /// 3. The output vector contains the predicted probabilities for each class
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and passes it through the network to get a prediction.
    /// 
    /// The process works like this:
    /// 1. Your input data enters the network (like an image or set of measurements)
    /// 2. The data travels through each layer in sequence
    /// 3. In each layer:
    ///    - Neurons receive values from the previous layer
    ///    - They apply weights, biases, and activation functions
    ///    - They output new values to the next layer
    /// 4. The final layer produces the prediction (like probabilities for each category)
    /// 
    /// For example, in a medical diagnosis system:
    /// - Input: Patient symptoms, test results, and vital signs
    /// - Process: The network analyzes patterns in this data
    /// - Output: Probability of different possible diagnoses
    /// 
    /// This is how you use the network after it's been trained to make predictions on new data.
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
    /// Updates the parameters (weights and biases) of the neural network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the entire network</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameters to each layer of the neural network.
    /// It's typically used during training when an optimization algorithm has calculated
    /// new parameter values to improve the network's performance.
    /// 
    /// The parameters vector must contain values for all trainable parameters in the network,
    /// arranged in the same order as the layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the internal values that the network has learned.
    /// 
    /// Neural networks learn by adjusting their "parameters" (weights and biases):
    /// - Weights determine how strongly neurons are connected to each other
    /// - Biases allow neurons to activate more or less easily
    /// 
    /// During training, the network figures out what parameters work best by:
    /// 1. Making predictions on training examples
    /// 2. Comparing predictions to correct answers
    /// 3. Calculating how to change parameters to improve accuracy
    /// 4. Using this method to update those parameters
    /// 
    /// Think of it like adjusting the settings on a complex machine to improve its performance.
    /// This method takes a long list of new parameter values and distributes them to the right
    /// places throughout the network.
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
    /// Saves the neural network's structure and parameters to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the data to</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when serialization encounters an error</exception>
    /// <remarks>
    /// <para>
    /// This method saves the complete state of the neural network, including:
    /// - The number and types of layers
    /// - All weights, biases, and other parameters
    /// 
    /// The saved data can later be loaded using the Deserialize method to recreate
    /// the exact same neural network without needing to train it again.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained neural network to a file.
    /// 
    /// After spending time training a neural network (which can take hours or days),
    /// you'll want to save it so you can use it later without training again. This method:
    /// 
    /// 1. Records all the network's structural information:
    ///    - How many layers it has
    ///    - What type each layer is
    ///    - How neurons are connected
    /// 
    /// 2. Saves all learned values:
    ///    - All weights between neurons
    ///    - All bias values
    ///    - Any other parameters the network has learned
    /// 
    /// Think of it like taking a snapshot of the network's "brain" that you can reload later.
    /// This is valuable because training is computationally expensive, but loading a pre-trained
    /// network takes just seconds.
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
    /// Loads a neural network's structure and parameters from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the data from</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when deserialization encounters an error</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs a neural network that was previously saved using the Serialize method.
    /// It recreates:
    /// - All layers with their original types
    /// - All weights, biases, and other parameters
    /// 
    /// After deserialization, the neural network is ready to use for making predictions
    /// without needing to be trained again.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved neural network from a file.
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
    /// For example, if you trained a network to recognize handwriting and saved it,
    /// you could load it weeks later and immediately use it to recognize text without
    /// having to retrain it on thousands of examples again.
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