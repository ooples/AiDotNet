namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Hierarchical Temporal Memory (HTM) network, a biologically-inspired sequence learning algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Hierarchical Temporal Memory (HTM) is a machine learning model that mimics the structural and algorithmic properties 
/// of the neocortex. It is particularly designed for sequence learning, prediction, and anomaly detection in 
/// time-series data. HTM networks consist of two main components: a Spatial Pooler that creates sparse distributed 
/// representations of inputs, and a Temporal Memory that learns sequences of these representations.
/// </para>
/// <para><b>For Beginners:</b> HTM is a special type of neural network inspired by how the human brain works.
/// 
/// Think of HTM like a system that learns patterns over time:
/// - It's similar to how your brain recognizes songs or predicts what comes next in a familiar sequence
/// - It's especially good at learning from time-series data (information that changes over time)
/// - It can recognize patterns even when they contain noise or slight variations
/// 
/// HTM networks have two main parts:
/// - Spatial Pooler: This converts incoming data into a special format that highlights important patterns
///   (like how your brain might focus on the melody of a song rather than background noise)
/// - Temporal Memory: This learns sequences and can predict what might come next
///   (like how you can anticipate the next note in a familiar song)
/// 
/// HTM is particularly useful for:
/// - Anomaly detection (finding unusual patterns)
/// - Sequence prediction (guessing what comes next)
/// - Pattern recognition in noisy data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HTMNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the size of the input vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input size defines the dimensionality of the input vectors that the HTM network can process.
    /// This value is determined by the input shape specified in the network architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many data points the network receives at once.
    /// 
    /// For example:
    /// - If you're analyzing a stock price, this might be 1 (just the price)
    /// - If you're analyzing weather, this might be several values (temperature, humidity, pressure, etc.)
    /// - If you're processing an image, this would be the total number of pixels
    /// 
    /// Think of this as the number of "sensors" your HTM network has to detect information from the world.
    /// </para>
    /// </remarks>
    private int _inputSize { get; }

    /// <summary>
    /// Gets the number of columns in the spatial pooler.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The column count determines the number of columns in the spatial pooler component of the HTM network.
    /// Each column represents a group of cells that share the same feedforward receptive field. A higher column count 
    /// increases the representational capacity of the network but also increases computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the number of "pattern detectors" in the network.
    /// 
    /// Columns in HTM are groups of cells that work together to recognize specific patterns in the input.
    /// 
    /// Think of each column as a specialist that learns to recognize particular features:
    /// - Some columns might learn to recognize rising trends
    /// - Others might learn to recognize sudden changes
    /// - Others might learn to recognize specific repeated patterns
    /// 
    /// More columns mean the network can recognize more different patterns, but requires more
    /// computational power. The default value of 2048 works well for many applications.
    /// </para>
    /// </remarks>
    private int _columnCount { get; }

    /// <summary>
    /// Gets the number of cells per column in the temporal memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property specifies how many cells are in each column within the temporal memory component.
    /// Each cell can learn different temporal contexts for the same spatial input pattern. More cells per column
    /// allow the network to disambiguate between different sequences that share common elements.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how well the network can understand context in sequences.
    /// 
    /// In HTM, each column contains multiple cells. These cells help the network understand
    /// that the same input can mean different things depending on what came before it.
    /// 
    /// For example, in English the word "bank" means different things in these sequences:
    /// - "I'll deposit money at the bank"
    /// - "I'll sit on the river bank"
    /// 
    /// Having multiple cells per column helps the network learn these different contexts.
    /// More cells (the default is 32) let the network handle more complex sequences
    /// where the same pattern can have different meanings based on context.
    /// </para>
    /// </remarks>
    private int _cellsPerColumn { get; }

    /// <summary>
    /// Gets the target sparsity for the spatial pooler output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sparsity threshold defines the target percentage of active columns in the spatial pooler output.
    /// A typical value is around 2% (0.02), which means that roughly 2% of columns will be active at any given time.
    /// Sparse representations are a key feature of HTM networks, as they enable efficient learning and generalization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how "selective" the network is about what it considers important.
    /// 
    /// Sparsity is a fundamental concept in HTM:
    /// - It means that only a small percentage of neurons are active at any time
    /// - In the human brain, typically only about 2% of neurons are active at once
    /// - This default value (0.02 or 2%) mimics this biological principle
    /// 
    /// Low sparsity (like 2%) helps the network:
    /// - Focus on the most important patterns
    /// - Generalize better to new, similar examples
    /// - Use memory and processing power efficiently
    /// - Avoid getting confused by noise
    /// 
    /// Think of it like highlighting only the most important words in a textbook rather than
    /// everything - it helps you focus on what really matters.
    /// </para>
    /// </remarks>
    private double _sparsityThreshold { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="HTMNetwork{T}"/> class with the specified architecture and parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="columnCount">The number of columns in the spatial pooler. Default is 2048.</param>
    /// <param name="cellsPerColumn">The number of cells per column in the temporal memory. Default is 32.</param>
    /// <param name="sparsityThreshold">The target sparsity for the spatial pooler output. Default is 0.02 (2%).</param>
    /// <exception cref="InvalidOperationException">Thrown when the input shape is not specified in the architecture.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates an HTM network with the specified architecture and parameters. It initializes
    /// the spatial pooler and temporal memory components based on the provided configuration values.
    /// The default parameters are based on common values used in HTM research and applications.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new HTM network with your chosen settings.
    /// 
    /// When creating an HTM network, you can customize several important aspects:
    /// 
    /// 1. Architecture: The basic structure of your network
    /// 
    /// 2. Column Count: How many pattern detectors (default is 2048)
    ///    - More columns = more capacity to learn different patterns
    ///    - Fewer columns = faster but less capable
    /// 
    /// 3. Cells Per Column: How much context the network can understand (default is 32)
    ///    - More cells = better at understanding complex sequences
    ///    - Fewer cells = simpler but may confuse similar sequences
    /// 
    /// 4. Sparsity Threshold: How selective the network is (default is 0.02 or 2%)
    ///    - Lower values = more selective, focuses on fewer elements
    ///    - Higher values = less selective, considers more elements important
    /// 
    /// The default values work well for many applications, but you can adjust them
    /// based on your specific needs.
    /// </para>
    /// </remarks>
    public HTMNetwork(
        NeuralNetworkArchitecture<T> architecture, 
        int columnCount = 2048, 
        int cellsPerColumn = 32, 
        double sparsityThreshold = 0.02) 
        : base(architecture)
    {
        var inputShape = Architecture.GetInputShape();
        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for HTM network.");
        }
    
        _inputSize = inputShape[0];
        _columnCount = columnCount;
        _cellsPerColumn = cellsPerColumn;
        _sparsityThreshold = sparsityThreshold;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the HTM network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the HTM network. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for an HTM network are created,
    /// including a spatial pooler layer and a temporal memory layer configured with the specified parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your HTM network.
    /// 
    /// When initializing the network:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates standard HTM layers automatically
    /// 
    /// The standard HTM layers typically include:
    /// 1. Spatial Pooler: Converts input to sparse distributed representations
    /// 2. Temporal Memory: Learns sequences and makes predictions
    /// 3. Optional additional layers: May include encoder/decoder layers or other processing
    /// 
    /// This process is like assembling all the components before the network starts learning.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultHTMLayers(Architecture, _columnCount, _cellsPerColumn, _sparsityThreshold));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network to generate a prediction from an input vector.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method processes an input vector through all layers of the HTM network sequentially, transforming
    /// it at each step according to the layer's function, and returns the final output vector. For HTM networks,
    /// this typically means passing through the spatial pooler, temporal memory, and any additional layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and passes it through 
    /// the network to get a prediction without updating the network's knowledge.
    /// 
    /// The process works like this:
    /// 1. Your input data enters the Spatial Pooler, which converts it to a sparse representation
    /// 2. This representation enters the Temporal Memory, which interprets it based on previous patterns
    /// 3. Any additional layers process the information further
    /// 4. The final layer produces a prediction about what might come next
    /// 
    /// This method is used when you want the network to make a prediction
    /// without learning from the new data (for example, when testing the network's performance).
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
    /// Processes an input vector through the network and updates the network's internal state based on the input.
    /// </summary>
    /// <param name="input">The input vector to learn from.</param>
    /// <exception cref="ArgumentException">Thrown when the input size doesn't match the expected size.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't contain the expected layer types.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the learning process in an HTM network. It first propagates the input through
    /// the spatial pooler and temporal memory layers. Then it performs learning operations on both components,
    /// allowing the network to adapt to the input sequence. The temporal memory layer's state is updated to
    /// maintain the sequence context for the next input.
    /// </para>
    /// <para><b>For Beginners:</b> This method both predicts AND learns from new data.
    /// 
    /// When you call this method with new input data:
    /// 
    /// 1. The network first processes the input through the Spatial Pooler
    ///    - Converts your input to a sparse representation
    ///    - Highlights the most important patterns
    /// 
    /// 2. Then it processes the result through the Temporal Memory
    ///    - Makes predictions based on what it has learned so far
    ///    - Creates a representation of the current state
    /// 
    /// 3. The Temporal Memory learns from this new information
    ///    - Updates its understanding of sequences
    ///    - Adjusts its predictions for the future
    /// 
    /// 4. The network stores the current state to use as context for the next input
    ///    - This is how it maintains an understanding of sequences over time
    /// 
    /// 5. The Spatial Pooler also learns from the input
    ///    - Improves its ability to identify important patterns
    ///    - Adapts to changing input statistics
    /// 
    /// This continuous learning process allows the network to improve over time as it sees more data.
    /// </para>
    /// </remarks>
    public void Learn(Vector<T> input)
    {
        if (input.Length != _inputSize)
            throw new ArgumentException($"Input size mismatch. Expected {_inputSize}, got {input.Length}.");

        // Forward pass through Spatial Pooler
        if (!(Layers[0] is SpatialPoolerLayer<T> spatialPoolerLayer))
            throw new InvalidOperationException("The first layer is not a SpatialPoolerLayer.");
        var spatialPoolerOutput = spatialPoolerLayer.Forward(Tensor<T>.FromVector(input)).ToVector();

        // Forward pass through Temporal Memory
        if (!(Layers[1] is TemporalMemoryLayer<T> temporalMemoryLayer))
            throw new InvalidOperationException("The second layer is not a TemporalMemoryLayer.");
        var temporalMemoryOutput = temporalMemoryLayer.Forward(Tensor<T>.FromVector(spatialPoolerOutput)).ToVector();

        // Learning in Temporal Memory
        temporalMemoryLayer.Learn(spatialPoolerOutput, temporalMemoryLayer.PreviousState);

        // Update the previous state for the next iteration
        temporalMemoryLayer.PreviousState = temporalMemoryOutput;

        // Learning in Spatial Pooler
        spatialPoolerLayer.Learn(input);

        // Forward pass through remaining layers
        var current = temporalMemoryOutput;
        for (int i = 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the network using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing updated parameters for all layers.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter values to each layer in the network. It extracts
    /// the appropriate segment of the parameter vector for each layer based on the layer's parameter count.
    /// For HTM networks, this is typically used for fine-tuning parameters after initial learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the configuration values in the network.
    /// 
    /// While HTM networks primarily learn through the specific Learning methods in each layer,
    /// this method allows you to directly update the parameters of all layers at once:
    /// 
    /// 1. It takes a long list of parameter values
    /// 2. Determines which values belong to which layers
    /// 3. Updates each layer with its corresponding values
    /// 
    /// This might be used for:
    /// - Fine-tuning a network after initial training
    /// - Applying optimized parameters found through experimentation
    /// - Resetting certain aspects of the network
    /// 
    /// This method is less commonly used in HTM networks compared to other types of neural networks,
    /// as HTMs typically learn through their specialized learning mechanisms.
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
    /// Serializes the HTM network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized network to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or when a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the HTM network's structure and parameters to a binary format that can be stored
    /// and later loaded. It writes the number of layers and then serializes each layer individually,
    /// preserving the network's learned knowledge and configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained HTM network to a file.
    /// 
    /// After training an HTM network, which can take time and data, you'll want to save it
    /// so you can use it later without training again. This method:
    /// 
    /// 1. Counts how many layers your network has
    /// 2. Writes this count to the file
    /// 3. For each layer:
    ///    - Writes the type of layer
    ///    - Saves all the learned patterns and parameters for that layer
    /// 
    /// This is like saving a document so you can open it again later without redoing all your work.
    /// The saved file contains all the knowledge the network has accumulated through training.
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
    /// Deserializes the HTM network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized network from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when an empty layer type name is encountered, when a layer type cannot be found, when a type does not implement the required interface, or when a layer instance cannot be created.</exception>
    /// <remarks>
    /// <para>
    /// This method loads a previously serialized HTM network from a binary format. It reads the number of layers
    /// and then deserializes each layer individually, recreating the network's structure and learned parameters.
    /// This allows the network to continue from its previously learned state.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved HTM network from a file.
    /// 
    /// When you want to use a network that was trained earlier, this method:
    /// 
    /// 1. Reads how many layers the network should have
    /// 2. Creates a new, empty network
    /// 3. For each layer:
    ///    - Reads what type of layer it should be
    ///    - Creates that type of layer
    ///    - Loads all the learned patterns and parameters for that layer
    /// 
    /// This is like opening a saved document to continue working where you left off.
    /// 
    /// For example, if you trained an HTM network to detect anomalies in server performance,
    /// you could save it after training on historical data and then load it later to
    /// monitor live systems without having to retrain.
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