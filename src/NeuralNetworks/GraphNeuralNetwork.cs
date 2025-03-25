namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Graph Neural Network that can process data represented as graphs.
/// </summary>
/// <remarks>
/// <para>
/// A Graph Neural Network (GNN) is designed to work with data structured as graphs, 
/// where nodes represent entities and edges represent relationships between these entities.
/// This implementation supports various activation functions for different layers and
/// provides methods for predicting outputs from both vector inputs and graph inputs.
/// </para>
/// <para><b>For Beginners:</b> A Graph Neural Network is a type of neural network that works with connected data.
/// 
/// Think of it like analyzing a social network:
/// - Each person is a "node" in the graph
/// - Friendships between people are "edges" connecting the nodes
/// - People have attributes (like age, location, interests) which are "node features"
/// 
/// GNNs are useful when:
/// - The relationships between items are as important as the items themselves
/// - You're working with network-like data (social networks, molecules, road systems)
/// - You need to make predictions about how nodes influence each other
/// 
/// For example, GNNs can help predict which products a customer might like based on
/// what similar customers have purchased, by analyzing the connections between customers and products.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the vector activation function used in graph convolutional layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the output of graph convolutional layers and operates on entire vectors.
    /// Vector activation functions transform multiple values at once, potentially capturing relationships between them.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how signals are processed in the graph layers.
    /// 
    /// An activation function:
    /// - Adds non-linearity to the network (helps it learn complex patterns)
    /// - Decides whether a node should pass information to the next layer
    /// - Usually transforms values to be within a certain range (like 0 to 1)
    /// 
    /// This specific activation works on groups of numbers together, rather than individually.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _graphConvolutionalVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in standard activation layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied in dedicated activation layers and operates on entire vectors.
    /// These layers typically follow other types of layers to introduce non-linearity.
    /// </para>
    /// <para><b>For Beginners:</b> This activation function is used in specific layers
    /// dedicated just to adding non-linearity to the network.
    /// 
    /// Think of it like a filter that processes multiple values at once to decide
    /// which information should continue through the network.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _activationLayerVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in the final dense layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the output of the final dense layer and operates on entire vectors.
    /// The final dense layer typically maps features to output classes or values.
    /// </para>
    /// <para><b>For Beginners:</b> This activation function processes the information 
    /// in the last main layer of the network.
    /// 
    /// It helps transform the final calculations into the format needed for your prediction.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _finalDenseLayerVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in the final activation layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied in the final dedicated activation layer and operates on entire vectors.
    /// The final activation layer often determines the range and characteristics of the network's output.
    /// </para>
    /// <para><b>For Beginners:</b> This is the very last processing step that determines
    /// the final output format of your network.
    /// 
    /// For example:
    /// - If you're classifying items into categories, this might ensure outputs are probabilities (0-1)
    /// - If you're predicting a continuous value, this might allow outputs to be any number
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _finalActivationLayerVectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in graph convolutional layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in graph convolutional layers.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This processes each individual number in graph layers separately.
    /// 
    /// Unlike the vector activation which works on groups of numbers together,
    /// this one applies the same transformation to each number independently.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _graphConvolutionalScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in standard activation layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in dedicated activation layers.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This processes each value individually in layers
    /// dedicated to adding non-linearity.
    /// 
    /// It's like having a separate filter for each piece of information.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _activationLayerScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in the final dense layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in the final dense layer.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This processes each number individually in the 
    /// last main layer of the network.
    /// 
    /// Each output gets its own separate transformation.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _finalDenseLayerScalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in the final activation layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to individual scalars in the final activation layer.
    /// Scalar activation functions transform each value independently.
    /// </para>
    /// <para><b>For Beginners:</b> This is the very last processing step applied
    /// individually to each output value.
    /// 
    /// It determines the final form of each individual output number.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _finalActivationLayerScalarActivation { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphNeuralNetwork{T}"/> class with vector activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="graphConvolutionalVectorActivation">The vector activation function for graph convolutional layers.</param>
    /// <param name="activationLayerVectorActivation">The vector activation function for standard activation layers.</param>
    /// <param name="finalDenseLayerVectorActivation">The vector activation function for the final dense layer.</param>
    /// <param name="finalActivationLayerVectorActivation">The vector activation function for the final activation layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a graph neural network with the specified architecture and vector activation functions.
    /// Vector activation functions operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new graph neural network where the 
    /// activation functions work on groups of numbers together.
    /// 
    /// When creating your network, you specify:
    /// - The overall structure (architecture)
    /// - Which activation functions to use at different stages
    /// 
    /// Vector activation functions process multiple values as a group, which can help
    /// capture relationships between different values.
    /// </para>
    /// </remarks>
    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture, IVectorActivationFunction<T>? graphConvolutionalVectorActivation = null, 
        IVectorActivationFunction<T>? activationLayerVectorActivation = null, IVectorActivationFunction<T>? finalDenseLayerVectorActivation = null, 
        IVectorActivationFunction<T>? finalActivationLayerVectorActivation = null) : base(architecture)
    {
        _graphConvolutionalVectorActivation = graphConvolutionalVectorActivation;
        _activationLayerVectorActivation = activationLayerVectorActivation;
        _finalDenseLayerVectorActivation = finalDenseLayerVectorActivation;
        _finalActivationLayerVectorActivation = finalActivationLayerVectorActivation;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphNeuralNetwork{T}"/> class with scalar activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="graphConvolutionalActivation">The scalar activation function for graph convolutional layers.</param>
    /// <param name="activationLayerActivation">The scalar activation function for standard activation layers.</param>
    /// <param name="finalDenseLayerActivation">The scalar activation function for the final dense layer.</param>
    /// <param name="finalActivationLayerActivation">The scalar activation function for the final activation layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a graph neural network with the specified architecture and scalar activation functions.
    /// Scalar activation functions operate on individual elements rather than entire vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new graph neural network where the 
    /// activation functions work on individual numbers separately.
    /// 
    /// When creating your network, you specify:
    /// - The overall structure (architecture)
    /// - Which activation functions to use at different stages
    /// 
    /// Scalar activation functions process each value independently, applying the same
    /// transformation to each number without considering relationships between values.
    /// </para>
    /// </remarks>
    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture, IActivationFunction<T>? graphConvolutionalActivation = null, 
        IActivationFunction<T>? activationLayerActivation = null, IActivationFunction<T>? finalDenseLayerActivation = null, 
        IActivationFunction<T>? finalActivationLayerActivation = null) : base(architecture)
    {
        _graphConvolutionalScalarActivation = graphConvolutionalActivation;
        _activationLayerScalarActivation = activationLayerActivation;
        _finalDenseLayerScalarActivation = finalDenseLayerActivation;
        _finalActivationLayerScalarActivation = finalActivationLayerActivation;
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the graph neural network. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for a graph neural network are created.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your neural network.
    /// 
    /// When initializing layers:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates a standard set of layers for graph processing
    /// 
    /// Think of this like assembling the components of the network before training begins.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultGNNLayers(Architecture));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network to generate a prediction from a vector input.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method processes an input vector through all layers of the network sequentially, transforming
    /// it at each step according to the layer's function, and returns the final output vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and passes it through 
    /// all the layers of the network to get a prediction.
    /// 
    /// The process is like an assembly line:
    /// - Data enters the first layer
    /// - Each layer transforms the data in some way
    /// - The transformed data is passed to the next layer
    /// - The final layer produces the prediction
    /// 
    /// This is the basic way to use the network after it's been trained.
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
    /// Performs a forward pass through the network to generate a prediction from graph data.
    /// </summary>
    /// <param name="nodeFeatures">A tensor containing features for each node in the graph.</param>
    /// <param name="adjacencyMatrix">A tensor representing the connections between nodes in the graph.</param>
    /// <returns>A tensor containing the prediction for the graph data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either nodeFeatures or adjacencyMatrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when nodeFeatures and adjacencyMatrix have incompatible dimensions.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the network encounters an unsupported layer type or invalid output shape.</exception>
    /// <remarks>
    /// <para>
    /// This method processes graph data through the network. It takes node features and an adjacency matrix
    /// as input and passes them through graph-specific and standard layers, applying appropriate transformations
    /// at each step. The method concludes with hybrid pooling to generate the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes a graph (like a social network) 
    /// through the neural network to make predictions.
    /// 
    /// You provide two pieces of information:
    /// - nodeFeatures: Information about each node (like age, interests for each person)
    /// - adjacencyMatrix: Information about how nodes are connected (like who is friends with whom)
    /// 
    /// The method:
    /// - Passes this information through specialized graph layers
    /// - Also passes it through standard neural network layers
    /// - Combines the results using a technique called "hybrid pooling"
    /// - Returns a prediction based on the entire graph structure
    /// 
    /// This is useful for tasks like predicting which users might become friends,
    /// which products a customer might like, or how a molecule might behave.
    /// </para>
    /// </remarks>
    public Tensor<T> PredictGraph(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
    {
        if (nodeFeatures == null || adjacencyMatrix == null)
            throw new ArgumentNullException(nodeFeatures == null ? nameof(nodeFeatures) : nameof(adjacencyMatrix));

        if (nodeFeatures.Shape[0] != adjacencyMatrix.Shape[0] || nodeFeatures.Shape[1] != adjacencyMatrix.Shape[1])
            throw new ArgumentException("Node features and adjacency matrix dimensions are incompatible.");

        var current = nodeFeatures;
        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLayer<T> graphLayer)
            {
                current = graphLayer.Forward(current, adjacencyMatrix);
            }
            else if (layer is ILayer<T> standardLayer)
            {
                // Handle non-graph layers (e.g., Dense, Activation)
                current = standardLayer.Forward(current);
            }
            else
            {
                throw new InvalidOperationException($"Unsupported layer type: {layer.GetType().Name}");
            }

            // Ensure the output maintains the expected shape
            if (current.Rank < 2)
                throw new InvalidOperationException($"Layer {layer.GetType().Name} produced an invalid output shape.");
        }

        // Implement hybrid pooling
        return HybridPooling(current);
    }

    /// <summary>
    /// Performs hybrid pooling on node features to generate a final output representation.
    /// </summary>
    /// <param name="nodeFeatures">A tensor containing processed features for each node.</param>
    /// <returns>A tensor containing the pooled output representation.</returns>
    /// <remarks>
    /// <para>
    /// This method combines multiple pooling strategies to create a comprehensive representation of the entire graph.
    /// It applies mean, max, and sum pooling, concatenates the results, and then processes them through a small
    /// neural network to learn the best combination of these pooling methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method combines information from all nodes in the graph
    /// to make an overall prediction.
    /// 
    /// Imagine having data about many individual users and needing to make a single
    /// prediction about the entire social network. This method:
    /// 
    /// 1. Combines node information in three different ways:
    ///    - Mean pooling: Takes the average of all nodes (like the average age of all users)
    ///    - Max pooling: Takes the maximum value from all nodes (like the age of the oldest user)
    ///    - Sum pooling: Adds up values from all nodes (like the total number of posts)
    /// 
    /// 2. Puts these three summaries together
    /// 
    /// 3. Uses a small neural network to figure out the best way to combine these summaries
    /// 
    /// This approach is more powerful than just using one type of pooling because
    /// different problems might need different ways of combining node information.
    /// </para>
    /// </remarks>
    private Tensor<T> HybridPooling(Tensor<T> nodeFeatures)
    {
        // Perform different types of pooling
        var meanPooled = nodeFeatures.MeanOverAxis(1);
        var maxPooled = nodeFeatures.MaxOverAxis(1);
        var sumPooled = nodeFeatures.SumOverAxis(1);

        // Concatenate the pooling results
        var concatenated = Tensor<T>.Concatenate([meanPooled, maxPooled, sumPooled], axis: 1);

        // Apply a small neural network to learn the best combination
        var denseLayer = new DenseLayer<T>(concatenated.Shape[1], concatenated.Shape[1] / 2, 
            _finalDenseLayerVectorActivation ?? new ReLUActivation<T>());
        // No separate activation layer needed since activation is included in the dense layer
        var outputLayer = new DenseLayer<T>(concatenated.Shape[1] / 2, meanPooled.Shape[1], 
            _finalActivationLayerVectorActivation ?? new IdentityActivation<T>());
        var activation = new ReLUActivation<T>();

        var hidden = denseLayer.Forward(concatenated);
        hidden = activation.Activate(hidden);
        var output = outputLayer.Forward(hidden);

        return output;
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
    /// Think of it like updating the settings on different parts of a machine to make it
    /// work better based on what it has learned.
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
    /// After training a network (which can take a lot of time), you'll want to save it
    /// so you can use it later without training again. This method:
    /// 
    /// 1. Counts how many layers your network has
    /// 2. Writes this count to the file
    /// 3. For each layer:
    ///    - Writes the type of layer (what it does)
    ///    - Saves all the learned values (parameters) for that layer
    /// 
    /// This is like saving a document so you can open it again later without redoing all your work.
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
    /// When you want to use a network that was trained earlier, this method:
    /// 
    /// 1. Reads how many layers the network should have
    /// 2. Creates a new, empty network
    /// 3. For each layer:
    ///    - Reads what type of layer it should be
    ///    - Creates that type of layer
    ///    - Loads all the learned values (parameters) for that layer
    /// 
    /// This is like opening a saved document to continue working where you left off.
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