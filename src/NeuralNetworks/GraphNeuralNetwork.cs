using AiDotNet.NeuralNetworks.Layers;

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
public class GraphNeuralNetwork<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (graph smoothness regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Graph smoothness regularization encourages connected nodes to have similar representations.
    /// This is based on the principle that nodes with edges between them should have similar features,
    /// which is a common assumption in many graph-based learning tasks.
    /// </para>
    /// <para><b>For Beginners:</b> Graph smoothness is like encouraging friends to be similar.
    ///
    /// In a graph:
    /// - Nodes that are connected (like friends in a social network) should have similar features
    /// - This auxiliary loss penalizes the network when connected nodes have very different representations
    /// - It helps the network learn more meaningful patterns that respect the graph structure
    ///
    /// For example:
    /// - In a social network, friends often have similar interests
    /// - In a molecule, bonded atoms influence each other's properties
    /// - In a citation network, papers that cite each other often cover similar topics
    ///
    /// Enabling this helps the network learn representations that are consistent with the graph structure.
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the graph smoothness auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much the graph smoothness regularization contributes to the total loss.
    /// The total loss is: main_loss + (auxiliary_weight * smoothness_loss).
    /// Typical values range from 0.01 to 0.1.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the network should enforce similarity between connected nodes.
    ///
    /// The weight determines the balance between:
    /// - Task accuracy (main loss) - making correct predictions
    /// - Graph smoothness (auxiliary loss) - keeping connected nodes similar
    ///
    /// Common values:
    /// - 0.05 (default): Balanced smoothness regularization
    /// - 0.01-0.03: Light smoothness enforcement
    /// - 0.08-0.1: Strong smoothness enforcement
    ///
    /// Higher values make the network focus more on keeping connected nodes similar,
    /// which can help with generalization but may reduce flexibility.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastGraphSmoothnessLoss;
    private Tensor<T>? _lastNodeRepresentations = null;
    private Tensor<T>? _lastAdjacencyMatrix = null;
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
    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? graphConvolutionalVectorActivation = null,
        IVectorActivationFunction<T>? activationLayerVectorActivation = null, IVectorActivationFunction<T>? finalDenseLayerVectorActivation = null,
        IVectorActivationFunction<T>? finalActivationLayerVectorActivation = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.05);
        _lastGraphSmoothnessLoss = NumOps.Zero;

        _graphConvolutionalVectorActivation = graphConvolutionalVectorActivation;
        _activationLayerVectorActivation = activationLayerVectorActivation;
        _finalDenseLayerVectorActivation = finalDenseLayerVectorActivation;
        _finalActivationLayerVectorActivation = finalActivationLayerVectorActivation;

        InitializeLayers();
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
    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? graphConvolutionalActivation = null,
        IActivationFunction<T>? activationLayerActivation = null, IActivationFunction<T>? finalDenseLayerActivation = null,
        IActivationFunction<T>? finalActivationLayerActivation = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.05);
        _lastGraphSmoothnessLoss = NumOps.Zero;

        _graphConvolutionalScalarActivation = graphConvolutionalActivation;
        _activationLayerScalarActivation = activationLayerActivation;
        _finalDenseLayerScalarActivation = finalDenseLayerActivation;
        _finalActivationLayerScalarActivation = finalActivationLayerActivation;

        InitializeLayers();
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
    /// Computes the auxiliary loss for graph smoothness regularization.
    /// </summary>
    /// <returns>The computed graph smoothness auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the graph smoothness loss, which encourages connected nodes
    /// to have similar representations. The loss is computed as the sum of squared differences
    /// between representations of connected nodes, weighted by the adjacency matrix.
    /// Formula: L_smooth = Σ_edges ||h_i - h_j||² * A_{ij}
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how different connected nodes are from each other.
    ///
    /// Graph smoothness works by:
    /// 1. Looking at each pair of connected nodes in the graph
    /// 2. Measuring how different their learned representations are
    /// 3. Penalizing large differences between connected nodes
    /// 4. Summing up these penalties across all edges
    ///
    /// This helps because:
    /// - It encourages the network to respect the graph structure
    /// - Connected nodes learn similar representations
    /// - The network generalizes better to new graph data
    ///
    /// For example, in a social network, friends (connected nodes) will have
    /// similar learned features, which makes sense since friends often share interests.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastNodeRepresentations == null || _lastAdjacencyMatrix == null)
        {
            _lastGraphSmoothnessLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute graph smoothness loss: Σ_ij A_ij * ||h_i - h_j||²
        // where A is the adjacency matrix and h are node representations
        T smoothnessLoss = NumOps.Zero;
        int numNodes = _lastNodeRepresentations.Shape[0];

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                // Get edge weight from adjacency matrix
                T edgeWeight = _lastAdjacencyMatrix[new int[] { i, j }];

                // Skip if no edge
                if (NumOps.Equals(edgeWeight, NumOps.Zero))
                    continue;

                // Compute squared difference between node representations
                var nodeI = _lastNodeRepresentations.GetRow(i);
                var nodeJ = _lastNodeRepresentations.GetRow(j);
                var diff = nodeI.Subtract(nodeJ);
                T squaredDiff = diff.DotProduct(diff);

                // Add weighted difference to smoothness loss
                smoothnessLoss = NumOps.Add(smoothnessLoss, NumOps.Multiply(edgeWeight, squaredDiff));
            }
        }

        // Normalize by number of edges
        int edgeCount = 0;
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                if (!NumOps.Equals(_lastAdjacencyMatrix[new int[] { i, j }], NumOps.Zero))
                    edgeCount++;
            }
        }

        if (edgeCount > 0)
        {
            smoothnessLoss = NumOps.Divide(smoothnessLoss, NumOps.FromDouble(edgeCount));
        }

        _lastGraphSmoothnessLoss = smoothnessLoss;
        return smoothnessLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the graph smoothness auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about auxiliary losses.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about the graph smoothness regularization, including
    /// the computed smoothness loss, weight applied, and whether the feature is enabled.
    /// This information is useful for monitoring training progress and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how graph smoothness regularization is working.
    ///
    /// The diagnostics include:
    /// - Total smoothness loss (how different connected nodes are)
    /// - Weight applied to the smoothness loss
    /// - Whether smoothness regularization is enabled
    /// - Whether node representations are being tracked
    ///
    /// This helps you:
    /// - Monitor if smoothness regularization is helping training
    /// - Debug issues with graph structure learning
    /// - Understand the impact of smoothness enforcement on learning
    ///
    /// You can use this information to adjust the auxiliary loss weight for better results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalGraphSmoothnessLoss", _lastGraphSmoothnessLoss?.ToString() ?? "0" },
            { "SmoothnessWeight", AuxiliaryLossWeight?.ToString() ?? "0.05" },
            { "UseGraphSmoothness", UseAuxiliaryLoss.ToString() },
            { "NodeRepresentationsCached", (_lastNodeRepresentations != null).ToString() },
            { "AdjacencyMatrixCached", (_lastAdjacencyMatrix != null).ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
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
    /// If graph smoothness regularization is enabled, it caches the node representations for auxiliary loss computation.
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
    /// - If smoothness regularization is enabled, saves intermediate representations
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

        // Cache for auxiliary loss computation
        if (UseAuxiliaryLoss)
        {
            _lastAdjacencyMatrix = adjacencyMatrix;
        }

        var current = nodeFeatures;
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                // Use the SetAdjacencyMatrix/Forward pattern for all graph layers
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
                current = layer.Forward(current);
            }
            else
            {
                // Handle non-graph layers (e.g., Dense, Activation)
                current = layer.Forward(current);
            }

            // Ensure the output maintains the expected shape
            if (current.Rank < 2)
                throw new InvalidOperationException($"Layer {layer.GetType().Name} produced an invalid output shape.");
        }

        // Cache node representations for auxiliary loss
        if (UseAuxiliaryLoss)
        {
            _lastNodeRepresentations = current;
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
    /// Performs a forward pass through the network to make a prediction using a standard input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input data through each layer of the network sequentially.
    /// For a GraphNeuralNetwork, this method assumes the input tensor already contains the necessary
    /// node features and adjacency information in a preprocessed format, or that the input is
    /// for a portion of the network that uses standard layers rather than graph-specific ones.
    /// </para>
    /// <para><b>For Beginners:</b> While GNNs work best with explicit graph data
    /// (provided through the PredictGraph method), this method allows the network to
    /// process pre-processed graph data or operate on non-graph portions of the network.
    /// 
    /// It works by:
    /// 1. Starting with your input data
    /// 2. Passing it through each layer of the network in sequence
    /// 3. Letting each layer transform the data based on its specific function
    /// 
    /// The result is a prediction based on the trained network's understanding of graph patterns.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Reset any layer states
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }

        // Forward pass through all layers
        Tensor<T> current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            // Skip graph-specific layers if this is a standard prediction
            if (Layers[i] is IGraphConvolutionLayer<T>)
            {
                // For graph layers, we need adjacency information which is not available
                // Just pass through without modification for standard prediction
                continue;
            }
            else
            {
                // Process through standard layers
                current = Layers[i].Forward(current);
            }
        }

        return current;
    }

    /// <summary>
    /// Trains the Graph Neural Network on a single input-output pair.
    /// </summary>
    /// <param name="input">The input tensor containing node features and adjacency information.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method trains the network on a single batch of data. For graph neural networks,
    /// the input tensor needs to contain both node features and adjacency information in a
    /// structured format. This implementation assumes the input tensor contains node features
    /// in the first half and adjacency matrix in the second half.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the neural network using example data.
    /// 
    /// For a graph neural network, the training process:
    /// 1. Extracts node features and connection information from the input
    /// 2. Makes a prediction using this graph data
    /// 3. Compares the prediction to the expected output to calculate error
    /// 4. Updates the network's internal values to reduce the error
    /// 
    /// Over time, with many examples, the network learns to make accurate predictions
    /// by understanding how nodes in a graph influence each other.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!IsTrainingMode)
        {
            SetTrainingMode(true);
        }

        // Extract node features and adjacency matrix from the input tensor
        int featuresDimension = input.Shape[0] / 2; // First half contains features

        // Extract node features
        Tensor<T> nodeFeatures = input.Slice(0, featuresDimension);

        // Extract adjacency matrix
        Tensor<T> adjacencyMatrix = input.Slice(featuresDimension, featuresDimension);

        // Forward pass with graph data
        Tensor<T> prediction = PredictGraph(nodeFeatures, adjacencyMatrix);

        // Calculate main loss
        var flattenedPredictions = prediction.ToVector();
        var flattenedExpected = expectedOutput.ToVector();
        LastLoss = LossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        // Add auxiliary loss if enabled
        if (UseAuxiliaryLoss)
        {
            T auxLoss = ComputeAuxiliaryLoss();
            T weightedAuxLoss = NumOps.Multiply(AuxiliaryLossWeight, auxLoss);
            LastLoss = NumOps.Add(LastLoss, weightedAuxLoss);
        }

        // Calculate output gradients
        var outputGradients = LossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);

        // Backpropagate to get parameter gradients
        Vector<T> gradients = Backpropagate(Tensor<T>.FromVector(outputGradients)).ToVector();

        // Get parameter gradients for all trainable layers
        Vector<T> parameterGradients = GetParameterGradients();

        // Clip gradients to prevent exploding gradients
        parameterGradients = ClipGradient(parameterGradients);

        // Create optimizer
        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Get current parameters
        Vector<T> currentParameters = GetParameters();

        // Update parameters using the optimizer
        Vector<T> updatedParameters = optimizer.UpdateParameters(currentParameters, parameterGradients);

        // Apply updated parameters
        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Trains the Graph Neural Network directly on graph data.
    /// </summary>
    /// <param name="nodeFeatures">A tensor containing features for each node in the graph.</param>
    /// <param name="adjacencyMatrix">A tensor representing the connections between nodes in the graph.</param>
    /// <param name="expectedOutput">The expected output tensor for the given graph input.</param>
    /// <remarks>
    /// <para>
    /// This method provides a more direct interface for training the network on graph data by explicitly
    /// accepting node features and adjacency matrix as separate parameters. This is often more intuitive
    /// than combining them into a single input tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This is a more straightforward way to train your graph neural network.
    /// 
    /// Instead of combining node information and connection information into one input,
    /// you can provide them separately:
    /// - nodeFeatures: Information about each node (e.g., user profiles in a social network)
    /// - adjacencyMatrix: Information about connections (e.g., who is friends with whom)
    /// - expectedOutput: What the network should predict for this graph
    /// 
    /// The network then learns to make predictions based on both the node attributes
    /// and how nodes are connected to each other.
    /// </para>
    /// </remarks>
    public void TrainGraph(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix, Tensor<T> expectedOutput)
    {
        if (!IsTrainingMode)
        {
            SetTrainingMode(true);
        }

        // Forward pass with graph data
        Tensor<T> prediction = PredictGraph(nodeFeatures, adjacencyMatrix);

        // Calculate main loss
        var flattenedPredictions = prediction.ToVector();
        var flattenedExpected = expectedOutput.ToVector();
        var loss = new MeanSquaredErrorLoss<T>().CalculateLoss(flattenedPredictions, flattenedExpected);

        // Add auxiliary loss if enabled
        if (UseAuxiliaryLoss)
        {
            T auxLoss = ComputeAuxiliaryLoss();
            T weightedAuxLoss = NumOps.Multiply(AuxiliaryLossWeight, auxLoss);
            loss = NumOps.Add(loss, weightedAuxLoss);
        }

        // Calculate output gradients
        var outputGradients = new MeanSquaredErrorLoss<T>().CalculateDerivative(flattenedPredictions, flattenedExpected);

        // Back-propagate the gradients
        Vector<T> backpropGradients = Backpropagate(Tensor<T>.FromVector(outputGradients)).ToVector();

        // Get parameter gradients
        Vector<T> parameterGradients = GetParameterGradients();

        // Apply gradient clipping
        parameterGradients = ClipGradient(parameterGradients);

        // Use adaptive optimizer (Adam)
        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Get current parameters
        Vector<T> currentParameters = GetParameters();

        // Update parameters
        Vector<T> updatedParameters = optimizer.UpdateParameters(currentParameters, parameterGradients);

        // Apply updated parameters
        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Gets metadata about the Graph Neural Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the Graph Neural Network, including its model type,
    /// number of layers, types of activation functions used, and serialized model data. This information
    /// is useful for model management and serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides a summary of your GNN's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (GraphNeuralNetwork)
    /// - Details about the network structure (layers, activations)
    /// - Performance metrics
    /// - Data needed to save and load the model
    /// 
    /// This is useful for:
    /// - Keeping track of different models you've created
    /// - Understanding a model's properties
    /// - Saving the model for later use
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GraphNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LayerCount", Layers.Count },
                { "GraphLayerCount", Layers.Count(l => l is IGraphConvolutionLayer<T>) },
                { "StandardLayerCount", Layers.Count(l => !(l is IGraphConvolutionLayer<T>)) },
                { "ParameterCount", GetParameterCount() },
                { "ActivationTypes", string.Join(", ", GetActivationTypes()) }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Gets the types of activation functions used in the network.
    /// </summary>
    /// <returns>A collection of activation function types used in the network.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method identifies what kinds of activation functions
    /// your neural network is using at different stages.
    /// 
    /// Activation functions determine how signals flow through the network and
    /// are a key part of how neural networks learn complex patterns.
    /// </para>
    /// </remarks>
    private IEnumerable<string> GetActivationTypes()
    {
        var activationTypes = new List<string>();

        // Add scalar activations if they exist
        if (_graphConvolutionalScalarActivation != null)
            activationTypes.Add(_graphConvolutionalScalarActivation.GetType().Name);

        if (_activationLayerScalarActivation != null)
            activationTypes.Add(_activationLayerScalarActivation.GetType().Name);

        if (_finalDenseLayerScalarActivation != null)
            activationTypes.Add(_finalDenseLayerScalarActivation.GetType().Name);

        if (_finalActivationLayerScalarActivation != null)
            activationTypes.Add(_finalActivationLayerScalarActivation.GetType().Name);

        // Add vector activations if they exist
        if (_graphConvolutionalVectorActivation != null)
            activationTypes.Add(_graphConvolutionalVectorActivation.GetType().Name);

        if (_activationLayerVectorActivation != null)
            activationTypes.Add(_activationLayerVectorActivation.GetType().Name);

        if (_finalDenseLayerVectorActivation != null)
            activationTypes.Add(_finalDenseLayerVectorActivation.GetType().Name);

        if (_finalActivationLayerVectorActivation != null)
            activationTypes.Add(_finalActivationLayerVectorActivation.GetType().Name);

        return activationTypes.Distinct();
    }

    /// <summary>
    /// Serializes Graph Neural Network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the Graph Neural Network's specific configuration data to a binary stream.
    /// This includes activation function types and any other GNN-specific parameters. This data
    /// is needed to reconstruct the GNN when deserializing.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the special configuration of your GNN.
    /// 
    /// Think of it like writing down the recipe for your neural network:
    /// - What activation functions it uses at different stages
    /// - How its graph-specific components are configured
    /// - Any other special settings that make this GNN unique
    /// 
    /// These details are crucial because they define how your GNN processes information,
    /// and they need to be saved along with the weights for the model to work correctly
    /// when loaded later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Serialize activation functions if they exist
        SerializationHelper<T>.SerializeInterface(writer, _graphConvolutionalScalarActivation);
        SerializationHelper<T>.SerializeInterface(writer, _activationLayerScalarActivation);
        SerializationHelper<T>.SerializeInterface(writer, _finalDenseLayerScalarActivation);
        SerializationHelper<T>.SerializeInterface(writer, _finalActivationLayerScalarActivation);

        SerializationHelper<T>.SerializeInterface(writer, _graphConvolutionalVectorActivation);
        SerializationHelper<T>.SerializeInterface(writer, _activationLayerVectorActivation);
        SerializationHelper<T>.SerializeInterface(writer, _finalDenseLayerVectorActivation);
        SerializationHelper<T>.SerializeInterface(writer, _finalActivationLayerVectorActivation);
    }

    /// <summary>
    /// Deserializes Graph Neural Network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the Graph Neural Network's specific configuration data from a binary stream.
    /// This includes activation function types and any other GNN-specific parameters. After reading this data,
    /// the GNN's state is fully restored to what it was when saved.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved GNN configuration.
    /// 
    /// Think of it like following a recipe to rebuild your neural network:
    /// - Reading what activation functions were used at different stages
    /// - Setting up the graph-specific components with the right configuration
    /// - Restoring any other special settings that make this GNN unique
    /// 
    /// This ensures that your loaded model will process information exactly the same way
    /// as when you saved it.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Deserialize activation functions
        _graphConvolutionalScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        _activationLayerScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        _finalDenseLayerScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        _finalActivationLayerScalarActivation = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);

        _graphConvolutionalVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        _activationLayerVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        _finalDenseLayerVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        _finalActivationLayerVectorActivation = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a new instance with the same architecture and activation functions
        // Determine which constructor to use based on which activation functions are set
        bool hasVectorActivations = _graphConvolutionalVectorActivation != null ||
                                    _activationLayerVectorActivation != null ||
                                    _finalDenseLayerVectorActivation != null ||
                                    _finalActivationLayerVectorActivation != null;

        bool hasScalarActivations = _graphConvolutionalScalarActivation != null ||
                                    _activationLayerScalarActivation != null ||
                                    _finalDenseLayerScalarActivation != null ||
                                    _finalActivationLayerScalarActivation != null;

        // Validate that we don't have a mix of vector and scalar activations
        if (hasVectorActivations && hasScalarActivations)
        {
            throw new InvalidOperationException(
                "Cannot create new instance with mixed vector and scalar activation functions. " +
                "All activation functions must be either vector-based or scalar-based, not a combination of both.");
        }

        if (hasVectorActivations)
        {
            return new GraphNeuralNetwork<T>(
                Architecture,
                LossFunction,
                _graphConvolutionalVectorActivation,
                _activationLayerVectorActivation,
                _finalDenseLayerVectorActivation,
                _finalActivationLayerVectorActivation);
        }
        else
        {
            return new GraphNeuralNetwork<T>(
                Architecture,
                LossFunction,
                _graphConvolutionalScalarActivation,
                _activationLayerScalarActivation,
                _finalDenseLayerScalarActivation,
                _finalActivationLayerScalarActivation);
        }
    }
}
