namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Graph Convolutional Network (GCN) layer for processing graph-structured data.
/// </summary>
/// <remarks>
/// <para>
/// A Graph Convolutional Layer applies convolution operations to graph-structured data by leveraging
/// an adjacency matrix that defines connections between nodes in the graph. This layer learns
/// representations for nodes in a graph by aggregating feature information from a node's local neighborhood.
/// The layer performs the transformation: output = adjacency_matrix * input * weights + bias.
/// </para>
/// <para><b>For Beginners:</b> This layer helps neural networks understand data that's organized like a network or graph.
/// 
/// Think of a social network where people are connected to friends:
/// - Each person is a "node" with certain features (age, interests, etc.)
/// - Connections between people are "edges"
/// - This layer helps the network learn patterns by looking at each person AND their connections
/// 
/// For example, in a social network recommendation system, this layer can help understand that 
/// a person might like something because their friends like it, even if their personal profile 
/// doesn't suggest they would.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphConvolutionalLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>, IGraphConvolutionLayer<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether auxiliary loss is enabled for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the layer computes a graph smoothness auxiliary loss that encourages connected nodes
    /// to have similar learned representations. This helps the network learn more coherent graph embeddings.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the layer uses an additional learning signal.
    ///
    /// When enabled (true):
    /// - The layer encourages connected nodes to learn similar features
    /// - This helps the network understand that connected nodes should be related
    /// - Training may be more stable and produce better results
    ///
    /// When disabled (false):
    /// - Only the main task loss is used for training
    /// - This is the default setting
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the auxiliary loss contribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines how much the graph smoothness loss contributes to the total loss.
    /// The default value of 0.01 provides a good balance between the main task and smoothness regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much importance to give to the smoothness penalty.
    ///
    /// The weight affects training:
    /// - Higher values (e.g., 0.1) make the network prioritize smooth features more strongly
    /// - Lower values (e.g., 0.001) make the smoothness penalty less important
    /// - The default (0.01) works well for most graph learning tasks
    ///
    /// If your graph has very clear structure, you might increase this value.
    /// If the main task is more important, you might decrease it.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Gets the number of input features per node.
    /// </summary>
    public int InputFeatures { get; private set; }

    /// <summary>
    /// Gets the number of output features per node.
    /// </summary>
    public int OutputFeatures { get; private set; }

    /// <summary>
    /// Stores the last computed graph smoothness loss for diagnostic purposes.
    /// </summary>
    private T _lastGraphSmoothnessLoss;

    /// <summary>
    /// The weight tensor that transforms input features to output features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the learnable parameters that transform the input features into the output features.
    /// Shape: [inputFeatures, outputFeatures].
    /// </para>
    /// <para><b>For Beginners:</b> Think of weights as the "importance factors" for each feature.
    ///
    /// These weights determine:
    /// - How much attention to pay to each input feature
    /// - How to combine features to create new, meaningful outputs
    /// - The patterns the layer is looking for in the data
    ///
    /// During training, these weights are adjusted to help the network make better predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights;

    /// <summary>
    /// The bias tensor that is added to the output of the transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the learnable bias parameters that are added to the output of the transformation.
    /// Shape: [outputFeatures].
    /// Adding a bias allows the layer to shift the activation function's output.
    /// </para>
    /// <para><b>For Beginners:</b> The bias is like a "default value" or "starting point" for each output.
    ///
    /// It helps the layer by:
    /// - Allowing outputs to be non-zero even when inputs are zero
    /// - Giving the model flexibility to fit data better
    /// - Providing an adjustable "baseline" for predictions
    ///
    /// Think of it as setting the initial position before fine-tuning.
    /// </para>
    /// </remarks>
    private Tensor<T> _bias;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stores the output tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// The adjacency matrix that defines the graph structure.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor represents the connections between nodes in the graph. A non-zero value at position [i,j]
    /// indicates that node i is connected to node j. The adjacency matrix must be set before calling Forward.
    /// </para>
    /// <para><b>For Beginners:</b> The adjacency matrix is like a map of connections in your data.
    /// 
    /// Imagine a map showing which cities have roads between them:
    /// - A value of 1 means "there is a direct connection"
    /// - A value of 0 means "there is no direct connection"
    /// - Other values can represent the "strength" of connections
    /// 
    /// This matrix tells the layer which nodes should share information with each other.
    /// </para>
    /// </remarks>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached reshaped adjacency matrix (3D) for backward pass.
    /// </summary>
    private Tensor<T>? _adjForBatch;

    /// <summary>
    /// Edge source node indices for sparse graph representation.
    /// When set via SetEdges, enables memory-efficient scatter-based aggregation.
    /// </summary>
    private Tensor<int>? _edgeSourceIndices;

    /// <summary>
    /// Edge target node indices for sparse graph representation.
    /// When set via SetEdges, enables memory-efficient scatter-based aggregation.
    /// </summary>
    private Tensor<int>? _edgeTargetIndices;

    /// <summary>
    /// Indicates whether to use sparse (edge-based) or dense (adjacency matrix) aggregation.
    /// </summary>
    private bool _useSparseAggregation = false;

    /// <summary>
    /// Stores the gradients for the weights calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Stores the gradients for the bias calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _biasGradient;

    /// <summary>
    /// Stores the node features from the last forward pass for auxiliary loss computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output node features after the graph convolution operation.
    /// These features are used to compute the Laplacian smoothness regularization loss,
    /// which encourages connected nodes to have similar feature representations.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the features for each node after processing.
    ///
    /// The node features:
    /// - Represent the learned characteristics of each node in the graph
    /// - Are used to compute regularization losses
    /// - Help ensure that connected nodes have similar representations
    ///
    /// This is useful for graph-based learning where we want neighboring nodes
    /// to have similar properties while still maintaining their unique characteristics.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastNodeFeatures;

    /// <summary>
    /// Stores the list of edges in the graph for auxiliary loss computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores pairs of node indices representing edges in the graph.
    /// Each tuple contains (sourceNode, targetNode) indices. These edges are extracted
    /// from the adjacency matrix and used to compute Laplacian smoothness regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This stores which nodes are connected to each other.
    ///
    /// The edge list:
    /// - Contains pairs of node indices that are connected
    /// - Is derived from the adjacency matrix
    /// - Is used to compute smoothness regularization
    ///
    /// For example, if nodes 0 and 2 are connected, the list would include (0, 2).
    /// This helps the layer encourage connected nodes to have similar features.
    /// </para>
    /// </remarks>
    private List<(int Source, int Target)>? _graphEdges;

    /// <summary>
    /// Tracks whether edges have been extracted from the current adjacency matrix.
    /// </summary>
    private bool _edgesExtracted = false;

    /// <summary>
    /// Gets or sets the weight for Laplacian smoothness regularization.
    /// </summary>
    /// <value>
    /// The weight to apply to the smoothness loss. Default is 0.001.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property controls the strength of Laplacian smoothness regularization applied to
    /// node features. Higher values encourage more similar representations for connected nodes,
    /// while lower values allow more variation between neighbors.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly to encourage smooth features across edges.
    ///
    /// Smoothness regularization:
    /// - Encourages connected nodes to have similar features
    /// - Helps the network learn coherent representations across the graph
    /// - Can improve generalization by enforcing local consistency
    ///
    /// Typical values range from 0.0001 to 0.01. Set to 0 to disable smoothness regularization.
    /// </para>
    /// </remarks>
    public T SmoothnessWeight { get; set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The GraphConvolutionalLayer always returns true because it contains trainable weights and biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// This layer always supports training because it has weights and biases that can be updated.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    public override int ParameterCount => _weights.Length + _bias.Length;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphConvolutionalLayer{T}"/> class with the specified dimensions and activation function.
    /// </summary>
    /// <param name="inputFeatures">The number of features in the input data for each node.</param>
    /// <param name="outputFeatures">The number of features to output for each node.</param>
    /// <param name="activationFunction">The activation function to apply after the convolution. Defaults to identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Graph Convolutional Layer with randomly initialized weights and zero biases.
    /// The activation function is applied element-wise to the output of the convolution operation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new layer with specific input and output sizes.
    /// 
    /// When you create this layer, you specify:
    /// - How many features each node in your graph has (inputFeatures)
    /// - How many features you want in the output for each node (outputFeatures)
    /// - An optional activation function that adds non-linearity (making the network more powerful)
    /// 
    /// For example, if your graph represents molecules where each atom has 8 features, and you want
    /// to transform this into 16 features per atom, you would use inputFeatures=8 and outputFeatures=16.
    /// </para>
    /// </remarks>
    public GraphConvolutionalLayer(int inputFeatures, int outputFeatures, IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        if (inputFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputFeatures), "Input features must be positive.");
        }

        if (outputFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputFeatures), "Output features must be positive.");
        }

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastGraphSmoothnessLoss = NumOps.Zero;

        _weights = new Tensor<T>([inputFeatures, outputFeatures]);
        _bias = new Tensor<T>([outputFeatures]);

        SmoothnessWeight = NumOps.FromDouble(0.001);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphConvolutionalLayer{T}"/> class with the specified dimensions and vector activation function.
    /// </summary>
    /// <param name="inputFeatures">The number of features in the input data for each node.</param>
    /// <param name="outputFeatures">The number of features to output for each node.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after the convolution. Defaults to identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Graph Convolutional Layer with randomly initialized weights and zero biases.
    /// The vector activation function is applied to vectors of output features rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different outputs.
    /// </para>
    /// </remarks>
    public GraphConvolutionalLayer(int inputFeatures, int outputFeatures, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputFeatures], [outputFeatures], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        if (inputFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputFeatures), "Input features must be positive.");
        }

        if (outputFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputFeatures), "Output features must be positive.");
        }

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastGraphSmoothnessLoss = NumOps.Zero;

        _weights = new Tensor<T>([inputFeatures, outputFeatures]);
        _bias = new Tensor<T>([outputFeatures]);

        SmoothnessWeight = NumOps.FromDouble(0.001);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using a scaled random initialization scheme, which helps with 
    /// training stability. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the layer's weights and biases.
    /// 
    /// For weights:
    /// - Values are randomized to break symmetry (prevent all neurons from learning the same thing)
    /// - The scale factor helps prevent the signals from growing too large or too small during forward and backward passes
    /// - This specific method is designed to work well for many types of neural networks
    /// 
    /// For biases:
    /// - All values start at zero
    /// - They will adjust during training to fit the data better
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_weights.Shape[0] + _weights.Shape[1])));
        InitializeTensor(_weights, scale);

        _bias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the provided tensor with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a tensor with random values for starting weights.
    ///
    /// The method:
    /// - Generates random numbers between -0.5 and 0.5
    /// - Multiplies them by a scale factor to control their size
    /// - Fills each position in the tensor with these scaled random values
    ///
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        // Create random tensor using Engine operations
        var randomTensor = Tensor<T>.CreateRandom(tensor.Shape);

        // Shift to [-0.5, 0.5] range: randomTensor - 0.5
        var halfTensor = new Tensor<T>(tensor.Shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the scale factor
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        // Copy to tensor
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = scaled.GetFlat(i);
        }
    }

    /// <summary>
    /// Sets the adjacency matrix that defines the graph structure.
    /// </summary>
    /// <param name="adjacencyMatrix">The adjacency matrix tensor.</param>
    /// <remarks>
    /// <para>
    /// This method sets the adjacency matrix that defines the graph structure. The adjacency matrix must be set
    /// before calling the Forward method. A non-zero value at position [i,j] indicates that node i is connected
    /// to node j. This method also extracts the edge list from the adjacency matrix for use in auxiliary loss
    /// computation.
    /// </para>
    /// <para>
    /// <b>Important Limitation:</b> Edge extraction only examines the first batch element (batch index 0).
    /// This assumes all samples in a batch share the same graph structure. If different samples have different
    /// graph topologies, the smoothness loss computation will only reflect the structure of the first sample.
    /// For per-sample graph structures, consider extracting edges dynamically or using separate forward passes.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells the layer how the nodes in your graph are connected.
    ///
    /// The adjacency matrix is like a road map:
    /// - It shows which nodes can directly communicate with each other
    /// - It determines how information flows through your graph
    /// - It must be provided before processing data through the layer
    ///
    /// For example, in a social network, the adjacency matrix would show who is friends with whom.
    /// In a molecule, it would show which atoms are bonded to each other.
    /// </para>
    /// </remarks>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        // Check if we need to re-extract edges (new matrix or first time)
        bool needsExtraction = _adjacencyMatrix != adjacencyMatrix || !_edgesExtracted;

        _adjacencyMatrix = adjacencyMatrix;

        // Extract edges from adjacency matrix for auxiliary loss computation only if needed
        // We only extract edges from the first batch (assuming all batches have the same graph structure)
        if (needsExtraction)
        {
            _graphEdges = new List<(int, int)>();

            if (adjacencyMatrix.Shape.Length >= 3)
            {
                int numNodes = adjacencyMatrix.Shape[1];
                for (int i = 0; i < numNodes; i++)
                {
                    for (int j = 0; j < numNodes; j++)
                    {
                        // Check if there's an edge between nodes i and j
                        if (!MathHelper.AlmostEqual(adjacencyMatrix[0, i, j], NumOps.Zero))
                        {
                            _graphEdges.Add((i, j));
                        }
                    }
                }
            }

            _edgesExtracted = true;
        }
    }

    /// <summary>
    /// Gets the adjacency matrix currently being used by this layer.
    /// </summary>
    /// <returns>The adjacency matrix tensor, or null if not set.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves the adjacency matrix that was set using SetAdjacencyMatrix.
    /// It may return null if the adjacency matrix has not been set yet.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you check what graph structure the layer is using.
    ///
    /// This can be useful for:
    /// - Verifying the correct graph was loaded
    /// - Debugging graph connectivity issues
    /// - Visualizing the graph structure
    /// </para>
    /// </remarks>
    public Tensor<T>? GetAdjacencyMatrix()
    {
        return _adjacencyMatrix;
    }
    /// <summary>
    /// Sets the edge list representation of the graph structure for sparse aggregation.
    /// </summary>
    /// <param name="sourceIndices">Tensor containing source node indices for each edge. Shape: [numEdges].</param>
    /// <param name="targetIndices">Tensor containing target node indices for each edge. Shape: [numEdges].</param>
    /// <remarks>
    /// <para>
    /// This method provides an edge-list representation of the graph, enabling memory-efficient
    /// sparse aggregation using scatter operations. This is the recommended approach for production
    /// GNN workloads, especially for large sparse graphs where O(E) complexity is much better than
    /// O(N^2) dense adjacency matrix operations.
    /// </para>
    /// </remarks>
    public void SetEdges(Tensor<int> sourceIndices, Tensor<int> targetIndices)
    {
        if (sourceIndices == null)
            throw new ArgumentNullException(nameof(sourceIndices));

        if (targetIndices == null)
            throw new ArgumentNullException(nameof(targetIndices));

        if (sourceIndices.Length != targetIndices.Length)
            throw new ArgumentException($"Source and target index tensors must have the same length. Got {sourceIndices.Length} and {targetIndices.Length}.");

        _edgeSourceIndices = sourceIndices;
        _edgeTargetIndices = targetIndices;
        _useSparseAggregation = true;

        _graphEdges = new List<(int, int)>();
        for (int i = 0; i < sourceIndices.Length; i++)
        {
            _graphEdges.Add((sourceIndices.GetFlat(i), targetIndices.GetFlat(i)));
        }
        _edgesExtracted = true;
    }

    /// <summary>
    /// Gets whether sparse (edge-based) aggregation is currently enabled.
    /// </summary>
    public bool UsesSparseAggregation => _useSparseAggregation;

    /// <summary>
    /// Clears the edge list and switches back to dense adjacency matrix aggregation.
    /// </summary>
    public void ClearEdges()
    {
        _edgeSourceIndices = null;
        _edgeTargetIndices = null;
        _useSparseAggregation = false;
    }


    /// <summary>
    /// Performs the forward pass of the graph convolutional layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after graph convolution and activation.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the adjacency matrix has not been set.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the graph convolutional layer according to the formula:
    /// output = activation(adjacency_matrix * input * weights + bias). The input tensor should have shape
    /// [batchSize, numNodes, inputFeatures], and the output will have shape [batchSize, numNodes, outputFeatures].
    /// </para>
    /// <para><b>For Beginners:</b> This method processes data through the graph convolutional layer.
    /// 
    /// During the forward pass:
    /// 1. The layer checks if you've provided a map of connections (adjacency matrix)
    /// 2. It multiplies the input features by the weights to transform them
    /// 3. It uses the adjacency matrix to gather information from connected nodes
    /// 4. It adds a bias value to each output
    /// 5. It applies an activation function to add non-linearity
    /// 
    /// This process allows each node to update its features based on both its own data
    /// and data from its neighbors in the graph.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Check that either adjacency matrix or edge indices are set
        if (_adjacencyMatrix == null && !_useSparseAggregation)
        {
            throw new InvalidOperationException("Graph structure must be set using SetAdjacencyMatrix or SetEdges before calling Forward.");
        }

        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims for rank > 3
        Tensor<T> processInput;
        int batchSize;

        if (rank == 2)
        {
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0], input.Shape[1]]);
        }
        else if (rank == 3)
        {
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        _lastInput = processInput;
        int numNodes = processInput.Shape[1];
        int outputFeatures = _weights.Shape[1];

        // Perform graph convolution: A * X * W
        // First: X * W using reshape pattern for 3D @ 2D
        var xw = BatchedMatMul3Dx2D(processInput, _weights, batchSize, numNodes, processInput.Shape[2], outputFeatures);

        Tensor<T> output;

        if (_useSparseAggregation && _edgeSourceIndices != null && _edgeTargetIndices != null)
        {
            // Sparse aggregation using scatter operations (production-recommended for large graphs)
            // For each batch, gather source features and scatter-add to target nodes
            output = new Tensor<T>([batchSize, numNodes, outputFeatures]);
            output.Fill(NumOps.Zero);

            int numEdges = _edgeSourceIndices.Length;

            for (int b = 0; b < batchSize; b++)
            {
                // Get the slice for this batch: [numNodes, outputFeatures]
                var batchSlice = xw.Slice(b);

                // Gather source node features for all edges using Engine.Gather
                // messages[e, :] = batchSlice[src[e], :]
                var messages = Engine.Gather(batchSlice, _edgeSourceIndices, axis: 0);

                // Scatter-add messages to target nodes
                var aggregated = Engine.ScatterAdd(messages, _edgeTargetIndices, dim: 0, outputSize: numNodes);

                // Set the aggregated results in output tensor for this batch
                output.SetSlice(b, aggregated);
            }

            // Store for backward pass (null for sparse path)
            _adjForBatch = null;
        }
        else
        {
            // Dense aggregation using adjacency matrix multiplication
            if (_adjacencyMatrix == null)
            {
                throw new InvalidOperationException(
                    "Adjacency matrix is required for dense aggregation. " +
                    "Either call SetAdjacencyMatrix() or use SetEdges() for sparse aggregation.");
            }

            // Ensure adjacency matrix has matching rank for batch operation
            // If adjacency is 2D and xw is 3D, broadcast adjacency to 3D
            Tensor<T> adjForBatch;
            if (_adjacencyMatrix.Shape.Length == 2 && xw.Shape.Length == 3)
            {
                // Reshape 2D adjacency [nodes, nodes] to 3D [1, nodes, nodes] and broadcast to [batchSize, nodes, nodes]
                if (batchSize == 1)
                {
                    adjForBatch = _adjacencyMatrix.Reshape([1, numNodes, numNodes]);
                }
                else
                {
                    // Broadcast: repeat adjacency matrix for each batch item using Engine.TensorTile
                    // First reshape to [1, nodes, nodes] then tile along batch dimension
                    var adjReshaped = _adjacencyMatrix.Reshape([1, numNodes, numNodes]);
                    adjForBatch = Engine.TensorTile(adjReshaped, [batchSize, 1, 1]);
                }
            }
            else
            {
                adjForBatch = _adjacencyMatrix;
            }

            // Store for backward pass
            _adjForBatch = adjForBatch;

            // Then: A * (X * W) using batched matmul for 3D @ 3D
            output = Engine.BatchMatMul(adjForBatch, xw);
        }

        // Add bias by broadcasting across batch and node dimensions
        var biasBroadcast = BroadcastBias(_bias, batchSize, numNodes);
        output = Engine.TensorAdd(output, biasBroadcast);

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
            // Store node features for auxiliary loss computation
            _lastNodeFeatures = result;
        }

        // Restore original shape for any-rank tensor support
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 2)
            {
                // Was 2D, return [nodes, outputFeatures]
                return result.Reshape([numNodes, outputFeatures]);
            }
            else
            {
                // Higher-rank: restore leading dimensions
                var newShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 1; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 1] = outputFeatures;
                return result.Reshape(newShape);
            }
        }

        return result;
    }

    /// <summary>
    /// Broadcasts a bias tensor across batch and node dimensions.
    /// </summary>
    /// <param name="bias">The bias tensor of shape [outputFeatures].</param>
    /// <param name="batchSize">The batch size for broadcasting.</param>
    /// <param name="numNodes">The number of nodes for broadcasting.</param>
    /// <returns>A tensor of shape [batchSize, numNodes, outputFeatures] with biases broadcast.</returns>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize, int numNodes)
    {
        int outputFeatures = bias.Length;

        // Reshape bias from [outputFeatures] to [1, 1, outputFeatures]
        var biasReshaped = bias.Reshape([1, 1, outputFeatures]);

        // Tile across batch and node dimensions: [batchSize, numNodes, outputFeatures]
        var broadcast = Engine.TensorTile(biasReshaped, [batchSize, numNodes, 1]);

        return broadcast;
    }

    /// <summary>
    /// Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix.
    /// Input: [batch, rows, cols] @ weights: [cols, output_cols] -> [batch, rows, output_cols]
    /// </summary>
    private Tensor<T> BatchedMatMul3Dx2D(Tensor<T> input3D, Tensor<T> weights2D, int batch, int rows, int cols, int outputCols)
    {
        // Flatten batch dimension: [batch, rows, cols] -> [batch * rows, cols]
        var flattened = input3D.Reshape([batch * rows, cols]);
        // Standard 2D matmul: [batch * rows, cols] @ [cols, output_cols] -> [batch * rows, output_cols]
        var result = Engine.TensorMatMul(flattened, weights2D);
        // Unflatten: [batch * rows, output_cols] -> [batch, rows, output_cols]
        return result.Reshape([batch, rows, outputCols]);
    }

    /// <summary>
    /// Performs the backward pass of the graph convolutional layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the graph convolutional layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients for the weights and
    /// biases, and returns the gradient with respect to the input for further backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output should change to reduce the overall error
    /// 2. It calculates how its weights and biases should change to produce better output
    /// 3. It calculates how its input should change, which will be used by earlier layers
    /// 
    /// This complex calculation considers how information flows through the graph structure
    /// and ensures that connected nodes properly influence each other during learning.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Reshape outputGradient to match _lastOutput's 3D shape if needed
        var gradForBackward = outputGradient;
        if (_originalInputShape != null && _originalInputShape.Length != 3 && outputGradient.Shape.Length != _lastOutput.Shape.Length)
        {
            gradForBackward = outputGradient.Reshape(_lastOutput.Shape);
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, gradForBackward);

        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];
        int inputFeatures = _lastInput.Shape[2];
        int outputFeatures = _weights.Shape[1];

        // Calculate bias gradient using ReduceSum: sum over batch and nodes (axes 0 and 1)
        _biasGradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        // Calculate weights gradient: dW = sum_b (X^T @ A^T @ dY)
        // For batched: we need to transpose adjacency and input, then multiply
        // dW = sum over batches of: input[b]^T @ (adj[b]^T @ actGrad[b])
        _weightsGradient = new Tensor<T>([inputFeatures, outputFeatures]);
        _weightsGradient.Fill(NumOps.Zero);

        // Use the stored reshaped adjacency matrix for backward pass
        var adjBatched = _adjForBatch ?? _adjacencyMatrix;

        // Transpose adjacency matrix (batched transpose - swap last two dims for each batch element)
        var adjTransposed = Engine.TensorPermute(adjBatched, [0, 2, 1]);

        // For each batch, compute: input^T @ adj^T @ activationGradient
        // This equals: (adj @ input)^T @ activationGradient for forward was: A @ X @ W
        // Gradient: dW = X^T @ A^T @ dY (summed over batches)
        for (int b = 0; b < batchSize; b++)
        {
            // Extract batch slices
            var inputBatch = Engine.TensorSlice(_lastInput, [b, 0, 0], [1, numNodes, inputFeatures]).Reshape([numNodes, inputFeatures]);
            var adjTBatch = Engine.TensorSlice(adjTransposed, [b, 0, 0], [1, numNodes, numNodes]).Reshape([numNodes, numNodes]);
            var gradBatch = Engine.TensorSlice(activationGradient, [b, 0, 0], [1, numNodes, outputFeatures]).Reshape([numNodes, outputFeatures]);

            // inputT @ adjT @ gradBatch
            var inputT = Engine.TensorTranspose(inputBatch); // [inputFeatures, numNodes]
            var adjTGrad = Engine.TensorMatMul(adjTBatch, gradBatch); // [numNodes, outputFeatures]
            var batchWeightGrad = Engine.TensorMatMul(inputT, adjTGrad); // [inputFeatures, outputFeatures]

            _weightsGradient = Engine.TensorAdd(_weightsGradient, batchWeightGrad);
        }

        // Calculate input gradient: dX = A^T @ dY @ W^T
        var weightsT = Engine.TensorTranspose(_weights); // [outputFeatures, inputFeatures]

        // For batched computation
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        inputGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var adjTBatch = Engine.TensorSlice(adjTransposed, [b, 0, 0], [1, numNodes, numNodes]).Reshape([numNodes, numNodes]);
            var gradBatch = Engine.TensorSlice(activationGradient, [b, 0, 0], [1, numNodes, outputFeatures]).Reshape([numNodes, outputFeatures]);

            // adjT @ gradBatch @ weightsT
            var adjTGrad = Engine.TensorMatMul(adjTBatch, gradBatch); // [numNodes, outputFeatures]
            var inputGradBatch = Engine.TensorMatMul(adjTGrad, weightsT); // [numNodes, inputFeatures]

            // Set slice in inputGradient
            inputGradient = Engine.TensorSetSlice(inputGradient, inputGradBatch.Reshape([1, numNodes, inputFeatures]), [b, 0, 0]);
        }

        // Reshape back to original input shape if it was not 3D
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation with production-grade pattern:
    /// - Uses cached forward pass values for activation derivative computation
    /// - Uses Tensor.FromRowMatrix/FromVector for efficient conversions
    /// - Builds minimal autodiff graph for gradient routing
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Reshape outputGradient to match _lastOutput's 3D shape if needed
        var gradForBackward = outputGradient;
        if (_originalInputShape != null && _originalInputShape.Length != 3 && outputGradient.Shape.Length != _lastOutput.Shape.Length)
        {
            gradForBackward = outputGradient.Reshape(_lastOutput.Shape);
        }

        // Production-grade: Compute activation derivative using cached output
        Tensor<T> preActivationGradient;
        if (VectorActivation != null)
        {
            var actDeriv = VectorActivation.Derivative(_lastOutput);
            preActivationGradient = Engine.TensorMultiply(gradForBackward, actDeriv);
        }
        else if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            var activation = ScalarActivation;
            var activationDerivative = _lastOutput.Transform((x, _) => activation.Derivative(x));
            preActivationGradient = Engine.TensorMultiply(gradForBackward, activationDerivative);
        }
        else
        {
            preActivationGradient = gradForBackward;
        }

        // Create computation nodes (weights/bias already Tensor<T>)
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var adjNode = Autodiff.TensorOperations<T>.Variable(_adjacencyMatrix, "adjacency", requiresGradient: false);
        var weightsNode = Autodiff.TensorOperations<T>.Variable(_weights, "weights", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_bias, "bias", requiresGradient: true);

        // Build minimal autodiff graph for linear operations (activation derivative already applied)
        var preActivationNode = Autodiff.TensorOperations<T>.GraphConv(inputNode, adjNode, weightsNode, biasNode);

        // Set gradient on pre-activation node (activation derivative already applied)
        preActivationNode.Gradient = preActivationGradient;

        // Inline topological sort and backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((preActivationNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients (already Tensor<T>)
        _weightsGradient = weightsNode.Gradient;
        _biasGradient = biasNode.Gradient;

        // Return input gradient
        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");

        // Reshape back to original input shape if it was not 3D
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the layer based on the gradients calculated during the
    /// backward pass. The learning rate controls the size of the parameter updates. This is typically called
    /// after the backward pass during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - The weights and biases are adjusted to reduce prediction errors
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually improving
    /// its ability to extract useful patterns from graph-structured data.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Use Engine operations for parameter updates
        var scaledWeightsGrad = Engine.TensorMultiplyScalar(_weightsGradient, learningRate);
        _weights = Engine.TensorSubtract(_weights, scaledWeightsGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasGradient, learningRate);
        _bias = Engine.TensorSubtract(_bias, scaledBiasGrad);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) and combines them into a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving and loading
    /// model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include weights and biases
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector.Concatenate to efficiently combine all parameters
        return Vector<T>.Concatenate(
            new Vector<T>(_weights.ToArray()),
            new Vector<T>(_bias.ToArray())
        );
    }

    /// <summary>
    /// Sets the trainable parameters of the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the weights and biases of the layer from a single vector of parameters. The vector must
    /// have the correct length to match the total number of parameters in the layer. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The first part of the vector is used for the weights
    /// - The second part of the vector is used for the biases
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int weightsSize = _weights.Shape[0] * _weights.Shape[1];
        int biasSize = _bias.Length;
        int totalParams = weightsSize + biasSize;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set weights using Tensor.FromVector
        var weightsParams = parameters.SubVector(index, weightsSize);
        _weights = Tensor<T>.FromVector(weightsParams).Reshape(_weights.Shape);
        index += weightsSize;

        // Set bias using Tensor.FromVector
        var biasParams = parameters.SubVector(index, biasSize);
        _bias = Tensor<T>.FromVector(biasParams);
    }

    /// <summary>
    /// Gets the gradients of all trainable parameters in this layer.
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        if (_weightsGradient == null || _biasGradient == null)
        {
            return new Vector<T>(ParameterCount);
        }

        return Vector<T>.Concatenate(
            new Vector<T>(_weightsGradient.ToArray()),
            new Vector<T>(_biasGradient.ToArray())
        );
    }

    /// <summary>
    /// Clears the stored gradients for this layer.
    /// </summary>
    public override void ClearGradients()
    {
        if (_weightsGradient != null)
        {
            _weightsGradient.Fill(NumOps.Zero);
        }

        if (_biasGradient != null)
        {
            _biasGradient.Fill(NumOps.Zero);
        }
    }

    /// <summary>
    /// Computes the Laplacian smoothness regularization loss on node features.
    /// </summary>
    /// <returns>The Laplacian smoothness loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes graph Laplacian smoothness regularization to encourage connected
    /// nodes to have similar feature representations. The loss is calculated as the sum of
    /// squared L2 distances between features of connected nodes, normalized by the number of edges.
    /// This is equivalent to trace(X^T * L * X) where L is the graph Laplacian matrix (L = D - A).
    /// </para>
    /// <para><b>For Beginners:</b> This method encourages connected nodes to have similar features.
    ///
    /// Laplacian smoothness regularization:
    /// - Measures how different neighboring nodes are from each other
    /// - Adds a penalty when connected nodes have very different features
    /// - Helps the network learn coherent representations across the graph structure
    ///
    /// The process:
    /// 1. For each edge (i, j) in the graph
    /// 2. Calculate the squared distance between node i's features and node j's features
    /// 3. Sum all these distances
    /// 4. Normalize by the number of edges
    ///
    /// A higher loss means connected nodes have very different features.
    /// A lower loss means connected nodes have similar features, indicating a smooth representation.
    ///
    /// This loss encourages the network to learn representations where nearby nodes in the graph
    /// have similar feature vectors, which often improves generalization on graph-structured data.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (_lastNodeFeatures is null || _graphEdges is null || _graphEdges.Count == 0)
        {
            return NumOps.Zero;
        }

        // Compute Laplacian smoothness via tensor ops:
        // For each edge (i,j), compute ||x_i - x_j||^2, weight by adjacency, sum, then normalize.
        // Shapes: _lastNodeFeatures [batch, nodes, features], adjacency [batch?, nodes, nodes]
        int batchSize = _lastNodeFeatures.Shape[0];
        int numNodes = _lastNodeFeatures.Shape[1];

        // Expand adjacency to [batch, nodes, nodes] (if single matrix provided, broadcast first dim)
        Tensor<T> adj;
        if (_adjacencyMatrix != null && _adjacencyMatrix.Shape.Length == 2)
        {
            // [nodes, nodes] -> [batch, nodes, nodes]
            adj = Engine.TensorRepeatElements(_adjacencyMatrix.Reshape([1, numNodes, numNodes]), batchSize, axis: 0);
        }
        else if (_adjacencyMatrix != null)
        {
            adj = _adjacencyMatrix;
        }
        else
        {
            // Default to identity adjacency (tensor built locally)
            var identityTensor = new Tensor<T>([numNodes, numNodes]);
            identityTensor.Fill(NumOps.Zero);
            for (int i = 0; i < numNodes; i++)
            {
                identityTensor[i, i] = NumOps.One;
            }
            adj = Engine.TensorRepeatElements(identityTensor.Reshape([1, numNodes, numNodes]), batchSize, axis: 0);
        }

        // Compute pairwise differences for all node pairs: x_i - x_j
        var features = _lastNodeFeatures; // [B, N, F]
        var featuresI = Engine.TensorRepeatElements(features, numNodes, axis: 1); // [B, N*N, F], but reshape differently
        var featuresJ = Engine.TensorTile(features, new[] { 1, numNodes, 1 });    // [B, N*N, F]

        // We need a consistent ordering for edges: flatten adjacency and align
        var adjFlat = adj.Reshape([batchSize, numNodes * numNodes, 1]); // [B, N*N, 1]

        // Compute squared L2 per edge: sum over features
        var diff = Engine.TensorSubtract(featuresI, featuresJ);           // [B, N*N, F]
        var diffSquared = Engine.TensorMultiply(diff, diff);             // [B, N*N, F]
        var squaredDistance = Engine.ReduceSum(diffSquared, new[] { 2 }, keepDims: false); // [B, N*N]

        // Weighted by adjacency
        var weighted = Engine.TensorMultiply(squaredDistance, adjFlat.Reshape([batchSize, numNodes * numNodes])); // [B, N*N]

        // Sum all edges per batch
        var sumPerBatch = Engine.ReduceSum(weighted, new[] { 1 }, keepDims: false); // [B]

        // Total weight per batch for normalization
        var totalWeightPerBatch = Engine.ReduceSum(adjFlat.Reshape([batchSize, numNodes * numNodes]), new[] { 1 }, keepDims: false); // [B]

        // Avoid divide by zero by max with epsilon
        var epsilon = NumOps.FromDouble(1e-10);
        var safeWeights = Engine.TensorMax(totalWeightPerBatch, epsilon);
        var normalized = Engine.TensorDivide(sumPerBatch, safeWeights); // [B]

        // Mean across batch
        var meanLoss = Engine.ReduceMean(normalized, new[] { 0 }, keepDims: false); // scalar tensor
        _lastGraphSmoothnessLoss = NumOps.Multiply(meanLoss.GetFlat(0), SmoothnessWeight);
        return _lastGraphSmoothnessLoss;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// This is useful when starting to process a new sequence or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    ///
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Gradient information is cleared
    /// - The layer forgets any information from previous data
    ///
    /// This is important for:
    /// - Processing a new, unrelated graph
    /// - Preventing information from one training batch affecting another
    /// - Starting a new training episode
    ///
    /// For example, if you've processed one graph and want to start with a new graph,
    /// you should reset the state to prevent the new graph from being influenced by the previous one.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastNodeFeatures = null;
        _adjForBatch = null;
        _weightsGradient = null;
        _biasGradient = null;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about the auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method returns diagnostic information that can be used to monitor the auxiliary loss during training.
    /// The diagnostics include the total smoothness loss, the weight applied to it, and whether auxiliary loss is enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information to help you understand how the auxiliary loss is working.
    ///
    /// The diagnostics show:
    /// - TotalSmoothnessLoss: The computed penalty for feature differences between connected nodes
    /// - SmoothnessWeight: How much this penalty affects the overall training
    /// - UseSmoothnessLoss: Whether this penalty is currently enabled
    ///
    /// You can use this information to:
    /// - Monitor if the smoothness penalty is too high or too low
    /// - Debug training issues
    /// - Understand how the graph structure affects learning
    ///
    /// Example: If TotalSmoothnessLoss is very high, it might mean your network is learning very different
    /// features for connected nodes, which might indicate the need to adjust hyperparameters.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalSmoothnessLoss", $"{_lastGraphSmoothnessLoss}" },
            { "SmoothnessWeight", $"{SmoothnessWeight}" },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
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
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_weights == null || _bias == null)
            throw new InvalidOperationException("Layer not initialized. Call Initialize() first.");

        if (_adjacencyMatrix == null)
            throw new InvalidOperationException("Adjacency matrix not set. Call SetAdjacencyMatrix() first.");

        // Create symbolic input [numNodes, inputFeatures]
        var symbolicInput = new Tensor<T>([1, .. InputShape]);
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Convert adjacency matrix to constant node
        var adjNode = TensorOperations<T>.Constant(_adjacencyMatrix, "adjacency");

        // Weights are already Tensor<T>, use them directly
        var weightsNode = TensorOperations<T>.Constant(_weights, "weights");

        // Use GraphConv operation: output = adjacency @ input @ weights
        var convOutput = TensorOperations<T>.GraphConv(inputNode, adjNode, weightsNode);

        // Bias is already Tensor<T>, use directly
        var biasNode = TensorOperations<T>.Constant(_bias, "bias");
        var output = TensorOperations<T>.Add(convOutput, biasNode);

        // Apply activation if present
        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
        {
            output = ScalarActivation.ApplyToGraph(output);
        }
        else if (VectorActivation != null && VectorActivation.SupportsJitCompilation)
        {
            output = VectorActivation.ApplyToGraph(output);
        }

        return output;
    }

    public override bool SupportsJitCompilation => _weights != null && _bias != null && _adjacencyMatrix != null;

}
