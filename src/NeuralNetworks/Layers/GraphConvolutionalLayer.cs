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
public class GraphConvolutionalLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight matrix that transforms input features to output features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix contains the learnable parameters that transform the input features into the output features.
    /// The dimensions are [inputFeatures, outputFeatures].
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
    private Matrix<T> _weights = default!;

    /// <summary>
    /// The bias vector that is added to the output of the transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the learnable bias parameters that are added to the output of the transformation.
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
    private Vector<T> _bias = default!;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

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
    /// Stores the gradients for the weights calculated during the backward pass.
    /// </summary>
    private Matrix<T>? _weightsGradient;

    /// <summary>
    /// Stores the gradients for the bias calculated during the backward pass.
    /// </summary>
    private Vector<T>? _biasGradient;

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
        _weights = new Matrix<T>(inputFeatures, outputFeatures);
        _bias = new Vector<T>(outputFeatures);

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
        _weights = new Matrix<T>(inputFeatures, outputFeatures);
        _bias = new Vector<T>(outputFeatures);

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
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_weights.Rows + _weights.Columns)));
        InitializeMatrix(_weights, scale);

        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a matrix with scaled random values.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the provided matrix with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a matrix with random values for starting weights.
    /// 
    /// The method:
    /// - Generates random numbers between -0.5 and 0.5
    /// - Multiplies them by a scale factor to control their size
    /// - Fills each position in the matrix with these scaled random values
    /// 
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// </para>
    /// </remarks>
    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
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
    /// to node j.
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
        _adjacencyMatrix = adjacencyMatrix;
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
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException("Adjacency matrix must be set using the SetAdjacencyMatrix method before calling Forward.");
        }

        _lastInput = input;

        int batchSize = input.Shape[0];
        int numNodes = input.Shape[1];
        int inputFeatures = input.Shape[2];
        int outputFeatures = _weights.Columns;

        // Perform graph convolution: A * X * W
        var xw = input.Multiply(_weights);
        var output = _adjacencyMatrix.Multiply(xw.ToMatrix());

        // Add bias
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < outputFeatures; f++)
                {
                    output[b, n, f] = NumOps.Add(output[b, n, f], _bias[f]);
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
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
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];
        int inputFeatures = _lastInput.Shape[2];
        int outputFeatures = _weights.Columns;

        // Calculate gradients for weights and bias
        _weightsGradient = new Matrix<T>(inputFeatures, outputFeatures);
        _biasGradient = new Vector<T>(outputFeatures);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    for (int f_in = 0; f_in < inputFeatures; f_in++)
                    {
                        for (int f_out = 0; f_out < outputFeatures; f_out++)
                        {
                            T gradValue = NumOps.Multiply(_adjacencyMatrix[b, i, j], 
                                NumOps.Multiply(_lastInput[b, j, f_in], activationGradient[b, i, f_out]));
                            _weightsGradient[f_in, f_out] = NumOps.Add(_weightsGradient[f_in, f_out], gradValue);
                        }
                    }
                }
            }
        }

        // Calculate bias gradient
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < outputFeatures; f++)
                {
                    _biasGradient[f] = NumOps.Add(_biasGradient[f], activationGradient[b, n, f]);
                }
            }
        }

        // Calculate input gradient
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    for (int f_in = 0; f_in < inputFeatures; f_in++)
                    {
                        for (int f_out = 0; f_out < outputFeatures; f_out++)
                        {
                            T gradValue = NumOps.Multiply(_adjacencyMatrix[b, j, i],
                                NumOps.Multiply(activationGradient[b, j, f_out], _weights[f_in, f_out]));
                            inputGradient[b, i, f_in] = NumOps.Add(inputGradient[b, i, f_in], gradValue);
                        }
                    }
                }
            }
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

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
        _bias = _bias.Subtract(_biasGradient.Multiply(learningRate));
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
        // Calculate total number of parameters
        int totalParams = _weights.Rows * _weights.Columns + _bias.Length;
        var parameters = new Vector<T>(totalParams);

        int index = 0;

        // Copy weights parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }

        // Copy bias parameters
        for (int i = 0; i < _bias.Length; i++)
        {
            parameters[index++] = _bias[i];
        }

        return parameters;
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
        if (parameters.Length != _weights.Rows * _weights.Columns + _bias.Length)
        {
            throw new ArgumentException($"Expected {_weights.Rows * _weights.Columns + _bias.Length} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set weights parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }

        // Set bias parameters
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = parameters[index++];
        }
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
        _weightsGradient = null;
        _biasGradient = null;
    }
}