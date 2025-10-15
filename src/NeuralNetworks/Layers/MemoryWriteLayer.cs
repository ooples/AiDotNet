namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that writes to a memory tensor using an attention mechanism.
/// </summary>
/// <remarks>
/// <para>
/// The MemoryWriteLayer implements a form of attention-based memory writing. It computes attention scores
/// between the input and memory tensors, using these scores to determine where to write new information.
/// This approach allows the layer to selectively update memory based on the current input. The layer uses
/// a query-key-value attention mechanism where queries and keys determine where to write, and values determine
/// what to write.
/// </para>
/// <para><b>For Beginners:</b> This layer helps a neural network store information in memory.
/// 
/// Think of it like deciding what to write in a notebook:
/// - You have some new information (your current input)
/// - You have a notebook with existing notes (your memory)
/// - The layer decides which pages of the notebook are relevant to your new information
/// - It then writes the new information on those pages, focusing more on the most relevant ones
/// 
/// For example, if your input represents new information about "France has a beautiful capital city",
/// the layer would focus on memory locations related to France and update them with this new information.
/// 
/// This is similar to how we humans selectively update our memories with new information, rather than
/// storing everything in completely new locations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MemoryWriteLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight matrix used to transform the input into query vectors.
    /// </summary>
    /// <remarks>
    /// This matrix transforms the input vector into query vectors used to determine where to write in memory.
    /// </remarks>
    private Matrix<T> _queryWeights = default!;
    
    /// <summary>
    /// The weight matrix used to transform the input into key vectors.
    /// </summary>
    /// <remarks>
    /// This matrix transforms the input vector into key vectors used with memory keys for attention calculation.
    /// </remarks>
    private Matrix<T> _keyWeights = default!;
    
    /// <summary>
    /// The weight matrix used to transform the input into value vectors.
    /// </summary>
    /// <remarks>
    /// This matrix transforms the input vector into value vectors that determine what to write to memory.
    /// </remarks>
    private Matrix<T> _valueWeights = default!;
    
    /// <summary>
    /// The weight matrix applied to the output after value transformation.
    /// </summary>
    /// <remarks>
    /// This matrix applies a final transformation to the output before adding the bias.
    /// </remarks>
    private Matrix<T> _outputWeights = default!;
    
    /// <summary>
    /// The bias vector added to the output.
    /// </summary>
    /// <remarks>
    /// This vector is added to the output after all weight transformations.
    /// </remarks>
    private Vector<T> _outputBias = default!;

    /// <summary>
    /// The input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastInput;
    
    /// <summary>
    /// The memory tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the memory tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastMemory;
    
    /// <summary>
    /// The output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the output tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastOutput;
    
    /// <summary>
    /// The attention scores tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the attention scores from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastAttentionScores;

    /// <summary>
    /// The gradient of the loss with respect to the query weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the query weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Matrix<T>? _queryWeightsGradient;
    
    /// <summary>
    /// The gradient of the loss with respect to the key weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the key weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Matrix<T>? _keyWeightsGradient;
    
    /// <summary>
    /// The gradient of the loss with respect to the value weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the value weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Matrix<T>? _valueWeightsGradient;
    
    /// <summary>
    /// The gradient of the loss with respect to the output weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the output weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Matrix<T>? _outputWeightsGradient;
    
    /// <summary>
    /// The gradient of the loss with respect to the output bias.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the output bias, which is used to update the bias
    /// during the parameter update step.
    /// </remarks>
    private Vector<T>? _outputBiasGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because the MemoryWriteLayer has trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that MemoryWriteLayer can be trained through backpropagation. The layer
    /// has trainable parameters (weights and biases) that are updated during training to optimize
    /// the memory writing process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values (weights and biases) that change during training
    /// - It will improve its performance as it sees more data
    /// - It learns to better focus attention on relevant parts of memory for writing
    /// 
    /// During training, the layer learns:
    /// - Which features in the input are important for determining where to write
    /// - How to transform input information into memory updates
    /// - How to selectively update memory instead of overwriting everything
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryWriteLayer{T}"/> class with the specified dimensions
    /// and a scalar activation function.
    /// </summary>
    /// <param name="inputDimension">The size of the input vector.</param>
    /// <param name="memoryDimension">The size of each memory entry.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MemoryWriteLayer with the specified dimensions and activation function.
    /// The layer is initialized with random weights scaled according to the layer dimensions to facilitate
    /// stable training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary dimensions and activation function.
    /// 
    /// When creating a MemoryWriteLayer, you need to specify:
    /// - inputDimension: The size of your input vector (e.g., 128 for a 128-feature input)
    /// - memoryDimension: The size of each memory entry (e.g., 256 for memory entries with 256 features)
    /// - activationFunction: The function that processes the final output (optional)
    /// 
    /// The constructor creates weight matrices of the appropriate sizes and initializes them
    /// with small random values to start the learning process. The initialization scale
    /// is carefully chosen to prevent training issues like vanishing or exploding gradients.
    /// </para>
    /// </remarks>
    public MemoryWriteLayer(int inputDimension, int memoryDimension, IActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [memoryDimension], activationFunction ?? new IdentityActivation<T>())
    {
        _queryWeights = new Matrix<T>(inputDimension, memoryDimension);
        _keyWeights = new Matrix<T>(inputDimension, memoryDimension);
        _valueWeights = new Matrix<T>(inputDimension, memoryDimension);
        _outputWeights = new Matrix<T>(memoryDimension, memoryDimension);
        _outputBias = new Vector<T>(memoryDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryWriteLayer{T}"/> class with the specified dimensions
    /// and a vector activation function.
    /// </summary>
    /// <param name="inputDimension">The size of the input vector.</param>
    /// <param name="memoryDimension">The size of each memory entry.</param>
    /// <param name="activationFunction">The vector activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MemoryWriteLayer with the specified dimensions and vector activation function.
    /// A vector activation function operates on entire vectors rather than individual elements.
    /// The layer is initialized with random weights scaled according to the layer dimensions to facilitate
    /// stable training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary dimensions and a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different outputs in your memory writing operation.
    /// </para>
    /// </remarks>
    public MemoryWriteLayer(int inputDimension, int memoryDimension, IVectorActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [memoryDimension], activationFunction ?? new IdentityActivation<T>())
    {
        _queryWeights = new Matrix<T>(inputDimension, memoryDimension);
        _keyWeights = new Matrix<T>(inputDimension, memoryDimension);
        _valueWeights = new Matrix<T>(inputDimension, memoryDimension);
        _outputWeights = new Matrix<T>(memoryDimension, memoryDimension);
        _outputBias = new Vector<T>(memoryDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the layer's weights and biases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using a scaling factor derived from the dimensions
    /// of the weight matrices. The scaling helps prevent vanishing or exploding gradients
    /// during training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's weights and biases.
    /// 
    /// Proper initialization is important for neural networks because:
    /// - Starting with good values helps the network learn faster
    /// - It helps prevent problems during training like vanishing or exploding gradients
    ///   (when values become too small or too large)
    /// 
    /// This method:
    /// - Calculates a scaling factor based on the size of the matrices
    /// - Initializes weights to small random values multiplied by this scale
    /// - Sets all bias values to zero
    /// 
    /// This approach (known as "He initialization") works well for many types of neural networks.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Rows + _queryWeights.Columns)));
        InitializeMatrix(_queryWeights, scale);
        InitializeMatrix(_keyWeights, scale);
        InitializeMatrix(_valueWeights, scale);
        InitializeMatrix(_outputWeights, scale);

        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a matrix with random values scaled by the given factor.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the matrix with random values between -0.5 and 0.5, scaled by the provided factor.
    /// This approach helps to establish good initial conditions for training, especially for deeper networks
    /// where proper weight initialization is crucial for convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a matrix with small random numbers.
    /// 
    /// When initializing a neural network:
    /// - We need to start with random values to break symmetry
    /// - Values that are too large or too small can cause problems
    /// - The scale parameter helps control how large the initial values are
    /// 
    /// This method goes through each position in the matrix and assigns it a random
    /// value between -0.5 and 0.5, multiplied by the scale factor. This gives a
    /// controlled amount of randomness that helps the network start learning effectively.
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
    /// Performs the forward pass of the memory write layer with input and memory tensors.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <param name="memory">The memory tensor to update.</param>
    /// <returns>The output tensor containing updated memory.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the memory write layer. It computes queries, keys, and values
    /// from the input, computes attention scores between the queries and memory, applies softmax to get
    /// attention weights, and then uses these weights to selectively update memory with the computed values.
    /// The method uses a scaled dot-product attention mechanism, dividing the attention scores by the square
    /// root of the key dimension for stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs the actual memory writing operation based on the input.
    /// 
    /// The forward pass works in these steps:
    /// 1. Transform the input into three different representations:
    ///    - Queries: used to match against memory
    ///    - Keys: used to transform the input for attention scoring
    ///    - Values: the actual content to be written to memory
    /// 2. Compare queries with memory to get attention scores
    /// 3. Scale the scores for stability and convert to weights using softmax
    /// 4. Use these weights to determine where to write the values
    /// 5. Apply additional transformations and activation function for the final output
    /// 
    /// This process allows the layer to selectively update different parts of memory
    /// based on the relevance of the current input, rather than writing to all
    /// memory locations equally.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> memory)
    {
        _lastInput = input;
        _lastMemory = memory;

        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        var attentionScores = queries.Multiply(memory.Transpose([1, 0]));
        attentionScores = attentionScores.Multiply(NumOps.FromDouble(1.0 / Math.Sqrt(keys.Shape[1])));

        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        var writeValues = values.Multiply(attentionWeights);
        var output = writeValues.Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    /// <summary>
    /// Performs the backward pass of the memory write layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the memory write layer, which is used during training to propagate
    /// error gradients back through the network. It computes the gradients of all weights and biases, as well as
    /// the gradient with respect to the input tensor. The computed weight and bias gradients are stored for later
    /// use in the parameter update step.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how all parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// - The layer receives gradients indicating how the output (updated memory) should change
    /// - It calculates how each weight, bias, and input value should change
    /// - These gradients are used later to update the parameters during training
    /// 
    /// The backward pass is complex because it needs to:
    /// - Calculate gradients for query, key, and value weights
    /// - Calculate gradients for the output weights and bias
    /// - Handle the chain rule through the softmax attention mechanism
    /// - Combine gradients from multiple paths
    /// 
    /// This process enables the layer to learn more effective memory writing strategies over time.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMemory == null || _lastOutput == null || _lastAttentionScores == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        _outputWeightsGradient = activationGradient.Transpose([1, 0]).Multiply(_lastOutput).ToMatrix();
        _outputBiasGradient = activationGradient.Sum([0]).ToVector();

        var writeValuesGradient = activationGradient.Multiply(_outputWeights.Transpose());

        var softmaxActivation = new SoftmaxActivation<T>();
        var softmaxDerivative = softmaxActivation.Derivative(_lastAttentionScores);
        var attentionWeightsGradient = softmaxDerivative.ElementwiseMultiply(
            writeValuesGradient.Multiply(_lastInput.Multiply(_valueWeights).Transpose([1, 0])));

        var queriesGradient = attentionWeightsGradient.Multiply(_lastMemory);
        var keysGradient = attentionWeightsGradient.Transpose([1, 0]).Multiply(_lastInput);
        var valuesGradient = _lastAttentionScores.Transpose([1, 0]).Multiply(writeValuesGradient);

        _queryWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(queriesGradient).ToMatrix();
        _keyWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(keysGradient).ToMatrix();
        _valueWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(valuesGradient).ToMatrix();

        var inputGradient = queriesGradient.Multiply(_queryWeights.Transpose())
                            .Add(keysGradient.Multiply(_keyWeights.Transpose()))
                            .Add(valuesGradient.Multiply(_valueWeights.Transpose()));

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the memory write layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates all trainable parameters of the layer (query weights, key weights, value weights,
    /// output weights, and output bias) based on the gradients calculated during the backward pass. The learning
    /// rate controls the size of the parameter updates. Each parameter is updated by subtracting the corresponding
    /// gradient multiplied by the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's weights and biases during training.
    /// 
    /// After the backward pass calculates how parameters should change, this method:
    /// - Takes each weight matrix and bias vector
    /// - Subtracts the corresponding gradient scaled by the learning rate
    /// - This moves the parameters in the direction that reduces errors
    /// 
    /// The learning rate controls how big each update step is:
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer gradually improves its memory writing abilities over many training iterations.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null || _outputWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _queryWeights = _queryWeights.Subtract(_queryWeightsGradient.Multiply(learningRate));
        _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
        _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
        _outputWeights = _outputWeights.Subtract(_outputWeightsGradient.Multiply(learningRate));
        _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Performs the forward pass of the memory write layer with just the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing with an empty memory tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base Forward method to handle the case where only an input tensor is provided.
    /// It creates an empty (zero-initialized) memory tensor of the appropriate size and calls the overloaded
    /// Forward method with both the input and the empty memory.
    /// </para>
    /// <para><b>For Beginners:</b> This method handles the case when no existing memory is provided.
    /// 
    /// When you call this method with just an input tensor:
    /// - The layer creates a blank memory tensor filled with zeros
    /// - It then calls the regular Forward method with both your input and this blank memory
    /// - The result is as if you're writing to a fresh, empty memory
    /// 
    /// This is useful when:
    /// - You're starting a new sequence and don't have previous memory
    /// - You want to initialize memory from scratch
    /// - You want to simplify your code by not having to create empty memory yourself
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // For a memory write layer, we need both input and memory
        // When only input is provided, we can create an empty memory tensor
        // or use a zero-initialized memory tensor of appropriate size
        int batchSize = input.Shape[0];
        int memoryDimension = _queryWeights.Columns;
    
        // Create an empty memory tensor with the same batch size as input
        // and the memory dimension of the layer
        var emptyMemory = new Tensor<T>([batchSize, memoryDimension]);
    
        // Initialize with zeros
        for (int i = 0; i < emptyMemory.Length; i++)
        {
            emptyMemory[i] = NumOps.Zero;
        }
    
        // Call the overloaded Forward method with the empty memory
        return Forward(input, emptyMemory);
    }

    /// <summary>
    /// Gets all trainable parameters from the memory write layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. It concatenates
    /// the query weights, key weights, value weights, output weights, and output bias into a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving
    /// and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values in the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include all the weights and biases from this layer
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The method carefully arranges all parameters in a specific order
    /// so they can be correctly restored later.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy query weights
        for (int i = 0; i < _queryWeights.Rows; i++)
        {
            for (int j = 0; j < _queryWeights.Columns; j++)
            {
                parameters[index++] = _queryWeights[i, j];
            }
        }
    
        // Copy key weights
        for (int i = 0; i < _keyWeights.Rows; i++)
        {
            for (int j = 0; j < _keyWeights.Columns; j++)
            {
                parameters[index++] = _keyWeights[i, j];
            }
        }
    
        // Copy value weights
        for (int i = 0; i < _valueWeights.Rows; i++)
        {
            for (int j = 0; j < _valueWeights.Columns; j++)
            {
                parameters[index++] = _valueWeights[i, j];
            }
        }
    
        // Copy output weights
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                parameters[index++] = _outputWeights[i, j];
            }
        }
    
        // Copy output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            parameters[index++] = _outputBias[i];
        }
    
        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters for the memory write layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters of the layer from a single vector. It extracts the appropriate
    /// portions of the input vector for each parameter (query weights, key weights, value weights, output weights,
    /// and output bias). This is useful for loading saved model weights or for implementing optimization algorithms
    /// that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The method extracts portions for each weight matrix and bias vector
    /// - It places each value in its correct position
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters,
    /// ensuring that all matrices and vectors maintain their correct dimensions.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set query weights
        for (int i = 0; i < _queryWeights.Rows; i++)
        {
            for (int j = 0; j < _queryWeights.Columns; j++)
            {
                _queryWeights[i, j] = parameters[index++];
            }
        }
    
        // Set key weights
        for (int i = 0; i < _keyWeights.Rows; i++)
        {
            for (int j = 0; j < _keyWeights.Columns; j++)
            {
                _keyWeights[i, j] = parameters[index++];
            }
        }
    
        // Set value weights
        for (int i = 0; i < _valueWeights.Rows; i++)
        {
            for (int j = 0; j < _valueWeights.Columns; j++)
            {
                _valueWeights[i, j] = parameters[index++];
            }
        }
    
        // Set output weights
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                _outputWeights[i, j] = parameters[index++];
            }
        }
    
        // Set output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the memory write layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the memory write layer, including the cached inputs, memory,
    /// outputs, attention scores, and all gradients. This is useful when starting to process a new sequence
    /// or batch of data, or when implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs, memory, outputs, and attention scores from previous processing are cleared
    /// - All calculated gradients are cleared
    /// - The layer forgets any information from previous data batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Ensuring clean state before a new training epoch
    /// - Preventing information from one batch affecting another
    /// 
    /// Resetting state helps ensure that each forward and backward pass is independent,
    /// which is important for correct behavior in many neural network architectures.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastMemory = null;
        _lastOutput = null;
        _lastAttentionScores = null;
    
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }
}