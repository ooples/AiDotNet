namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a self-attention layer that allows a sequence to attend to itself, capturing relationships between elements.
/// </summary>
/// <remarks>
/// <para>
/// The SelfAttentionLayer implements the self-attention mechanism, a key component of transformer architectures.
/// It allows each position in a sequence to attend to all positions within the same sequence, enabling the model
/// to capture long-range dependencies and relationships. The layer uses the scaled dot-product attention mechanism
/// with multiple attention heads, which allows it to focus on different aspects of the input simultaneously.
/// </para>
/// <para><b>For Beginners:</b> This layer helps a neural network understand relationships between different parts of a sequence.
/// 
/// Think of the SelfAttentionLayer like a group of spotlights at a theater performance:
/// - Each spotlight (attention head) can focus on different actors on stage
/// - For each actor, the spotlights decide which other actors are most relevant to them
/// - The spotlights assign importance scores to these relationships
/// - This helps the network understand who is interacting with whom, and how
/// 
/// For example, in a sentence like "The cat sat on the mat because it was tired":
/// - Traditional networks might struggle to figure out what "it" refers to
/// - Self-attention can learn that "it" has a strong relationship with "cat"
/// - This helps the network understand that the cat was tired, not the mat
/// 
/// Multi-head attention (using multiple "spotlights") allows the layer to focus on different types
/// of relationships simultaneously, such as grammatical structure, semantic meaning, and contextual clues.
/// 
/// Self-attention is a cornerstone of modern natural language processing and has revolutionized
/// how neural networks handle sequential data like text, time series, and even images.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SelfAttentionLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Matrix<double> of weights for transforming input embeddings into query vectors.
    /// </summary>
    /// <remarks>
    /// This matrix transforms input embeddings into query vectors, which are used to compute attention scores.
    /// Queries represent what each position in the sequence is looking for in other positions.
    /// </remarks>
    private Matrix<T> _queryWeights = default!;
    
    /// <summary>
    /// Matrix<double> of weights for transforming input embeddings into key vectors.
    /// </summary>
    /// <remarks>
    /// This matrix transforms input embeddings into key vectors, which are used to compute attention scores.
    /// Keys represent what each position in the sequence has to offer to other positions.
    /// </remarks>
    private Matrix<T> _keyWeights = default!;
    
    /// <summary>
    /// Matrix<double> of weights for transforming input embeddings into value vectors.
    /// </summary>
    /// <remarks>
    /// This matrix transforms input embeddings into value vectors, which contain the actual content
    /// that will be aggregated based on attention scores. Values represent the information that
    /// is being extracted from each position.
    /// </remarks>
    private Matrix<T> _valueWeights = default!;
    
    /// <summary>
    /// Vector<double> of biases added to the output of the attention mechanism.
    /// </summary>
    /// <remarks>
    /// This vector contains bias terms that are added to the output of the attention mechanism
    /// before applying the final activation function. Biases allow the network to adjust the
    /// baseline activation level of the attention output.
    /// </remarks>
    private Vector<T> _outputBias = default!;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute gradients. It holds the
    /// sequence of input embeddings that were processed in the most recent forward pass.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastInput;
    
    /// <summary>
    /// Stores the output tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached output is needed during the backward pass to compute certain derivatives.
    /// It holds the sequence of output embeddings that were produced in the most recent forward pass.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastOutput;
    
    /// <summary>
    /// Stores the attention score tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached tensor contains the attention weights (after softmax) computed during the forward pass.
    /// These weights represent how much each position attends to every other position in the sequence.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastAttentionScores;

    /// <summary>
    /// Stores the gradients of the loss with respect to the query weight parameters.
    /// </summary>
    /// <remarks>
    /// This matrix holds the accumulated gradients for the query weight parameters during the backward pass.
    /// It has the same dimensions as the _queryWeights matrix and is used to update the query weights during
    /// the parameter update step. The matrix is null before the first backward pass or after a reset.
    /// </remarks>
    private Matrix<T>? _queryWeightsGradient;
    
    /// <summary>
    /// Stores the gradients of the loss with respect to the key weight parameters.
    /// </summary>
    /// <remarks>
    /// This matrix holds the accumulated gradients for the key weight parameters during the backward pass.
    /// It has the same dimensions as the _keyWeights matrix and is used to update the key weights during
    /// the parameter update step. The matrix is null before the first backward pass or after a reset.
    /// </remarks>
    private Matrix<T>? _keyWeightsGradient;
    
    /// <summary>
    /// Stores the gradients of the loss with respect to the value weight parameters.
    /// </summary>
    /// <remarks>
    /// This matrix holds the accumulated gradients for the value weight parameters during the backward pass.
    /// It has the same dimensions as the _valueWeights matrix and is used to update the value weights during
    /// the parameter update step. The matrix is null before the first backward pass or after a reset.
    /// </remarks>
    private Matrix<T>? _valueWeightsGradient;
    
    /// <summary>
    /// Stores the gradients of the loss with respect to the output bias parameters.
    /// </summary>
    /// <remarks>
    /// This vector holds the accumulated gradients for the output bias parameters during the backward pass.
    /// It has the same length as the _outputBias vector and is used to update the output biases during
    /// the parameter update step. The vector is null before the first backward pass or after a reset.
    /// </remarks>
    private Vector<T>? _outputBiasGradient;

    /// <summary>
    /// The number of attention heads used in the multi-head attention mechanism.
    /// </summary>
    /// <remarks>
    /// This value determines how many different attention patterns the layer can learn simultaneously.
    /// Each head can focus on different relationships within the sequence. More heads can capture more
    /// complex relationships but require more computation.
    /// </remarks>
    private int _headCount;
    
    /// <summary>
    /// The dimension of each attention head.
    /// </summary>
    /// <remarks>
    /// This value determines the size of each attention head, which is typically the embedding dimension
    /// divided by the number of heads. It affects the expressive power of each individual attention head.
    /// </remarks>
    private int _headDimension;
    
    /// <summary>
    /// The dimension of the input and output embeddings.
    /// </summary>
    /// <remarks>
    /// This value determines the size of the input and output embeddings. It is typically the same for
    /// both input and output to maintain the dimensionality of the sequence representation.
    /// </remarks>
    private int _embeddingDimension;
    
    /// <summary>
    /// The length of the input sequence.
    /// </summary>
    /// <remarks>
    /// This value determines the number of positions in the input sequence. Each position will attend
    /// to all other positions in the sequence during the attention computation.
    /// </remarks>
    private int _sequenceLength;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for SelfAttentionLayer, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the SelfAttentionLayer has trainable parameters (query, key, and value weights,
    /// as well as output biases) that can be optimized during the training process using backpropagation. The gradients
    /// of these parameters are calculated during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values (weights and biases) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// When you train a neural network containing this layer, it will automatically learn
    /// which relationships between sequence positions are important for your specific task.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SelfAttentionLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of the input and output embeddings.</param>
    /// <param name="headCount">The number of attention heads. Defaults to 8.</param>
    /// <param name="activationFunction">The activation function to apply to the output. Defaults to Identity if not specified.</param>
    /// <exception cref="ArgumentException">Thrown when the embedding dimension is not divisible by the number of heads.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new SelfAttentionLayer with the specified dimensions and a scalar activation function.
    /// It validates that the embedding dimension is divisible by the number of heads and initializes the weight matrices
    /// and bias vector with appropriate values. A scalar activation function is applied element-wise to each output
    /// embedding independently.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new self-attention layer for your neural network using a simple activation function.
    /// 
    /// When you create this layer, you specify:
    /// - sequenceLength: How many items (like words) are in your sequence
    /// - embeddingDimension: How many features each item has
    /// - headCount: How many different "spotlights" the attention mechanism uses (default: 8)
    /// - activationFunction: How to transform the output (defaults to Identity, which makes no changes)
    /// 
    /// For example, in a language model:
    /// - sequenceLength might be 512 (the maximum number of words/tokens in a text)
    /// - embeddingDimension might be 768 (the number of features per word/token)
    /// - Using 8 attention heads lets the model focus on 8 different types of relationships
    /// 
    /// The embedding dimension must be divisible by the number of heads (e.g., 768 ÷ 8 = 96),
    /// so each head has the same dimension.
    /// </para>
    /// </remarks>
    public SelfAttentionLayer(
        int sequenceLength, 
        int embeddingDimension, 
        int headCount = 8, 
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, embeddingDimension], 
            [sequenceLength, embeddingDimension], 
            activationFunction ?? new IdentityActivation<T>())
    {
        _queryWeights = Matrix<T>.Empty();
        _keyWeights = Matrix<T>.Empty();
        _valueWeights = Matrix<T>.Empty();
        _outputBias = Vector<T>.Empty();

        InitializeLayer(sequenceLength, embeddingDimension, headCount);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SelfAttentionLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of the input and output embeddings.</param>
    /// <param name="headCount">The number of attention heads. Defaults to 8.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply to the output. Defaults to Identity if not specified.</param>
    /// <exception cref="ArgumentException">Thrown when the embedding dimension is not divisible by the number of heads.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new SelfAttentionLayer with the specified dimensions and a vector activation function.
    /// It validates that the embedding dimension is divisible by the number of heads and initializes the weight matrices
    /// and bias vector with appropriate values. A vector activation function is applied to the entire output vector at once,
    /// which allows for interactions between different output elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new self-attention layer for your neural network using an advanced activation function.
    /// 
    /// When you create this layer, you specify the same parameters as in the scalar version, but with a vector activation:
    /// - sequenceLength: How many items are in your sequence
    /// - embeddingDimension: How many features each item has
    /// - headCount: How many different "spotlights" the attention mechanism uses
    /// - vectorActivationFunction: How to transform the entire output as a group
    /// 
    /// A vector activation can consider relationships between different positions in the output,
    /// which might be useful for certain advanced applications.
    /// 
    /// This constructor works the same as the scalar version, but allows for more sophisticated
    /// activation patterns across the output sequence.
    /// </para>
    /// </remarks>
    public SelfAttentionLayer(
        int sequenceLength, 
        int embeddingDimension, 
        int headCount = 8, 
        IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(
            [sequenceLength, embeddingDimension], 
            [sequenceLength, embeddingDimension], 
            vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _queryWeights = Matrix<T>.Empty();
        _keyWeights = Matrix<T>.Empty();
        _valueWeights = Matrix<T>.Empty();
        _outputBias = Vector<T>.Empty();

        InitializeLayer(sequenceLength, embeddingDimension, headCount);
    }

    /// <summary>
    /// Performs the forward pass of the self-attention layer.
    /// </summary>
    /// <param name="input">The input tensor to process, with shape [batchSize, sequenceLength, embeddingDimension].</param>
    /// <returns>The output tensor after self-attention, with the same shape as the input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the self-attention layer. It transforms the input into queries,
    /// keys, and values, then computes attention scores between each position and all other positions. These scores
    /// are normalized using the softmax function and used to compute a weighted sum of the values. The result is
    /// transformed back to the original embedding dimension and passed through an activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your sequence data through the self-attention mechanism.
    /// 
    /// During the forward pass:
    /// 1. The input sequence is transformed into three different representations:
    ///    - Queries: What each position is looking for
    ///    - Keys: What each position has to offer
    ///    - Values: The actual content at each position
    /// 2. For each position, attention scores are computed by comparing its query with all keys
    /// 3. These scores are scaled and normalized to create attention weights
    /// 4. Each position's output is a weighted sum of all values, based on the attention weights
    /// 5. The result is transformed and passed through an activation function
    /// 
    /// Imagine a classroom where each student (position) asks a question (query) to the entire class.
    /// Other students offer answers (keys) and knowledge (values). Each student pays more attention
    /// to the most relevant answers and combines that knowledge to form their own understanding.
    /// 
    /// The multi-head mechanism allows this process to happen in parallel with different "perspectives"
    /// or types of questions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape[1];
        int embeddingDimension = input.Shape[2];

        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        queries = queries.Reshape(batchSize, sequenceLength, _headCount, _headDimension);
        keys = keys.Reshape(batchSize, sequenceLength, _headCount, _headDimension);
        values = values.Reshape(batchSize, sequenceLength, _headCount, _headDimension);

        var attentionScores = queries.Multiply(keys.Reshape(batchSize, sequenceLength, _headDimension, _headCount));
        attentionScores = attentionScores.Multiply(NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension)));

        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        var attentionOutput = attentionWeights.Multiply(values);
        attentionOutput = attentionOutput.Reshape(batchSize, sequenceLength, embeddingDimension);

        var output = attentionOutput.Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    /// <summary>
    /// Performs the backward pass of the self-attention layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the self-attention layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients of the loss
    /// with respect to the layer's parameters (query, key, and value weights, as well as output biases)
    /// and with respect to the layer's input. The calculation involves complex tensor operations that
    /// essentially reverse the computations done in the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the layer's parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives error gradients indicating how the output should change
    /// 2. It calculates how each of its internal components contributed to the error:
    ///    - How the query weights should change
    ///    - How the key weights should change
    ///    - How the value weights should change
    ///    - How the output biases should change
    /// 3. It also calculates how the error should propagate back to the previous layer
    /// 
    /// This involves complex matrix mathematics, but the basic idea is:
    /// - Finding which attention patterns led to errors
    /// - Adjusting the weights to improve these patterns
    /// - Sending appropriate feedback to the previous layer
    /// 
    /// The backward pass is what allows the self-attention mechanism to learn which relationships
    /// in the sequence are important for the specific task.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastAttentionScores == null)
        throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];
        int embeddingDimension = _lastInput.Shape[2];

        var attentionOutputGradient = activationGradient;

        // Sum over batch and sequence dimensions, then convert to Vector<double>
        _outputBiasGradient = attentionOutputGradient.Sum([0, 1]).ToVector();

        // Reshape attentionOutputGradient for multi-head attention
        attentionOutputGradient = attentionOutputGradient.Reshape([batchSize, sequenceLength, _headCount, _headDimension]);
    
        // Transpose to align dimensions for matrix multiplication
        attentionOutputGradient = attentionOutputGradient.Transpose([0, 2, 1, 3]);

        // Calculate gradients for values
        var valuesGradient = _lastAttentionScores.Transpose([0, 1, 3, 2]).Multiply(attentionOutputGradient);

        // Reshape and transpose input for further calculations
        var reshapedLastInput = _lastInput.Reshape([batchSize, sequenceLength, _headCount, _headDimension]);
        var transposedLastInput = reshapedLastInput.Transpose([0, 2, 1, 3]);
        var attentionScoresGradient = attentionOutputGradient.Multiply(transposedLastInput);

        var softmaxActivation = new SoftmaxActivation<T>();
        var softmaxDerivative = softmaxActivation.Derivative(_lastAttentionScores);
        var attentionWeightsGradient = softmaxDerivative.ElementwiseMultiply(attentionScoresGradient);

        var queriesGradient = attentionWeightsGradient.Multiply(reshapedLastInput.Transpose([0, 2, 3, 1]));
        var keysGradient = attentionWeightsGradient.Transpose([0, 1, 3, 2]).Multiply(transposedLastInput);

        queriesGradient = queriesGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        keysGradient = keysGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        valuesGradient = valuesGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);

        // Calculate gradients for the weight matrices
        var batchGradientQ = _lastInput.Transpose([0, 2, 1]).Multiply(queriesGradient);
        var batchGradientK = _lastInput.Transpose([0, 2, 1]).Multiply(keysGradient);
        var batchGradientV = _lastInput.Transpose([0, 2, 1]).Multiply(valuesGradient);

        // Sum over the batch dimension to get the final weight gradients
        _queryWeightsGradient = batchGradientQ.Sum([0]).Reshape([embeddingDimension, embeddingDimension]).ToMatrix();
        _keyWeightsGradient = batchGradientK.Sum([0]).Reshape([embeddingDimension, embeddingDimension]).ToMatrix();
        _valueWeightsGradient = batchGradientV.Sum([0]).Reshape([embeddingDimension, embeddingDimension]).ToMatrix();

        var inputGradient = queriesGradient.Multiply(_queryWeights.Transpose())
                            .Add(keysGradient.Multiply(_keyWeights.Transpose()))
                            .Add(valuesGradient.Multiply(_valueWeights.Transpose()));

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the self-attention layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the query weights, key weights, value weights, and output biases of the self-attention
    /// layer based on the gradients calculated during the backward pass. The learning rate controls the size of the
    /// parameter updates. This method should be called after the backward pass to apply the calculated updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. The query weight values are adjusted based on their gradients
    /// 2. The key weight values are adjusted based on their gradients
    /// 3. The value weight values are adjusted based on their gradients
    /// 4. The output bias values are adjusted based on their gradients
    /// 5. The learning rate controls how big each update step is
    /// 
    /// These updates help the self-attention mechanism:
    /// - Focus on more relevant relationships between positions
    /// - Ignore irrelevant relationships
    /// - Better understand the structure of your sequences
    /// 
    /// Smaller learning rates mean slower but more stable learning, while larger learning rates
    /// mean faster but potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _queryWeights = _queryWeights.Subtract(_queryWeightsGradient.Multiply(learningRate));
        _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
        _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
        _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the self-attention layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (query weights, key weights, value weights, and output biases).</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the self-attention layer as a single vector. The query weights
    /// are stored first, followed by the key weights, value weights, and finally the output biases. This is useful for
    /// optimization algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the self-attention layer.
    /// 
    /// The parameters:
    /// - Are the weights and biases that the self-attention layer learns during training
    /// - Control how the layer processes sequence information
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The query weights are stored first in the vector, followed by the key weights, value weights,
    /// and finally the output biases.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
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
    
        // Copy output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            parameters[index++] = _outputBias[i];
        }
    
        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the self-attention layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (query weights, key weights, value weights, and output biases) to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters of the self-attention layer from a single vector. The vector should
    /// contain the query weight values first, followed by the key weight values, value weight values, and finally
    /// the output bias values. This is useful for loading saved model weights or for implementing optimization
    /// algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the weights and biases in the self-attention layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct total length
    /// - The first part of the vector is used for the query weights
    /// - The second part of the vector is used for the key weights
    /// - The third part of the vector is used for the value weights
    /// - The last part of the vector is used for the output biases
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
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
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
    
        // Set output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the self-attention layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the self-attention layer, including the cached inputs, outputs,
    /// attention scores from the forward pass, and the gradients from the backward pass. This is useful when
    /// starting to process a new batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs, outputs, and attention scores from previous calculations are cleared
    /// - Calculated gradients for all weights and biases are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Managing memory usage efficiently
    /// 
    /// Since the self-attention layer caches quite a bit of information during the forward
    /// and backward passes, resetting the state helps prevent memory leaks and ensures
    /// each new sequence is processed independently.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastAttentionScores = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputBiasGradient = null;
    }

    /// <summary>
    /// Initializes the layer's internal parameters based on the sequence length, embedding dimension, and head count.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of the input and output embeddings.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <exception cref="ArgumentException">Thrown when the embedding dimension is not divisible by the number of heads.</exception>
    /// <remarks>
    /// <para>
    /// This private method initializes the internal parameters of the self-attention layer based on the specified
    /// dimensions. It validates that the embedding dimension is divisible by the number of heads, calculates the
    /// dimension of each head, and then calls InitializeParameters to set up the weight matrices and bias vector.
    /// This method is called by both constructors.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the internal structure of the self-attention layer.
    /// 
    /// During initialization:
    /// - The method saves the basic dimensions (sequence length, embedding size, head count)
    /// - It calculates how large each attention head should be
    /// - It verifies that the embedding dimension can be evenly divided by the head count
    /// - It triggers the creation of all the weight matrices with proper initial values
    /// 
    /// The head dimension calculation is important - if you have an embedding size of 512 and
    /// 8 attention heads, each head will have a dimension of 64 (512 ÷ 8). This allows each
    /// head to specialize in different aspects of the input sequence.
    /// 
    /// This method throws an error if the embedding dimension isn't divisible by the head count
    /// because the attention mechanism requires equal-sized heads.
    /// </para>
    /// </remarks>
    private void InitializeLayer(int sequenceLength, int embeddingDimension, int headCount)
    {
        _sequenceLength = sequenceLength;
        _embeddingDimension = embeddingDimension;
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;

        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException("Embedding dimension must be divisible by the number of heads.");
        }

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weight matrices and bias vector with proper scaling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This private method initializes the query, key, and value weight matrices with small random values
    /// scaled according to the dimensions of the matrices. This scaling helps prevent vanishing or exploding
    /// gradients during training. The output bias vector is initialized to zeros. This method is called
    /// during the initialization of the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for all the weights and biases.
    /// 
    /// During initialization:
    /// - The query, key, and value weight matrices are filled with small random values
    /// - These values are scaled using a special formula (Xavier/Glorot initialization)
    /// - The output biases are set to zero
    /// 
    /// The scaling is important because:
    /// - Too large initial weights can cause unstable training (exploding gradients)
    /// - Too small initial weights can cause slow or stalled training (vanishing gradients)
    /// - The Xavier/Glorot initialization helps find a good middle ground
    /// 
    /// Setting the biases to zero is a common practice that lets the weights do the initial learning,
    /// with the biases adjusting later to fine-tune the output values.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Rows + _queryWeights.Columns)));
        InitializeMatrix(_queryWeights, scale);
        InitializeMatrix(_keyWeights, scale);
        InitializeMatrix(_valueWeights, scale);

        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a matrix with small random values scaled by the provided factor.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This private helper method fills the specified matrix with small random values between -0.5 and 0.5,
    /// scaled by the provided factor. This approach, known as Xavier/Glorot initialization, helps ensure
    /// that the activations and gradients have appropriate magnitudes, which improves training dynamics.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a weight matrix with properly sized random values.
    /// 
    /// During initialization:
    /// - The method loops through every position in the matrix
    /// - At each position, it generates a random number between -0.5 and 0.5
    /// - It multiplies this number by a scaling factor to get the right magnitude
    /// - The result becomes the initial weight value at that position
    /// 
    /// This random initialization is crucial because:
    /// - Starting with all zeros or the same value would make all neurons learn the same patterns
    /// - Starting with values that are too large or small would cause training problems
    /// - The slight randomness breaks symmetry and allows different neurons to specialize
    /// 
    /// The scaling factor ensures that these random values are appropriately sized based on
    /// the dimensions of the matrix, helping training to proceed smoothly.
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
}