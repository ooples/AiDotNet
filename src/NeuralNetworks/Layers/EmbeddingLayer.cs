namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an embedding layer that converts discrete token indices into dense vector representations.
/// </summary>
/// <remarks>
/// <para>
/// An embedding layer maps discrete tokens (represented as indices) to continuous vector representations.
/// This is particularly useful for natural language processing tasks where words or tokens need to be
/// represented as dense vectors that capture semantic relationships. Each token is assigned a unique
/// vector in a high-dimensional space, allowing the model to learn meaningful representations.
/// </para>
/// <para><b>For Beginners:</b> An embedding layer turns words or other symbols into lists of numbers that capture their meaning.
/// 
/// Imagine you have a dictionary where:
/// - Each word has an ID number (like "cat" = 5, "dog" = 10)
/// - The embedding layer gives each ID a unique "coordinate" in a multi-dimensional space
/// - Words with similar meanings end up with similar coordinates
/// 
/// For example:
/// - "Cat" might become [0.2, -0.5, 0.1, 0.8]
/// - "Kitten" might become [0.25, -0.4, 0.15, 0.7]
/// - "Computer" might become [-0.8, 0.2, 0.5, -0.3]
/// 
/// The embedding layer learns these representations during training, so that:
/// - Similar words end up close to each other
/// - Related concepts form clusters
/// - The vectors capture meaningful semantic relationships
/// 
/// This allows neural networks to work with text and other discrete tokens in a way
/// that captures their meaning and relationships.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class EmbeddingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The embedding matrix that stores vector representations for each token in the vocabulary.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the learnable embedding vectors for each token in the vocabulary. The rows
    /// correspond to token indices, and the columns represent the dimensions of the embedding space.
    /// Each row in the matrix is the embedding vector for the corresponding token.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "dictionary" that maps each token ID to its vector representation.
    /// 
    /// The embedding matrix works like this:
    /// - Each row corresponds to one token (word, character, etc.)
    /// - Each column is one dimension of the embedding space
    /// - If you have 10,000 words and 300 dimensions, the matrix will be 10,000 × 300
    /// 
    /// For example, with a vocabulary of 5 words and 4 dimensions:
    /// ```
    /// Word ID | Embedding Vector<double>
    /// --------|-----------------
    /// 0       | [0.1, 0.2, -0.3, 0.5]
    /// 1       | [-0.5, 0.8, 0.1, -0.2]
    /// 2       | [0.4, -0.1, -0.7, 0.3]
    /// 3       | [0.2, 0.5, 0.6, -0.4]
    /// 4       | [-0.3, -0.2, 0.4, 0.8]
    /// ```
    /// 
    /// During training, these values are adjusted to make similar tokens have similar vectors.
    /// </para>
    /// </remarks>
    private Matrix<T> _embeddingMatrix = default!;

    /// <summary>
    /// The gradients for the embedding matrix, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the gradients of the loss with respect to each element in the embedding matrix.
    /// These gradients are used to update the embeddings during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each embedding value.
    /// 
    /// During training:
    /// - The network calculates how each embedding vector contributed to errors
    /// - These gradients show how to change each value to improve performance
    /// - Larger gradients mean bigger adjustments are needed
    /// 
    /// For example, if the network predicts incorrectly using the embedding for "cat",
    /// the gradients will indicate how to adjust that specific embedding vector to
    /// improve future predictions.
    /// 
    /// Only the embeddings for tokens that were actually used in the current batch
    /// will receive gradient updates.
    /// </para>
    /// </remarks>
    private Matrix<T>? _embeddingGradient;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input indices received during the last forward pass. These indices are
    /// necessary for computing the gradients during the backward pass, as they indicate which embeddings
    /// were accessed.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers which token IDs were processed in the latest calculation.
    /// 
    /// During training:
    /// - The layer needs to remember which tokens it looked up
    /// - This helps when calculating how to improve the embeddings
    /// - Only the embeddings for these specific tokens will be updated
    /// 
    /// For example, if the input was the sequence [5, 10, 3] (representing three tokens),
    /// only the embeddings for token IDs 5, 10, and 3 will receive updates during training.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because this layer has trainable parameters (the embedding matrix).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the embedding layer supports training through backpropagation.
    /// The layer has trainable embeddings that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its embeddings during training
    /// - It will improve its representations as it sees more data
    /// - It has parameters (the embedding matrix) that are updated to make better predictions
    /// 
    /// Unlike static word embeddings (like pre-trained word vectors), these embeddings
    /// adapt and improve specifically for your task during training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingLayer{T}"/> class.
    /// </summary>
    /// <param name="vocabularySize">The number of unique tokens in the vocabulary.</param>
    /// <param name="embeddingDimension">The dimension of the embedding vectors.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new embedding layer with the specified vocabulary size and embedding dimension.
    /// The embedding matrix is initialized with small random values scaled to help with training convergence.
    /// The input shape is set to [1] because the layer expects token indices, and the output shape is
    /// set to [embeddingDimension] as each token is mapped to a vector of that dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the embedding layer with the vocabulary size and embedding dimensions you need.
    /// 
    /// When creating an embedding layer, you need to specify:
    /// - Vocabulary size: How many different tokens (words, characters, etc.) your model will handle
    /// - Embedding dimension: How many numbers to use for each token's representation
    /// 
    /// For example:
    /// ```csharp
    /// // Create an embedding layer for 10,000 words with 300-dimensional embeddings
    /// var wordEmbedding = new EmbeddingLayer<float>(10000, 300);
    /// 
    /// // Create an embedding layer for 128 characters with 50-dimensional embeddings
    /// var charEmbedding = new EmbeddingLayer<float>(128, 50);
    /// ```
    /// 
    /// Typical embedding dimensions:
    /// - For words: 100-300 dimensions
    /// - For characters: 25-100 dimensions
    /// - For special tokens: 50-200 dimensions
    /// 
    /// Larger dimensions can capture more information but require more computation and memory.
    /// </para>
    /// </remarks>
    public EmbeddingLayer(int vocabularySize, int embeddingDimension)
        : base([1], [embeddingDimension])
    {
        _embeddingMatrix = new Matrix<T>(vocabularySize, embeddingDimension);
        InitializeParameters();
    }

    /// <summary>
    /// Initializes the embedding matrix with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the embedding matrix with small random values scaled by a factor
    /// that depends on the embedding dimension. This scaling helps in achieving good convergence
    /// during training by preventing the initial values from being too large or too small.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial random values for all embeddings.
    /// 
    /// Before training begins:
    /// - Each embedding needs some starting value
    /// - We use small random values, centered around zero
    /// - The values are scaled based on the embedding dimension
    /// 
    /// This initialization is important because:
    /// - Too large values could cause training instability
    /// - Too small values could slow down learning
    /// - The scaling factor helps find a good middle ground
    /// 
    /// As training progresses, these random initial values will gradually be replaced
    /// with meaningful representations learned from data.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Initialize embedding matrix with small random values
        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _embeddingMatrix.Columns));

        for (int i = 0; i < _embeddingMatrix.Rows; i++)
        {
            for (int j = 0; j < _embeddingMatrix.Columns; j++)
            {
                _embeddingMatrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Performs the forward pass of the embedding layer, converting token indices to vector representations.
    /// </summary>
    /// <param name="input">The input tensor containing token indices. Shape: [sequenceLength, batchSize, 1].</param>
    /// <returns>The output tensor containing embedding vectors. Shape: [sequenceLength, batchSize, embeddingDimension].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the embedding layer. It takes a tensor of token indices
    /// and returns a tensor of embedding vectors by looking up each index in the embedding matrix.
    /// The input tensor should have the shape [sequenceLength, batchSize, 1], where each element
    /// is an integer index into the embedding matrix. The output tensor will have the shape
    /// [sequenceLength, batchSize, embeddingDimension].
    /// </para>
    /// <para><b>For Beginners:</b> This method looks up the vector for each token ID in your input.
    /// 
    /// The forward pass works like this:
    /// 1. Take a sequence of token IDs as input (like [5, 10, 3])
    /// 2. For each ID, look up its corresponding row in the embedding matrix
    /// 3. Copy that row (the embedding vector) to the output
    /// 
    /// For example, with an input sequence [5, 10, 3]:
    /// - Look up row 5 in the embedding matrix → output row 1
    /// - Look up row 10 in the embedding matrix → output row 2
    /// - Look up row 3 in the embedding matrix → output row 3
    /// 
    /// The result is a sequence of embedding vectors, one for each input token.
    /// This transforms your discrete tokens into continuous vectors that the neural
    /// network can process more effectively.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int sequenceLength = input.Shape[0];
        int batchSize = input.Shape[1];

        var output = new Tensor<T>([sequenceLength, batchSize, _embeddingMatrix.Columns]);

        for (int t = 0; t < sequenceLength; t++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                int index = Convert.ToInt32(input[t, b, 0]);
                for (int d = 0; d < _embeddingMatrix.Columns; d++)
                {
                    output[t, b, d] = _embeddingMatrix[index, d];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the embedding layer, computing gradients for the embedding matrix.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [sequenceLength, batchSize, embeddingDimension].</param>
    /// <returns>A zero-filled tensor with the same shape as the input, as gradients don't flow back to indices.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the embedding layer. It computes
    /// the gradients for the embedding matrix by accumulating the gradients from the output for each
    /// token index that was used in the forward pass. Since the input to the embedding layer is
    /// indices rather than computed values, no meaningful gradients can be computed for the input.
    /// Therefore, this method returns a zero-filled tensor with the same shape as the input.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the embedding layer learns from its mistakes during training.
    /// 
    /// During the backward pass:
    /// 1. For each token in the input sequence:
    ///    - Look up which embedding was used (based on the token ID)
    ///    - Add the corresponding gradient to that specific embedding
    /// 2. Return a dummy gradient for the input (since we can't backpropagate through token IDs)
    /// 
    /// For example, if token ID 5 appears three times in different positions:
    /// - All three gradient contributions will be added together for embedding #5
    /// - This accumulates learning from all occurrences of that token
    /// 
    /// This is different from most layers because:
    /// - We only update the embeddings that were actually used in this batch
    /// - We don't pass meaningful gradients back to the input (the token IDs themselves don't change)
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int sequenceLength = _lastInput.Shape[0];
        int batchSize = _lastInput.Shape[1];

        _embeddingGradient = new Matrix<T>(_embeddingMatrix.Rows, _embeddingMatrix.Columns);

        for (int t = 0; t < sequenceLength; t++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                int index = Convert.ToInt32(_lastInput[t, b, 0]);
                for (int d = 0; d < _embeddingMatrix.Columns; d++)
                {
                    _embeddingGradient[index, d] = NumOps.Add(_embeddingGradient[index, d], outputGradient[t, b, d]);
                }
            }
        }

        // We don't compute input gradients for embedding layer
        return new Tensor<T>(_lastInput.Shape);
    }

    /// <summary>
    /// Updates the embedding matrix using the calculated gradients and the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the embedding matrix based on the gradients calculated during the backward pass.
    /// Only the embeddings for tokens that appeared in the input during the forward pass will be updated.
    /// The learning rate determines the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method actually changes the embeddings to improve future predictions.
    /// 
    /// After figuring out how each embedding should change:
    /// - The embedding matrix is updated by subtracting the gradients
    /// - Each value is adjusted proportionally to its gradient
    /// - The learning rate controls how big these adjustments are
    /// 
    /// For example:
    /// - If embedding for token #5 has a gradient of [0.1, -0.2, 0.3]
    /// - With learning rate of 0.01
    /// - The embedding will change by [-0.001, 0.002, -0.003]
    /// 
    /// Only embeddings for tokens that appeared in the recent input batch will be updated.
    /// Frequently used tokens will get more updates over time.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_embeddingGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _embeddingMatrix = _embeddingMatrix.Subtract(_embeddingGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (the entire embedding matrix) as a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving
    /// and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the embedding values into a single list.
    /// 
    /// The parameters include:
    /// - All values from the embedding matrix, arranged in a single long list
    /// - Each embedding vector is placed one after another
    /// 
    /// This is useful for:
    /// - Saving the embeddings to disk
    /// - Loading pre-trained embeddings
    /// - Applying specific optimization techniques
    /// 
    /// For example, a vocabulary of 1,000 tokens with 100-dimensional embeddings
    /// would produce a vector of 100,000 values.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _embeddingMatrix.Rows * _embeddingMatrix.Columns;
        var parameters = new Vector<T>(totalParams);

        int index = 0;

        // Copy embedding matrix parameters
        for (int i = 0; i < _embeddingMatrix.Rows; i++)
        {
            for (int j = 0; j < _embeddingMatrix.Columns; j++)
            {
                parameters[index++] = _embeddingMatrix[i, j];
            }
        }

        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters (the entire embedding matrix) from a single vector.
    /// This is useful for loading saved model weights or pre-trained embeddings.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all embedding values from a provided list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the exact right length
    /// - The values are distributed back to the embedding matrix
    /// - This allows loading previously trained or pre-trained embeddings
    /// 
    /// Use cases include:
    /// - Loading embeddings trained on another task
    /// - Initializing with pre-trained word vectors (like Word2Vec or GloVe)
    /// - Restoring a saved model
    /// 
    /// For example, you might initialize your embeddings with GloVe vectors
    /// that were pre-trained on a large corpus, giving your model a head start.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _embeddingMatrix.Rows * _embeddingMatrix.Columns)
        {
            throw new ArgumentException($"Expected {_embeddingMatrix.Rows * _embeddingMatrix.Columns} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set embedding matrix parameters
        for (int i = 0; i < _embeddingMatrix.Rows; i++)
        {
            for (int j = 0; j < _embeddingMatrix.Columns; j++)
            {
                _embeddingMatrix[i, j] = parameters[index++];
            }
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input and embedding gradients
    /// from previous forward and backward passes. This is useful when starting to process a new batch of
    /// data or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input token IDs are cleared
    /// - The calculated gradients are cleared
    /// - The layer forgets previous calculations it performed
    /// 
    /// This is typically called:
    /// - Between training batches to free up memory
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    /// 
    /// It doesn't affect the learned embeddings themselves, just the temporary
    /// working data used during computation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _embeddingGradient = null;
    }
}