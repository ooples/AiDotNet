

using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

public enum EmbeddingInputMode
{
    Auto,
    Indices,
    Continuous
}

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
/// <para>
/// <b>Thread Safety:</b> This layer is not thread-safe. Each layer instance maintains internal state
/// during forward and backward passes. If you need concurrent execution, use separate layer instances
/// per thread or synchronize access to shared instances.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class EmbeddingLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// The embedding tensor that stores vector representations for each token in the vocabulary.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable embedding vectors for each token in the vocabulary. The rows
    /// correspond to token indices, and the columns represent the dimensions of the embedding space.
    /// Each row in the tensor is the embedding vector for the corresponding token.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "dictionary" that maps each token ID to its vector representation.
    ///
    /// The embedding tensor works like this:
    /// - Each row corresponds to one token (word, character, etc.)
    /// - Each column is one dimension of the embedding space
    /// - If you have 10,000 words and 300 dimensions, the tensor will be 10,000 × 300
    ///
    /// For example, with a vocabulary of 5 words and 4 dimensions:
    /// ```
    /// Word ID | Embedding Vector
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
    private Tensor<T> _embeddingTensor;

    /// <summary>
    /// Projection weights for continuous input (lazy initialized).
    /// Used when input contains continuous values instead of integer token indices.
    /// </summary>
    private Tensor<T>? _projectionWeights;

    private Tensor<T>? _projectionWeightsGradient;
    private bool _lastInputWasContinuous;
    private bool? _autoDetectedContinuous;

    /// <summary>
    /// The gradients for the embedding tensor, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each element in the embedding tensor.
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
    private Tensor<T>? _embeddingGradient;

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
    /// Stores the last computed embedding regularization loss for diagnostics.
    /// </summary>
    private T _lastEmbeddingRegularizationLoss;

    /// <summary>
    /// Gets or sets whether to use auxiliary loss (embedding regularization) during training.
    /// Default is false. Enable to prevent embeddings from becoming too large or collapsing.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for embedding regularization.
    /// Default is 0.0001. Controls L2 regularization strength on embedding weights.
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    private EmbeddingInputMode _inputMode = EmbeddingInputMode.Auto;
    public EmbeddingInputMode InputMode
    {
        get => _inputMode;
        set
        {
            if (_inputMode == value)
            {
                return;
            }

            _inputMode = value;
            if (_inputMode == EmbeddingInputMode.Auto)
            {
                _autoDetectedContinuous = null;
            }
        }
    }


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
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <value>
    /// The number of elements in the embedding matrix (vocabulary size × embedding dimension).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts the total number of adjustable values in the layer.
    /// For an embedding layer with 10,000 vocabulary size and 300 dimensions,
    /// the parameter count would be 10,000 × 300 = 3,000,000 parameters.
    /// </para>
    /// </remarks>
    public override int ParameterCount
        => _embeddingTensor.Shape[0] * _embeddingTensor.Shape[1] +
           (_projectionWeights?.Length ?? 0);

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
        AuxiliaryLossWeight = NumOps.FromDouble(0.0001);
        _lastEmbeddingRegularizationLoss = NumOps.Zero;

        _embeddingTensor = new Tensor<T>([vocabularySize, embeddingDimension]);
        InitializeParameters();
    }

    /// <summary>
    /// Initializes the embedding tensor with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the embedding tensor with small random values scaled by a factor
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
        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];

        // Initialize embedding tensor with small random values using Engine operations
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(NumOps.FromDouble(1.0), NumOps.FromDouble(embeddingDim)));

        // Create random tensor [0, 1], shift to [-0.5, 0.5], then scale
        var randomTensor = Tensor<T>.CreateRandom(vocabSize, embeddingDim);
        var halfTensor = new Tensor<T>([vocabSize, embeddingDim]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        _embeddingTensor = Engine.TensorMultiplyScalar(shifted, scale);
    }

    /// <summary>
    /// The original input shape, saved for backward pass.
    /// </summary>
    private int[] _originalInputShape = [];

    /// <summary>
    /// Performs the forward pass of the embedding layer, converting token indices to vector representations.
    /// </summary>
    /// <param name="input">The input tensor containing token indices. Supports any-rank tensors:
    /// - 1D: [seqLen] - single sequence
    /// - 2D: [batch, seqLen] - batch of sequences (industry standard)
    /// - 3D: [batch, seqLen, 1] - compatible with legacy format
    /// </param>
    /// <returns>The output tensor containing embedding vectors with the same leading dimensions plus embeddingDim.</returns>
    /// <remarks>
    /// <para>
    /// <b>Industry Standard:</b> Like PyTorch's nn.Embedding, this layer supports any-rank input tensors.
    /// The indices in the last dimension(s) are looked up in the embedding table, and the result has
    /// the same shape with the last dimension replaced by the embedding dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method looks up the vector for each token ID in your input.
    ///
    /// The forward pass works like this:
    /// 1. Take a sequence of token IDs as input (like [5, 10, 3])
    /// 2. For each ID, look up its corresponding row in the embedding matrix
    /// 3. Copy that row (the embedding vector) to the output
    ///
    /// For example, with an input sequence [5, 10, 3]:
    /// - Look up row 5 in the embedding matrix -> output row 1
    /// - Look up row 10 in the embedding matrix -> output row 2
    /// - Look up row 3 in the embedding matrix -> output row 3
    ///
    /// The result is a sequence of embedding vectors, one for each input token.
    /// This transforms your discrete tokens into continuous vectors that the neural
    /// network can process more effectively.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        _originalInputShape = input.Shape;

        int embeddingDim = _embeddingTensor.Shape[1];
        int vocabularySize = _embeddingTensor.Shape[0];

        // Industry standard: Support any-rank input tensors
        // 1D: [seqLen] -> [seqLen, embeddingDim]
        // 2D: [batch, seqLen] -> [batch, seqLen, embeddingDim]
        // 3D: [batch, seqLen, 1] -> [batch, seqLen, embeddingDim]

        // Detect if input is continuous (float values, not integer indices)
        // Continuous input: use linear projection instead of embedding lookup
        bool isContinuousInput = InputMode switch
        {
            EmbeddingInputMode.Continuous => true,
            EmbeddingInputMode.Indices => false,
            _ => _autoDetectedContinuous ??= IsContinuousInput(input, vocabularySize)
        };
        _lastInputWasContinuous = isContinuousInput;

        Tensor<T> flatOutput;

        if (isContinuousInput)
        {
            // Use linear projection for continuous input
            // Project from input features to embedding dimension
            int inputFeatures = input.Shape[input.Rank - 1];

            // Create projection weights if needed (lazy initialization)
            if (_projectionWeights == null || _projectionWeights.Shape[0] != inputFeatures)
            {
                _projectionWeights = new Tensor<T>([inputFeatures, embeddingDim]);
                // Xavier initialization
                T scale = NumOps.FromDouble(Math.Sqrt(2.0 / (inputFeatures + embeddingDim)));
                var random = RandomHelper.CreateSecureRandom();
                for (int i = 0; i < _projectionWeights.Length; i++)
                {
                    _projectionWeights.SetFlat(i, NumOps.Multiply(scale, NumOps.FromDouble(random.NextDouble() * 2 - 1)));
                }
            }

            // Flatten input to 2D [total_samples, inputFeatures] for projection
            int totalSamples = input.Length / inputFeatures;
            var input2D = input.Reshape([totalSamples, inputFeatures]);
            flatOutput = input2D.MatrixMultiply(_projectionWeights);
        }
        else
        {
            // Standard embedding lookup for integer token indices
            int totalIndices = input.Length;
            var flatIndices = new Tensor<int>([totalIndices]);

            for (int i = 0; i < totalIndices; i++)
            {
                int index = Convert.ToInt32(NumOps.ToDouble(input.Data[i]));
                flatIndices[i] = index;
            }

            // Use Engine embedding lookup operation
            flatOutput = Engine.TensorEmbeddingLookup<T, int>(_embeddingTensor, flatIndices);
        }

        // Calculate output shape
        int[] outputShape;
        if (isContinuousInput)
        {
            // For continuous input (linear projection): replace last dimension with embeddingDim
            // input[..., inputFeatures] -> output[..., embeddingDim]
            outputShape = new int[input.Rank];
            for (int i = 0; i < input.Rank - 1; i++)
            {
                outputShape[i] = input.Shape[i];
            }
            outputShape[^1] = embeddingDim;
        }
        else if (input.Rank == 1)
        {
            // [seqLen] -> [seqLen, embeddingDim]
            outputShape = [input.Shape[0], embeddingDim];
        }
        else if (input.Rank == 2)
        {
            // [batch, seqLen] -> [batch, seqLen, embeddingDim]
            outputShape = [input.Shape[0], input.Shape[1], embeddingDim];
        }
        else if (input.Rank == 3 && input.Shape[2] == 1)
        {
            // Legacy format [batch, seqLen, 1] -> [batch, seqLen, embeddingDim]
            outputShape = [input.Shape[0], input.Shape[1], embeddingDim];
        }
        else
        {
            // Generic case for any rank: input shape [...] -> [..., embeddingDim]
            // This matches PyTorch's nn.Embedding behavior which accepts any shape
            // and appends the embedding dimension to the output
            outputShape = new int[input.Rank + 1];
            for (int i = 0; i < input.Rank; i++)
            {
                outputShape[i] = input.Shape[i];
            }
            outputShape[^1] = embeddingDim;
        }

        return flatOutput.Reshape(outputShape);
    }

    private bool IsContinuousInput(Tensor<T> input, int vocabularySize)
    {
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input.Data[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                return true;
            }
            int intVal = (int)val;
            if (Math.Abs(val - intVal) > 1e-6 || intVal < 0 || intVal >= vocabularySize)
            {
                return true;
            }
        }

        return false;
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
        if (_lastInputWasContinuous)
        {
            return BackwardContinuous(outputGradient);
        }

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
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];
        int totalIndices = _lastInput.Length;

        // Flatten input indices to 1D
        var flatIndices = new Tensor<int>([totalIndices]);
        for (int i = 0; i < totalIndices; i++)
        {
            flatIndices[i] = Convert.ToInt32(NumOps.ToDouble(_lastInput.Data[i]));
        }

        // Flatten outputGradient: [..., embeddingDim] -> [totalIndices, embeddingDim]
        var flatGradOutput = outputGradient.Reshape([totalIndices, embeddingDim]);

        // Use Engine scatter-add operation for gradient accumulation
        _embeddingGradient = Engine.TensorEmbeddingLookupBackward<T, int>(flatGradOutput, flatIndices, vocabSize, embeddingDim);

        // We don't compute input gradients for embedding layer (indices are not differentiable)
        return new Tensor<T>(_originalInputShape);
    }

    private Tensor<T> BackwardContinuous(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _projectionWeights == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int inputFeatures = _projectionWeights.Shape[0];
        int embeddingDim = _projectionWeights.Shape[1];
        int totalSamples = _lastInput.Length / inputFeatures;

        var input2D = _lastInput.Reshape([totalSamples, inputFeatures]);
        var grad2D = outputGradient.Reshape([totalSamples, embeddingDim]);

        var input2DTransposed = Engine.TensorTranspose(input2D);
        _projectionWeightsGradient = Engine.TensorMatMul(input2DTransposed, grad2D);

        var projectionWeightsTransposed = Engine.TensorTranspose(_projectionWeights);
        var inputGrad2D = Engine.TensorMatMul(grad2D, projectionWeightsTransposed);

        _embeddingGradient = null;
        return inputGrad2D.Reshape(_originalInputShape);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients.
    /// It builds a computation graph with EmbeddingLookup operation which handles
    /// the scatter-add gradient accumulation during the backward pass.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Create variables
        // Input indices do not require gradients
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "indices", requiresGradient: false);
        // Embeddings require gradients
        var embeddingNode = Autodiff.TensorOperations<T>.Variable(_embeddingTensor, "embeddings", requiresGradient: true);

        // 2. Build graph
        var output = Autodiff.TensorOperations<T>.EmbeddingLookup(embeddingNode, inputNode);

        // 3. Set gradient
        output.Gradient = outputGradient;

        // 4. Topo sort and backward pass
        output.Backward();

        // 5. Extract gradient
        _embeddingGradient = embeddingNode.Gradient;

        // Return zero gradient for input (indices are not differentiable)
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
        if (_embeddingGradient == null && _projectionWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        if (_embeddingGradient != null)
        {
            var scaledGradient = Engine.TensorMultiplyScalar(_embeddingGradient, learningRate);
            _embeddingTensor = Engine.TensorSubtract(_embeddingTensor, scaledGradient);
        }

        if (_projectionWeightsGradient != null && _projectionWeights != null)
        {
            var scaledProjectionGradient = Engine.TensorMultiplyScalar(_projectionWeightsGradient, learningRate);
            _projectionWeights = Engine.TensorSubtract(_projectionWeights, scaledProjectionGradient);
        }
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
        // Use ToArray() for production-grade parameter extraction
        var embeddingParams = new Vector<T>(_embeddingTensor.ToArray());
        if (_projectionWeights == null)
        {
            return embeddingParams;
        }

        var projectionParams = new Vector<T>(_projectionWeights.ToArray());
        return Vector<T>.Concatenate(embeddingParams, projectionParams);
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
        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];
        int expectedParams = vocabSize * embeddingDim;

        if (parameters.Length < expectedParams)
        {
            throw new ArgumentException($"Expected {expectedParams} parameters, but got {parameters.Length}");
        }

        // Restore embeddings without hot-path conversions
        _embeddingTensor = new Tensor<T>([vocabSize, embeddingDim], parameters.Slice(0, expectedParams));

        int projectionCount = parameters.Length - expectedParams;
        if (projectionCount == 0)
        {
            _projectionWeights = null;
            return;
        }

        if (projectionCount % embeddingDim != 0)
        {
            throw new ArgumentException($"Projection parameter count {projectionCount} is not divisible by embedding dimension {embeddingDim}.");
        }

        int inputFeatures = projectionCount / embeddingDim;
        _projectionWeights = new Tensor<T>([inputFeatures, embeddingDim], parameters.Slice(expectedParams, projectionCount));
    }

    /// <summary>
    /// Computes the auxiliary loss for the EmbeddingLayer, which is embedding regularization.
    /// </summary>
    /// <returns>The embedding regularization loss value.</returns>
    /// <remarks>
    /// <para>
    /// Embedding regularization prevents embedding vectors from becoming too large or too similar,
    /// which can lead to overfitting. It applies L2 regularization on the embedding weights:
    /// Loss = (1/2) * Σ||embedding||²
    ///
    /// This regularization:
    /// - Prevents embeddings from growing unboundedly
    /// - Encourages smaller, more generalizable embedding values
    /// - Helps prevent overfitting to the training data
    /// - Promotes diverse embedding representations
    /// </para>
    /// <para><b>For Beginners:</b> This calculates a penalty for embeddings that become too large.
    ///
    /// Embedding regularization:
    /// - Measures how large the embedding vectors are
    /// - Penalizes very large embedding values
    /// - Encourages the model to use smaller, more manageable numbers
    /// - Prevents the model from memorizing training data too closely
    ///
    /// Why this is important:
    /// - Large embedding values can indicate overfitting
    /// - Regularization promotes better generalization to new data
    /// - Keeps embedding vectors at reasonable scales
    /// - Prevents embeddings from collapsing or diverging
    ///
    /// Think of it like a referee that prevents embeddings from becoming too extreme,
    /// keeping them in a reasonable range for better model performance.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            // Reset cached loss to avoid stale diagnostics
            _lastEmbeddingRegularizationLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];

        // Compute L2 regularization on embedding weights using Engine operation: (1/2) * Σ||embedding||²
        T sumSquaredNorms = Engine.TensorSumOfSquares(_embeddingTensor);

        // Average over all embedding values and scale by 0.5 (standard L2 regularization)
        int totalElements = vocabSize * embeddingDim;
        T regularizationLoss = NumericalStabilityHelper.SafeDiv(sumSquaredNorms, NumOps.FromDouble(totalElements * 2));

        // Store unweighted loss for diagnostics
        _lastEmbeddingRegularizationLoss = regularizationLoss;

        // Return weighted auxiliary loss
        return NumOps.Multiply(AuxiliaryLossWeight, regularizationLoss);
    }

    /// <summary>
    /// Gets diagnostic information about the embedding regularization.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about embedding health.</returns>
    /// <remarks>
    /// <para>
    /// This method provides insights into embedding behavior, including:
    /// - Embedding regularization loss
    /// - Average embedding magnitude
    /// - Regularization weight
    /// </para>
    /// <para><b>For Beginners:</b> This gives you information to monitor embedding quality.
    ///
    /// The diagnostics include:
    /// - Embedding Regularization Loss: Measure of embedding magnitude
    /// - Regularization Weight: How much the penalty influences training
    /// - Average Embedding Magnitude: Typical size of embedding vectors
    /// - Use Auxiliary Loss: Whether regularization is enabled
    ///
    /// These values help you:
    /// - Monitor if embeddings are growing too large
    /// - Detect potential overfitting in embedding layer
    /// - Tune the regularization weight
    /// - Ensure embeddings remain at reasonable scales
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        string regLossStr = Convert.ToString(_lastEmbeddingRegularizationLoss) ?? "0";
        string weightStr = Convert.ToString(AuxiliaryLossWeight) ?? "0.0001";

        var diagnostics = new Dictionary<string, string>
        {
            { "EmbeddingRegularizationLoss", regLossStr },
            { "RegularizationWeight", weightStr },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
        };

        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];

        // Calculate average embedding magnitude using Engine operations
        // Sum of squares gives us Σ||embedding_i||² across all embeddings
        T totalSumOfSquares = Engine.TensorSumOfSquares(_embeddingTensor);

        // For average magnitude: sqrt(sum_of_squares / num_elements) * num_rows / num_rows
        // Simplified: average magnitude ≈ sqrt(total_sum_of_squares / total_elements) * sqrt(embedding_dim)
        // This is an approximation, but avoids per-row loops
        if (vocabSize > 0)
        {
            T avgSquaredMagnitude = NumericalStabilityHelper.SafeDiv(totalSumOfSquares, NumOps.FromDouble(vocabSize));
            T avgMagnitude = NumOps.Sqrt(avgSquaredMagnitude);
            string avgMagStr = Convert.ToString(avgMagnitude) ?? "0";
            diagnostics["AverageEmbeddingMagnitude"] = avgMagStr;
        }

        return diagnostics;
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
        _projectionWeightsGradient = null;
        _lastInputWasContinuous = false;
        _autoDetectedContinuous = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because embedding lookup can be JIT compiled.
    /// </value>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the embedding layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the embedded vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the embedding lookup operation.
    /// The graph uses the embedding matrix as a constant and performs an EmbeddingLookup operation
    /// based on the input indices.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an optimized version of the embedding lookup.
    ///
    /// The computation graph:
    /// - Takes input indices (token IDs)
    /// - Looks up corresponding rows in the embedding matrix
    /// - Returns the embedding vectors for each token
    ///
    /// This is JIT compiled for faster inference.
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create placeholder for input indices
        // Input shape for embeddings: [batchSize, sequenceLength] or [batchSize, 1]
        var inputPlaceholder = new Tensor<T>(new int[] { 1, 1 });
        var inputNode = Autodiff.TensorOperations<T>.Variable(inputPlaceholder, "input_indices");
        inputNodes.Add(inputNode);

        // Create constant node for embedding tensor [vocab_size, embedding_dim]
        var embeddingNode = Autodiff.TensorOperations<T>.Constant(_embeddingTensor, "embeddings");

        // Use EmbeddingLookup operation which supports gradients
        return Autodiff.TensorOperations<T>.EmbeddingLookup(embeddingNode, inputNode);
    }
}
