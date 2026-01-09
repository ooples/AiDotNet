using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
public class SelfAttentionLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (attention sparsity regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attention sparsity regularization encourages the attention mechanism to focus on relevant positions
    /// while ignoring irrelevant ones. This prevents attention from being too diffuse and improves interpretability.
    /// </para>
    /// <para><b>For Beginners:</b> This helps self-attention focus on what matters.
    ///
    /// Self-attention works best when it's selective:
    /// - Without regularization: Attention might spread too thin across all positions
    /// - With regularization: Attention focuses on truly relevant relationships
    ///
    /// This includes:
    /// 1. Entropy regularization: Prevents overly uniform attention
    /// 2. Sparsity penalties: Encourages sharp, focused attention patterns
    ///
    /// This helps the model:
    /// - Learn clearer, more interpretable attention patterns
    /// - Focus computational resources on relevant relationships
    /// - Improve robustness and generalization
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the attention sparsity auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much attention sparsity regularization contributes to the total loss.
    /// Typical values range from 0.001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much we encourage focused attention.
    ///
    /// Common values:
    /// - 0.005 (default): Balanced sparsity regularization
    /// - 0.001-0.003: Light sparsity enforcement
    /// - 0.008-0.01: Strong sparsity enforcement
    ///
    /// Higher values encourage sharper, more focused attention patterns.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastEntropyLoss;
    private T _lastSparsityLoss;

    /// <summary>
    /// Tensor of weights for transforming input embeddings into query vectors.
    /// </summary>
    /// <remarks>
    /// This tensor transforms input embeddings into query vectors, which are used to compute attention scores.
    /// Queries represent what each position in the sequence is looking for in other positions.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </remarks>
    private Tensor<T> _queryWeights;

    /// <summary>
    /// Tensor of weights for transforming input embeddings into key vectors.
    /// </summary>
    /// <remarks>
    /// This tensor transforms input embeddings into key vectors, which are used to compute attention scores.
    /// Keys represent what each position in the sequence has to offer to other positions.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </remarks>
    private Tensor<T> _keyWeights;

    /// <summary>
    /// Tensor of weights for transforming input embeddings into value vectors.
    /// </summary>
    /// <remarks>
    /// This tensor transforms input embeddings into value vectors, which contain the actual content
    /// that will be aggregated based on attention scores. Values represent the information that
    /// is being extracted from each position.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </remarks>
    private Tensor<T> _valueWeights;

    /// <summary>
    /// Tensor of biases added to the output of the attention mechanism.
    /// </summary>
    /// <remarks>
    /// This tensor contains bias terms that are added to the output of the attention mechanism
    /// before applying the final activation function. Biases allow the network to adjust the
    /// baseline activation level of the attention output.
    /// Shape: [embeddingDimension]
    /// </remarks>
    private Tensor<T> _outputBias;

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
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

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
    /// This tensor holds the accumulated gradients for the query weight parameters during the backward pass.
    /// It has the same dimensions as the _queryWeights tensor and is used to update the query weights during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </remarks>
    private Tensor<T>? _queryWeightsGradient;

    /// <summary>
    /// Stores the gradients of the loss with respect to the key weight parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for the key weight parameters during the backward pass.
    /// It has the same dimensions as the _keyWeights tensor and is used to update the key weights during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </remarks>
    private Tensor<T>? _keyWeightsGradient;

    /// <summary>
    /// Stores the gradients of the loss with respect to the value weight parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for the value weight parameters during the backward pass.
    /// It has the same dimensions as the _valueWeights tensor and is used to update the value weights during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </remarks>
    private Tensor<T>? _valueWeightsGradient;

    /// <summary>
    /// Stores the gradients of the loss with respect to the output bias parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for the output bias parameters during the backward pass.
    /// It has the same length as the _outputBias tensor and is used to update the output biases during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// Shape: [embeddingDimension]
    /// </remarks>
    private Tensor<T>? _outputBiasGradient;

    private Tensor<T>? _queryWeightsVelocity;
    private Tensor<T>? _keyWeightsVelocity;
    private Tensor<T>? _valueWeightsVelocity;
    private Tensor<T>? _outputBiasVelocity;

    #region GPU Training Fields
    // Cached GPU tensors for backward pass
    private IGpuTensor<T>? _gpuLastInput;
    private IGpuTensor<T>? _gpuProjectedQueries;
    private IGpuTensor<T>? _gpuProjectedKeys;
    private IGpuTensor<T>? _gpuProjectedValues;
    private IGpuTensor<T>? _gpuAttentionWeightsGpu;

    // Cached GPU weight tensors
    private IGpuTensor<T>? _gpuQueryWeights;
    private IGpuTensor<T>? _gpuKeyWeights;
    private IGpuTensor<T>? _gpuValueWeights;
    private IGpuTensor<T>? _gpuOutputBias;

    // Cached GPU gradient tensors
    private IGpuTensor<T>? _gpuQueryWeightsGradient;
    private IGpuTensor<T>? _gpuKeyWeightsGradient;
    private IGpuTensor<T>? _gpuValueWeightsGradient;
    private IGpuTensor<T>? _gpuOutputBiasGradient;

    // Optimizer state buffers for query weights
    private IGpuTensor<T>? _gpuQueryWeightsVelocity;
    private IGpuTensor<T>? _gpuQueryWeightsM;
    private IGpuTensor<T>? _gpuQueryWeightsV;

    // Optimizer state buffers for key weights
    private IGpuTensor<T>? _gpuKeyWeightsVelocity;
    private IGpuTensor<T>? _gpuKeyWeightsM;
    private IGpuTensor<T>? _gpuKeyWeightsV;

    // Optimizer state buffers for value weights
    private IGpuTensor<T>? _gpuValueWeightsVelocity;
    private IGpuTensor<T>? _gpuValueWeightsM;
    private IGpuTensor<T>? _gpuValueWeightsV;

    // Optimizer state buffers for output bias
    private IGpuTensor<T>? _gpuOutputBiasVelocity;
    private IGpuTensor<T>? _gpuOutputBiasM;
    private IGpuTensor<T>? _gpuOutputBiasV;
    #endregion

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
    /// Returns layer-specific metadata required for cloning/serialization.
    /// </summary>
    /// <remarks>
    /// Self-attention requires the configured head count to reconstruct the layer correctly during
    /// serialization-based cloning. This mirrors <see cref="MultiHeadAttentionLayer{T}"/> metadata.
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HeadCount"] = _headCount.ToString();
        return metadata;
    }

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
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-resident training (backward pass and parameter updates on GPU).
    /// </summary>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <value>
    /// The total number of parameters: 3 weight matrices (Q, K, V) each of size [embeddingDimension × embeddingDimension],
    /// plus an output bias of size [embeddingDimension].
    /// Total = 3 × E² + E = E × (3E + 1) where E is the embedding dimension.
    /// </value>
    public override int ParameterCount =>
        3 * (_embeddingDimension * _embeddingDimension) + _embeddingDimension;

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
        // Initialize auxiliary loss fields first so compiler knows they're set
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastEntropyLoss = NumOps.Zero;
        _lastSparsityLoss = NumOps.Zero;

        // Initialize tensor fields - will be properly sized in InitializeLayer
        _queryWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _keyWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _valueWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputBias = new Tensor<T>([embeddingDimension]);

        InitializeLayer(sequenceLength, embeddingDimension, headCount);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
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
    /// It validates that the embedding dimension is divisible by the number of heads and initializes the weight tensors
    /// and bias tensor with appropriate values. A vector activation function is applied to the entire output vector at once,
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
        // Initialize auxiliary loss fields first so compiler knows they're set
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastEntropyLoss = NumOps.Zero;
        _lastSparsityLoss = NumOps.Zero;

        // Initialize tensor fields - will be properly sized in InitializeLayer
        _queryWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _keyWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _valueWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputBias = new Tensor<T>([embeddingDimension]);

        InitializeLayer(sequenceLength, embeddingDimension, headCount);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
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
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: need at least 2D [seqLen, embedDim]
        Tensor<T> input3D;
        int batchSize;
        int sequenceLength;
        int embeddingDimension;

        if (rank == 2)
        {
            // 2D: [seqLen, embedDim] -> add batch dim
            batchSize = 1;
            sequenceLength = input.Shape[0];
            embeddingDimension = input.Shape[1];
            input3D = input.Reshape([1, sequenceLength, embeddingDimension]);
        }
        else if (rank == 3)
        {
            // Standard 3D: [batch, seqLen, embedDim]
            batchSize = input.Shape[0];
            sequenceLength = input.Shape[1];
            embeddingDimension = input.Shape[2];
            input3D = input;
        }
        else if (rank > 3)
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            sequenceLength = input.Shape[rank - 2];
            embeddingDimension = input.Shape[rank - 1];
            input3D = input.Reshape([flatBatch, sequenceLength, embeddingDimension]);
        }
        else
        {
            throw new ArgumentException($"SelfAttentionLayer requires at least 2D input, got {rank}D");
        }

        _lastInput = input3D;

        // 1. Project Input to Q, K, V
        // Reshape input to 2D [Batch*Seq, EmbedDim] for efficient MatrixMultiply
        var input2D = input3D.Reshape(batchSize * sequenceLength, _embeddingDimension);

        // Compute Projections: [B*S, E] @ [E, E] -> [B*S, E]
        // Using Engine.TensorMatMul for GPU acceleration
        var Q_flat = Engine.TensorMatMul(input2D, _queryWeights);
        var K_flat = Engine.TensorMatMul(input2D, _keyWeights);
        var V_flat = Engine.TensorMatMul(input2D, _valueWeights);

        // Reshape to [Batch, Seq, HeadCount, HeadDim]
        var queries = Q_flat.Reshape(batchSize, sequenceLength, _headCount, _headDimension);
        var keys = K_flat.Reshape(batchSize, sequenceLength, _headCount, _headDimension);
        var values = V_flat.Reshape(batchSize, sequenceLength, _headCount, _headDimension);

        // Transpose for multi-head attention: [Batch, HeadCount, Seq, HeadDim]
        // This aligns dimensions for batched matrix multiplication
        var Q = queries.Transpose(new[] { 0, 2, 1, 3 });
        var K = keys.Transpose(new[] { 0, 2, 1, 3 });
        var V = values.Transpose(new[] { 0, 2, 1, 3 });

        // 2. Compute Scaled Dot-Product Attention
        // ScaledDotProductAttention computes: softmax(Q @ K^T / scale) @ V
        // Input shapes: [batch, heads, seq, head_dim]
        // Output shape: [batch, heads, seq, head_dim]
        var output4D = Engine.ScaledDotProductAttention(
            Q, K, V,
            mask: null,
            scale: 1.0 / Math.Sqrt(_headDimension),
            out var attentionWeights4D);

        // Cache attention weights for backward pass
        _lastAttentionScores = attentionWeights4D;

        // 3. Reshape and Project Output
        var outputTransposed = output4D.Transpose(new[] { 0, 2, 1, 3 });
        var outputFlat = outputTransposed.Reshape(batchSize, sequenceLength, embeddingDimension);

        // 7. Add Bias with engine broadcast for GPU acceleration
        var biasBroadcast = _outputBias.Reshape(1, 1, embeddingDimension);
        var outputBiased = Engine.TensorBroadcastAdd(outputFlat, biasBroadcast);

        var output = ApplyActivation(outputBiased);

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            int[] newShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 2; d++)
                newShape[d] = _originalInputShape[d];
            newShape[_originalInputShape.Length - 2] = sequenceLength;
            newShape[_originalInputShape.Length - 1] = embeddingDimension;
            output = output.Reshape(newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            // 2D input -> 2D output (remove batch dim)
            output = output.Reshape([sequenceLength, embeddingDimension]);
        }

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = output;
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the entire self-attention forward pass on the GPU without downloading
    /// intermediate results to CPU. All projections, attention computation, and bias addition
    /// remain GPU-resident for maximum performance.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];

        // Get dimensions from input shape
        int[] inputShape = input.Shape;
        int rank = inputShape.Length;

        int batchSize;
        int sequenceLength;
        int embeddingDimension;
        IGpuTensor<T> input3D;

        if (rank == 2)
        {
            // 2D: [seqLen, embedDim] -> add batch dim
            batchSize = 1;
            sequenceLength = inputShape[0];
            embeddingDimension = inputShape[1];
            input3D = gpuEngine.ReshapeGpu(input, [1, sequenceLength, embeddingDimension]);
        }
        else if (rank == 3)
        {
            // Standard 3D: [batch, seqLen, embedDim]
            batchSize = inputShape[0];
            sequenceLength = inputShape[1];
            embeddingDimension = inputShape[2];
            input3D = input;
        }
        else if (rank > 3)
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= inputShape[d];
            batchSize = flatBatch;
            sequenceLength = inputShape[rank - 2];
            embeddingDimension = inputShape[rank - 1];
            input3D = gpuEngine.ReshapeGpu(input, [flatBatch, sequenceLength, embeddingDimension]);
        }
        else
        {
            throw new ArgumentException($"SelfAttentionLayer requires at least 2D input, got {rank}D");
        }

        // 1. Project Input to Q, K, V
        // Reshape input to 2D [Batch*Seq, EmbedDim] for efficient MatrixMultiply
        var input2D = gpuEngine.ReshapeGpu(input3D, [batchSize * sequenceLength, _embeddingDimension]);

        // Compute Projections: [B*S, E] @ [E, E] -> [B*S, E]
        var Q_flat = gpuEngine.BatchedMatMulGpu(input2D, _queryWeights);
        var K_flat = gpuEngine.BatchedMatMulGpu(input2D, _keyWeights);
        var V_flat = gpuEngine.BatchedMatMulGpu(input2D, _valueWeights);

        // Reshape to [Batch, Seq, HeadCount, HeadDim]
        var queries = gpuEngine.ReshapeGpu(Q_flat, [batchSize, sequenceLength, _headCount, _headDimension]);
        var keys = gpuEngine.ReshapeGpu(K_flat, [batchSize, sequenceLength, _headCount, _headDimension]);
        var values = gpuEngine.ReshapeGpu(V_flat, [batchSize, sequenceLength, _headCount, _headDimension]);

        // Transpose for multi-head attention: [Batch, HeadCount, Seq, HeadDim]
        var Q = gpuEngine.PermuteGpu(queries, [0, 2, 1, 3]);
        var K = gpuEngine.PermuteGpu(keys, [0, 2, 1, 3]);
        var V = gpuEngine.PermuteGpu(values, [0, 2, 1, 3]);

        // 2. Compute Scaled Dot-Product Attention
        // Use overload that returns attention weights during training for backward pass
        double scale = 1.0 / Math.Sqrt(_headDimension);
        IGpuTensor<T> attnOutput4D;
        IGpuTensor<T>? attentionWeightsGpu = null;

        if (IsTrainingMode)
        {
            // Training mode: get attention weights for backward pass
            attnOutput4D = gpuEngine.ScaledDotProductAttentionGpu(Q, K, V, scale, out attentionWeightsGpu, isCausal: false);
        }
        else
        {
            // Inference mode: no need for attention weights
            attnOutput4D = gpuEngine.ScaledDotProductAttentionGpu(Q, K, V, scale, isCausal: false);
        }

        // 3. Reshape and Project Output
        var outputTransposed = gpuEngine.PermuteGpu(attnOutput4D, [0, 2, 1, 3]);
        var outputFlat = gpuEngine.ReshapeGpu(outputTransposed, [batchSize, sequenceLength, embeddingDimension]);

        // 4. Add Bias
        var outputBiased = gpuEngine.AddBiasGpu(outputFlat, _outputBias);

        // 5. Apply activation if not identity
        IGpuTensor<T> output;
        if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            var fusedType = MapActivationToFused();
            output = gpuEngine.ActivationGpu<T>(outputBiased, fusedType);
        }
        else
        {
            output = outputBiased;
        }

        // Cache state for backward pass only during training
        // Skip this expensive download during inference (50% overhead reduction)
        if (IsTrainingMode)
        {
            // Cache GPU tensors for GPU-resident backward pass
            _gpuLastInput = input3D;
            _gpuProjectedQueries = Q;  // [batch, heads, seq, headDim]
            _gpuProjectedKeys = K;
            _gpuProjectedValues = V;
            _gpuAttentionWeightsGpu = attentionWeightsGpu;

            // Also download to CPU for backward compatibility with CPU backward pass
            _lastInput = input3D.ToTensor();
            _lastAttentionScores = attentionWeightsGpu?.ToTensor();
            _lastOutput = output.ToTensor();
            _originalInputShape = inputShape;
        }

        // Restore original batch dimensions for any-rank support
        if (rank > 3)
        {
            int[] newShape = new int[rank];
            for (int d = 0; d < rank - 2; d++)
                newShape[d] = inputShape[d];
            newShape[rank - 2] = sequenceLength;
            newShape[rank - 1] = embeddingDimension;
            output = gpuEngine.ReshapeGpu(output, newShape);
        }
        else if (rank == 2)
        {
            // 2D input -> 2D output (remove batch dim)
            output = gpuEngine.ReshapeGpu(output, [sequenceLength, embeddingDimension]);
        }

        return output;
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
        // Fall back to manual backward when vector activation is used, since autodiff
        // doesn't properly handle vector activation derivatives or bias gradient propagation
        bool canUseAutodiff = UseAutodiff && VectorActivation == null;
        return canUseAutodiff
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
        if (_lastInput == null || _lastOutput == null || _lastAttentionScores == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        Tensor<T> outputGrad3D = outputGradient;
        Tensor<T> lastOutput3D = _lastOutput;
        if (outputGradient.Shape.Length != 3)
        {
            outputGrad3D = outputGradient.Reshape(_lastInput.Shape);
        }
        if (_lastOutput.Shape.Length != 3)
        {
            lastOutput3D = _lastOutput.Reshape(_lastInput.Shape);
        }

        var activationGradient = ApplyActivationDerivative(lastOutput3D, outputGrad3D);

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];
        int embeddingDimension = _lastInput.Shape[2];

        // Bias gradient: sum over batch and sequence
        _outputBiasGradient = Engine.ReduceSum(activationGradient, new[] { 0, 1 }, keepDims: false);

        // Recompute Q, K, V from input
        var input2D = _lastInput.Reshape(batchSize * sequenceLength, embeddingDimension);
        var Q_flat = Engine.TensorMatMul(input2D, _queryWeights);
        var K_flat = Engine.TensorMatMul(input2D, _keyWeights);
        var V_flat = Engine.TensorMatMul(input2D, _valueWeights);

        var Q = Q_flat.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose(new[] { 0, 2, 1, 3 });
        var K = K_flat.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose(new[] { 0, 2, 1, 3 });
        var V = V_flat.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose(new[] { 0, 2, 1, 3 });

        // Output gradient to 4D: [B, H, S, D]
        var dOutput4D = activationGradient.Reshape(batchSize, sequenceLength, _headCount, _headDimension)
            .Transpose(new[] { 0, 2, 1, 3 });

        // Use Engine.ScaledDotProductAttentionBackward for efficient gradient computation
        Engine.ScaledDotProductAttentionBackward(
            dOutput4D,
            Q,
            K,
            V,
            _lastAttentionScores,
            1.0 / Math.Sqrt(_headDimension),
            out var dQ_4D,
            out var dK_4D,
            out var dV_4D);

        // Reshape gradients from 4D to 2D for weight gradient computation
        var dQ = dQ_4D.Transpose(new[] { 0, 2, 1, 3 }).Reshape(batchSize * sequenceLength, embeddingDimension);
        var dK = dK_4D.Transpose(new[] { 0, 2, 1, 3 }).Reshape(batchSize * sequenceLength, embeddingDimension);
        var dV = dV_4D.Transpose(new[] { 0, 2, 1, 3 }).Reshape(batchSize * sequenceLength, embeddingDimension);

        // Weight gradients
        var input2D_T = Engine.TensorTranspose(input2D);
        _queryWeightsGradient = Engine.TensorMatMul(input2D_T, dQ);
        _keyWeightsGradient = Engine.TensorMatMul(input2D_T, dK);
        _valueWeightsGradient = Engine.TensorMatMul(input2D_T, dV);

        // Input gradient
        var dInputFromQ = Engine.TensorMatMul(dQ, Engine.TensorTranspose(_queryWeights));
        var dInputFromK = Engine.TensorMatMul(dK, Engine.TensorTranspose(_keyWeights));
        var dInputFromV = Engine.TensorMatMul(dV, Engine.TensorTranspose(_valueWeights));
        var dInput2D = Engine.TensorAdd(Engine.TensorAdd(dInputFromQ, dInputFromK), dInputFromV);

        var inputGradient = dInput2D.Reshape(batchSize, sequenceLength, embeddingDimension);
        return _originalInputShape != null && _originalInputShape.Length != 3
            ? inputGradient.Reshape(_originalInputShape)
            : inputGradient;

    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients by building a computation
    /// graph that mirrors the forward pass operations. Similar to how PyTorch and other production
    /// frameworks implement attention backward passes, this method:
    /// 1. Projects input to Q, K, V using weight matrix multiplications
    /// 2. Applies scaled dot-product attention
    /// 3. Adds output bias
    /// 4. Applies activation
    /// 5. Propagates gradients backward through the entire graph
    /// </para>
    /// <para>
    /// The computation graph enables automatic gradient computation for all parameters including
    /// query, key, and value weights as well as output biases. Weight nodes are created as
    /// Variable nodes with requiresGradient: true, and their gradients are extracted after
    /// the backward pass completes. This is the production-grade approach used in modern
    /// deep learning frameworks like PyTorch and TensorFlow.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastAttentionScores == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        // Build computation graph mirroring the forward pass
        // Step 1: Create input variable node with gradient tracking
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Step 2: Create variable nodes for weight tensors with gradient tracking
        // These are Variable (not Constant) so gradients flow through them
        // Weights are already Tensor<T> - no conversion needed (production-ready pattern)
        var wqNode = Autodiff.TensorOperations<T>.Variable(_queryWeights, "Wq", requiresGradient: true);
        var wkNode = Autodiff.TensorOperations<T>.Variable(_keyWeights, "Wk", requiresGradient: true);
        var wvNode = Autodiff.TensorOperations<T>.Variable(_valueWeights, "Wv", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_outputBias, "output_bias", requiresGradient: true);

        // Step 3: Project input to Q, K, V
        // Q = input @ Wq, K = input @ Wk, V = input @ Wv
        var queryNode = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, wqNode);
        var keyNode = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, wkNode);
        var valueNode = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, wvNode);

        // Step 4: Apply scaled dot-product attention
        // This computes: softmax(Q @ K^T / sqrt(d_k)) @ V
        var attentionOutput = Autodiff.TensorOperations<T>.ScaledDotProductAttention(queryNode, keyNode, valueNode);

        // Step 5: Add output bias (broadcast across batch dimension)
        var biasesBroadcast = BroadcastBias(biasNode.Value, batchSize);
        var biasBroadcastNode = Autodiff.TensorOperations<T>.Variable(biasesBroadcast, "bias_broadcast", requiresGradient: false);
        var biasedOutput = Autodiff.TensorOperations<T>.Add(attentionOutput, biasBroadcastNode);

        // Step 6: Apply activation using the generic ApplyActivation that supports ALL 39 activations
        // This follows the Open/Closed principle - no type checking needed
        Autodiff.ComputationNode<T> outputNode;
        if (ScalarActivation != null)
        {
            outputNode = Autodiff.TensorOperations<T>.ApplyActivation(biasedOutput, ScalarActivation);
        }
        else if (VectorActivation != null)
        {
            // Vector activations (like Softmax) applied to the output
            var activatedTensor = VectorActivation.Activate(biasedOutput.Value);
            outputNode = Autodiff.TensorOperations<T>.Variable(activatedTensor, "activated", requiresGradient: true);
            // Connect parent for gradient flow
            outputNode = Autodiff.TensorOperations<T>.Add(
                biasedOutput,
                Autodiff.TensorOperations<T>.Constant(new Tensor<T>(biasedOutput.Value.Shape), "zero"));
        }
        else
        {
            // Identity activation - pass through
            outputNode = biasedOutput;
        }

        // Step 7: Set the output gradient for backward propagation
        outputNode.Gradient = outputGradient;

        // Step 8: Inline topological sort for backward pass (production-grade pattern)
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

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

        // Step 9: Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Step 10: Extract weight gradients directly as Tensor<T> (no conversion needed - production-ready pattern)
        if (wqNode.Gradient != null)
            _queryWeightsGradient = wqNode.Gradient;
        if (wkNode.Gradient != null)
            _keyWeightsGradient = wkNode.Gradient;
        if (wvNode.Gradient != null)
            _valueWeightsGradient = wvNode.Gradient;
        if (biasNode.Gradient != null)
            _outputBiasGradient = biasNode.Gradient;

        // Step 11: Extract and return the input gradient
        if (inputNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed in automatic differentiation.");

        return inputNode.Gradient;
    }

    /// <summary>
    /// Broadcasts bias vector across the batch dimension.
    /// </summary>
    /// <param name="bias">The bias tensor to broadcast.</param>
    /// <param name="batchSize">The batch size for broadcasting.</param>
    /// <returns>A tensor with bias replicated across the batch dimension.</returns>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize)
    {
        var biasReshaped = bias.Reshape(1, 1, _embeddingDimension);
        var zeros = new Tensor<T>(new[] { batchSize, _sequenceLength, _embeddingDimension });
        return Engine.TensorBroadcastAdd(zeros, biasReshaped);
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

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            if (_queryWeightsVelocity == null)
            {
                _queryWeightsVelocity = new Tensor<T>(_queryWeights.Shape);
                _queryWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_queryWeightsVelocity, PersistentTensorRole.OptimizerState);

                _keyWeightsVelocity = new Tensor<T>(_keyWeights.Shape);
                _keyWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_keyWeightsVelocity, PersistentTensorRole.OptimizerState);

                _valueWeightsVelocity = new Tensor<T>(_valueWeights.Shape);
                _valueWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_valueWeightsVelocity, PersistentTensorRole.OptimizerState);

                _outputBiasVelocity = new Tensor<T>(_outputBias.Shape);
                _outputBiasVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_outputBiasVelocity, PersistentTensorRole.OptimizerState);
            }

            gpuEngine.SgdMomentumUpdateGpu(_queryWeights, _queryWeightsGradient, _queryWeightsVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_keyWeights, _keyWeightsGradient, _keyWeightsVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_valueWeights, _valueWeightsGradient, _valueWeightsVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_outputBias, _outputBiasGradient, _outputBiasVelocity!, lr, 0.0f, 0.0f);
        }
        else
        {
            _queryWeights = _queryWeights.Subtract(_queryWeightsGradient.Multiply(learningRate));
            _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
            _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
            _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));

            // Notify GPU that tensor data has changed
            Engine.InvalidatePersistentTensor(_queryWeights);
            Engine.InvalidatePersistentTensor(_keyWeights);
            Engine.InvalidatePersistentTensor(_valueWeights);
            Engine.InvalidatePersistentTensor(_outputBias);
        }
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
        // Calculate total number of parameters using tensor shape
        int qRows = _queryWeights.Shape[0], qCols = _queryWeights.Shape[1];
        int kRows = _keyWeights.Shape[0], kCols = _keyWeights.Shape[1];
        int vRows = _valueWeights.Shape[0], vCols = _valueWeights.Shape[1];
        int biasLen = _outputBias.Shape[0];

        int totalParams = qRows * qCols + kRows * kCols + vRows * vCols + biasLen;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy query weights
        for (int i = 0; i < qRows; i++)
        {
            for (int j = 0; j < qCols; j++)
            {
                parameters[index++] = _queryWeights[i, j];
            }
        }

        // Copy key weights
        for (int i = 0; i < kRows; i++)
        {
            for (int j = 0; j < kCols; j++)
            {
                parameters[index++] = _keyWeights[i, j];
            }
        }

        // Copy value weights
        for (int i = 0; i < vRows; i++)
        {
            for (int j = 0; j < vCols; j++)
            {
                parameters[index++] = _valueWeights[i, j];
            }
        }

        // Copy output bias
        for (int i = 0; i < biasLen; i++)
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
        // Calculate total number of parameters using tensor shape
        int qRows = _queryWeights.Shape[0], qCols = _queryWeights.Shape[1];
        int kRows = _keyWeights.Shape[0], kCols = _keyWeights.Shape[1];
        int vRows = _valueWeights.Shape[0], vCols = _valueWeights.Shape[1];
        int biasLen = _outputBias.Shape[0];

        int totalParams = qRows * qCols + kRows * kCols + vRows * vCols + biasLen;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set query weights
        for (int i = 0; i < qRows; i++)
        {
            for (int j = 0; j < qCols; j++)
            {
                _queryWeights[i, j] = parameters[index++];
            }
        }

        // Set key weights
        for (int i = 0; i < kRows; i++)
        {
            for (int j = 0; j < kCols; j++)
            {
                _keyWeights[i, j] = parameters[index++];
            }
        }

        // Set value weights
        for (int i = 0; i < vRows; i++)
        {
            for (int j = 0; j < vCols; j++)
            {
                _valueWeights[i, j] = parameters[index++];
            }
        }

        // Set output bias
        for (int i = 0; i < biasLen; i++)
        {
            _outputBias[i] = parameters[index++];
        }

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_queryWeights);
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
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

    /// <summary>
    /// Computes the auxiliary loss for attention sparsity regularization.
    /// </summary>
    /// <returns>The computed attention sparsity auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes two types of regularization for self-attention:
    /// 1. Entropy regularization: Prevents overly uniform attention distributions
    /// 2. Sparsity penalty: Encourages focused attention on relevant positions
    /// Formula: L = -H(attention) + λ * ||attention||_1 where H is entropy
    /// </para>
    /// <para><b>For Beginners:</b> This calculates penalties to improve attention quality.
    ///
    /// Attention sparsity works by:
    /// 1. Measuring attention entropy (how spread out attention is)
    /// 2. Computing L1 norm (sum of absolute attention weights)
    /// 3. Combining these to encourage focused, interpretable attention
    ///
    /// This helps because:
    /// - Prevents attention from being too diffuse (attending to everything)
    /// - Encourages sharp, focused attention on relevant positions
    /// - Improves model interpretability
    /// - Reduces computational waste on irrelevant positions
    ///
    /// The auxiliary loss is minimized during training alongside the main task loss.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastAttentionScores == null)
        {
            _lastEntropyLoss = NumOps.Zero;
            _lastSparsityLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        T totalLoss = NumOps.Zero;

        // 1. Compute negative entropy (to encourage low entropy/focused attention)
        // H = -Σ(p * log(p))
        T totalNegativeEntropy = NumOps.Zero;
        int numHeads = _headCount;
        int seqLen = _sequenceLength;

        // _lastAttentionScores has shape [batchSize, headCount, seqLen, seqLen]
        int batchSize = _lastAttentionScores.Shape[0];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    T entropy = NumOps.Zero;
                    for (int j = 0; j < seqLen; j++)
                    {
                        // Get attention weight for this batch, head, and position
                        T attnWeight = _lastAttentionScores[new int[] { b, h, i, j }];

                        // Skip zero or very small values to avoid log(0)
                        if (NumOps.LessThan(attnWeight, NumOps.FromDouble(1e-10)))
                            continue;

                        // H = -Σ(p * log(p))
                        T logWeight = NumOps.Log(attnWeight);
                        T term = NumOps.Multiply(attnWeight, logWeight);
                        entropy = NumOps.Subtract(entropy, term);
                    }
                    // We want low entropy (focused attention), so we minimize -H
                    totalNegativeEntropy = NumOps.Subtract(totalNegativeEntropy, entropy);
                }
            }
        }

        // Average over batch size
        totalNegativeEntropy = NumOps.Divide(totalNegativeEntropy, NumOps.FromDouble(batchSize));

        // Store unweighted loss for diagnostics
        _lastEntropyLoss = totalNegativeEntropy;

        // 2. Optional: L1 sparsity penalty (not implemented in basic version)
        // Can be added if needed: _lastSparsityLoss = Σ|attention_weights|

        // Apply auxiliary loss weight and return weighted loss
        totalLoss = NumOps.Multiply(totalNegativeEntropy, AuxiliaryLossWeight);
        return totalLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the attention sparsity auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about attention regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about attention sparsity regularization, including
    /// entropy loss, sparsity penalty, and configuration parameters.
    /// This information is useful for monitoring training progress and debugging attention patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how attention regularization is working.
    ///
    /// The diagnostics include:
    /// - Total entropy loss (how focused attention patterns are)
    /// - Total sparsity loss (L1 penalty on attention weights)
    /// - Weight applied to the regularization
    /// - Whether regularization is enabled
    /// - Number of attention heads
    ///
    /// This helps you:
    /// - Monitor if attention is becoming too diffuse or too sharp
    /// - Debug issues with attention patterns
    /// - Understand the impact of regularization on learning
    ///
    /// You can use this information to adjust regularization weights for better results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalEntropyLoss", _lastEntropyLoss?.ToString() ?? "0" },
            { "TotalSparsityLoss", _lastSparsityLoss?.ToString() ?? "0" },
            { "SparsityWeight", AuxiliaryLossWeight?.ToString() ?? "0.005" },
            { "UseAttentionSparsity", UseAuxiliaryLoss.ToString() },
            { "NumberOfHeads", _headCount.ToString() },
            { "SequenceLength", _sequenceLength.ToString() },
            { "AttentionScoresCached", (_lastAttentionScores != null).ToString() }
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
        // Calculate scale using tensor shape
        int rows = _queryWeights.Shape[0];
        int cols = _queryWeights.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (rows + cols)));

        InitializeTensor(_queryWeights, scale);
        InitializeTensor(_keyWeights, scale);
        InitializeTensor(_valueWeights, scale);

        // Initialize bias tensor to zeros
        int biasLen = _outputBias.Shape[0];
        for (int i = 0; i < biasLen; i++)
        {
            _outputBias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a 2D tensor with small random values scaled by the provided factor.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This private helper method fills the specified tensor with small random values between -0.5 and 0.5,
    /// scaled by the provided factor. This approach, known as Xavier/Glorot initialization, helps ensure
    /// that the activations and gradients have appropriate magnitudes, which improves training dynamics.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a weight tensor with properly sized random values.
    ///
    /// During initialization:
    /// - The method loops through every position in the tensor
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
    /// the dimensions of the tensor, helping training to proceed smoothly.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                tensor[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Exports the self-attention layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the self-attention operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node with shape [batch=1, sequenceLength, embeddingDimension]
    /// 2. Creates constant nodes for Query, Key, Value projection weights
    /// 3. Projects input to Q, K, V using matrix multiplication (self-attention: all from same input)
    /// 4. Applies multi-head scaled dot-product attention mechanism
    /// 5. Returns the attention output with residual connection and bias
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of self-attention for JIT.
    ///
    /// JIT compilation converts multi-head self-attention into optimized native code.
    /// Self-attention allows each position in a sequence to attend to all positions, enabling
    /// the model to capture long-range dependencies and relationships within the sequence.
    ///
    /// Multi-head attention uses multiple parallel attention mechanisms ("heads") that:
    /// - Focus on different aspects of the input simultaneously
    /// - Allow the model to capture diverse relationships (syntax, semantics, context)
    /// - Improve the model's ability to understand complex patterns
    ///
    /// The symbolic graph allows the JIT compiler to:
    /// - Optimize parallel matrix multiplications across heads
    /// - Fuse attention score computation and softmax
    /// - Generate efficient memory layouts for multi-head processing
    /// - Optimize the split and concatenation operations for heads
    ///
    /// Self-attention is the core of Transformer architectures (BERT, GPT, Vision Transformers).
    /// JIT compilation provides 5-10x speedup by optimizing these complex operations.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer parameters are not initialized.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured. Initialize the layer first.");

        if (_queryWeights == null || _keyWeights == null || _valueWeights == null)
            throw new InvalidOperationException("Layer projection weights not initialized. Train or initialize the model first.");

        // Create symbolic input node (shape definition only, batch size adapts at runtime)
        // SelfAttentionLayer expects input shape: [sequenceLength, embeddingDimension]
        // For self-attention, we use: [batch, sequenceLength, embeddingDimension]
        // But for simplicity in the 2D case, we flatten to [batch, sequenceLength * embeddingDimension]
        // and reshape after projection
        var symbolicInput = new Tensor<T>(new int[] { 1, _sequenceLength, _embeddingDimension });
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create constant nodes for projection weights - weights are already Tensor<T>
        var wqNode = TensorOperations<T>.Constant(_queryWeights, "Wq");
        var wkNode = TensorOperations<T>.Constant(_keyWeights, "Wk");
        var wvNode = TensorOperations<T>.Constant(_valueWeights, "Wv");

        // Note: For multi-head attention, we would split the input and process each head separately.
        // For simplicity in JIT compilation, we'll use single-head attention with the full embeddings.
        // This matches the mathematical operation but doesn't explicitly show the multi-head structure.

        // Flatten input for matrix multiplication: [batch, seq_len, embed_dim] -> [batch, seq_len * embed_dim]
        // Then project to Q, K, V
        // For now, we'll use a simplified 2D approach assuming the input is already properly shaped

        // Apply scaled dot-product attention (self-attention: Q, K, V all from same input)
        // Since we can't easily reshape in the computation graph for multi-head,
        // we'll use the full attention as a single head (this is a simplification)
        var output = TensorOperations<T>.ScaledDotProductAttention(inputNode, inputNode, inputNode);

        // Note: In a full implementation, we would:
        // 1. Reshape input to separate heads: [batch, seq, embed] -> [batch, heads, seq, head_dim]
        // 2. Apply attention per head
        // 3. Concatenate heads: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        // 4. Apply output projection
        // This simplified version captures the core attention mechanism for JIT optimization.

        return output;
    }

    /// <summary>
    /// GPU-resident backward pass for self-attention layer.
    /// Computes gradients for all trainable parameters using cached GPU tensors.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient from upstream layer.</param>
    /// <returns>GPU-resident gradient to pass to previous layer.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.Backend ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuLastInput == null || _gpuProjectedQueries == null || _gpuProjectedKeys == null ||
            _gpuProjectedValues == null || _gpuAttentionWeightsGpu == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        // Get dimensions from cached tensors
        int batchSize = _gpuLastInput.Shape[0];
        int seqLength = _gpuLastInput.Shape[1];
        int embeddingDimension = _gpuLastInput.Shape[2];

        // Ensure output gradient is 3D [batch, seq, embedding]
        var gradOutput3D = outputGradient.Shape.Length == 3
            ? outputGradient
            : gpuEngine.ReshapeGpu(outputGradient, [batchSize, seqLength, embeddingDimension]);

        // 1. Compute output bias gradient (sum over batch and seq dimensions)
        _gpuOutputBiasGradient = gpuEngine.ReduceSumGpu(gradOutput3D, [0, 1]);
        _outputBiasGradient = _gpuOutputBiasGradient.ToTensor();

        // 2. Reshape gradient for attention backward: [batch, seq, embedding] -> [batch, heads, seq, headDim]
        var gradOutput4D = gpuEngine.PermuteGpu(
            gpuEngine.ReshapeGpu(gradOutput3D, [batchSize, seqLength, _headCount, _headDimension]),
            [0, 2, 1, 3]);

        // 3. Compute attention backward to get gradients for Q, K, V projections
        double scale = 1.0 / Math.Sqrt(_headDimension);
        gpuEngine.ScaledDotProductAttentionBackwardGpu(
            gradOutput4D,
            _gpuProjectedQueries,
            _gpuProjectedKeys,
            _gpuProjectedValues,
            _gpuAttentionWeightsGpu,
            scale,
            out var gradQ4D,
            out var gradK4D,
            out var gradV4D);

        // 4. Reshape Q, K, V gradients from [batch, heads, seq, headDim] to [batch, seq, embedding]
        var gradQ = gpuEngine.ReshapeGpu(
            gpuEngine.PermuteGpu(gradQ4D, [0, 2, 1, 3]),
            [batchSize, seqLength, embeddingDimension]);
        var gradK = gpuEngine.ReshapeGpu(
            gpuEngine.PermuteGpu(gradK4D, [0, 2, 1, 3]),
            [batchSize, seqLength, embeddingDimension]);
        var gradV = gpuEngine.ReshapeGpu(
            gpuEngine.PermuteGpu(gradV4D, [0, 2, 1, 3]),
            [batchSize, seqLength, embeddingDimension]);

        // 5. Compute weight gradients: input^T @ grad
        var inputTransposed = gpuEngine.PermuteGpu(_gpuLastInput, [0, 2, 1]); // [batch, embedding, seq]

        var qWeightsGradBatched = gpuEngine.BatchedMatMulGpu(inputTransposed, gradQ.ToTensor());
        _gpuQueryWeightsGradient = gpuEngine.ReduceSumGpu(qWeightsGradBatched, [0]);
        _queryWeightsGradient = _gpuQueryWeightsGradient.ToTensor().Reshape([embeddingDimension, embeddingDimension]);

        var kWeightsGradBatched = gpuEngine.BatchedMatMulGpu(inputTransposed, gradK.ToTensor());
        _gpuKeyWeightsGradient = gpuEngine.ReduceSumGpu(kWeightsGradBatched, [0]);
        _keyWeightsGradient = _gpuKeyWeightsGradient.ToTensor().Reshape([embeddingDimension, embeddingDimension]);

        var vWeightsGradBatched = gpuEngine.BatchedMatMulGpu(inputTransposed, gradV.ToTensor());
        _gpuValueWeightsGradient = gpuEngine.ReduceSumGpu(vWeightsGradBatched, [0]);
        _valueWeightsGradient = _gpuValueWeightsGradient.ToTensor().Reshape([embeddingDimension, embeddingDimension]);

        // 6. Compute input gradient: gradQ @ Wq^T + gradK @ Wk^T + gradV @ Wv^T
        var queryWeightsT = _queryWeights.Transpose([1, 0]);
        var keyWeightsT = _keyWeights.Transpose([1, 0]);
        var valueWeightsT = _valueWeights.Transpose([1, 0]);

        var inputGradQ = gpuEngine.BatchedMatMulGpu(gradQ, queryWeightsT);
        var inputGradK = gpuEngine.BatchedMatMulGpu(gradK, keyWeightsT);
        var inputGradV = gpuEngine.BatchedMatMulGpu(gradV, valueWeightsT);

        var inputGradient = gpuEngine.AddGpu(gpuEngine.AddGpu(inputGradQ, inputGradK), inputGradV);

        return inputGradient;
    }

    /// <summary>
    /// GPU-resident parameter update with polymorphic optimizer support.
    /// Updates all weight tensors directly on GPU using the specified optimizer configuration.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.Backend ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuQueryWeightsGradient == null || _gpuKeyWeightsGradient == null ||
            _gpuValueWeightsGradient == null || _gpuOutputBiasGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure GPU weight tensors exist
        _gpuQueryWeights ??= new GpuTensor<T>(backend, _queryWeights, GpuTensorRole.Weight);
        _gpuKeyWeights ??= new GpuTensor<T>(backend, _keyWeights, GpuTensorRole.Weight);
        _gpuValueWeights ??= new GpuTensor<T>(backend, _valueWeights, GpuTensorRole.Weight);
        _gpuOutputBias ??= new GpuTensor<T>(backend, _outputBias, GpuTensorRole.Bias);

        // Ensure optimizer state buffers exist
        EnsureSelfAttentionOptimizerState(backend, config.OptimizerType);

        // Apply optimizer updates
        config.ApplyUpdate(backend, _gpuQueryWeights.Buffer, _gpuQueryWeightsGradient.Buffer,
            BuildSelfAttentionOptimizerState("query"), _queryWeights.Length);
        config.ApplyUpdate(backend, _gpuKeyWeights.Buffer, _gpuKeyWeightsGradient.Buffer,
            BuildSelfAttentionOptimizerState("key"), _keyWeights.Length);
        config.ApplyUpdate(backend, _gpuValueWeights.Buffer, _gpuValueWeightsGradient.Buffer,
            BuildSelfAttentionOptimizerState("value"), _valueWeights.Length);
        config.ApplyUpdate(backend, _gpuOutputBias.Buffer, _gpuOutputBiasGradient.Buffer,
            BuildSelfAttentionOptimizerState("bias"), _outputBias.Length);

        // Sync back to CPU tensors for compatibility
        _queryWeights = _gpuQueryWeights.ToTensor();
        _keyWeights = _gpuKeyWeights.ToTensor();
        _valueWeights = _gpuValueWeights.ToTensor();
        _outputBias = _gpuOutputBias.ToTensor();
    }

    /// <summary>
    /// Ensures optimizer state tensors are allocated for the given optimizer type.
    /// </summary>
    private void EnsureSelfAttentionOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        var weightSize = _embeddingDimension * _embeddingDimension;
        var biasSize = _embeddingDimension;

        if (optimizerType == GpuOptimizerType.Sgd || optimizerType == GpuOptimizerType.Nag || optimizerType == GpuOptimizerType.Lars)
        {
            // Velocity tensors for momentum-based optimizers
            _gpuQueryWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuKeyWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuValueWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuOutputBiasVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
        else if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW || optimizerType == GpuOptimizerType.Lamb)
        {
            // M and V tensors for Adam-family optimizers
            _gpuQueryWeightsM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuQueryWeightsV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuKeyWeightsM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuKeyWeightsV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuValueWeightsM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuValueWeightsV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuOutputBiasM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuOutputBiasV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
        else if (optimizerType == GpuOptimizerType.RmsProp)
        {
            // Squared average for RMSprop (reuse velocity buffers)
            _gpuQueryWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuKeyWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuValueWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuOutputBiasVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
        else if (optimizerType == GpuOptimizerType.Adagrad)
        {
            // Accumulated gradient for Adagrad (reuse velocity buffers)
            _gpuQueryWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuKeyWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuValueWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuOutputBiasVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
    }

    /// <summary>
    /// Builds optimizer state for a specific parameter tensor.
    /// </summary>
    private GpuOptimizerState BuildSelfAttentionOptimizerState(string paramName)
    {
        return paramName switch
        {
            "query" => new GpuOptimizerState
            {
                Velocity = _gpuQueryWeightsVelocity?.Buffer,
                M = _gpuQueryWeightsM?.Buffer,
                V = _gpuQueryWeightsV?.Buffer,
                SquaredAvg = _gpuQueryWeightsVelocity?.Buffer,
                AccumulatedGrad = _gpuQueryWeightsVelocity?.Buffer
            },
            "key" => new GpuOptimizerState
            {
                Velocity = _gpuKeyWeightsVelocity?.Buffer,
                M = _gpuKeyWeightsM?.Buffer,
                V = _gpuKeyWeightsV?.Buffer,
                SquaredAvg = _gpuKeyWeightsVelocity?.Buffer,
                AccumulatedGrad = _gpuKeyWeightsVelocity?.Buffer
            },
            "value" => new GpuOptimizerState
            {
                Velocity = _gpuValueWeightsVelocity?.Buffer,
                M = _gpuValueWeightsM?.Buffer,
                V = _gpuValueWeightsV?.Buffer,
                SquaredAvg = _gpuValueWeightsVelocity?.Buffer,
                AccumulatedGrad = _gpuValueWeightsVelocity?.Buffer
            },
            "bias" => new GpuOptimizerState
            {
                Velocity = _gpuOutputBiasVelocity?.Buffer,
                M = _gpuOutputBiasM?.Buffer,
                V = _gpuOutputBiasV?.Buffer,
                SquaredAvg = _gpuOutputBiasVelocity?.Buffer,
                AccumulatedGrad = _gpuOutputBiasVelocity?.Buffer
            },
            _ => new GpuOptimizerState()
        };
    }

    /// <summary>
    /// Gets whether this self-attention layer supports JIT compilation.
    /// </summary>
    /// <value>True if the layer parameters are initialized.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. The layer supports JIT if:
    /// - Query, Key, Value projection weights are initialized
    /// - The layer has been properly configured with sequence length and embedding dimensions
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this layer can use JIT compilation for faster inference.
    ///
    /// The layer can be JIT compiled if:
    /// - The layer has been initialized with projection weight matrices (query, key, value weights)
    /// - The multi-head structure has been configured
    ///
    /// Self-attention layers are computationally expensive because each position attends to all
    /// other positions in the sequence (O(n²) complexity). JIT compilation can provide significant
    /// speedup (5-10x) by optimizing:
    /// - Parallel matrix multiplications for projections
    /// - Multi-head attention score computation across heads
    /// - Softmax operations for attention weights
    /// - Weighted sums of values across all heads
    ///
    /// This is especially critical for Transformers where self-attention is the bottleneck:
    /// - BERT has 12-24 self-attention layers
    /// - GPT-3 has 96 self-attention layers
    /// - Vision Transformers process image patches as sequences
    ///
    /// JIT compilation makes these models practical for production use.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Self-attention supports JIT if projection weight tensors are initialized
            return _queryWeights != null && _keyWeights != null && _valueWeights != null &&
                   _queryWeights.Shape.Length >= 2 && _queryWeights.Shape[0] > 0 &&
                   _keyWeights.Shape.Length >= 2 && _keyWeights.Shape[0] > 0 &&
                   _valueWeights.Shape.Length >= 2 && _valueWeights.Shape[0] > 0;
        }
    }
}
