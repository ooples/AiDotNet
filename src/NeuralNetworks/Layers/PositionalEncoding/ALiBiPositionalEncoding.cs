namespace AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Implements ALiBi (Attention with Linear Biases) as used in models like Bloom.
/// </summary>
/// <remarks>
/// <para>
/// ALiBi is a positional encoding method that adds a linear bias to attention scores
/// based on the distance between tokens. Unlike traditional positional encodings that
/// modify token representations, ALiBi directly modifies the attention mechanism itself.
/// It has shown excellent ability to extrapolate to longer sequences than seen during training.
/// </para>
/// <para><b>For Beginners:</b> ALiBi modifies how words "pay attention" to each other
/// based on distance.
/// 
/// Instead of adding position information to each word:
/// - ALiBi directly changes how attention works
/// - It adds a penalty that increases with distance between words
/// - Words that are far apart pay less attention to each other
/// - This bias is applied consistently at all layers
/// 
/// This is like telling the model: "Pay more attention to nearby words, and less to distant ones."
/// 
/// ALiBi is especially good at handling sequences longer than what the model was trained on.
/// For example, if you train on 1,000-word documents, ALiBi helps the model work well on 
/// 10,000-word documents too.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ALiBiPositionalEncoding<T> : PositionalEncodingBase<T>
{
    /// <summary>
    /// The pre-computed bias matrix.
    /// </summary>
    private readonly Tensor<T> _biasMatrix = default!;
    
    /// <summary>
    /// The slope parameter that controls how quickly the penalty increases with distance.
    /// </summary>
    private readonly T _slope = default!;
    
    /// <summary>
    /// A flag indicating whether we've adjusted the input format yet.
    /// </summary>
    private bool _inputFormatAdjusted = false;
    
    /// <summary>
    /// Original input shape for backward pass.
    /// </summary>
    private int[]? _originalInputShape;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ALiBiPositionalEncoding{T}"/> class.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    /// <param name="slope">The slope parameter that controls how quickly the penalty increases with distance. Default is -0.1.</param>
    public ALiBiPositionalEncoding(int maxSequenceLength, int embeddingSize, double slope = -0.1)
        : base(maxSequenceLength, embeddingSize)
    {
        _slope = NumOps.FromDouble(slope);
        _biasMatrix = new Tensor<T>([maxSequenceLength, maxSequenceLength]);
        InitializeBiasMatrix();
    }
    
    /// <summary>
    /// Initializes the bias matrix used to modify attention scores.
    /// </summary>
    private void InitializeBiasMatrix()
    {
        // ALiBi uses a linear bias matrix where the bias increases
        // with the distance between tokens
        for (int i = 0; i < _maxSequenceLength; i++)
        {
            for (int j = 0; j < _maxSequenceLength; j++)
            {
                // Calculate the absolute distance between positions
                int distance = Math.Abs(i - j);
                
                // The bias is the negative distance multiplied by the slope
                // This creates a bias that decreases as distance increases
                _biasMatrix[i, j] = NumOps.Multiply(_slope, NumOps.FromDouble(distance));
            }
        }
    }
    
    /// <summary>
    /// Applies ALiBi positional encoding to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with ALiBi positional encodings applied.</returns>
    /// <remarks>
    /// Note: This implementation assumes that attention scores will be computed directly 
    /// from the output of this layer. ALiBi is most effective when directly applied to 
    /// attention scores, so in practice, this would often be integrated into the attention
    /// mechanism rather than as a standalone encoding layer.
    /// </remarks>
    protected override Tensor<T> ApplyPositionalEncoding(Tensor<T> input)
    {
        // Store original input shape for backward pass
        _originalInputShape = input.Shape;
        
        // ALiBi works differently than other positional encodings.
        // It should be applied to attention scores rather than embeddings.
        // This implementation returns the input with a note that it needs
        // to be integrated with the attention mechanism.
        
        // As a placeholder implementation, we'll just return the input unchanged,
        // with the expectation that the bias matrix will be used in the attention layer.
        
        // In a real implementation, you'd modify the attention mechanism to add the bias
        // matrix to the attention scores before softmax.
        
        if (!_inputFormatAdjusted)
        {
            Console.WriteLine("Note: ALiBi positional encoding is most effective when integrated directly with the attention mechanism.");
            Console.WriteLine("This implementation returns the input unchanged, expecting the bias matrix to be used in the attention layer.");
            _inputFormatAdjusted = true;
        }
        
        return input;
    }
    
    /// <summary>
    /// Gets the bias matrix that should be added to attention scores.
    /// </summary>
    /// <param name="sequenceLength">The current sequence length.</param>
    /// <returns>The bias matrix for the specified sequence length.</returns>
    public Tensor<T> GetBiasMatrix(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxSequenceLength}");
        }
        
        return _biasMatrix.Slice(0, 0, sequenceLength, sequenceLength);
    }
    
    /// <summary>
    /// Performs the backward pass for ALiBi positional encoding.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    protected override Tensor<T> BackwardPositionalEncoding(Tensor<T> outputGradient)
    {
        // Since ALiBi doesn't modify the input embeddings directly,
        // gradients flow through unchanged
        return outputGradient;
    }
    
    /// <summary>
    /// Encodes positional information for a sequence.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing positional encodings for each position in the sequence.</returns>
    /// <remarks>
    /// ALiBi doesn't directly encode positions into embeddings. Instead, it provides bias values
    /// to be added to attention scores. This method returns a zero tensor as ALiBi operates
    /// differently from traditional positional encodings.
    /// </remarks>
    public override Tensor<T> Encode(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxSequenceLength}");
        }
        
        // ALiBi doesn't add anything to embeddings - it modifies attention scores
        // Return zero tensor to maintain compatibility with the interface
        return new Tensor<T>([sequenceLength, _embeddingSize]);
    }
}