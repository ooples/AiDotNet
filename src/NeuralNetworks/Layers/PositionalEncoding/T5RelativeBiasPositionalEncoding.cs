namespace AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Implements T5-style relative positional bias, used in text-to-text transfer transformer models.
/// </summary>
/// <remarks>
/// <para>
/// T5 (Text-to-Text Transfer Transformer) uses a relative positional bias approach where
/// learned relative position representations are added as biases to the attention scores.
/// This approach buckets relative positions into a smaller set of learnable parameters,
/// which can improve efficiency and generalization.
/// </para>
/// <para><b>For Beginners:</b> T5 uses a clever way to handle positions in long texts.
/// 
/// Instead of having a separate position code for every possible distance between words:
/// - It groups similar distances together (for example, "distance 10-20" might be one group)
/// - It learns a specific bias value for each group
/// - These biases are added to attention scores rather than to word representations
/// 
/// This approach:
/// - Requires fewer parameters than having one value per possible distance
/// - Helps the model generalize better
/// - Works well for various text-to-text tasks like translation, summarization, etc.
/// 
/// This is the method used in Google's T5 model, which achieves state-of-the-art results
/// on many natural language processing tasks.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class T5RelativeBiasPositionalEncoding<T> : PositionalEncodingBase<T>
{
    /// <summary>
    /// The number of buckets used to discretize relative positions.
    /// </summary>
    private readonly int _numBuckets;
    
    /// <summary>
    /// The maximum relative distance before clipping.
    /// </summary>
    private readonly int _maxDistance;
    
    /// <summary>
    /// Whether to use bidirectional bucketing.
    /// </summary>
    private readonly bool _bidirectional;
    
    /// <summary>
    /// The learnable relative position embeddings.
    /// </summary>
    private Vector<T> _relativeBias = default!;
    
    
    /// <summary>
    /// Flag indicating if input format has been adjusted.
    /// </summary>
    private bool _inputFormatAdjusted = false;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="T5RelativeBiasPositionalEncoding{T}"/> class.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    /// <param name="numBuckets">The number of buckets to use for relative positions. Default is 32.</param>
    /// <param name="maxDistance">The maximum relative distance to consider before clipping. Default is 128.</param>
    /// <param name="bidirectional">Whether to use bidirectional bucketing. Default is true.</param>
    public T5RelativeBiasPositionalEncoding(
        int maxSequenceLength, 
        int embeddingSize,
        int numBuckets = 32,
        int maxDistance = 128,
        bool bidirectional = true)
        : base(maxSequenceLength, embeddingSize)
    {
        _numBuckets = numBuckets;
        _maxDistance = maxDistance;
        _bidirectional = bidirectional;
        
        // Initialize the relative bias vector with small random values
        _relativeBias = new Vector<T>(_numBuckets);
        InitializeRelativeBias();
    }
    
    /// <summary>
    /// Initializes the relative bias vector with small random values.
    /// </summary>
    private void InitializeRelativeBias()
    {
        // Initialize with values from a normal distribution with small variance
        Random random = new Random(42); // Fixed seed for reproducibility
        double stdDev = 0.02; // Small standard deviation
        
        for (int i = 0; i < _numBuckets; i++)
        {
            // Box-Muller transform to generate normal distribution
            double u1 = 1.0 - random.NextDouble(); // Uniform(0,1) random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            double randNormal = randStdNormal * stdDev;
            
            _relativeBias[i] = NumOps.FromDouble(randNormal);
        }
    }
    
    /// <summary>
    /// Applies T5-style relative positional encoding to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with T5-style relative positional encodings applied.</returns>
    /// <remarks>
    /// Note: This implementation assumes integration with the attention mechanism.
    /// In practice, the relative bias would be added to attention scores before softmax.
    /// </remarks>
    protected override Tensor<T> ApplyPositionalEncoding(Tensor<T> input)
    {
        // T5 relative positional bias works differently than other encodings.
        // It should be applied to attention scores rather than embeddings.
        // This implementation returns the input unchanged with a note that it needs
        // to be integrated with the attention mechanism.
        
        if (!_inputFormatAdjusted)
        {
            Console.WriteLine("Note: T5 relative positional bias is typically integrated directly with the attention mechanism.");
            Console.WriteLine("This implementation returns the input unchanged, expecting the bias to be used in the attention layer.");
            _inputFormatAdjusted = true;
        }
        
        return input;
    }
    
    /// <summary>
    /// Computes the bucket index for a relative position.
    /// </summary>
    /// <param name="relPos">The relative position.</param>
    /// <returns>The bucket index for the relative position.</returns>
    public int RelativePositionBucket(int relPos)
    {
        int relPosAbs = Math.Abs(relPos);
        
        // Half of the buckets are for exact relative positions
        int maxExact = _numBuckets / 2;
        bool isSmall = relPosAbs < maxExact;
        
        // For the other half, we divide the range from maxExact to maxDistance into buckets
        int bucketPos = isSmall ? relPosAbs : maxExact + (relPosAbs - maxExact) * (_numBuckets - maxExact) / (_maxDistance - maxExact);
        
        // If bidirectional, use different buckets for negative relative positions
        return _bidirectional && relPos < 0 ? _numBuckets - bucketPos - 1 : bucketPos;
    }
    
    /// <summary>
    /// Gets the bias value for a relative position.
    /// </summary>
    /// <param name="relPos">The relative position.</param>
    /// <returns>The bias value for the relative position.</returns>
    public T GetBias(int relPos)
    {
        int bucket = RelativePositionBucket(relPos);
        return _relativeBias[bucket];
    }
    
    /// <summary>
    /// Gets the relative position bias matrix for a specific sequence length.
    /// </summary>
    /// <param name="sequenceLength">The current sequence length.</param>
    /// <returns>A tensor containing bias values for each query-key pair.</returns>
    public Tensor<T> GetRelativeBiasMatrix(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxSequenceLength}");
        }
        
        var biasMatrix = new Tensor<T>([sequenceLength, sequenceLength]);
        
        for (int i = 0; i < sequenceLength; i++)
        {
            for (int j = 0; j < sequenceLength; j++)
            {
                int relPos = j - i;  // Relative position
                biasMatrix[i, j] = GetBias(relPos);
            }
        }
        
        return biasMatrix;
    }
    
    /// <summary>
    /// Performs the backward pass for T5-style relative positional encoding.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    protected override Tensor<T> BackwardPositionalEncoding(Tensor<T> outputGradient)
    {
        // Since T5 relative positional bias doesn't modify the input embeddings directly,
        // gradients flow through unchanged
        return outputGradient;
    }
    
    /// <summary>
    /// Updates the parameters of the layer using the provided gradients.
    /// </summary>
    /// <param name="biasGradients">Gradients for the relative position biases.</param>
    public void UpdateBiasParameters(Vector<T> biasGradients, T learningRate)
    {
        if (biasGradients.Length != _numBuckets)
        {
            throw new ArgumentException($"Expected bias gradients vector of length {_numBuckets}, but got {biasGradients.Length}");
        }
        
        for (int i = 0; i < _numBuckets; i++)
        {
            _relativeBias[i] = NumOps.Subtract(
                _relativeBias[i],
                NumOps.Multiply(learningRate, biasGradients[i]));
        }
    }
    
    /// <summary>
    /// Gets all trainable parameters from the layer.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    public override Vector<T> GetParameters()
    {
        return _relativeBias;
    }
    
    /// <summary>
    /// Updates the parameters of the layer using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing new values for all trainable parameters.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != _numBuckets)
        {
            throw new ArgumentException($"Expected parameter vector of length {_numBuckets}, but got {parameters.Length}");
        }
        
        _relativeBias = parameters;
    }
    
    /// <summary>
    /// Gets the number of trainable parameters in this layer.
    /// </summary>
    public override int ParameterCount => _numBuckets;
    
    /// <summary>
    /// Encodes positional information for a sequence.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing positional encodings for each position in the sequence.</returns>
    /// <remarks>
    /// T5 relative bias doesn't directly encode positions into embeddings. Instead, it provides
    /// bias values to be added to attention scores. This method returns a zero tensor as T5
    /// operates differently from traditional positional encodings.
    /// </remarks>
    public override Tensor<T> Encode(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxSequenceLength}");
        }
        
        // T5 relative bias doesn't add anything to embeddings - it modifies attention scores
        // Return zero tensor to maintain compatibility with the interface
        return new Tensor<T>([sequenceLength, _embeddingSize]);
    }
}