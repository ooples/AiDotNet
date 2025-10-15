namespace AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Implements relative positional encoding as used in models like Transformer-XL.
/// </summary>
/// <remarks>
/// <para>
/// Relative positional encoding focuses on the relative distance between tokens rather than their
/// absolute positions. This helps models better understand the relationships between tokens and
/// can improve performance on tasks that require understanding context across long distances.
/// </para>
/// <para><b>For Beginners:</b> This encoding focuses on "how far apart" words are, not their exact positions.
/// 
/// Instead of saying "this is word #5 in the sequence":
/// - We focus on relationships like "this word is 3 positions away from that word"
/// - This helps the model understand relative relationships better
/// - It can be especially helpful for understanding long texts or complex relationships
/// 
/// This approach is used in Transformer-XL and similar models that need to process very long
/// sequences or understand relationships between distant parts of a text.
/// 
/// The main advantage is that the model can more easily generalize to sequence lengths
/// it hasn't seen during training, since it focuses on relative distances rather than
/// absolute positions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RelativePositionalEncoding<T> : PositionalEncodingBase<T>
{
    /// <summary>
    /// The relative position embeddings.
    /// </summary>
    private readonly Tensor<T> _relativeEmbeddings = default!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="RelativePositionalEncoding{T}"/> class.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    public RelativePositionalEncoding(int maxSequenceLength, int embeddingSize)
        : base(maxSequenceLength, embeddingSize)
    {
        // We need 2*maxSequenceLength-1 relative positions (-maxSequenceLength+1 to maxSequenceLength-1)
        int numRelativePositions = 2 * maxSequenceLength - 1;
        _relativeEmbeddings = new Tensor<T>([numRelativePositions, embeddingSize]);
        InitializeRelativeEmbeddings();
    }
    
    /// <summary>
    /// Initializes the relative position embeddings using sinusoidal encoding.
    /// </summary>
    private void InitializeRelativeEmbeddings()
    {
        int numRelativePositions = 2 * _maxSequenceLength - 1;
        int centerPosition = _maxSequenceLength - 1; // Index for relative position 0
        
        for (int relPos = -_maxSequenceLength + 1; relPos < _maxSequenceLength; relPos++)
        {
            int index = relPos + centerPosition; // Convert to array index
            
            for (int i = 0; i < _embeddingSize; i++)
            {
                double angle = Math.Abs(relPos) / Math.Pow(10000, (2 * (i / 2)) / (double)_embeddingSize);
                if (i % 2 == 0)
                {
                    _relativeEmbeddings[index, i] = NumOps.FromDouble(Math.Sin(angle));
                }
                else
                {
                    _relativeEmbeddings[index, i] = NumOps.FromDouble(Math.Cos(angle));
                }
                
                // Add sign information for odd dimensions to make encoding direction-aware
                if (i % 2 == 1 && relPos < 0)
                {
                    _relativeEmbeddings[index, i] = NumOps.Negate(_relativeEmbeddings[index, i]);
                }
            }
        }
    }
    
    /// <summary>
    /// Applies relative positional encoding to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with relative positional encodings applied.</returns>
    protected override Tensor<T> ApplyPositionalEncoding(Tensor<T> input)
    {
        int sequenceLength = input.Shape[0];
        var output = input.Clone();
        int centerPosition = _maxSequenceLength - 1; // Index for relative position 0
        
        // For each position in the sequence
        for (int pos = 0; pos < sequenceLength; pos++)
        {
            // Calculate relative encodings based on positions of other tokens
            for (int refPos = 0; refPos < sequenceLength; refPos++)
            {
                int relPos = refPos - pos; // Relative position
                int relIndex = relPos + centerPosition; // Convert to array index
                
                // Add a scaled version of the relative embedding to the token representation
                var scalar = NumOps.FromDouble(1.0 / Math.Sqrt(sequenceLength));
                for (int i = 0; i < _embeddingSize; i++)
                {
                    output[pos, i] = NumOps.Add(
                        output[pos, i],
                        NumOps.Multiply(scalar, _relativeEmbeddings[relIndex, i]));
                }
            }
        }
        
        return output;
    }
    
    /// <summary>
    /// Performs the backward pass for relative positional encoding.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    protected override Tensor<T> BackwardPositionalEncoding(Tensor<T> outputGradient)
    {
        // Since we're using a fixed encoding, gradient flows through unchanged
        return outputGradient;
    }
    
    /// <summary>
    /// Encodes positional information for a sequence.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing positional encodings for each position in the sequence.</returns>
    /// <remarks>
    /// For relative positional encoding, this returns a tensor where each position contains
    /// an average of relative encodings with respect to all other positions in the sequence.
    /// </remarks>
    public override Tensor<T> Encode(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxSequenceLength}");
        }
        
        var encodings = new Tensor<T>([sequenceLength, _embeddingSize]);
        int centerPosition = _maxSequenceLength - 1; // Index for relative position 0
        
        // For each position, create an encoding based on its relative positions to all others
        for (int pos = 0; pos < sequenceLength; pos++)
        {
            // Average the relative encodings for this position
            for (int refPos = 0; refPos < sequenceLength; refPos++)
            {
                int relPos = refPos - pos; // Relative position
                int relIndex = relPos + centerPosition; // Convert to array index
                
                var scalar = NumOps.FromDouble(1.0 / sequenceLength);
                for (int i = 0; i < _embeddingSize; i++)
                {
                    encodings[pos, i] = NumOps.Add(
                        encodings[pos, i],
                        NumOps.Multiply(scalar, _relativeEmbeddings[relIndex, i]));
                }
            }
        }
        
        return encodings;
    }
}