using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Implements the original sinusoidal position encoding from the "Attention Is All You Need" paper.
/// </summary>
/// <remarks>
/// <para>
/// This implementation uses sine and cosine functions of different frequencies to create unique
/// positional embeddings. For each position and dimension, the encoding follows:
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// </para>
/// <para><b>For Beginners:</b> This is the classic way of encoding positions in transformers.
/// 
/// Think of it like creating a unique fingerprint for each position using a mathematical formula:
/// - For even dimensions (0, 2, 4...), we use a sine function
/// - For odd dimensions (1, 3, 5...), we use a cosine function
/// - We use different frequencies for different dimensions
/// 
/// This creates a unique pattern for each position that:
/// - Changes smoothly as you move along positions
/// - Has mathematical properties that help the model understand relative positions
/// - Can work with sequences longer than those seen during training
/// 
/// This is the method used in the original transformer paper, and it remains widely used
/// because of its simplicity and effectiveness.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SinusoidalPositionalEncoding<T> : PositionalEncodingBase<T>
{
    /// <summary>
    /// The pre-computed positional encodings tensor.
    /// </summary>
    private readonly Tensor<T> _encodings = default!;
    
    
    /// <summary>
    /// Initializes a new instance of the <see cref="SinusoidalPositionalEncoding{T}"/> class.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    public SinusoidalPositionalEncoding(int maxSequenceLength, int embeddingSize)
        : base(maxSequenceLength, embeddingSize)
    {
        _encodings = new Tensor<T>([maxSequenceLength, embeddingSize]);
        InitializeEncodings();
    }
    
    /// <summary>
    /// Initializes the positional encodings using sine and cosine functions.
    /// </summary>
    private void InitializeEncodings()
    {
        for (int pos = 0; pos < _maxSequenceLength; pos++)
        {
            for (int i = 0; i < _embeddingSize; i++)
            {
                double angle = pos / Math.Pow(10000, (2 * (i / 2)) / (double)_embeddingSize);
                if (i % 2 == 0)
                {
                    _encodings[pos, i] = NumOps.FromDouble(Math.Sin(angle));
                }
                else
                {
                    _encodings[pos, i] = NumOps.FromDouble(Math.Cos(angle));
                }
            }
        }
    }
    
    /// <summary>
    /// Applies sinusoidal positional encoding to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with positional encodings added.</returns>
    protected override Tensor<T> ApplyPositionalEncoding(Tensor<T> input)
    {
        var slicedEncodings = _encodings.Slice(0, 0, input.Shape[0], _embeddingSize);
        // Use tensor addition method instead of direct operator to follow INumericOperations pattern
        return input.Add(slicedEncodings);
    }
    
    /// <summary>
    /// Performs the backward pass for sinusoidal positional encoding.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    protected override Tensor<T> BackwardPositionalEncoding(Tensor<T> outputGradient)
    {
        // With sinusoidal encodings, gradients flow through unchanged
        return outputGradient;
    }
    
    /// <summary>
    /// Generates positional encodings for a specific sequence length.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing the positional encodings.</returns>
    public override Tensor<T> Encode(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum sequence length {_maxSequenceLength}");
        }
        
        // Return a slice of the pre-computed encodings
        return _encodings.Slice(0, 0, sequenceLength, _embeddingSize);
    }
}