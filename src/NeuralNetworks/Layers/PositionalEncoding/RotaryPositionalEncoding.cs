using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Implements Rotary Position Embedding (RoPE) as used in models like GPT-Neo-X and PaLM.
/// </summary>
/// <remarks>
/// <para>
/// Rotary Position Embedding (RoPE) encodes position information by applying a rotation
/// to each element of the input embedding. The rotation is performed in a way that preserves
/// the inner product between tokens at the same relative positions. This makes it particularly
/// effective for attention mechanisms.
/// </para>
/// <para><b>For Beginners:</b> This method "rotates" word representations based on their position.
/// 
/// Imagine each word embedding as a point in multi-dimensional space:
/// - RoPE applies a mathematical rotation to these points
/// - The amount of rotation depends on the position in the sequence
/// - This rotation creates a unique pattern for each position
/// - At the same time, it preserves the relationships between words
/// 
/// This clever approach has become popular in modern large language models like PaLM and GPT-Neo-X
/// because it combines the benefits of both absolute and relative positional encoding.
/// 
/// RoPE has strong mathematical properties that make it well-suited for transformer models and
/// allows them to generalize well to different sequence lengths.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RotaryPositionalEncoding<T> : PositionalEncodingBase<T>
{
    /// <summary>
    /// The pre-computed cosine values for the rotations.
    /// </summary>
    private readonly Tensor<T> _cos = default!;
    
    /// <summary>
    /// The pre-computed sine values for the rotations.
    /// </summary>
    private readonly Tensor<T> _sin = default!;
    
    /// <summary>
    /// The base frequency for the rotations.
    /// </summary>
    private readonly double _base;
    
    
    /// <summary>
    /// Initializes a new instance of the <see cref="RotaryPositionalEncoding{T}"/> class.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    /// <param name="base">The base value for frequency calculations. Default is 10000.0.</param>
    public RotaryPositionalEncoding(int maxSequenceLength, int embeddingSize, double @base = 10000.0)
        : base(maxSequenceLength, embeddingSize)
    {
        _base = @base;
        _cos = new Tensor<T>([maxSequenceLength, embeddingSize / 2]);
        _sin = new Tensor<T>([maxSequenceLength, embeddingSize / 2]);
        
        // Ensure embedding size is even for RoPE
        if (embeddingSize % 2 != 0)
        {
            throw new ArgumentException("RoPE requires an even embedding size");
        }
        
        InitializeRotations();
    }
    
    /// <summary>
    /// Initializes the cosine and sine rotation values.
    /// </summary>
    private void InitializeRotations()
    {
        int halfDim = _embeddingSize / 2;
        
        for (int pos = 0; pos < _maxSequenceLength; pos++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                double freq = 1.0 / Math.Pow(_base, (2 * i) / (double)_embeddingSize);
                double theta = pos * freq;
                
                _cos[pos, i] = NumOps.FromDouble(Math.Cos(theta));
                _sin[pos, i] = NumOps.FromDouble(Math.Sin(theta));
            }
        }
    }
    
    /// <summary>
    /// Applies rotary positional encoding to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with rotary positional encodings applied.</returns>
    protected override Tensor<T> ApplyPositionalEncoding(Tensor<T> input)
    {
        int sequenceLength = input.Shape[0];
        int halfDim = _embeddingSize / 2;
        var output = input.Clone();
        
        for (int pos = 0; pos < sequenceLength; pos++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                // Get paired dimensions (xs, ys) for rotation
                int idx1 = i * 2;
                int idx2 = i * 2 + 1;
                
                // Get original values
                T x = input[pos, idx1];
                T y = input[pos, idx2];
                
                // Get cos and sin values for rotation
                T cos = _cos[pos, i];
                T sin = _sin[pos, i];
                
                // Apply rotation: [cos_θ, -sin_θ; sin_θ, cos_θ] * [x; y]
                output[pos, idx1] = NumOps.Subtract(
                    NumOps.Multiply(cos, x),
                    NumOps.Multiply(sin, y));
                
                output[pos, idx2] = NumOps.Add(
                    NumOps.Multiply(sin, x),
                    NumOps.Multiply(cos, y));
            }
        }
        
        return output;
    }
    
    /// <summary>
    /// Performs the backward pass for rotary positional encoding.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    protected override Tensor<T> BackwardPositionalEncoding(Tensor<T> outputGradient)
    {
        int sequenceLength = outputGradient.Shape[0];
        int halfDim = _embeddingSize / 2;
        var inputGradient = outputGradient.Clone();
        
        for (int pos = 0; pos < sequenceLength; pos++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                // Get paired dimensions (xs, ys) for rotation
                int idx1 = i * 2;
                int idx2 = i * 2 + 1;
                
                // Get gradient values
                T gradX = outputGradient[pos, idx1];
                T gradY = outputGradient[pos, idx2];
                
                // Get cos and sin values for rotation
                T cos = _cos[pos, i];
                T sin = _sin[pos, i];
                
                // Apply inverse rotation: [cos_θ, sin_θ; -sin_θ, cos_θ] * [gradX; gradY]
                // (transpose of the original rotation matrix)
                inputGradient[pos, idx1] = NumOps.Add(
                    NumOps.Multiply(cos, gradX),
                    NumOps.Multiply(sin, gradY));
                
                inputGradient[pos, idx2] = NumOps.Subtract(
                    NumOps.Multiply(cos, gradY),
                    NumOps.Multiply(sin, gradX));
            }
        }
        
        return inputGradient;
    }
    
    /// <summary>
    /// Encodes positional information for a sequence.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing positional encodings for each position in the sequence.</returns>
    /// <remarks>
    /// For RoPE, this returns the rotation coefficients that would be applied to embeddings.
    /// The actual encoding is performed by rotating the input embeddings, so this method
    /// returns the identity transformation to maintain interface compatibility.
    /// </remarks>
    public override Tensor<T> Encode(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxSequenceLength}");
        }
        
        // RoPE doesn't add encodings - it rotates embeddings
        // Return an identity-like encoding (zeros) for interface compatibility
        var encodings = new Tensor<T>([sequenceLength, _embeddingSize]);
        
        // Optionally, we could return the rotation parameters themselves
        // but that would require a different tensor structure
        // For now, return zeros as RoPE is applied through rotation, not addition
        return encodings;
    }
}