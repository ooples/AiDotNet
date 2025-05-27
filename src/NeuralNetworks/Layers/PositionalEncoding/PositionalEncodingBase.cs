using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Base class for different positional encoding implementations.
/// </summary>
/// <remarks>
/// <para>
/// Positional encoding is a critical component of transformer models, providing sequence order information
/// that would otherwise be lost in the self-attention mechanism. This base class provides a common
/// interface for all positional encoding implementations.
/// </para>
/// <para><b>For Beginners:</b> Transformers need to know the order of words in a sentence.
/// 
/// Since transformers process all words at once (in parallel), they need some way to know which word
/// comes first, second, third, etc. Positional encodings add this information to each word's representation.
/// 
/// This base class defines the common structure for all the different ways we can encode position information,
/// like using mathematical patterns or letting the model learn position representations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public abstract class PositionalEncodingBase<T> : LayerBase<T>, IPositionalEncoding<T>
{
    /// <summary>
    /// The maximum sequence length that this encoding can handle.
    /// </summary>
    protected readonly int _maxSequenceLength;
    
    /// <summary>
    /// The size of each embedding vector.
    /// </summary>
    protected readonly int _embeddingSize;
    
    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because positional encoding layers support backpropagation.
    /// </value>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the maximum sequence length that this encoding can handle.
    /// </summary>
    /// <value>
    /// The maximum number of elements in a sequence that can be encoded.
    /// </value>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>
    /// Gets the size of each embedding vector.
    /// </summary>
    /// <value>
    /// The dimensionality of the embedding vectors.
    /// </value>
    public int EmbeddingSize => _embeddingSize;
    
    
    /// <summary>
    /// Initializes a new instance of a positional encoding layer.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    protected PositionalEncodingBase(int maxSequenceLength, int embeddingSize)
        : base([maxSequenceLength, embeddingSize], [maxSequenceLength, embeddingSize])
    {
        _maxSequenceLength = maxSequenceLength;
        _embeddingSize = embeddingSize;
    }
    
    /// <summary>
    /// Performs the forward pass of the positional encoding layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with positional encodings added.</returns>
    /// <exception cref="ArgumentException">Thrown when the input sequence length exceeds the maximum sequence length.</exception>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape[0] > _maxSequenceLength)
        {
            throw new ArgumentException($"Input sequence length {input.Shape[0]} exceeds maximum sequence length {_maxSequenceLength}");
        }
        
        return ApplyPositionalEncoding(input);
    }
    
    /// <summary>
    /// Applies positional encoding to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with positional encodings applied.</returns>
    protected abstract Tensor<T> ApplyPositionalEncoding(Tensor<T> input);
    
    /// <summary>
    /// Performs the backward pass of the positional encoding layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return BackwardPositionalEncoding(outputGradient);
    }
    
    /// <summary>
    /// Applies the backward pass for the specific positional encoding implementation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    protected abstract Tensor<T> BackwardPositionalEncoding(Tensor<T> outputGradient);
    
    /// <summary>
    /// Updates the parameters of the positional encoding layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Default implementation does nothing - override in learnable encodings
    }
    
    /// <summary>
    /// Gets all trainable parameters from the positional encoding layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters, or an empty vector if there are none.</returns>
    public override Vector<T> GetParameters()
    {
        // Default implementation returns an empty vector - override in learnable encodings
        return Vector<T>.Empty();
    }
    
    /// <summary>
    /// Resets the internal state of the positional encoding layer.
    /// </summary>
    public override void ResetState()
    {
        // Default implementation does nothing - override if needed
    }

    /// <summary>
    /// Encodes positional information for a sequence.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing positional encodings for each position in the sequence.</returns>
    public abstract Tensor<T> Encode(int sequenceLength);

    /// <summary>
    /// Adds positional encoding to input embeddings.
    /// </summary>
    /// <param name="embeddings">The input embeddings to add positional information to.</param>
    /// <returns>The embeddings with positional information added.</returns>
    public virtual Tensor<T> AddPositionalEncoding(Tensor<T> embeddings)
    {
        var positionalEncodings = Encode(embeddings.Shape[0]);
        return embeddings + positionalEncodings;
    }
}