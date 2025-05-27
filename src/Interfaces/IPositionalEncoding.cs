using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for positional encoding implementations used in transformer architectures.
/// </summary>
/// <remarks>
/// <para>
/// Positional encoding is essential for transformer models to understand the order of elements in a sequence.
/// Since transformers process all elements in parallel, they need explicit position information to understand
/// the sequential nature of the input data.
/// </para>
/// <para><b>For Beginners:</b> This interface defines what all positional encoding methods must be able to do.
/// 
/// Think of positional encoding like adding address labels to houses on a street:
/// - Without addresses, you can't tell which house comes first, second, etc.
/// - Similarly, transformers need position information to understand word order in sentences
/// - This interface ensures all positional encoding methods can provide this information consistently
/// 
/// Different implementations might use:
/// - Mathematical patterns (sine/cosine waves)
/// - Learned position embeddings
/// - Relative position information
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public interface IPositionalEncoding<T>
{
    /// <summary>
    /// Gets the maximum sequence length that this encoding can handle.
    /// </summary>
    /// <value>
    /// The maximum number of elements in a sequence that can be encoded.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates the maximum length of sequences that this positional encoding can handle.
    /// Different encoding methods may have different limitations based on their implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you the longest sequence this encoding can handle.
    /// 
    /// For example:
    /// - If MaxSequenceLength is 512, you can encode sequences up to 512 elements long
    /// - This might be 512 words in a sentence or 512 tokens in a document
    /// - Longer sequences would need to be split into smaller chunks
    /// </para>
    /// </remarks>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the size of each embedding vector.
    /// </summary>
    /// <value>
    /// The dimensionality of the embedding vectors.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates the size of the embedding vectors that this positional encoding works with.
    /// The positional encoding must match the embedding size of the model it's used with.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many dimensions each position encoding has.
    /// 
    /// For example:
    /// - If EmbeddingSize is 768, each position gets a 768-dimensional vector
    /// - This must match the model's embedding size (like 512, 768, or 1024)
    /// - Higher dimensions allow for more complex position representations
    /// </para>
    /// </remarks>
    int EmbeddingSize { get; }

    /// <summary>
    /// Encodes positional information for a sequence.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing positional encodings for each position in the sequence.</returns>
    /// <remarks>
    /// <para>
    /// This method generates positional encodings for a sequence of the specified length.
    /// The returned tensor has shape [sequenceLength, embeddingSize], where each row
    /// contains the positional encoding for that position in the sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates position information for each element in a sequence.
    /// 
    /// For example, if you have a sentence with 10 words:
    /// - You call Encode(10)
    /// - You get back a tensor with 10 rows (one for each word position)
    /// - Each row contains the position encoding for that word's position
    /// - Position 0 gets one encoding, position 1 gets another, etc.
    /// 
    /// The encoding helps the model understand "this is the first word, this is the second word, etc."
    /// </para>
    /// </remarks>
    Tensor<T> Encode(int sequenceLength);

    /// <summary>
    /// Adds positional encoding to input embeddings.
    /// </summary>
    /// <param name="embeddings">The input embeddings to add positional information to.</param>
    /// <returns>The embeddings with positional information added.</returns>
    /// <remarks>
    /// <para>
    /// This method adds positional information to existing embeddings. The input tensor should have
    /// shape [sequenceLength, embeddingSize], and the method returns a tensor of the same shape
    /// with positional encodings added to each embedding vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method combines word meanings with position information.
    /// 
    /// Think of it like adding GPS coordinates to a list of landmarks:
    /// - You start with embeddings that represent what each word means
    /// - You add position encodings that represent where each word is located
    /// - The result combines both "what" (meaning) and "where" (position)
    /// 
    /// For example:
    /// - Input: word embeddings for "The cat sat"
    /// - Output: the same embeddings but now each word also knows its position
    /// - "The" knows it's first, "cat" knows it's second, "sat" knows it's third
    /// </para>
    /// </remarks>
    Tensor<T> AddPositionalEncoding(Tensor<T> embeddings);
}