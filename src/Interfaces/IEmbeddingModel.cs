using System.Threading;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for embedding models that convert text into vector representations.
/// </summary>
/// <remarks>
/// <para>
/// An embedding model transforms text into dense numerical vectors that capture semantic meaning.
/// These vectors enable similarity comparisons and are fundamental to retrieval-augmented generation.
/// The interface supports both single and batch embeddings with configurable dimensions.
/// </para>
/// <para><b>For Beginners:</b> An embedding model is like a translator that converts words into numbers.
/// 
/// Think of it like a coordinate system for meaning:
/// - Each word or sentence becomes a point in high-dimensional space
/// - Similar meanings end up close together (like "happy" near "joyful")
/// - Different meanings are far apart (like "happy" far from "sad")
/// - This lets computers understand and compare text by measuring distances
/// 
/// For example, the embedding for "cat" might be close to "kitten" and "feline",
/// but far from "democracy" or "algorithm".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public interface IEmbeddingModel<T>
{
    /// <summary>
    /// Gets the dimensionality of the embedding vectors produced by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The embedding dimension determines the size of the vector representation.
    /// Common dimensions range from 128 to 1536, with larger dimensions typically
    /// capturing more nuanced semantic relationships at the cost of memory and computation.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many numbers represent each piece of text.
    /// 
    /// Think of it like describing a person:
    /// - Low dimension (128): Basic traits like height, weight, age
    /// - High dimension (768): Detailed description including personality, preferences, habits
    /// - Very high dimension (1536): Extremely detailed profile
    /// 
    /// More dimensions = more detailed understanding, but also more storage space needed.
    /// </para>
    /// </remarks>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Embeds a single text string into a vector representation.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <returns>A vector representing the semantic meaning of the input text.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a text string into a dense vector that captures its semantic meaning.
    /// The resulting vector can be used for similarity comparisons, clustering, or as input to
    /// downstream models. Implementations should return normalized vectors for consistent similarity calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This converts one piece of text into a list of numbers.
    /// 
    /// For example:
    /// - Input: "The weather is beautiful today"
    /// - Output: A vector like [0.23, -0.45, 0.78, ..., 0.12] with EmbeddingDimension numbers
    /// 
    /// These numbers capture the meaning so similar sentences get similar number patterns.
    /// </para>
    /// </remarks>
    Vector<T> Embed(string text);

    /// <summary>
    /// Asynchronously embeds a single text string into a vector representation.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <param name="cancellationToken">Token to cancel the asynchronous operation.</param>
    /// <returns>A task representing the async operation, with the resulting vector.</returns>
    Task<Vector<T>> EmbedAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Embeds multiple text strings into vector representations in a single batch operation.
    /// </summary>
    /// <param name="texts">The collection of texts to embed.</param>
    /// <returns>A matrix where each row represents the embedding of the corresponding input text.</returns>
    /// <remarks>
    /// <para>
    /// Batch embedding is more efficient than embedding texts individually, as it allows
    /// the model to process multiple inputs simultaneously. The returned matrix has dimensions
    /// [textCount, EmbeddingDimension], where each row corresponds to one input text.
    /// </para>
    /// <para><b>For Beginners:</b> This converts many pieces of text into numbers all at once.
    /// 
    /// Think of it like processing a batch of photos:
    /// - Instead of converting one photo at a time (slow)
    /// - You process the entire album together (fast)
    /// 
    /// For example, embedding 100 sentences together is much faster than
    /// calling Embed() 100 separate times.
    /// </para>
    /// </remarks>
    Matrix<T> EmbedBatch(IEnumerable<string> texts);

    /// <summary>
    /// Asynchronously embeds multiple text strings into vector representations in a single batch operation.
    /// </summary>
    /// <param name="texts">The collection of texts to embed.</param>
    /// <param name="cancellationToken">Token to cancel the asynchronous operation.</param>
    /// <returns>A task representing the async operation, with the resulting matrix.</returns>
    Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the maximum length of text (in tokens) that this model can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Most embedding models have a maximum context length beyond which text must be truncated.
    /// Common limits range from 512 to 8192 tokens. Implementations should handle text exceeding
    /// this limit gracefully, either by truncation or raising an exception.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum amount of text the model can understand at once.
    /// 
    /// Think of it like a reader's attention span:
    /// - Short span (512 tokens): Can read about a paragraph
    /// - Medium span (2048 tokens): Can read a few pages
    /// - Long span (8192 tokens): Can read a short chapter
    /// 
    /// If your text is longer, it needs to be split into chunks.
    /// (A token is roughly a word, so 512 tokens ≈ 1-2 paragraphs)
    /// </para>
    /// </remarks>
    int MaxTokens { get; }
}
