using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for compressing context documents to reduce token usage while preserving relevance.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// A context compressor reduces the size of retrieved documents before passing them to a language model,
/// helping to stay within token limits while maintaining the most relevant information.
/// </para>
/// <para><b>For Beginners:</b> A context compressor is like a summarizer for search results.
/// 
/// Think of it like preparing a briefing:
/// - You retrieve 10 long documents (might be 50,000 tokens)
/// - Your language model can only handle 8,000 tokens
/// - The compressor extracts key sentences and information
/// - Result: Compressed to 5,000 tokens with the most important content
/// 
/// This ensures you can:
/// - Use more retrieved documents without hitting token limits
/// - Focus on the most relevant parts of each document
/// - Reduce costs by using fewer tokens
/// - Still get accurate answers from the compressed context
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ContextCompressor")]
public interface IContextCompressor<T>
{
    /// <summary>
    /// Compresses a collection of documents while preserving relevance to the query.
    /// </summary>
    /// <param name="documents">The documents to compress.</param>
    /// <param name="query">The query text used to determine relevance.</param>
    /// <param name="options">Optional compression parameters.</param>
    /// <returns>The compressed documents with reduced content but maintained relevance.</returns>
    /// <remarks>
    /// <para>
    /// This method takes documents and reduces their content size while keeping the parts
    /// most relevant to the query. The compression strategy varies by implementation - some
    /// may use extractive methods (selecting key sentences), others may use abstractive
    /// methods (generating summaries), or hybrid approaches.
    /// </para>
    /// <para><b>For Beginners:</b> This reduces document size while keeping important information.
    /// 
    /// For example:
    /// - Input: 3 documents, each 2000 tokens (6000 total)
    /// - Query: "How do I train a neural network?"
    /// - Processing: Extract sentences about training, remove unrelated content
    /// - Output: 3 documents, each 500 tokens (1500 total)
    /// 
    /// The compressed documents still contain the key information needed to answer
    /// the question, but take up much less space.
    /// </para>
    /// </remarks>
    List<Document<T>> Compress(
        List<Document<T>> documents,
        string query,
        Dictionary<string, object>? options = null);
}
