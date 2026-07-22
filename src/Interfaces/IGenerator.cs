using System.Threading;
using System.Threading.Tasks;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for text generation models used in retrieval-augmented generation.
/// </summary>
/// <remarks>
/// <para>
/// A generator produces text responses based on input prompts, optionally augmented with
/// retrieved context. In RAG systems, generators take the user's query along with relevant
/// document snippets and produce grounded answers that cite their sources. The interface
/// extends IModel to integrate with the broader AiDotNet ecosystem.
/// </para>
/// <para><b>For Beginners:</b> A generator is like a smart writer that creates answers.
/// 
/// Think of it like a research assistant:
/// - You ask a question: "What is machine learning?"
/// - The assistant reads relevant documents you provide
/// - The assistant writes an answer based on those documents
/// - The assistant includes references to show where information came from
/// 
/// In RAG systems:
/// 1. Retriever finds relevant documents (research phase)
/// 2. Generator reads those documents and writes the answer (writing phase)
/// 3. The answer is "grounded" because it's based on real documents, not imagination
/// 
/// For example:
/// - Question: "How do transformers work?"
/// - Retrieved docs: 3 papers about transformer architecture
/// - Generated answer: "Transformers use self-attention mechanisms [1] to process
///   sequences in parallel [2], making them efficient for NLP tasks [3]."
/// - Citations [1], [2], [3] point to the source documents
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("Generator")]
public interface IGenerator<T> : ITextGenerator
{
    // string Generate(string prompt) is inherited from ITextGenerator — buffered text generation from a
    // prompt (in RAG, a prompt already augmented with retrieved context). See ITextGenerator for details.

    /// <summary>
    /// Generates a grounded answer using provided context documents.
    /// </summary>
    /// <param name="query">The user's original query or question.</param>
    /// <param name="context">The retrieved documents providing context for the answer.</param>
    /// <returns>A grounded answer with the generated text, source documents, and extracted citations.</returns>
    /// <remarks>
    /// <para>
    /// This method is the core of RAG systems. It combines the user's query with retrieved
    /// context documents to generate an answer that is grounded in the provided sources.
    /// The method handles prompt construction, citation extraction, and source attribution
    /// automatically.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an answer with proof of where it came from.
    /// 
    /// Think of it like writing a research paper:
    /// - query: Your research question
    /// - context: The papers and books you read
    /// - GroundedAnswer: Your written answer with proper citations
    /// 
    /// For example:
    /// - Query: "What are the benefits of exercise?"
    /// - Context: 5 health research articles
    /// - Generated Answer: "Exercise improves cardiovascular health [1], reduces stress [2],
    ///   and strengthens muscles [3]."
    /// - Citations: [1] = Article about heart health, [2] = Stress study, etc.
    /// 
    /// The "grounded" part means every claim in the answer can be traced back to
    /// a specific source document - it's not made up!
    /// </para>
    /// </remarks>
    GroundedAnswer<T> GenerateGrounded(string query, IEnumerable<Document<T>> context);

    /// <summary>
    /// Asynchronously generates a text response for a prompt, honoring cancellation.
    /// </summary>
    /// <param name="prompt">The input prompt (typically already augmented with retrieved context).</param>
    /// <param name="cancellationToken">A token to observe for cancellation requests.</param>
    /// <returns>A task producing the generated text response.</returns>
    /// <remarks>
    /// <para>
    /// Asynchronous counterpart to <see cref="ITextGenerator.Generate(string)"/>. Generators backed by a
    /// remote chat model perform genuine non-blocking work here; local generators complete synchronously.
    /// </para>
    /// </remarks>
    Task<string> GenerateAsync(string prompt, CancellationToken cancellationToken = default);

    /// <summary>
    /// Asynchronously generates a grounded answer using provided context documents, honoring cancellation.
    /// </summary>
    /// <param name="query">The user's original query or question.</param>
    /// <param name="context">The retrieved documents providing context for the answer.</param>
    /// <param name="cancellationToken">A token to observe for cancellation requests.</param>
    /// <returns>A task producing a grounded answer with the generated text, source documents, and extracted citations.</returns>
    /// <remarks>
    /// <para>
    /// Asynchronous counterpart to <see cref="GenerateGrounded(string, IEnumerable{Document{T}})"/>.
    /// </para>
    /// </remarks>
    Task<GroundedAnswer<T>> GenerateGroundedAsync(string query, IEnumerable<Document<T>> context, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the maximum number of tokens this generator can process in a single request.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The context window determines how much text (prompt + retrieved context) can be
    /// processed at once. Larger context windows allow including more retrieved documents
    /// but may be slower. Common sizes range from 2048 to 128000 tokens.
    /// </para>
    /// <para><b>For Beginners:</b> This is how much text the generator can read at once.
    /// 
    /// Think of it like a reader's working memory:
    /// - Small (2048 tokens): Can read about 2-3 pages
    /// - Medium (8192 tokens): Can read about 10-15 pages  
    /// - Large (32000+ tokens): Can read a small book
    /// 
    /// Why does this matter?
    /// If you retrieve 10 documents (5000 tokens) but the context window is only 2048 tokens,
    /// you'll need to either:
    /// - Use fewer documents
    /// - Summarize the documents
    /// - Use a model with a larger context window
    /// 
    /// (Note: 1 token ≈ 0.75 words, so 2048 tokens ≈ 1500 words ≈ 2-3 pages)
    /// </para>
    /// </remarks>
    int MaxContextTokens { get; }

    /// <summary>
    /// Gets the maximum number of tokens this generator can generate in a response.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This limits the length of generated responses. It's typically smaller than the
    /// context window to reserve space for the input prompt and retrieved context.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum length of answers the generator can write.
    /// 
    /// For example:
    /// - MaxGenerationTokens: 500 tokens ≈ 375 words ≈ 2-3 paragraphs
    /// - MaxGenerationTokens: 2000 tokens ≈ 1500 words ≈ 1-2 pages
    /// 
    /// This prevents the generator from writing book-length answers when you just need
    /// a concise response.
    /// </para>
    /// </remarks>
    int MaxGenerationTokens { get; }
}
