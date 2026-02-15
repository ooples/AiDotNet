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
public interface IGenerator<T>
{
    /// <summary>
    /// Generates a text response based on a prompt.
    /// </summary>
    /// <param name="prompt">The input prompt or question.</param>
    /// <returns>The generated text response.</returns>
    /// <remarks>
    /// <para>
    /// This method generates text based solely on the provided prompt, without
    /// additional context. It's suitable for general-purpose text generation tasks.
    /// In RAG systems, this is typically called with prompts that have been augmented
    /// with retrieved context.
    /// </para>
    /// <para><b>For Beginners:</b> This generates text from a prompt.
    /// 
    /// For example:
    /// - Prompt: "Explain photosynthesis in simple terms"
    /// - Generated: "Photosynthesis is how plants make food using sunlight..."
    /// 
    /// In RAG, the prompt usually includes both the question and retrieved documents:
    /// - Prompt: "Context: [3 documents about photosynthesis]\n\nQuestion: Explain photosynthesis"
    /// - Generated: Answer based on those specific documents
    /// </para>
    /// </remarks>
    string Generate(string prompt);

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
