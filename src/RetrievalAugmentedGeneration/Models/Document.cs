namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Represents a document with content, metadata, and optional relevance scoring.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// A document is the fundamental unit of information in a retrieval-augmented generation system.
/// It contains the actual text content, metadata for filtering and tracking, and optional
/// relevance scores assigned during retrieval or reranking processes.
/// </para>
/// <para><b>For Beginners:</b> A document is like a file or article in your system.
/// 
/// Think of it like a book entry in a library catalog:
/// - Id: The unique catalog number
/// - Content: The actual text from the book
/// - Metadata: Information about the book (author, date, category, etc.)
/// - RelevanceScore: How well this book matches what you're looking for
/// 
/// For example, when you search for "climate change", documents about environmental
/// science get high relevance scores, while documents about sports get low scores.
/// </para>
/// </remarks>
public class Document<T>
{
    /// <summary>
    /// Gets or sets the unique identifier for this document.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The document ID should be unique within a document collection and persistent
    /// across sessions to enable consistent referencing and citation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a barcode or ISBN that uniquely identifies this document.
    /// No two documents should have the same ID.
    /// </para>
    /// </remarks>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the text content of the document.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the main textual content that will be searched, retrieved, and used
    /// to generate answers. The content can range from a single sentence to multiple paragraphs,
    /// depending on the chunking strategy employed.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual text from the document.
    /// 
    /// For example:
    /// - A product description
    /// - A paragraph from a research paper
    /// - An answer from a FAQ
    /// - A section from a technical manual
    /// </para>
    /// </remarks>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets metadata associated with this document.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Metadata provides additional information about the document that can be used for
    /// filtering, categorization, and source attribution. Common metadata includes:
    /// - Source file or URL
    /// - Author or creator
    /// - Creation or modification date
    /// - Document type or category
    /// - Section or chapter information
    /// </para>
    /// <para><b>For Beginners:</b> Metadata is information *about* the document, not the content itself.
    /// 
    /// Think of it like tags on a YouTube video:
    /// - Title, description, upload date (metadata)
    /// - The actual video content (stored in Content property)
    /// 
    /// Metadata helps you filter documents, like "show me only documents from 2024"
    /// or "only documents written by Dr. Smith".
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the relevance score assigned to this document by a retriever or reranker.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The relevance score indicates how well this document matches a query.
    /// Higher scores indicate stronger relevance. The score scale and interpretation
    /// depend on the retrieval or reranking algorithm used. Use HasRelevanceScore to check
    /// if a score has been assigned before accessing this value.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a match percentage showing how relevant this document is.
    /// 
    /// Think of it like search results:
    /// - Score 0.95: Almost perfect match, highly relevant
    /// - Score 0.50: Somewhat relevant
    /// - Score 0.10: Barely relevant
    /// - Check HasRelevanceScore first to see if scored
    /// 
    /// Documents with higher scores are more likely to contain the answer to your question.
    /// </para>
    /// </remarks>
    public T RelevanceScore { get; set; }

    /// <summary>
    /// Gets or sets whether this document has a relevance score assigned.
    /// </summary>
    public bool HasRelevanceScore { get; set; }

    /// <summary>
    /// Initializes a new instance of the Document class.
    /// </summary>
    public Document()
    {
    }

    /// <summary>
    /// Initializes a new instance of the Document class with specified content.
    /// </summary>
    /// <param name="id">The unique identifier for the document.</param>
    /// <param name="content">The text content of the document.</param>
    public Document(string id, string content)
    {
        Id = id;
        Content = content;
    }

    /// <summary>
    /// Initializes a new instance of the Document class with content and metadata.
    /// </summary>
    /// <param name="id">The unique identifier for the document.</param>
    /// <param name="content">The text content of the document.</param>
    /// <param name="metadata">Metadata associated with the document.</param>
    public Document(string id, string content, Dictionary<string, object> metadata)
    {
        Id = id;
        Content = content;
        Metadata = metadata;
    }
}
