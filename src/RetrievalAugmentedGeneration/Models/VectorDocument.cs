using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Represents a document paired with its vector embedding for storage and retrieval.
/// </summary>
/// <remarks>
/// <para>
/// A VectorDocument combines a Document with its vector embedding, creating a complete
/// unit ready for indexing in a vector store. The vector embedding captures the semantic
/// meaning of the document's content in a numerical form suitable for similarity calculations.
/// </para>
/// <para><b>For Beginners:</b> A VectorDocument is like a book with its catalog card.
/// 
/// Think of it as two pieces working together:
/// - Document: The actual book (content, title, author, etc.)
/// - Embedding: The numerical "fingerprint" describing what the book is about
/// 
/// Why combine them?
/// When you add documents to a search system, you need both:
/// - The vector (for finding similar documents through math)
/// - The document (for returning the actual content to users)
/// 
/// For example:
/// - Document: "Climate change affects global temperatures..."
/// - Embedding: [0.23, -0.45, 0.78, ..., 0.12] (768 numbers)
/// 
/// The system uses the numbers to search, then returns the text.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for the vector embedding (typically float or double).</typeparam>
public class VectorDocument<T>
{
    /// <summary>
    /// Gets or sets the document containing the text content and metadata.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the actual document with all its information.
    /// Contains the text, ID, metadata - everything except the vector.
    /// </para>
    /// </remarks>
    public Document<T> Document { get; set; } = new();

    /// <summary>
    /// Gets or sets the vector embedding representing the document's semantic meaning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The embedding is a dense vector that encodes the document's content into a numerical
    /// representation. The vector dimension must match the embedding model's output dimension.
    /// Embeddings enable efficient similarity search through mathematical distance calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is the numerical "fingerprint" of the document.
    /// 
    /// Think of it like a GPS coordinate:
    /// - Just as GPS uses (latitude, longitude) to represent a location
    /// - An embedding uses hundreds of numbers to represent meaning
    /// - Documents with similar meanings have similar numbers (close GPS coordinates)
    /// - Different meanings have different numbers (far apart coordinates)
    /// 
    /// For example, embeddings for:
    /// - "cat" and "kitten" would be close together (similar meaning)
    /// - "cat" and "democracy" would be far apart (different meaning)
    /// </para>
    /// </remarks>
    public Vector<T> Embedding { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Initializes a new instance of the VectorDocument class.
    /// </summary>
    public VectorDocument()
    {
    }

    /// <summary>
    /// Initializes a new instance of the VectorDocument class with a document and embedding.
    /// </summary>
    /// <param name="document">The document containing content and metadata.</param>
    /// <param name="embedding">The vector embedding of the document.</param>
    public VectorDocument(Document<T> document, Vector<T> embedding)
    {
        Guard.NotNull(document);
        Document = document;
        Guard.NotNull(embedding);
        Embedding = embedding;
    }
}
