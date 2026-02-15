using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for document stores that index and retrieve vectorized documents.
/// </summary>
/// <remarks>
/// <para>
/// A document store manages a collection of documents with their vector embeddings,
/// enabling efficient similarity-based retrieval. Implementations can range from simple
/// in-memory storage to distributed vector databases. The interface supports adding documents,
/// similarity search, and metadata-based filtering.
/// </para>
/// <para><b>For Beginners:</b> A document store is like a smart library that organizes information by meaning.
/// 
/// Think of it like a special filing cabinet:
/// - Regular filing cabinet: Organized alphabetically or by date
/// - Document store: Organized by *meaning* using math
/// 
/// When you search for "climate change", it finds documents about environmental issues
/// even if they don't contain those exact words, because it understands the *meaning*.
/// 
/// It's like having a librarian who truly understands what each book is about and can
/// find exactly what you need based on your question, not just keywords.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("DocumentStore")]
public interface IDocumentStore<T>
{
    /// <summary>
    /// Gets the number of documents currently stored in the document store.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many documents are in the store,
    /// like counting how many books are in a library.
    /// </para>
    /// </remarks>
    int DocumentCount { get; }

    /// <summary>
    /// Gets the dimensionality of the vectors stored in this document store.
    /// </summary>
    /// <remarks>
    /// <para>
    /// All vectors in the store must have the same dimension, which is determined by
    /// the embedding model used. This ensures consistent similarity calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the number lists representing each document.
    /// All documents must use the same size so they can be fairly compared.
    /// </para>
    /// </remarks>
    int VectorDimension { get; }

    /// <summary>
    /// Adds a single vectorized document to the store.
    /// </summary>
    /// <param name="document">The document to add.</param>
    /// <param name="embedding">The vector embedding representing the document's semantic meaning.</param>
    /// <remarks>
    /// <para>
    /// This method indexes a document along with its vector embedding for later retrieval.
    /// If a document with the same ID already exists, the behavior depends on the implementation
    /// (typically either update or throw an exception).
    /// </para>
    /// <para><b>For Beginners:</b> This adds a new document to the library.
    /// 
    /// Like adding a new book:
    /// - document: The book itself (with its content and info)
    /// - embedding: A numeric "fingerprint" of what the book is about
    /// 
    /// The fingerprint lets the system quickly find similar books later.
    /// </para>
    /// </remarks>
    void Add(VectorDocument<T> vectorDocument);

    /// <summary>
    /// Adds multiple vectorized documents to the store in a batch operation.
    /// </summary>
    /// <param name="vectorDocuments">The documents to add.</param>
    /// <remarks>
    /// <para>
    /// Batch addition is more efficient than adding documents individually. The embeddings matrix
    /// should have dimensions [documentCount, VectorDimension], with each row representing
    /// one document's embedding in the same order as the documents enumerable.
    /// </para>
    /// <para><b>For Beginners:</b> This adds many documents at once, which is faster.
    /// 
    /// Like processing a whole shipment of new books to the library:
    /// - Instead of cataloging one book at a time
    /// - You process the entire box together
    /// - Much more efficient for large collections
    /// </para>
    /// </remarks>
    void AddBatch(IEnumerable<VectorDocument<T>> vectorDocuments);

    /// <summary>
    /// Retrieves the top-k most similar documents to a given query vector.
    /// </summary>
    /// <param name="queryVector">The vector to search for similar documents.</param>
    /// <param name="topK">The number of most similar documents to return.</param>
    /// <returns>An enumerable of documents ordered by similarity (most similar first), with relevance scores populated.</returns>
    /// <remarks>
    /// <para>
    /// This method performs similarity search to find documents whose vector embeddings are
    /// closest to the query vector. The similarity metric (e.g., cosine similarity, Euclidean distance)
    /// is implementation-specific. Results are ordered by decreasing similarity/relevance.
    /// </para>
    /// <para><b>For Beginners:</b> This finds documents most similar to your search.
    /// 
    /// Think of it like asking the librarian:
    /// "Find me the 5 books most similar to this topic"
    /// 
    /// The system:
    /// - Compares your query's "fingerprint" to all document fingerprints
    /// - Finds the closest matches using math (measuring distances in number-space)
    /// - Returns the top matches, ordered from best to worst match
    /// 
    /// topK = 5 means "give me the 5 best matches"
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> GetSimilar(Vector<T> queryVector, int topK);

    /// <summary>
    /// Retrieves similar documents with additional metadata filtering.
    /// </summary>
    /// <param name="queryVector">The vector to search for similar documents.</param>
    /// <param name="topK">The number of most similar documents to return.</param>
    /// <param name="metadataFilters">Metadata filters to apply before similarity search.</param>
    /// <returns>An enumerable of filtered documents ordered by similarity, with relevance scores populated.</returns>
    /// <remarks>
    /// <para>
    /// This method combines similarity search with metadata filtering. First, documents are
    /// filtered based on metadata criteria, then similarity search is performed on the
    /// remaining candidates. This enables queries like "find similar documents from 2024"
    /// or "find similar documents by author X".
    /// </para>
    /// <para><b>For Beginners:</b> This finds similar documents but only from a specific subset.
    /// 
    /// Think of asking the librarian:
    /// "Find me the 5 most relevant books about climate change,
    ///  but only books published after 2020 and only in the Science section"
    /// 
    /// The filters narrow down which documents to search:
    /// - metadata["year"] >= 2020
    /// - metadata["section"] == "Science"
    /// 
    /// Then similarity search runs only on documents that pass these filters.
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> GetSimilarWithFilters(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters);

    /// <summary>
    /// Retrieves a document by its unique identifier.
    /// </summary>
    /// <param name="documentId">The unique identifier of the document to retrieve.</param>
    /// <returns>The document if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gets a specific document if you know its ID.
    /// 
    /// Like asking the librarian: "Give me the book with catalog number ABC123"
    /// </para>
    /// </remarks>
    Document<T>? GetById(string documentId);

    /// <summary>
    /// Removes a document from the store by its identifier.
    /// </summary>
    /// <param name="documentId">The unique identifier of the document to remove.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This removes a document from the store.
    /// 
    /// Like removing a book from the library catalog - it's no longer searchable.
    /// </para>
    /// </remarks>
    bool Remove(string documentId);

    /// <summary>
    /// Removes all documents from the store.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This empties the entire store, removing all documents.
    /// Use with caution - this cannot be undone!
    /// </para>
    /// </remarks>
    void Clear();

    /// <summary>
    /// Gets all documents currently stored in the document store.
    /// </summary>
    /// <returns>An enumerable of all documents in the store.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all documents without any filtering or sorting.
    /// Use with caution on large document stores as it may be memory-intensive.
    /// For production systems with large document collections, consider using
    /// pagination or streaming approaches.
    /// </para>
    /// <para><b>For Beginners:</b> This gets every single document from the store.
    /// 
    /// Like asking the librarian: "Show me every book in the library"
    /// 
    /// Careful: If you have millions of documents, this could take a while
    /// and use a lot of memory!
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> GetAll();
}

