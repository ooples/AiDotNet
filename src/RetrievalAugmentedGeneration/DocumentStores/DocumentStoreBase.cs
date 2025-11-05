using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Provides a base implementation for document stores with common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements the IDocumentStore interface and provides common functionality
/// for vector document storage and retrieval. It handles validation, document management, and
/// provides utility methods for similarity calculations while allowing derived classes to focus
/// on implementing storage-specific logic and search algorithms.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all document storage systems build upon.
/// 
/// Think of it like a template for building a library:
/// - It handles common tasks (checking inputs, managing documents, calculating similarity)
/// - Specific storage systems (in-memory, database) just fill in where/how documents are stored
/// - This ensures all document stores work consistently
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public abstract class DocumentStoreBase<T> : IDocumentStore<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the number of documents currently stored in the document store.
    /// </summary>
    public abstract int DocumentCount { get; }

    /// <summary>
    /// Gets the dimensionality of the vectors stored in this document store.
    /// </summary>
    public abstract int VectorDimension { get; }

    /// <summary>
    /// Adds a single vectorized document to the store.
    /// </summary>
    /// <param name="vectorDocument">The vector document to add.</param>
    public void Add(VectorDocument<T> vectorDocument)
    {
        ValidateVectorDocument(vectorDocument);
        AddCore(vectorDocument);
    }

    /// <summary>
    /// Adds multiple vectorized documents to the store in a batch operation.
    /// </summary>
    /// <param name="vectorDocuments">The vector documents to add.</param>
    public void AddBatch(IEnumerable<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments == null)
            throw new ArgumentNullException(nameof(vectorDocuments));

        var documentList = vectorDocuments.ToList();
        if (documentList.Count == 0)
            throw new ArgumentException("Vector document collection cannot be empty", nameof(vectorDocuments));

        foreach (var vectorDocument in documentList)
        {
            ValidateVectorDocument(vectorDocument);
        }

        AddBatchCore(documentList);
    }

    /// <summary>
    /// Retrieves the top-k most similar documents to a given query vector.
    /// </summary>
    /// <param name="queryVector">The vector to search for similar documents.</param>
    /// <param name="topK">The number of most similar documents to return.</param>
    /// <returns>An enumerable of documents ordered by similarity (most similar first), with relevance scores populated.</returns>
    public IEnumerable<Document<T>> GetSimilar(Vector<T> queryVector, int topK)
    {
        return GetSimilarWithFilters(queryVector, topK, new Dictionary<string, object>());
    }

    /// <summary>
    /// Retrieves similar documents with additional metadata filtering.
    /// </summary>
    /// <param name="queryVector">The vector to search for similar documents.</param>
    /// <param name="topK">The number of most similar documents to return.</param>
    /// <param name="metadataFilters">Metadata filters to apply before similarity search.</param>
    /// <returns>An enumerable of filtered documents ordered by similarity, with relevance scores populated.</returns>
    public IEnumerable<Document<T>> GetSimilarWithFilters(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        ValidateQueryVector(queryVector);
        ValidateTopK(topK);
        ValidateMetadataFilters(metadataFilters);

        return GetSimilarCore(queryVector, topK, metadataFilters);
    }

    /// <summary>
    /// Retrieves a document by its unique identifier.
    /// </summary>
    /// <param name="documentId">The unique identifier of the document to retrieve.</param>
    /// <returns>The document if found; otherwise, null.</returns>
    public Document<T>? GetById(string documentId)
    {
        ValidateDocumentId(documentId);
        return GetByIdCore(documentId);
    }

    /// <summary>
    /// Removes a document from the store by its identifier.
    /// </summary>
    /// <param name="documentId">The unique identifier of the document to remove.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    public bool Remove(string documentId)
    {
        ValidateDocumentId(documentId);
        return RemoveCore(documentId);
    }

    /// <summary>
    /// Removes all documents from the store.
    /// </summary>
    public abstract void Clear();

    /// <summary>
    /// Core logic for adding a single vector document.
    /// </summary>
    /// <param name="vectorDocument">The validated vector document to add.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement document storage.
    /// 
    /// You don't need to:
    /// - Validate the vector document (already done)
    /// - Check vector dimensions (already validated)
    /// - Handle null inputs (already validated)
    /// 
    /// Just focus on: Storing the document and its embedding in your storage system.
    /// </para>
    /// </remarks>
    protected abstract void AddCore(VectorDocument<T> vectorDocument);

    /// <summary>
    /// Core logic for adding multiple vector documents in a batch.
    /// </summary>
    /// <param name="vectorDocuments">The validated list of vector documents to add.</param>
    /// <remarks>
    /// <para>
    /// The default implementation calls AddCore for each document. Override this to provide
    /// more efficient batch insertion if your storage backend supports it.
    /// </para>
    /// <para><b>For Implementers:</b> Override this for efficient batch operations.
    /// 
    /// For example:
    /// - Database stores can use bulk insert
    /// - In-memory stores can preallocate arrays
    /// - Network stores can batch API calls
    /// 
    /// If you don't override, documents are added one at a time (slower but works).
    /// </para>
    /// </remarks>
    protected virtual void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        foreach (var vectorDocument in vectorDocuments)
        {
            AddCore(vectorDocument);
        }
    }

    /// <summary>
    /// Core logic for similarity search with optional filtering.
    /// </summary>
    /// <param name="queryVector">The validated query vector.</param>
    /// <param name="topK">The validated number of documents to return.</param>
    /// <param name="metadataFilters">The validated metadata filters.</param>
    /// <returns>Top-k similar documents ordered by similarity score.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement similarity search.
    /// 
    /// You should:
    /// - Calculate similarity between queryVector and all stored embeddings
    /// - Apply metadata filters if provided
    /// - Return top K most similar documents
    /// - Set RelevanceScore on each returned document
    /// - Order results by similarity (highest first)
    /// 
    /// Common similarity metrics:
    /// - Cosine similarity: Use StatisticsHelper&lt;T&gt;.CosineSimilarity(vector1, vector2)
    /// - Euclidean distance: Use StatisticsHelper&lt;T&gt;.EuclideanDistance(vector1, vector2)
    /// - Jaccard similarity: Use StatisticsHelper&lt;T&gt;.JaccardSimilarity(vector1, vector2)
    /// </para>
    /// </remarks>
    protected abstract IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters);

    /// <summary>
    /// Core logic for retrieving a document by ID.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>The document if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement ID-based lookup.
    /// Simple dictionary lookup for in-memory stores, database query for persistent stores.
    /// </para>
    /// </remarks>
    protected abstract Document<T>? GetByIdCore(string documentId);

    /// <summary>
    /// Core logic for removing a document by ID.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if removed; false if not found.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement document removal.
    /// Remember to remove both the document and its embedding from your storage.
    /// </para>
    /// </remarks>
    protected abstract bool RemoveCore(string documentId);

    /// <summary>
    /// Validates a vector document before adding it to the store.
    /// </summary>
    /// <param name="vectorDocument">The vector document to validate.</param>
    protected virtual void ValidateVectorDocument(VectorDocument<T> vectorDocument)
    {
        if (vectorDocument == null)
            throw new ArgumentNullException(nameof(vectorDocument));

        if (vectorDocument.Document == null)
            throw new ArgumentException("VectorDocument.Document cannot be null", nameof(vectorDocument));

        if (string.IsNullOrWhiteSpace(vectorDocument.Document.Id))
            throw new ArgumentException("Document ID cannot be null or empty", nameof(vectorDocument));

        if (vectorDocument.Embedding == null)
            throw new ArgumentException("VectorDocument.Embedding cannot be null", nameof(vectorDocument));

        // Validate vector dimension if store already has documents
        if (DocumentCount > 0 && vectorDocument.Embedding.Length != VectorDimension)
            throw new ArgumentException(
                $"Vector dimension mismatch. Expected {VectorDimension}, got {vectorDocument.Embedding.Length}",
                nameof(vectorDocument));
    }

    /// <summary>
    /// Validates a query vector.
    /// </summary>
    /// <param name="queryVector">The query vector to validate.</param>
    protected virtual void ValidateQueryVector(Vector<T> queryVector)
    {
        if (queryVector == null)
            throw new ArgumentNullException(nameof(queryVector));

        if (DocumentCount > 0 && queryVector.Length != VectorDimension)
            throw new ArgumentException(
                $"Query vector dimension mismatch. Expected {VectorDimension}, got {queryVector.Length}",
                nameof(queryVector));
    }

    /// <summary>
    /// Validates a document ID.
    /// </summary>
    /// <param name="documentId">The document ID to validate.</param>
    protected virtual void ValidateDocumentId(string documentId)
    {
        if (string.IsNullOrWhiteSpace(documentId))
            throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));
    }

    /// <summary>
    /// Validates the topK parameter.
    /// </summary>
    /// <param name="topK">The topK value to validate.</param>
    protected virtual void ValidateTopK(int topK)
    {
        if (topK <= 0)
            throw new ArgumentException("TopK must be greater than zero", nameof(topK));
    }

    /// <summary>
    /// Validates metadata filters.
    /// </summary>
    /// <param name="metadataFilters">The metadata filters to validate.</param>
    protected virtual void ValidateMetadataFilters(Dictionary<string, object> metadataFilters)
    {
        if (metadataFilters == null)
            throw new ArgumentNullException(nameof(metadataFilters));
    }



    /// <summary>
    /// Checks if a document matches the specified metadata filters.
    /// </summary>
    /// <param name="document">The document to check.</param>
    /// <param name="filters">The metadata filters to apply.</param>
    /// <returns>True if the document matches all filters; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This helper method supports equality and range comparisons for metadata filtering.
    /// Override this to add support for more complex filter operations.
    /// </para>
    /// <para><b>For Implementers:</b> Use this in GetSimilarCore to filter documents.
    /// 
    /// Supported filters:
    /// - Equality: metadata["category"] == "science"
    /// - Range: metadata["year"] >= 2020 (for numeric/comparable values)
    /// 
    /// For example:
    /// filters = { {"category", "science"}, {"year", 2020} }
    /// Matches if: doc.Metadata["category"] == "science" AND doc.Metadata["year"] >= 2020
    /// </para>
    /// </remarks>
    protected bool MatchesFilters(Document<T> document, Dictionary<string, object> filters)
    {
        if (filters.Count == 0)
            return true;

        foreach (var filter in filters)
        {
            if (!document.Metadata.TryGetValue(filter.Key, out var value))
                return false;

            // Equality comparison
            if (filter.Value is string || filter.Value is bool)
            {
                if (!filter.Value.Equals(value))
                    return false;
            }
            // Range comparison for numbers
            else if (filter.Value is IComparable filterValue && value is IComparable docValue)
            {
                if (filterValue.CompareTo(docValue) > 0)
                    return false;
            }
            else
            {
                if (!filter.Value.Equals(value))
                    return false;
            }
        }

        return true;
    }
}
