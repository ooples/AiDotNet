
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Hybrid document store combining vector and keyword search strategies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implementation combines two separate document stores to enable hybrid search - one for vector similarity
    /// and one for keyword matching (like BM25). Results from both stores are weighted and combined for improved
    /// retrieval quality. This approach often outperforms pure vector or pure keyword search alone.
    /// </para>
    /// <para><b>For Beginners:</b> Hybrid search is like having two search engines working together.
    /// 
    /// Think of it like asking two librarians:
    /// - Vector librarian: Finds books by *meaning* (semantic similarity)
    /// - Keyword librarian: Finds books by *exact words* (traditional search)
    /// - Results are combined for better accuracy
    /// 
    /// Why hybrid is better:
    /// - Vector search: Finds "climate change" when you search "global warming"
    /// - Keyword search: Finds exact product codes, names, IDs
    /// - Together: Best of both worlds!
    /// 
    /// Good for:
    /// - E-commerce product search
    /// - Document retrieval systems
    /// - Q&A systems
    /// - Any application needing both exact and semantic matches
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HybridDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly IDocumentStore<T> _vectorStore;
        private readonly IDocumentStore<T> _keywordStore;
        private readonly T _vectorWeight;
        private readonly T _keywordWeight;

        /// <summary>
        /// Gets the number of documents currently stored (from the vector store).
        /// </summary>
        public override int DocumentCount => _vectorStore.DocumentCount;

        /// <summary>
        /// Gets the dimensionality of vectors stored (from the vector store).
        /// </summary>
        public override int VectorDimension => _vectorStore.VectorDimension;

        /// <summary>
        /// Initializes a new instance of the HybridDocumentStore class.
        /// </summary>
        /// <param name="vectorStore">The document store for vector similarity search.</param>
        /// <param name="keywordStore">The document store for keyword-based search.</param>
        /// <param name="vectorWeight">The weight for vector search scores (typically 0.5-0.8).</param>
        /// <param name="keywordWeight">The weight for keyword search scores (typically 0.2-0.5).</param>
        /// <exception cref="ArgumentNullException">Thrown when either store is null.</exception>
        /// <remarks>
        /// <para><b>For Beginners:</b> Creates a hybrid search by combining two stores.
        /// 
        /// Example:
        /// <code>
        /// // Create the two underlying stores
        /// var vectorStore = new InMemoryDocumentStore&lt;float&gt;();
        /// var keywordStore = new BM25DocumentStore&lt;float&gt;();
        /// 
        /// // Combine them with 70% vector, 30% keyword weighting
        /// var hybridStore = new HybridDocumentStore&lt;float&gt;(
        ///     vectorStore,
        ///     keywordStore,
        ///     vectorWeight: 0.7f,
        ///     keywordWeight: 0.3f
        /// );
        /// </code>
        /// 
        /// The weights control how much each search strategy contributes to final ranking.
        /// Common strategies:
        /// - Semantic-heavy: 0.8 vector, 0.2 keyword (when meaning matters most)
        /// - Balanced: 0.5 vector, 0.5 keyword (equal importance)
        /// - Keyword-heavy: 0.3 vector, 0.7 keyword (when exact matches matter most)
        /// </para>
        /// </remarks>
        public HybridDocumentStore(
            IDocumentStore<T> vectorStore,
            IDocumentStore<T> keywordStore,
            T vectorWeight,
            T keywordWeight)
        {
            if (vectorStore == null)
                throw new ArgumentNullException(nameof(vectorStore));
            if (keywordStore == null)
                throw new ArgumentNullException(nameof(keywordStore));

            _vectorStore = vectorStore;
            _keywordStore = keywordStore;
            _vectorWeight = vectorWeight;
            _keywordWeight = keywordWeight;
        }

        /// <summary>
        /// Core logic for adding a single vector document to both underlying stores.
        /// </summary>
        /// <param name="vectorDocument">The validated vector document to add.</param>
        /// <remarks>
        /// <para>
        /// Adds the document to both the vector store and keyword store simultaneously.
        /// Both stores maintain their own indices for their respective search strategies.
        /// </para>
        /// </remarks>
        protected override void AddCore(VectorDocument<T> vectorDocument)
        {
            _vectorStore.Add(vectorDocument);
            _keywordStore.Add(vectorDocument);
        }

        /// <summary>
        /// Core logic for adding multiple vector documents to both underlying stores in a batch.
        /// </summary>
        /// <param name="vectorDocuments">The validated list of vector documents to add.</param>
        /// <remarks>
        /// <para>
        /// Adds all documents to both stores in batch operations for better performance.
        /// Both indices are updated simultaneously.
        /// </para>
        /// <para><b>For Beginners:</b> Adds documents to both search engines at once.
        /// 
        /// This is more efficient than adding one at a time:
        /// <code>
        /// // Fast - batch add to both stores
        /// store.AddBatch(documents);
        /// </code>
        /// </para>
        /// </remarks>
        protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
        {
            _vectorStore.AddBatch(vectorDocuments);
            _keywordStore.AddBatch(vectorDocuments);
        }

        /// <summary>
        /// Core logic for hybrid search combining vector similarity and keyword matching.
        /// </summary>
        /// <param name="queryVector">The validated query vector.</param>
        /// <param name="topK">The validated number of documents to return.</param>
        /// <param name="metadataFilters">The validated metadata filters.</param>
        /// <returns>Top-k documents ranked by weighted combination of vector and keyword scores.</returns>
        /// <remarks>
        /// <para>
        /// Performs hybrid search by:
        /// 1. Getting top-2K results from vector store (semantic similarity)
        /// 2. Getting top-2K results from keyword store (exact matches)
        /// 3. Weighting each result by configured weights
        /// 4. Combining and re-ranking by total score
        /// 5. Returning top-K final results
        /// </para>
        /// <para><b>For Beginners:</b> Searches both ways and combines the results.
        /// 
        /// How it works:
        /// 1. Ask vector librarian for top matches by meaning
        /// 2. Ask keyword librarian for top matches by exact words
        /// 3. Score each result:
        ///    - Vector score × vector weight
        ///    - Keyword score × keyword weight
        ///    - Total = sum of both
        /// 4. Sort by total score and return best matches
        /// 
        /// Example:
        /// Query: "laptop"
        /// - Vector finds: "notebook computer", "portable PC" (by meaning)
        /// - Keyword finds: "laptop" (exact match)
        /// - Combined: Best results considering both strategies
        /// 
        /// This gives better results than either search alone!
        /// </para>
        /// </remarks>
        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
        {
            var vectorResults = _vectorStore.GetSimilarWithFilters(queryVector, topK * 2, metadataFilters);
            var keywordResults = _keywordStore.GetSimilarWithFilters(queryVector, topK * 2, metadataFilters);

            var combinedScores = new Dictionary<string, T>();

            foreach (var doc in vectorResults.Where(d => d.HasRelevanceScore))
            {
                var score = NumOps.Multiply(_vectorWeight, doc.RelevanceScore);
                combinedScores[doc.Id] = score;
            }

            foreach (var doc in keywordResults.Where(doc => doc.HasRelevanceScore))
            {
                var keywordScore = NumOps.Multiply(_keywordWeight, doc.RelevanceScore);
                combinedScores[doc.Id] = combinedScores.TryGetValue(doc.Id, out var existingScore)
                    ? NumOps.Add(existingScore, keywordScore)
                    : keywordScore;
            }

            var allDocuments = vectorResults.Concat(keywordResults)
                .GroupBy(d => d.Id)
                .Select(g => g.First())
                .ToList();

            var results = allDocuments
                .Where(doc => combinedScores.ContainsKey(doc.Id))
                .OrderByDescending(doc => combinedScores[doc.Id])
                .Take(topK)
                .Select(doc =>
                {
                    doc.RelevanceScore = combinedScores[doc.Id];
                    doc.HasRelevanceScore = true;
                    return doc;
                })
                .ToList();

            return results;
        }

        /// <summary>
        /// Core logic for retrieving a document by ID from the vector store.
        /// </summary>
        /// <param name="documentId">The validated document ID.</param>
        /// <returns>The document if found; otherwise, null.</returns>
        /// <remarks>
        /// <para>
        /// Retrieves the document from the vector store. Since both stores contain the same documents,
        /// querying either one returns the same document.
        /// </para>
        /// <para><b>For Beginners:</b> Gets a specific document by ID.
        /// 
        /// Example:
        /// <code>
        /// var doc = store.GetById("product-123");
        /// </code>
        /// </para>
        /// </remarks>
        protected override Document<T>? GetByIdCore(string documentId)
        {
            return _vectorStore.GetById(documentId);
        }

        /// <summary>
        /// Core logic for removing a document from both underlying stores.
        /// </summary>
        /// <param name="documentId">The validated document ID.</param>
        /// <returns>True if the document was found and removed from either store; otherwise, false.</returns>
        /// <remarks>
        /// <para>
        /// Removes the document from both the vector store and keyword store to maintain synchronization.
        /// Returns true if removed from either store.
        /// </para>
        /// <para><b>For Beginners:</b> Deletes a document from both search engines.
        /// 
        /// Example:
        /// <code>
        /// if (store.Remove("product-123"))
        ///     Console.WriteLine("Product removed from both indices");
        /// </code>
        /// </para>
        /// </remarks>
        protected override bool RemoveCore(string documentId)
        {
            var removed1 = _vectorStore.Remove(documentId);
            var removed2 = _keywordStore.Remove(documentId);
            return removed1 || removed2;
        }

        /// <summary>
        /// Core logic for retrieving all documents from the vector store.
        /// </summary>
        /// <returns>An enumerable of all documents without their vector embeddings.</returns>
        /// <remarks>
        /// <para>
        /// Returns all documents from the vector store. Since both stores contain the same documents,
        /// querying either one returns the complete set.
        /// </para>
        /// <para><b>For Beginners:</b> Gets every document from the hybrid store.
        /// 
        /// Use cases:
        /// - Export all documents
        /// - Migrate to a different store
        /// - Bulk processing
        /// - Debugging
        /// 
        /// Warning: For large collections (> 10K documents), this can use significant memory.
        /// 
        /// Example:
        /// <code>
        /// var allDocs = store.GetAll().ToList();
        /// Console.WriteLine($"Total documents in hybrid store: {allDocs.Count}");
        /// </code>
        /// </para>
        /// </remarks>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            return _vectorStore.GetAll();
        }

        /// <summary>
        /// Removes all documents from both underlying stores.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Clears both the vector store and keyword store, removing all documents and resetting indices.
        /// Both stores are returned to their initial empty state.
        /// </para>
        /// <para><b>For Beginners:</b> Completely empties both search engines.
        /// 
        /// After calling Clear():
        /// - All documents removed from both stores
        /// - Both indices are empty
        /// - Ready for new documents
        /// 
        /// Use with caution - this cannot be undone!
        /// 
        /// Example:
        /// <code>
        /// store.Clear();
        /// Console.WriteLine($"Documents remaining: {store.DocumentCount}"); // 0
        /// </code>
        /// </para>
        /// </remarks>
        public override void Clear()
        {
            _vectorStore.Clear();
            _keywordStore.Clear();
        }
    }
}

