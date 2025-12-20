
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Retrieves precise small chunks for matching but returns complete parent documents for comprehensive context.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// ParentDocumentRetriever solves the "chunk size dilemma" in RAG systems: small chunks enable precise matching
/// but lack context, while large chunks provide context but reduce precision. This retriever uses a two-tier
/// approach—search against small chunks (e.g., paragraphs) for accuracy, then return their larger parent documents
/// (e.g., full sections or pages) for complete context. This ensures the LLM receives sufficient information to
/// generate accurate answers while maintaining high retrieval precision. The retriever can optionally include
/// neighboring chunks to expand context boundaries. This pattern is particularly effective for structured content
/// (technical docs, research papers, legal documents) where individual paragraphs are meaningful but answers require
/// broader context.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a two-step library search:
/// 
/// The Problem:
/// - Small chunks (paragraphs): Easy to match precisely, BUT not enough context
/// - Large chunks (whole pages): Plenty of context, BUT hard to match precisely
/// 
/// The Solution - Parent Document Retrieval:
/// 1. Search using SMALL chunks for precision
/// 2. Return the LARGE parent document for context
/// 
/// Real-world example:
/// Query: "How does photosynthesis work?"
/// 
/// Small chunk matches: "...chlorophyll absorbs light energy..." (paragraph 3)
/// Returns: Full page containing introduction + detailed process + diagram
/// 
/// ```csharp
/// var retriever = new ParentDocumentRetriever<double>(
///     documentStore,
///     chunkSize: 256,                    // Small chunks for precision
///     parentSize: 2048,                  // Large parents for context
///     includeNeighboringChunks: true     // Add nearby chunks too
/// );
/// 
/// var results = retriever.Retrieve("explain quantum entanglement", topK: 3);
/// // Finds precise paragraphs but returns full sections with complete explanation
/// ```
/// 
/// Why use ParentDocumentRetriever:
/// - Best of both worlds: precise matching + complete context
/// - Ideal for technical documentation and research papers
/// - Reduces LLM hallucinations (more context = better answers)
/// - Works great with structured content (headings, sections, chapters)
/// 
/// When NOT to use it:
/// - Very short documents (chunks = parents already)
/// - Documents with redundant content (wastes context window)
/// - When you need ONLY the matching excerpt (use regular retrieval)
/// - Memory-constrained systems (returns more content per match)
/// </para>
/// </remarks>
public class ParentDocumentRetriever<T> : RetrieverBase<T>
{
    private readonly IDocumentStore<T> _documentStore;
    private readonly IEmbeddingModel<T> _embeddingModel;
    private readonly int _chunkSize;
    private readonly int _parentSize;
    private readonly bool _includeNeighboringChunks;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParentDocumentRetriever{T}"/> class.
    /// </summary>
    /// <param name="documentStore">The document store containing chunked documents with parent metadata.</param>
    /// <param name="embeddingModel">The embedding model used to convert text queries into vector embeddings.</param>
    /// <param name="chunkSize">Character length of child chunks used for matching (typically 128-512 characters).</param>
    /// <param name="parentSize">Character length of parent documents returned (typically 1024-4096 characters).</param>
    /// <param name="includeNeighboringChunks">Whether to include adjacent chunks around the matched chunk (expands context boundaries).</param>
    /// <exception cref="ArgumentNullException">Thrown when documentStore or embeddingModel is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when chunkSize or parentSize is less than or equal to zero.</exception>
    /// <exception cref="ArgumentException">Thrown when parentSize is less than chunkSize.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This configures how the retriever balances precision vs. context.
    /// 
    /// Recommended configurations:
    /// 
    /// Technical Documentation:
    /// - chunkSize: 256 (1-2 paragraphs)
    /// - parentSize: 2048 (full section)
    /// - includeNeighboringChunks: true (add context before/after)
    /// 
    /// Research Papers:
    /// - chunkSize: 512 (paragraph or two)
    /// - parentSize: 4096 (entire subsection)
    /// - includeNeighboringChunks: false (rely on section boundaries)
    /// 
    /// General Content:
    /// - chunkSize: 128 (few sentences)
    /// - parentSize: 1024 (multiple paragraphs)
    /// - includeNeighboringChunks: true (smooth transitions)
    /// 
    /// The includeNeighboringChunks parameter is helpful when chunk boundaries might split important context.
    /// </para>
    /// </remarks>
    public ParentDocumentRetriever(
        IDocumentStore<T> documentStore,
        IEmbeddingModel<T> embeddingModel,
        int chunkSize,
        int parentSize,
        bool includeNeighboringChunks)
    {
        _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
        _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));

        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize), "Chunk size must be positive");

        if (parentSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(parentSize), "Parent size must be positive");

        if (parentSize < chunkSize)
            throw new ArgumentException("Parent size must be greater than or equal to chunk size");

        _chunkSize = chunkSize;
        _parentSize = parentSize;
        _includeNeighboringChunks = includeNeighboringChunks;
    }

    /// <summary>
    /// Retrieves parent documents by matching against child chunks and reconstructing full parent context.
    /// </summary>
    /// <param name="query">The validated search query (non-empty).</param>
    /// <param name="topK">The validated number of parent documents to return (positive integer).</param>
    /// <param name="metadataFilters">The validated metadata filters for document selection.</param>
    /// <returns>Parent documents ordered by their best child chunk's relevance score (highest first).</returns>
    /// <exception cref="ArgumentException">Thrown when query is null or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when topK is less than or equal to zero.</exception>
    /// <remarks>
    /// <para>
    /// This method implements a hierarchical retrieval pipeline:
    /// 1. Oversampling: Retrieves topK * 3 child chunks to ensure sufficient parent document coverage
    /// 2. Parent Grouping: Groups chunks by parent_id metadata field
    /// 3. Score Aggregation: Assigns each parent the MAXIMUM score of its child chunks (best match wins)
    /// 4. Optional Expansion: If includeNeighboringChunks=true, concatenates all matching chunks for each parent
    /// 5. Deduplication: Returns unique parent documents (multiple chunks may belong to same parent)
    /// 
    /// The retriever expects chunks to have metadata:
    /// - "parent_id": Identifier of the parent document
    /// - "chunk_index": Position of this chunk within parent (optional)
    /// - "chunk_start": Character offset where chunk begins (optional)
    /// 
    /// Parent documents are reconstructed by combining chunk content and filtering chunk-specific metadata.
    /// </para>
    /// <para><b>For Beginners:</b> Here's how the retrieval process works:
    /// 
    /// Step 1: Find the best matching small chunks
    /// - Query: "neural network backpropagation"
    /// - Matches: [Chunk 5 from Doc A (score=0.9), Chunk 12 from Doc A (score=0.7), Chunk 3 from Doc B (score=0.8)]
    /// 
    /// Step 2: Group chunks by parent document
    /// - Doc A: Chunk 5 (0.9), Chunk 12 (0.7)
    /// - Doc B: Chunk 3 (0.8)
    /// 
    /// Step 3: Assign parent score = best child score
    /// - Doc A: 0.9 (from Chunk 5)
    /// - Doc B: 0.8 (from Chunk 3)
    /// 
    /// Step 4: Return parent documents (full context!)
    /// - Doc A: Complete section on backpropagation including introduction, math, examples
    /// - Doc B: Complete chapter on neural network training
    /// 
    /// This gives your LLM ALL the context it needs to provide a complete answer!
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> RetrieveCore(
        string query,
        int topK,
        Dictionary<string, object> metadataFilters)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        // Retrieve chunks at higher K to ensure we get enough parent documents
        var chunkK = topK * 3;

        // Embed the query to get query vector
        var queryVector = _embeddingModel.Embed(query);

        // Use the document store to find similar chunks
        // The chunks should have metadata indicating their parent document
        var similarChunks = _documentStore.GetSimilarWithFilters(
            queryVector,
            chunkK,
            metadataFilters ?? new Dictionary<string, object>()
        ).ToList();

        // Group by parent document ID
        var parentDocuments = new Dictionary<string, (Document<T> doc, T maxScore)>();

        foreach (var chunk in similarChunks)
        {
            // Extract parent document ID from metadata
            if (!chunk.Metadata.ContainsKey("parent_id"))
                continue;

            var parentIdObj = chunk.Metadata["parent_id"];
            if (parentIdObj == null)
                continue;

            var parentId = parentIdObj.ToString();
            if (string.IsNullOrEmpty(parentId))
                continue;

            var score = chunk.RelevanceScore;

            if (!parentDocuments.ContainsKey(parentId))
            {
                // Create parent document by combining chunks
                var parentDoc = CreateParentDocument(chunk, parentId);
                parentDocuments[parentId] = (parentDoc, score);
            }
            else
            {
                // Update if this chunk has better score
                var existing = parentDocuments[parentId];
                if (NumOps.GreaterThan(score, existing.maxScore))
                {
                    parentDocuments[parentId] = (existing.doc, score);
                }

                // Append chunk content if including neighboring chunks
                if (_includeNeighboringChunks)
                {
                    parentDocuments[parentId].doc.Content += "\n\n" + chunk.Content;
                }
            }
        }

        // Return top-K parent documents sorted by best chunk score
        return parentDocuments.Values
            .OrderByDescending(p => p.maxScore)
            .Take(topK)
            .Select(p =>
            {
                p.doc.RelevanceScore = p.maxScore;
                p.doc.HasRelevanceScore = true;
                return p.doc;
            });
    }

    private Document<T> CreateParentDocument(Document<T> chunk, string parentId)
    {
        // Retrieve the full parent document from the store
        var fullParent = _documentStore.GetById(parentId);

        if (fullParent != null)
        {
            // Return the complete parent document from store
            return new Document<T>
            {
                Id = parentId,
                Content = fullParent.Content,
                Metadata = new Dictionary<string, object>(fullParent.Metadata ?? new Dictionary<string, object>())
            };
        }

        // Fallback: if parent not found, start with chunk content
        // This will be expanded as more chunks from same parent are processed
        var parentDoc = new Document<T>
        {
            Id = parentId,
            Content = chunk.Content,
            Metadata = new Dictionary<string, object>(chunk.Metadata ?? new Dictionary<string, object>())
        };

        // Remove chunk-specific metadata
        parentDoc.Metadata.Remove("chunk_index");
        parentDoc.Metadata.Remove("chunk_start");
        parentDoc.Metadata.Remove("chunk_end");
        parentDoc.Metadata.Remove("parent_id");

        return parentDoc;
    }
}
