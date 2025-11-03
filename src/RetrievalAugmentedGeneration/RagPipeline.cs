using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration;

/// <summary>
/// Orchestrates the complete retrieval-augmented generation pipeline.
/// </summary>
/// <remarks>
/// <para>
/// The RAG pipeline coordinates the entire process of retrieval-augmented generation:
/// retrieving relevant documents, optionally reranking them, and generating grounded answers.
/// It provides a high-level interface that hides the complexity of coordinating multiple
/// components while maintaining flexibility through dependency injection.
/// </para>
/// <para><b>For Beginners:</b> This is the main RAG system that puts everything together.
/// 
/// Think of it like an assembly line for answering questions:
/// 
/// Step 1 (Retrieval): Find relevant documents
/// - Your question goes to the Retriever
/// - Retriever searches the document store
/// - Returns top matching documents
/// 
/// Step 2 (Reranking - Optional): Improve the ranking
/// - Reranker takes the documents
/// - Scores them more carefully
/// - Reorders to put best ones first
/// 
/// Step 3 (Generation): Create the answer
/// - Generator reads the top documents
/// - Writes an answer based on them
/// - Includes citations to sources
/// 
/// For example:
/// - Question: "What is photosynthesis?"
/// - Retriever finds: 10 biology documents
/// - Reranker selects: Best 3 documents
/// - Generator writes: "Photosynthesis is the process... [1][2][3]"
/// - Returns: Answer + source documents + citations
/// 
/// This is the complete RAG system in one convenient class!
/// </para>
/// </remarks>
public class RagPipeline
{
    private readonly IRetriever _retriever;
    private readonly IReranker _reranker;
    private readonly IGenerator _generator;

    /// <summary>
    /// Gets the retriever used by this pipeline.
    /// </summary>
    public IRetriever Retriever => _retriever;

    /// <summary>
    /// Gets the reranker used by this pipeline.
    /// </summary>
    public IReranker Reranker => _reranker;

    /// <summary>
    /// Gets the generator used by this pipeline.
    /// </summary>
    public IGenerator Generator => _generator;

    /// <summary>
    /// Initializes a new instance of the RagPipeline class.
    /// </summary>
    /// <param name="retriever">The retriever for finding relevant documents.</param>
    /// <param name="reranker">The reranker for improving document ranking.</param>
    /// <param name="generator">The generator for producing grounded answers.</param>
    public RagPipeline(IRetriever retriever, IReranker reranker, IGenerator generator)
    {
        _retriever = retriever ?? throw new ArgumentNullException(nameof(retriever));
        _reranker = reranker ?? throw new ArgumentNullException(nameof(reranker));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
    }

    /// <summary>
    /// Generates a grounded answer for a query using the complete RAG pipeline.
    /// </summary>
    /// <param name="query">The user's question or query.</param>
    /// <param name="topK">The number of documents to retrieve (uses retriever's default if not specified).</param>
    /// <param name="topKAfterRerank">The number of documents to keep after reranking (uses all reranked docs if not specified).</param>
    /// <param name="metadataFilters">Optional metadata filters to apply during retrieval.</param>
    /// <returns>A grounded answer with source citations.</returns>
    /// <remarks>
    /// <para>
    /// This method executes the complete RAG pipeline:
    /// 1. Retrieve relevant documents from the document store
    /// 2. Rerank the retrieved documents for better relevance
    /// 3. Generate an answer using the reranked documents as context
    /// </para>
    /// <para><b>For Beginners:</b> This is the main method that answers questions.
    /// 
    /// Just call it with your question and get back an answer with sources:
    /// 
    /// ```csharp
    /// var answer = pipeline.Generate("What is photosynthesis?");
    /// Console.WriteLine(answer.Answer);  // The AI's answer
    /// foreach (var doc in answer.SourceDocuments) {
    ///     Console.WriteLine($"Source: {doc.Id}");  // Where the info came from
    /// }
    /// ```
    /// 
    /// Parameters:
    /// - query: Your question
    /// - topK: How many documents to retrieve (more = more context, but slower)
    /// - topKAfterRerank: How many to keep after reranking (usually fewer than topK)
    /// - metadataFilters: Optional filters like {"year": 2024, "category": "science"}
    /// </para>
    /// </remarks>
    public GroundedAnswer Generate(
        string query,
        int? topK = null,
        int? topKAfterRerank = null,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));

        // Step 1: Retrieve documents (honor metadata filters in both branches)
        var filters = metadataFilters ?? new Dictionary<string, object>();
        var retrievedDocs = topK.HasValue
            ? _retriever.Retrieve(query, topK.Value, filters)
            : _retriever.Retrieve(query, _retriever.DefaultTopK, filters);

        var retrievedList = retrievedDocs.ToList();

        if (retrievedList.Count == 0)
        {
            // No documents found, return empty answer
            return new GroundedAnswer
            {
                Query = query,
                Answer = "I couldn't find any relevant information to answer this question.",
                SourceDocuments = new List<Document>(),
                Citations = new List<string>(),
                ConfidenceScore = 0.0
            };
        }

        // Step 2: Rerank documents
        var rerankedDocs = _reranker.Rerank(query, retrievedList);
        
        // Limit to topKAfterRerank if specified
        if (topKAfterRerank.HasValue)
        {
            rerankedDocs = rerankedDocs.Take(topKAfterRerank.Value);
        }

        var contextDocs = rerankedDocs.ToList();

        // Step 3: Generate grounded answer
        return _generator.GenerateGrounded(query, contextDocs);
    }

    /// <summary>
    /// Generates a grounded answer asynchronously (placeholder for future async implementation).
    /// </summary>
    /// <param name="query">The user's question or query.</param>
    /// <param name="topK">The number of documents to retrieve.</param>
    /// <param name="topKAfterRerank">The number of documents to keep after reranking.</param>
    /// <param name="metadataFilters">Optional metadata filters.</param>
    /// <returns>A task representing the asynchronous operation, containing a grounded answer.</returns>
    /// <remarks>
    /// <para>
    /// Currently this is a synchronous implementation wrapped in a Task.
    /// Future versions will support true async operations when the underlying
    /// components (retriever, reranker, generator) support async.
    /// </para>
    /// </remarks>
    public Task<GroundedAnswer> GenerateAsync(
        string query,
        int? topK = null,
        int? topKAfterRerank = null,
        Dictionary<string, object>? metadataFilters = null)
    {
        return Task.FromResult(Generate(query, topK, topKAfterRerank, metadataFilters));
    }

    /// <summary>
    /// Retrieves documents without generating an answer.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="topK">The number of documents to retrieve.</param>
    /// <param name="applyReranking">Whether to apply reranking to the results.</param>
    /// <param name="metadataFilters">Optional metadata filters.</param>
    /// <returns>Retrieved (and optionally reranked) documents.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This retrieves documents without generating an answer.
    /// 
    /// Useful when you:
    /// - Just want to see what documents were found
    /// - Want to inspect the retrieved content
    /// - Are building a custom generation step
    /// - Need documents for analysis or debugging
    /// </para>
    /// </remarks>
    public IEnumerable<Document> RetrieveDocuments(
        string query,
        int? topK = null,
        bool applyReranking = true,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));

        var docs = topK.HasValue
            ? _retriever.Retrieve(query, topK.Value, metadataFilters ?? new Dictionary<string, object>())
            : _retriever.Retrieve(query);

        if (applyReranking)
        {
            docs = _reranker.Rerank(query, docs);
        }

        return docs.ToList();
    }
}
