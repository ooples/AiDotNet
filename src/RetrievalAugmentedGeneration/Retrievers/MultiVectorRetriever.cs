
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Retrieves documents by matching against multiple vector representations per document for improved precision.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// MultiVectorRetriever represents each document with multiple embedding vectors instead of a single one,
/// enabling capture of different semantic aspects (summaries, key sentences, topics) within the same document.
/// This approach is particularly effective for long documents with diverse content or when queries might match
/// different parts of a document. The retriever generates scores from each vector representation and aggregates
/// them using configurable methods (max, mean, weighted) to produce a final relevance score. This technique
/// improves both precision and recall compared to single-vector retrieval, especially for complex documents.
/// </para>
/// <para><b>For Beginners:</b> Think of this like having multiple "angles" to describe the same book:
/// 
/// Regular retrieval:
/// - Each book has ONE summary
/// - Search compares your question to that ONE summary
/// 
/// Multi-vector retrieval:
/// - Each book has MULTIPLE summaries (intro, chapters, conclusion, key topics)
/// - Search compares your question to ALL summaries
/// - Returns books that match ANY of their summaries
/// 
/// For example, searching "neural networks" in a machine learning textbook:
/// - Chapter 1 summary: "Introduction to AI and data"
/// - Chapter 5 summary: "Deep learning and neural networks" ← MATCH!
/// - Chapter 10 summary: "Practical applications"
/// 
/// ```csharp
/// var retriever = new MultiVectorRetriever<double>(
///     documentStore,
///     vectorsPerDocument: 5,        // 5 vectors per document
///     aggregationMethod: "max"       // Use best match
/// );
/// 
/// var results = retriever.Retrieve("quantum computing applications", topK: 5);
/// // Finds documents where ANY of the 5 vectors match your query well
/// ```
/// 
/// Why use MultiVectorRetriever:
/// - Better for long documents with multiple topics
/// - Captures diverse aspects of complex content
/// - Improves recall (finds more relevant matches)
/// - Ideal for technical papers, books, reports
/// 
/// When NOT to use it:
/// - Short documents (single vector is sufficient and faster)
/// - Storage-constrained systems (requires 3-5x more storage)
/// - Real-time systems (slower than single-vector retrieval)
/// - Documents with uniform content (no benefit from multiple vectors)
/// </para>
/// </remarks>
[ComponentType(ComponentType.Retriever)]
[PipelineStage(PipelineStage.Retrieval)]
public class MultiVectorRetriever<T> : RetrieverBase<T>
{
    private readonly int _vectorsPerDocument;
    private readonly string _aggregationMethod;

    private readonly IDocumentStore<T> _documentStore;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiVectorRetriever{T}"/> class.
    /// </summary>
    /// <param name="documentStore">The document store containing documents with multiple vector representations.</param>
    /// <param name="vectorsPerDocument">Number of embedding vectors per document (typically 3-5 for optimal balance).</param>
    /// <param name="aggregationMethod">Method for combining vector scores: "max" (best match), "mean" (average), or "weighted" (position-based weights).</param>
    /// <exception cref="ArgumentNullException">Thrown when documentStore or aggregationMethod is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when vectorsPerDocument is less than or equal to zero.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the retriever with your multi-vector document store.
    /// 
    /// Aggregation methods explained:
    /// - "max": Use the BEST matching vector (recommended for precise queries)
    /// - "mean": AVERAGE all vector scores (recommended for general queries)
    /// - "weighted": Give MORE importance to earlier vectors (e.g., summaries > details)
    /// 
    /// Common configurations:
    /// - Technical papers: 5 vectors, "max" aggregation
    /// - General documents: 3 vectors, "mean" aggregation
    /// - Hierarchical content: 5 vectors, "weighted" aggregation
    /// </para>
    /// </remarks>
    public MultiVectorRetriever(
        IDocumentStore<T> documentStore,
        int vectorsPerDocument,
        string aggregationMethod)
    {
        Guard.NotNull(documentStore);
        _documentStore = documentStore;

        if (vectorsPerDocument <= 0)
            throw new ArgumentOutOfRangeException(nameof(vectorsPerDocument), "Vectors per document must be positive");

        _vectorsPerDocument = vectorsPerDocument;
        Guard.NotNull(aggregationMethod);
        _aggregationMethod = aggregationMethod;
    }

    /// <summary>
    /// Retrieves documents by comparing the query against all vector representations and aggregating scores.
    /// </summary>
    /// <param name="query">The validated search query (non-empty).</param>
    /// <param name="topK">The validated number of documents to return (positive integer).</param>
    /// <param name="metadataFilters">The validated metadata filters for document selection.</param>
    /// <returns>Documents ordered by aggregated relevance score (highest first).</returns>
    /// <exception cref="ArgumentException">Thrown when query is null or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when topK is less than or equal to zero.</exception>
    /// <remarks>
    /// <para>
    /// This method implements multi-vector retrieval through:
    /// 1. Oversampling: Retrieves topK * vectorsPerDocument * 2 candidate vectors for comprehensive coverage
    /// 2. Grouping: Groups vectors by base document ID (format: "docId_vector_N")
    /// 3. Aggregation: Combines vector scores using the specified method:
    ///    - Max: Takes highest score among all vectors
    ///    - Mean: Computes arithmetic mean of all scores
    ///    - Weighted: Applies decreasing weights (1/1, 1/2, 1/3, ...) to vectors by position
    /// 4. Ranking: Sorts documents by aggregated scores and returns top-K
    /// 
    /// The aggregation method significantly impacts behavior:
    /// - Max favors documents with at least one strong match (high precision)
    /// - Mean favors documents with consistent matches across vectors (balanced)
    /// - Weighted favors documents with strong primary vector matches (hierarchical content)
    /// </para>
    /// <para><b>For Beginners:</b> This is where the actual searching happens:
    /// 
    /// Step 1: Compare your query to ALL vectors of ALL documents
    /// - Document A: [vector1=0.9, vector2=0.3, vector3=0.7]
    /// - Document B: [vector1=0.6, vector2=0.8, vector3=0.5]
    /// 
    /// Step 2: Combine scores based on aggregation method
    /// - Max: Document A = 0.9, Document B = 0.8
    /// - Mean: Document A = 0.63, Document B = 0.63
    /// - Weighted: Document A = 0.76, Document B = 0.67
    /// 
    /// Step 3: Return best documents
    /// 
    /// This ensures you find documents even if only PART of them matches your query!
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

        // Multi-vector retrieval (Khattab et al. 2021 PLAID, Santhanam et al. 2022
        // ColBERTv2) needs a token-level embedder to produce a per-token query
        // matrix — that embedder isn't bundled with this repo. Until one is
        // wired in, drive candidate selection off a BOUNDED top-K probe of
        // the document store (zero-vector + metadata filter) instead of
        // GetAll(), which is documented as potentially O(N) memory-intensive.
        // Oversampling by 4× gives the token-overlap aggregator headroom to
        // reorder tied candidates after the store's first-stage selection.
        // The aggregation step — group by base doc ID, combine scores — is
        // the paper-faithful late-interaction reducer and stays the same.
        //
        // Query-dependent fallback scoring: instead of aggregating the stale
        // `doc.RelevanceScore` (which would be unrelated to the current query
        // when other queries had previously written into it), compute a
        // per-token-overlap score against the current query. Matches the
        // ColBERTRetriever fallback's lexical-overlap composition so results
        // stay query-relevant even without the multi-vector embedder.
        var queryTokens = TokenizeForOverlap(query);

        // Oversample by `topK * _vectorsPerDocument * 2` so the per-base-doc
        // grouping step below has enough sub-vectors to aggregate. The store
        // returns ALL sub-vector entries for each underlying document
        // (vectorsPerDocument per doc), so without this multiplier the
        // candidate set sees only ~topK*2/vectorsPerDocument distinct base
        // docs and the late-interaction reducer under-samples — particularly
        // when vectorsPerDocument > 1, which is the whole point of this
        // retriever. Floor at 32 keeps the probe meaningful on tiny stores.
        int oversample = Math.Max(topK * Math.Max(1, _vectorsPerDocument) * 2, 32);
        var placeholderQuery = new Vector<T>(_documentStore.VectorDimension);
        var allDocuments = (metadataFilters == null || metadataFilters.Count == 0)
            ? _documentStore.GetSimilar(placeholderQuery, oversample)
            : _documentStore.GetSimilarWithFilters(placeholderQuery, oversample, metadataFilters);

        // Group documents by ID and aggregate their vector scores. Compute
        // per-document overlap on the fly — bounded-size accumulator per
        // base doc ID, no full-corpus list materialised.
        var documentScores = new Dictionary<string, (Document<T> doc, List<T> scores)>();

        foreach (var doc in allDocuments)
        {
            var docId = GetBaseDocumentId(doc.Id);

            if (!documentScores.ContainsKey(docId))
            {
                documentScores[docId] = (doc, new List<T>());
            }

            double overlap = ComputeTokenOverlap(queryTokens, doc.Content);
            documentScores[docId].scores.Add(NumOps.FromDouble(overlap));
        }

        // Aggregate scores based on the specified method
        var aggregatedResults = new List<(Document<T> doc, T score)>();

        foreach (var kvp in documentScores)
        {
            var docId = kvp.Key;
            var doc = kvp.Value.doc;
            var scores = kvp.Value.scores;
            T aggregatedScore = AggregateScores(scores);
            aggregatedResults.Add((doc, aggregatedScore));
        }

        // Sort by aggregated score and return top-K
        return aggregatedResults
            .OrderByDescending(x => x.score)
            .Take(topK)
            .Select(x =>
            {
                x.doc.RelevanceScore = x.score;
                x.doc.HasRelevanceScore = true;
                return x.doc;
            });
    }

    private static bool MatchesMetadata(Document<T> doc, Dictionary<string, object> filters)
    {
        // Exact-match metadata filter: every key must exist on the doc and equal
        // the requested value. Mirrors the per-store GetSimilarWithFilters
        // contract — unknown keys are non-matches.
        if (doc.Metadata == null) return false;
        foreach (var kvp in filters)
        {
            if (!doc.Metadata.TryGetValue(kvp.Key, out var actual))
                return false;
            if (!Equals(actual, kvp.Value))
                return false;
        }
        return true;
    }

    private string GetBaseDocumentId(string vectorDocId)
    {
        // Extract base document ID from vector ID (format: "docId_vector_N")
        var parts = vectorDocId.Split(new[] { "_vector_" }, StringSplitOptions.None);
        return parts.Length > 0 ? parts[0] : vectorDocId;
    }

    private T AggregateScores(List<T> scores)
    {
        if (scores.Count == 0)
            return NumOps.Zero;

        switch (_aggregationMethod.ToLower())
        {
            case "max":
                return scores.Max() ?? NumOps.Zero;

            case "mean":
            case "average":
                var sum = NumOps.Zero;
                foreach (var score in scores)
                {
                    sum = NumOps.Add(sum, score);
                }
                return NumOps.Divide(sum, NumOps.FromDouble(scores.Count));

            case "weighted":
                // Weight by position (first vectors get higher weight)
                var weightedSum = NumOps.Zero;
                var totalWeight = 0.0;

                for (int i = 0; i < scores.Count; i++)
                {
                    var weight = 1.0 / (i + 1.0); // Decreasing weights
                    weightedSum = NumOps.Add(
                        weightedSum,
                        NumOps.Multiply(scores[i], NumOps.FromDouble(weight))
                    );
                    totalWeight += weight;
                }

                return NumOps.Divide(weightedSum, NumOps.FromDouble(totalWeight));

            default:
                // Default to max
                return scores.Max() ?? NumOps.Zero;
        }
    }

    private static readonly char[] s_overlapSeparators = new[] { ' ', '\t', '\n', '\r', ',', '.', '!', '?', ';', ':' };

    /// <summary>
    /// Lowercases <paramref name="query"/> and splits on whitespace +
    /// common punctuation to produce the set of overlap tokens used by
    /// <see cref="ComputeTokenOverlap"/> in the embedder-less fallback.
    /// </summary>
    private static HashSet<string> TokenizeForOverlap(string query)
    {
        return query
            .ToLowerInvariant()
            .Split(s_overlapSeparators, StringSplitOptions.RemoveEmptyEntries)
            .ToHashSet();
    }

    /// <summary>
    /// Returns the fraction of <paramref name="queryTokens"/> present in
    /// <paramref name="docContent"/> — strict Jaccard-style overlap on the
    /// SET of distinct doc tokens. Result is always in [0, 1].
    /// </summary>
    /// <remarks>
    /// Counting every occurrence in the document and dividing by
    /// <c>queryTokens.Count</c> would let documents with repeated tokens
    /// score above 1.0, contradicting both the docstring and any callers
    /// that compose this with sigmoid/normalised aggregators downstream.
    /// Using a HashSet of doc tokens caps the numerator at
    /// <c>queryTokens.Count</c>.
    /// </remarks>
    private static double ComputeTokenOverlap(HashSet<string> queryTokens, string? docContent)
    {
        if (queryTokens.Count == 0 || string.IsNullOrWhiteSpace(docContent)) return 0.0;
        var docTokens = new HashSet<string>(
            (docContent ?? string.Empty)
                .ToLowerInvariant()
                .Split(s_overlapSeparators, StringSplitOptions.RemoveEmptyEntries));
        int hits = 0;
        foreach (var qt in queryTokens)
        {
            if (docTokens.Contains(qt)) hits++;
        }
        return (double)hits / queryTokens.Count;
    }
}
