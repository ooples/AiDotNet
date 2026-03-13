
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Rerankers;

/// <summary>
/// Reranks documents using a cross-encoder model that computes fine-grained relevance scores.
/// </summary>
/// <typeparam name="T">The numeric data type used for scoring.</typeparam>
/// <remarks>
/// <para>
/// Cross-encoder reranking is the gold standard for improving retrieval quality. Unlike bi-encoders
/// that encode query and document separately, cross-encoders process the query-document pair together,
/// allowing for richer interaction and more accurate relevance scoring. This is typically done as a
/// second-stage reranking step after initial retrieval.
/// </para>
/// <para><b>For Beginners:</b> Cross-encoder rerankers make search results much better.
/// 
/// Think of the difference between these two approaches:
/// 
/// **Regular Similarity (Bi-encoder):**
/// - Converts query to numbers: "best pizza" → [0.2, 0.8, 0.1, ...]
/// - Converts each document to numbers separately
/// - Compares numbers to find similar ones
/// - Fast but misses nuances
/// 
/// **Cross-Encoder Reranking:**
/// - Takes query + document together: "best pizza" + "Mario's serves authentic Italian pizza"
/// - Model reads both at once, understanding how they relate
/// - Produces a precise relevance score
/// - Slower but much more accurate
/// 
/// Real-world workflow:
/// 1. Initial retrieval: Fast method gets 100 candidates (e.g., vector search or BM25)
/// 2. Reranking: Cross-encoder carefully scores top 10-20 candidates
/// 3. Final results: Reordered by precise relevance scores
/// 
/// Why this works so well:
/// - Initial retrieval casts a wide net (high recall)
/// - Reranking refines the results (high precision)
/// - Best of both worlds: Speed + Accuracy
/// 
/// Common use cases:
/// - E-commerce search: Find the most relevant products
/// - Question answering: Find the paragraph that actually answers the question
/// - Document search: Rank by true relevance, not just keyword overlap
/// 
/// Performance impact:
/// - Reranking 10-20 docs: Fast enough for real-time
/// - Reranking 1000s of docs: Too slow, only rerank top candidates
/// </para>
/// </remarks>
public class CrossEncoderReranker<T> : RerankerBase<T>
{
    private readonly Func<string, string, T> _scoreFunction;
    private readonly int _maxPairsToScore;

    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    public override bool ModifiesScores => true;

    /// <summary>
    /// Initializes a new instance of the CrossEncoderReranker class.
    /// </summary>
    /// <param name="scoreFunction">A function that takes (query, documentContent) and returns a relevance score.</param>
    /// <param name="maxPairsToScore">Maximum number of query-document pairs to score (default: 20).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The scoreFunction is your cross-encoder model.
    /// 
    /// What it does:
    /// - Input: query string + document content string
    /// - Output: relevance score (higher = more relevant)
    /// 
    /// Example scoreFunction implementation:
    /// ```csharp
    /// Func&lt;string, string, double&gt; scoreFunc = (query, doc) => {
    ///     // Call your cross-encoder model (ONNX, API, etc.)
    ///     var score = crossEncoderModel.Score(query, doc);
    ///     return score;
    /// };
    /// ```
    /// 
    /// Popular cross-encoder models:
    /// - ms-marco-MiniLM-L-6-v2 (fast, good quality)
    /// - ms-marco-electra-base (slower, higher quality)
    /// - cross-encoder/ms-marco-TinyBERT-L-2-v2 (very fast, decent quality)
    /// 
    /// The maxPairsToScore parameter limits how many documents to rerank:
    /// - Too high: Slow, especially with large models
    /// - Too low: May miss relevant documents
    /// - Sweet spot: 10-20 for real-time, 50-100 for offline/batch
    /// </para>
    /// </remarks>
    public CrossEncoderReranker(Func<string, string, T> scoreFunction, int maxPairsToScore = 20)
    {
        if (scoreFunction == null)
            throw new ArgumentNullException(nameof(scoreFunction));

        if (maxPairsToScore <= 0)
            throw new ArgumentException("maxPairsToScore must be greater than zero", nameof(maxPairsToScore));

        _scoreFunction = scoreFunction;
        _maxPairsToScore = maxPairsToScore;
    }

    /// <summary>
    /// Reranks documents using the cross-encoder model.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="documents">The documents to rerank.</param>
    /// <returns>Documents reordered by cross-encoder relevance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method does the reranking magic.
    /// 
    /// What happens:
    /// 1. Takes your initial retrieval results
    /// 2. For each document, calls the cross-encoder model
    /// 3. Gets a precise relevance score for (query, document)
    /// 4. Sorts documents by these new scores
    /// 5. Returns reordered list
    /// 
    /// Performance tip:
    /// - Only pass top candidates to this method
    /// - Don't pass all 10,000 documents from your database
    /// - Typical flow: Retrieve 100 → Rerank top 20
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
    {
        var docList = documents.Take(_maxPairsToScore).ToList();

        if (docList.Count == 0)
            return Enumerable.Empty<Document<T>>();

        // Score each document with the cross-encoder
        var scoredDocs = new List<(Document<T> Doc, T Score)>();

        foreach (var doc in docList)
        {
            var score = _scoreFunction(query, doc.Content);
            scoredDocs.Add((doc, score));
        }

        // Sort by cross-encoder score (descending)
        var reranked = scoredDocs
            .OrderByDescending(x => Convert.ToDouble(x.Score))
            .Select(x =>
            {
                var doc = x.Doc;
                doc.RelevanceScore = x.Score;
                doc.HasRelevanceScore = true;
                return doc;
            })
            .ToList();

        return reranked;
    }
}
