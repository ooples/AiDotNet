using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for reranking retrieved documents to improve relevance ordering.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// A reranker refines the ordering of initially retrieved documents using more sophisticated
/// relevance scoring. While initial retrieval must be fast and may use simple similarity metrics,
/// reranking can employ computationally expensive methods like cross-encoders or large language
/// models to achieve better relevance rankings for the final result set.
/// </para>
/// <para><b>For Beginners:</b> A reranker is like a second opinion on search results.
/// 
/// Think of it like a two-stage hiring process:
/// 
/// Stage 1 (Initial Retrieval):
/// - Quick screening of 1000 applicants
/// - Filter to top 20 based on resume keywords
/// - Fast but might miss some good candidates
/// 
/// Stage 2 (Reranking):
/// - Detailed review of those 20 candidates
/// - Deeper analysis of experience and fit
/// - Slower but more accurate
/// - Final ranking of best 5
/// 
/// Similarly, reranking takes the initial search results and re-orders them using
/// more sophisticated analysis, ensuring the best results appear first.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Reranker")]
public interface IReranker<T>
{
    /// <summary>
    /// Reranks a collection of documents based on their relevance to a query.
    /// </summary>
    /// <param name="query">The query text used to assess relevance.</param>
    /// <param name="documents">The documents to rerank.</param>
    /// <returns>The documents reordered by relevance, with updated relevance scores.</returns>
    /// <remarks>
    /// <para>
    /// This method takes documents (typically from an initial retrieval stage) and reorders
    /// them using the reranker's scoring strategy. The returned documents have their
    /// RelevanceScore properties updated to reflect the reranker's assessment. Documents
    /// are returned in descending order of relevance.
    /// </para>
    /// <para><b>For Beginners:</b> This re-sorts search results to put the best matches first.
    /// 
    /// For example:
    /// - Input: 10 documents from initial search (roughly ordered)
    /// - Processing: Reranker deeply analyzes each one
    /// - Output: Same 10 documents, but better ordered with updated scores
    /// 
    /// The first document in the output is now the absolute best match,
    /// not just a good match like it might have been before reranking.
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> Rerank(string query, IEnumerable<Document<T>> documents);

    /// <summary>
    /// Reranks documents and returns only the top-k highest scoring results.
    /// </summary>
    /// <param name="query">The query text used to assess relevance.</param>
    /// <param name="documents">The documents to rerank.</param>
    /// <param name="topK">The number of top-ranked documents to return.</param>
    /// <returns>The top-k documents ordered by relevance, with updated relevance scores.</returns>
    /// <remarks>
    /// <para>
    /// This method combines reranking with result limiting. It's more efficient than
    /// reranking all documents when only the top results are needed, as some implementations
    /// can optimize by not fully scoring lower-ranked candidates.
    /// </para>
    /// <para><b>For Beginners:</b> This re-sorts and then keeps only the best results.
    /// 
    /// For example:
    /// - Input: 20 documents from initial search
    /// - Rerank: Analyze and re-sort all 20
    /// - Output: Only the top 5 best matches (if topK = 5)
    /// 
    /// This is useful when you retrieved many documents initially but only need
    /// the very best ones after careful analysis.
    /// </para>
    /// </remarks>
    IEnumerable<Document<T>> Rerank(string query, IEnumerable<Document<T>> documents, int topK);

    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some rerankers (like identity/pass-through rerankers) don't modify scores,
    /// while others compute new scores. This property indicates the behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the reranker actually does something.
    /// 
    /// - False: Pass-through reranker (no changes, just returns documents as-is)
    /// - True: Active reranker (recalculates scores and reorders documents)
    /// 
    /// A pass-through reranker is useful as a default when you don't need reranking,
    /// but want to keep the code structure consistent.
    /// </para>
    /// </remarks>
    bool ModifiesScores { get; }
}
