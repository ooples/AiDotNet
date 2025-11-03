using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Rerankers;

/// <summary>
/// A pass-through reranker that returns documents without modifying their order or scores.
/// </summary>
/// <remarks>
/// <para>
/// This implementation is a no-op reranker that simply returns the input documents unchanged.
/// It's useful as a default when reranking is not needed or as a placeholder in the pipeline.
/// The reranker interface can be swapped later with more sophisticated implementations without
/// changing the pipeline structure.
/// </para>
/// <para><b>For Beginners:</b> This reranker doesn't actually do anything - it's a placeholder.
/// 
/// Think of it like a rubber stamp that approves everything as-is:
/// - Documents come in → Documents go out unchanged
/// - No reordering, no new scores
/// - Just passes through
/// 
/// Why have a "do nothing" reranker?
/// - Keeps your RAG pipeline structure consistent
/// - Easy to swap in a real reranker later
/// - No code changes needed when you upgrade
/// - Clean architecture (always have a reranker, even if it does nothing)
/// 
/// For example:
/// - Input: [Doc A (score 0.9), Doc B (score 0.8), Doc C (score 0.7)]
/// - Output: [Doc A (score 0.9), Doc B (score 0.8), Doc C (score 0.7)]
/// - Exactly the same!
/// 
/// Later, replace with CrossEncoderReranker for better results without changing your code.
/// </para>
/// </remarks>
public class IdentityReranker : RerankerBase
{
    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This reranker does not modify scores, so this returns false.
    /// </para>
    /// </remarks>
    public override bool ModifiesScores => false;

    /// <summary>
    /// Core reranking logic that returns documents unchanged.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="documents">The validated and materialized list of documents.</param>
    /// <returns>The same documents in the same order with unchanged scores.</returns>
    protected override IEnumerable<Document> RerankCore(string query, IList<Document> documents)
    {
        // Pass through without changes
        return documents;
    }
}
