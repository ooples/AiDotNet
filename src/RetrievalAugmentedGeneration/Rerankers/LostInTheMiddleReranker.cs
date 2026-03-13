
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies;

/// <summary>
/// Addresses the "lost in the middle" problem by strategically reordering documents.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Research shows LLMs often ignore information in the middle of long contexts.
/// This reranker places most relevant documents at the beginning and end of the context.
/// </remarks>
public class LostInTheMiddleReranker<T> : RerankerBase<T>
{
    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    public override bool ModifiesScores => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="LostInTheMiddleReranker{T}"/> class.
    /// </summary>
    public LostInTheMiddleReranker()
    {
    }

    /// <summary>
    /// Reranks documents to avoid the "lost in the middle" problem.
    /// </summary>
    /// <remarks>
    /// Strategy: Place most relevant at start, 2nd most relevant at end, 3rd in middle,
    /// alternating to distribute important documents to positions LLMs pay attention to.
    /// </remarks>
    protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
    {
        if (documents.Count <= 2)
            return documents;

        var sorted = documents
            .OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : NumOps.Zero)
            .ToList();

        var reordered = new List<Document<T>>(new Document<T>[sorted.Count]);
        var startIdx = 0;
        var endIdx = sorted.Count - 1;
        var useStart = true;

        foreach (var doc in sorted)
        {
            if (useStart)
            {
                reordered[startIdx] = doc;
                startIdx++;
            }
            else
            {
                reordered[endIdx] = doc;
                endIdx--;
            }
            useStart = !useStart;
        }

        return reordered;
    }
}
