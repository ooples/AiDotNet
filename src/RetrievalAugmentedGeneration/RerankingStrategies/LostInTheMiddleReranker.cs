using AiDotNet.Helpers;
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
    /// Initializes a new instance of the <see cref="LostInTheMiddleReranker{T}"/> class.
    /// </summary>
    /// <param name="numericOperations">The numeric operations provider.</param>
    public LostInTheMiddleReranker(INumericOperations<T> numericOperations)
        : base(numericOperations)
    {
    }

    /// <summary>
    /// Reranks documents to avoid the "lost in the middle" problem.
    /// </summary>
    /// <remarks>
    /// Strategy: Place most relevant at start, 2nd most relevant at end, 3rd in middle,
    /// alternating to distribute important documents to positions LLMs pay attention to.
    /// </remarks>
    public override IEnumerable<Document<T>> Rerank(string query, IEnumerable<Document<T>> documents, int topK)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (documents == null)
            throw new ArgumentNullException(nameof(documents));

        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        var docList = documents
            .OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : NumOps.Zero)
            .Take(topK)
            .ToList();

        if (docList.Count <= 2)
            return docList;

        var reordered = new List<Document<T>>(new Document<T>[docList.Count]);
        var startIdx = 0;
        var endIdx = docList.Count - 1;
        var useStart = true;

        foreach (var doc in docList)
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
