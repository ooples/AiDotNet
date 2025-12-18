using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates the relevance of retrieved context to the query.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Measures how relevant the retrieved documents are to answering the query,
/// helping identify retrieval quality issues.
/// </remarks>
public class ContextRelevanceMetric<T> : RAGMetricBase<T>
{
    public override string Name => "Context Relevance";
    public override string Description => "Measures how relevant the retrieved documents are to answering the query";
    protected override bool RequiresGroundTruth => false;

    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (string.IsNullOrWhiteSpace(answer.Query) || !answer.SourceDocuments.Any())
            return NumOps.Zero;

        var totalRelevance = NumOps.Zero;
        var count = 0;

        foreach (var doc in answer.SourceDocuments)
        {
            var words1 = GetWords(answer.Query);
            var words2 = GetWords(doc.Content);

            var intersection = words1.Intersect(words2).Count();
            var union = words1.Union(words2).Count();

            if (union > 0)
            {
                var relevance = NumOps.Divide(NumOps.FromDouble(intersection), NumOps.FromDouble(union));
                totalRelevance = NumOps.Add(totalRelevance, relevance);
                count++;
            }
        }

        return count > 0
            ? NumOps.Divide(totalRelevance, NumOps.FromDouble(count))
            : NumOps.Zero;
    }
}
