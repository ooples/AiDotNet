using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
    /// <summary>
    /// Initializes a new instance of the <see cref="ContextRelevanceMetric{T}"/> class.
    /// </summary>
    /// <param name="numericOperations">The numeric operations provider.</param>
    public ContextRelevanceMetric(INumericOperations<T> numericOperations)
        : base(numericOperations)
    {
    }

    /// <summary>
    /// Evaluates context relevance.
    /// </summary>
    protected override T EvaluateCore(
        string query,
        string answer,
        IEnumerable<Document<T>> retrievedDocuments,
        string groundTruth)
    {
        if (string.IsNullOrWhiteSpace(query))
            return NumOps.Zero;

        var docs = retrievedDocuments?.ToList();
        if (docs == null || docs.Count == 0)
            return NumOps.Zero;

        var totalRelevance = NumOps.Zero;
        var count = 0;

        foreach (var doc in docs)
        {
            var relevance = StatisticsHelper.JaccardSimilarity(query, doc.Content);
            totalRelevance = NumOps.Add(totalRelevance, NumOps.FromDouble(relevance));
            count++;
        }

        return count > 0 
            ? NumOps.Divide(totalRelevance, NumOps.FromInt(count))
            : NumOps.Zero;
    }
}
