using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates the relevance of retrieved context to the query.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Measures how relevant the retrieved documents are to answering the query, helping identify
/// retrieval quality issues. When an embedding model is supplied, relevance is the mean cosine
/// similarity between the query and each retrieved document; otherwise the offline lexical Jaccard
/// word-overlap heuristic is used.
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class ContextRelevanceMetric<T> : RAGMetricBase<T>
{
    private readonly IEmbeddingModel<T>? _embeddingModel;

    /// <summary>
    /// Initializes a new instance of the <see cref="ContextRelevanceMetric{T}"/> class.
    /// </summary>
    /// <param name="embeddingModel">
    /// Optional embedding model. When supplied, query-to-document relevance is measured with cosine
    /// similarity; when <c>null</c>, the offline lexical Jaccard heuristic is used.
    /// </param>
    public ContextRelevanceMetric(IEmbeddingModel<T>? embeddingModel = null)
    {
        _embeddingModel = embeddingModel;
    }

    /// <summary>Gets the name of this metric.</summary>
    public override string Name => "Context Relevance";

    /// <summary>Gets the description of what this metric measures.</summary>
    public override string Description => "Measures how relevant the retrieved documents are to answering the query";

    /// <summary>Gets a value indicating whether this metric requires ground truth.</summary>
    protected override bool RequiresGroundTruth => false;

    /// <summary>
    /// Evaluates the mean relevance of retrieved documents to the query.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">Not used for this metric.</param>
    /// <returns>Relevance score (0-1).</returns>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (string.IsNullOrWhiteSpace(answer.Query) || !answer.SourceDocuments.Any())
            return NumOps.Zero;

        double total = 0.0;
        int count = 0;

        foreach (var doc in answer.SourceDocuments)
        {
            double relevance = _embeddingModel != null
                ? Math.Max(0.0, EmbeddingCosine(_embeddingModel, answer.Query, doc.Content))
                : JaccardSimilarity(answer.Query, doc.Content);

            total += relevance;
            count++;
        }

        return count > 0 ? NumOps.FromDouble(total / count) : NumOps.Zero;
    }
}
