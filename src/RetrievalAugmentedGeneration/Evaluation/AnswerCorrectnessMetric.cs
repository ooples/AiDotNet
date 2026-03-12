
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates the factual correctness of generated answers.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Assesses whether the generated answer contains factually correct information
/// by comparing against ground truth or using fact-checking mechanisms.
/// </remarks>
public class AnswerCorrectnessMetric<T> : RAGMetricBase<T>
{
    private readonly string _llmEndpoint;
    private readonly string _llmApiKey;

    public override string Name => "Answer Correctness";
    public override string Description => "Evaluates the factual correctness of generated answers by comparing against ground truth.";
    protected override bool RequiresGroundTruth => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="AnswerCorrectnessMetric{T}"/> class.
    /// </summary>
    /// <param name="llmEndpoint">The LLM API endpoint for fact checking.</param>
    /// <param name="llmApiKey">The API key for the LLM service.</param>
    public AnswerCorrectnessMetric(string llmEndpoint, string llmApiKey)
    {
        Guard.NotNull(llmEndpoint);
        _llmEndpoint = llmEndpoint;
        Guard.NotNull(llmApiKey);
        _llmApiKey = llmApiKey;
    }

    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (string.IsNullOrWhiteSpace(answer.Answer) || string.IsNullOrWhiteSpace(groundTruth))
            return NumOps.Zero;

        var words1 = GetWords(answer.Answer);
        var words2 = GetWords(groundTruth!);

        var intersection = words1.Intersect(words2).Count();
        var union = words1.Union(words2).Count();

        if (union == 0)
            return NumOps.Zero;

        return NumOps.Divide(NumOps.FromDouble(intersection), NumOps.FromDouble(union));
    }
}
