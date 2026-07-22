using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates whether the generated answer is faithful to the source documents.
/// </summary>
/// <remarks>
/// <para>
/// Faithfulness measures how well the generated answer adheres to the information in the
/// retrieved source documents. A faithful answer doesn't hallucinate or add information
/// not present in the sources. This metric checks for unsupported claims by analyzing
/// word overlap between the answer and source documents.
/// </para>
/// <para><b>For Beginners:</b> This checks if the AI made stuff up or stuck to the sources.
/// 
/// Think of it like plagiarism checking in reverse:
/// - High score: The answer only says things found in the source documents
/// - Low score: The answer includes information not in the sources (hallucination)
/// 
/// For example:
/// - Sources say: "Photosynthesis produces oxygen"
/// - Faithful answer: "Photosynthesis produces oxygen" ✓ (score: 1.0)
/// - Unfaithful answer: "Photosynthesis produces oxygen and nitrogen" ✗ (score: 0.5)
/// 
/// Why this matters:
/// - Prevents the AI from making up facts
/// - Ensures answers are verifiable
/// - Builds user trust
/// 
/// Note: This is a simplified metric. Production systems should use more sophisticated
/// techniques like NLI (Natural Language Inference) models.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class FaithfulnessMetric<T> : RAGMetricBase<T>
{
    private readonly ITextGenerator? _generator;

    /// <summary>
    /// Initializes a new instance of the <see cref="FaithfulnessMetric{T}"/> class.
    /// </summary>
    /// <param name="generator">
    /// Optional text generator used as an NLI-style judge. When supplied, the answer is broken
    /// into claims and each claim is checked for support against the retrieved context, giving a
    /// score of supported-claims / total-claims. When <c>null</c>, the metric falls back to the
    /// offline lexical word-overlap heuristic.
    /// </param>
    public FaithfulnessMetric(ITextGenerator? generator = null)
    {
        _generator = generator;
    }

    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    public override string Name => "Faithfulness";

    /// <summary>
    /// Gets the description of what this metric measures.
    /// </summary>
    public override string Description =>
        "Measures how well the answer adheres to information in source documents (no hallucination)";

    /// <summary>
    /// Gets a value indicating whether this metric requires ground truth.
    /// </summary>
    protected override bool RequiresGroundTruth => false;

    /// <summary>
    /// Evaluates faithfulness by measuring overlap between answer and sources.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">Not used for this metric.</param>
    /// <returns>Faithfulness score (0-1).</returns>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (!answer.SourceDocuments.Any())
            return NumOps.Zero;

        var sourceText = string.Join(" ", answer.SourceDocuments.Select(d => d.Content));

        if (_generator != null)
            return EvaluateWithJudge(answer.Answer, sourceText);

        // Offline lexical fallback: fraction of answer words present in the source documents.
        var answerWords = GetWords(answer.Answer);
        if (answerWords.Count == 0)
            return NumOps.Zero;

        var sourceWords = GetWords(sourceText);
        var supportedWords = answerWords.Intersect(sourceWords).Count();

        return NumOps.Divide(NumOps.FromDouble(supportedWords), NumOps.FromDouble(answerWords.Count));
    }

    /// <summary>
    /// NLI-style faithfulness: split the answer into claims and ask the generator whether each
    /// claim can be inferred from the context. Score = supported claims / total claims.
    /// </summary>
    private T EvaluateWithJudge(string answerText, string context)
    {
        var claims = SplitIntoSentences(answerText);
        if (claims.Count == 0)
            return NumOps.Zero;

        int supported = 0;
        foreach (var claim in claims)
        {
            var prompt =
                "You are a strict fact-checker. Determine whether the statement can be directly " +
                "inferred from the context. Reply with only 'yes' or 'no'.\n\n" +
                $"Context: {context}\n\nStatement: {claim}\n\nSupported (yes/no):";
            var reply = _generator!.Generate(prompt);
            if (ParseAffirmative(reply))
                supported++;
        }

        return NumOps.Divide(NumOps.FromDouble(supported), NumOps.FromDouble(claims.Count));
    }
}
