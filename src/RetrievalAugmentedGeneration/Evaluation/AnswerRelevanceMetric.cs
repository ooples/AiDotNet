using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates how relevant the generated answer is to the original query (RAGAS answer relevance).
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RAGAS answer relevance works by asking a generator to produce the questions that the answer would
/// answer, then measuring the mean cosine similarity between the embeddings of those generated
/// questions and the original query. A high score means the answer is on-topic and directly addresses
/// the question; a low score means the answer is evasive, incomplete, or off-topic.
/// </para>
/// <para>
/// This metric requires <b>both</b> a text generator (to reverse-generate questions) and an embedding
/// model (to compare them to the query). When either is missing it falls back to the offline lexical
/// Jaccard overlap between the answer and the query.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class AnswerRelevanceMetric<T> : RAGMetricBase<T>
{
    private readonly ITextGenerator? _generator;
    private readonly IEmbeddingModel<T>? _embeddingModel;
    private readonly int _numQuestions;

    /// <summary>
    /// Initializes a new instance of the <see cref="AnswerRelevanceMetric{T}"/> class.
    /// </summary>
    /// <param name="generator">Optional text generator used to reverse-generate questions from the answer.</param>
    /// <param name="embeddingModel">Optional embedding model used to compare generated questions to the query.</param>
    /// <param name="numQuestions">The number of questions to reverse-generate (default 3).</param>
    public AnswerRelevanceMetric(
        ITextGenerator? generator = null,
        IEmbeddingModel<T>? embeddingModel = null,
        int numQuestions = 3)
    {
        _generator = generator;
        _embeddingModel = embeddingModel;
        _numQuestions = numQuestions > 0 ? numQuestions : 3;
    }

    /// <summary>Gets the name of this metric.</summary>
    public override string Name => "Answer Relevance";

    /// <summary>Gets the description of what this metric measures.</summary>
    public override string Description =>
        "Measures how directly the generated answer addresses the original query (RAGAS answer relevance)";

    /// <summary>Gets a value indicating whether this metric requires ground truth.</summary>
    protected override bool RequiresGroundTruth => false;

    /// <summary>
    /// Evaluates answer relevance to the query.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">Not used for this metric.</param>
    /// <returns>Relevance score (0-1).</returns>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (string.IsNullOrWhiteSpace(answer.Query))
            return NumOps.Zero;

        if (_generator != null && _embeddingModel != null)
            return EvaluateWithModels(answer.Query, answer.Answer);

        // Offline lexical fallback: Jaccard overlap between the answer and the query.
        return NumOps.FromDouble(JaccardSimilarity(answer.Answer, answer.Query));
    }

    /// <summary>
    /// Reverse-generates questions from the answer and averages their cosine similarity to the query.
    /// </summary>
    private T EvaluateWithModels(string query, string answerText)
    {
        var prompt =
            $"Generate {_numQuestions} distinct questions that the following answer would correctly and " +
            "completely answer. Put each question on its own line.\n\n" +
            $"Answer: {answerText}\n\nQuestions:";
        var reply = _generator!.Generate(prompt);
        var questions = ParseListLines(reply);
        if (questions.Count == 0)
            return NumOps.Zero;

        double total = 0.0;
        foreach (var question in questions)
        {
            total += Math.Max(0.0, EmbeddingCosine(_embeddingModel!, question, query));
        }

        return NumOps.FromDouble(total / questions.Count);
    }
}
