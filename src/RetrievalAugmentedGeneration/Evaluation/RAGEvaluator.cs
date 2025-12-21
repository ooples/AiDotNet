
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates RAG system performance using multiple metrics.
/// </summary>
/// <remarks>
/// <para>
/// The RAG evaluator runs multiple evaluation metrics on grounded answers and aggregates
/// the results. This provides a comprehensive view of RAG system performance across different
/// quality dimensions (faithfulness, similarity, coverage, etc.).
/// </para>
/// <para><b>For Beginners:</b> This runs all your tests on the RAG system and gives you a report card.
/// 
/// Think of it like grading a student across multiple subjects:
/// - Math (Faithfulness): Did they show their work correctly?
/// - English (Similarity): How close is the essay to the example?
/// - Science (Coverage): Did they research enough sources?
/// 
/// The evaluator:
/// 1. Takes your RAG system's answer
/// 2. Runs all configured metrics on it
/// 3. Gives you scores for each metric
/// 4. Calculates an overall average score
/// 
/// Use this to:
/// - Compare different RAG configurations
/// - Track improvements over time
/// - Identify specific weaknesses
/// - Make data-driven optimization decisions
/// </para>
/// </remarks>
public class RAGEvaluator<T>
{
    private readonly IReadOnlyList<IRAGMetric<T>> _metrics;

    /// <summary>
    /// Gets the metrics used by this evaluator.
    /// </summary>
    public IReadOnlyList<IRAGMetric<T>> Metrics => _metrics;

    /// <summary>
    /// Initializes a new instance of the RAGEvaluator class with specified metrics.
    /// </summary>
    /// <param name="metrics">The metrics to use for evaluation.</param>
    public RAGEvaluator(IEnumerable<IRAGMetric<T>> metrics)
    {
        if (metrics == null)
            throw new ArgumentNullException(nameof(metrics));

        _metrics = metrics.ToList();

        if (_metrics.Count == 0)
            throw new ArgumentException("At least one metric must be provided", nameof(metrics));
    }

    /// <summary>
    /// Initializes a new instance of the RAGEvaluator class with default metrics.
    /// </summary>
    public RAGEvaluator()
    {
        _metrics = new List<IRAGMetric<T>>
        {
            new FaithfulnessMetric<T>(),
            new ContextCoverageMetric<T>()
        };
    }

    /// <summary>
    /// Evaluates a grounded answer using all configured metrics.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">The reference answer (optional, required by some metrics).</param>
    /// <returns>Evaluation results with scores for each metric.</returns>
    public EvaluationResult Evaluate(GroundedAnswer<T> answer, string? groundTruth = null)
    {
        if (answer == null)
            throw new ArgumentNullException(nameof(answer));

        var metricScores = new Dictionary<string, double>();

        foreach (var metric in _metrics)
        {
            try
            {
                var score = metric.Evaluate(answer, groundTruth);
                metricScores[metric.Name] = Convert.ToDouble(score);
            }
            catch (ArgumentNullException ex) when (ex.ParamName == "groundTruth")
            {
                throw new ArgumentException(
                    $"Metric '{metric.Name}' requires ground truth data. Please provide ground truth for evaluation.",
                    nameof(groundTruth),
                    ex);
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("ground truth"))
            {
                throw new ArgumentException(
                    $"Metric '{metric.Name}' requires ground truth data. Please provide ground truth for evaluation.",
                    nameof(groundTruth),
                    ex);
            }
        }

        var overallScore = metricScores.Values.Average();

        return new EvaluationResult
        {
            Query = answer.Query,
            Answer = answer.Answer,
            MetricScores = metricScores,
            OverallScore = overallScore,
            SourceDocumentCount = answer.SourceDocuments.Count(),
            ConfidenceScore = answer.ConfidenceScore
        };
    }

    /// <summary>
    /// Evaluates multiple grounded answers and returns aggregated results.
    /// </summary>
    /// <param name="answers">The answers to evaluate.</param>
    /// <param name="groundTruths">Corresponding ground truth answers (must match answer count if provided).</param>
    /// <returns>Collection of evaluation results.</returns>
    public IEnumerable<EvaluationResult> EvaluateBatch(
        IEnumerable<GroundedAnswer<T>> answers,
        IEnumerable<string>? groundTruths = null)
    {
        if (answers == null)
            throw new ArgumentNullException(nameof(answers));

        var answerList = answers.ToList();
        var truthList = groundTruths?.ToList() ?? new List<string>();

        if (truthList.Any() && truthList.Count != answerList.Count)
            throw new ArgumentException(
                "Ground truth count must match answer count",
                nameof(groundTruths));

        var results = new List<EvaluationResult>();

        for (int i = 0; i < answerList.Count; i++)
        {
            var truth = truthList.Any() ? truthList[i] : null;
            results.Add(Evaluate(answerList[i], truth));
        }

        return results;
    }

    /// <summary>
    /// Calculates aggregate statistics across multiple evaluation results.
    /// </summary>
    /// <param name="results">The evaluation results to aggregate.</param>
    /// <returns>Aggregated statistics.</returns>
    public AggregateStats GetAggregateStats(IEnumerable<EvaluationResult> results)
    {
        if (results == null)
            throw new ArgumentNullException(nameof(results));

        var resultList = results.ToList();
        if (resultList.Count == 0)
            throw new ArgumentException("Results collection cannot be empty", nameof(results));

        var stats = new Dictionary<string, (double Mean, double StdDev, double Min, double Max)>();

        // Aggregate each metric
        foreach (var metric in _metrics)
        {
            var scores = resultList
                .Where(r => r.MetricScores.ContainsKey(metric.Name))
                .Select(r => r.MetricScores[metric.Name])
                .ToList();

            if (scores.Any())
            {
                var scoresVector = new Vector<double>(scores.ToArray());
                var mean = Convert.ToDouble(StatisticsHelper<double>.CalculateMean(scoresVector));
                var stdDev = Convert.ToDouble(StatisticsHelper<double>.CalculateStandardDeviation(scoresVector));
                var min = scores.Min();
                var max = scores.Max();

                stats[metric.Name] = (mean, stdDev, min, max);
            }
        }

        return new AggregateStats
        {
            SampleCount = resultList.Count,
            MetricStatistics = stats,
            AverageOverallScore = resultList.Select(r => r.OverallScore).Average(),
            AverageConfidence = resultList.Select(r => r.ConfidenceScore).Average()
        };
    }
}

/// <summary>
/// Represents the evaluation results for a single grounded answer.
/// </summary>
public class EvaluationResult
{
    /// <summary>
    /// Gets or sets the query that was asked.
    /// </summary>
    public string Query { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the generated answer.
    /// </summary>
    public string Answer { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the scores for each metric.
    /// </summary>
    public Dictionary<string, double> MetricScores { get; set; } = new();

    /// <summary>
    /// Gets or sets the overall score (average of all metrics).
    /// </summary>
    public double OverallScore { get; set; }

    /// <summary>
    /// Gets or sets the number of source documents used.
    /// </summary>
    public int SourceDocumentCount { get; set; }

    /// <summary>
    /// Gets or sets the generator's confidence score.
    /// </summary>
    public double ConfidenceScore { get; set; }
}

/// <summary>
/// Represents aggregated statistics across multiple evaluations.
/// </summary>
public class AggregateStats
{
    /// <summary>
    /// Gets or sets the number of samples in the aggregate.
    /// </summary>
    public int SampleCount { get; set; }

    /// <summary>
    /// Gets or sets statistics for each metric.
    /// </summary>
    public Dictionary<string, (double Mean, double StdDev, double Min, double Max)> MetricStatistics { get; set; } = new();

    /// <summary>
    /// Gets or sets the average overall score across all samples.
    /// </summary>
    public double AverageOverallScore { get; set; }

    /// <summary>
    /// Gets or sets the average confidence score across all samples.
    /// </summary>
    public double AverageConfidence { get; set; }
}
