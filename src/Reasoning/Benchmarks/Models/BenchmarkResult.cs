using AiDotNet.LinearAlgebra;

namespace AiDotNet.Reasoning.Benchmarks.Models;

/// <summary>
/// Results from evaluating a reasoning system on a benchmark.
/// </summary>
/// <typeparam name="T">The numeric type used for metrics (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is like a report card for your reasoning system's performance
/// on a standardized test.
///
/// **Key metrics:**
/// - **Accuracy**: Percentage of problems answered correctly (most important)
/// - **Total Evaluated**: How many problems were tested
/// - **Correct Count**: How many were answered correctly
/// - **Average Confidence**: How confident the system was on average
///
/// **Example:**
/// ```
/// Benchmark: GSM8K (Grade School Math)
/// Problems Evaluated: 100
/// Correct: 87
/// Accuracy: 87.0%
/// Average Confidence: 0.92
/// Average Time: 3.2 seconds per problem
/// ```
///
/// This would indicate the system got 87 out of 100 math problems correct, with high confidence.
/// </para>
/// </remarks>
public class BenchmarkResult<T>
{
    /// <summary>
    /// Name of the benchmark that was evaluated.
    /// </summary>
    public string BenchmarkName { get; set; } = string.Empty;

    /// <summary>
    /// Total number of problems evaluated.
    /// </summary>
    public int TotalEvaluated { get; set; }

    /// <summary>
    /// Number of problems answered correctly.
    /// </summary>
    public int CorrectCount { get; set; }

    /// <summary>
    /// Overall accuracy (correct / total) as a value between 0.0 and 1.0.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is your "score" on the test.
    /// 0.87 means 87% correct, 1.0 means perfect score (100%).
    /// </para>
    /// </remarks>
    public T Accuracy { get; set; } = default!;

    /// <summary>
    /// Confidence scores for each evaluated problem (as a Vector).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each problem gets a confidence score (0.0-1.0) indicating
    /// how sure the system was about its answer. High confidence with wrong answers indicates
    /// the system doesn't know what it doesn't know (bad calibration).
    /// </para>
    /// </remarks>
    public Vector<T>? ConfidenceScores { get; set; }

    /// <summary>
    /// Average confidence across all evaluated problems.
    /// </summary>
    public T AverageConfidence { get; set; } = default!;

    /// <summary>
    /// Total time spent evaluating all problems.
    /// </summary>
    public TimeSpan TotalDuration { get; set; }

    /// <summary>
    /// Average time per problem.
    /// </summary>
    public TimeSpan AverageTimePerProblem =>
        TotalEvaluated > 0 ? TimeSpan.FromMilliseconds(TotalDuration.TotalMilliseconds / TotalEvaluated) : TimeSpan.Zero;

    /// <summary>
    /// Detailed results for each evaluated problem.
    /// </summary>
    public List<ProblemEvaluation<T>> ProblemResults { get; set; } = new();

    /// <summary>
    /// Breakdown of accuracy by category (if applicable).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows performance in different areas.
    /// For example, in math: {"algebra": 0.92, "geometry": 0.78, "arithmetic": 0.95}
    /// This helps identify strengths and weaknesses.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> AccuracyByCategory { get; set; } = new();

    /// <summary>
    /// Additional benchmark-specific metrics.
    /// </summary>
    public Dictionary<string, object> Metrics { get; set; } = new();

    /// <summary>
    /// Gets a summary string of the benchmark results.
    /// </summary>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Benchmark: {BenchmarkName}");
        sb.AppendLine($"Problems Evaluated: {TotalEvaluated}");
        sb.AppendLine($"Correct: {CorrectCount}");
        sb.AppendLine($"Accuracy: {Convert.ToDouble(Accuracy):P1}");
        sb.AppendLine($"Average Confidence: {Convert.ToDouble(AverageConfidence):F3}");
        sb.AppendLine($"Total Time: {TotalDuration.TotalSeconds:F1}s");
        sb.AppendLine($"Average Time per Problem: {AverageTimePerProblem.TotalSeconds:F2}s");

        if (AccuracyByCategory.Count > 0)
        {
            sb.AppendLine("\nAccuracy by Category:");
            foreach (var kvp in AccuracyByCategory.OrderByDescending(kvp => Convert.ToDouble(kvp.Value)))
            {
                string category = kvp.Key;
                T accuracy = kvp.Value;
                sb.AppendLine($"  {category}: {Convert.ToDouble(accuracy):P1}");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Returns a summary string.
    /// </summary>
    public override string ToString() => GetSummary();
}

/// <summary>
/// Result for a single problem evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This records what happened when the system tried to solve
/// one specific problem.
/// </para>
/// </remarks>
public class ProblemEvaluation<T>
{
    /// <summary>
    /// The problem ID.
    /// </summary>
    public string ProblemId { get; set; } = string.Empty;

    /// <summary>
    /// The problem statement.
    /// </summary>
    public string Problem { get; set; } = string.Empty;

    /// <summary>
    /// The correct answer.
    /// </summary>
    public string CorrectAnswer { get; set; } = string.Empty;

    /// <summary>
    /// The system's answer.
    /// </summary>
    public string SystemAnswer { get; set; } = string.Empty;

    /// <summary>
    /// Whether the answer was correct.
    /// </summary>
    public bool IsCorrect { get; set; }

    /// <summary>
    /// Confidence score for this answer.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Time taken to solve this problem.
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Category of this problem.
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Additional metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();
}
