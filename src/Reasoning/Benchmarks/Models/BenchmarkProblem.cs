namespace AiDotNet.Reasoning.Benchmarks.Models;

/// <summary>
/// Represents a single problem in a benchmark dataset.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is one test question with its correct answer.
/// Think of it like a single problem on a homework assignment or test.
///
/// **Example (GSM8K):**
/// ```
/// Problem: "Janet has 15 apples. She gives 40% to her friend. How many does she have left?"
/// CorrectAnswer: "9"
/// ```
///
/// **Example (HumanEval):**
/// ```
/// Problem: "Write a function that returns True if a number is prime, False otherwise"
/// CorrectAnswer: "def is_prime(n): ..." (reference implementation)
/// ```
/// </para>
/// </remarks>
public class BenchmarkProblem
{
    /// <summary>
    /// Unique identifier for this problem.
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// The problem statement or question.
    /// </summary>
    public string Problem { get; set; } = string.Empty;

    /// <summary>
    /// The correct answer or solution.
    /// </summary>
    public string CorrectAnswer { get; set; } = string.Empty;

    /// <summary>
    /// Category or topic of this problem (e.g., "algebra", "geometry", "sorting").
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Difficulty level (e.g., "easy", "medium", "hard").
    /// </summary>
    public string Difficulty { get; set; } = string.Empty;

    /// <summary>
    /// Additional metadata specific to the benchmark.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Gets a summary string of this problem.
    /// </summary>
    public override string ToString() =>
        $"[{Id}] {Category} - {Difficulty}: {Problem.Substring(0, Math.Min(50, Problem.Length))}...";
}
