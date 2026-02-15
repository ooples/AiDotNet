using AiDotNet.Reasoning.Benchmarks.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for reasoning benchmarks that evaluate model performance.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A benchmark is like a standardized test for AI reasoning systems.
/// Just like students take SAT or ACT tests to measure their abilities, AI systems are evaluated
/// on benchmarks to measure their reasoning capabilities.
///
/// **Common benchmarks:**
/// - **GSM8K**: Grade school math problems (8,000 questions)
/// - **MATH**: Competition-level mathematics
/// - **HumanEval**: Code generation tasks
/// - **MMLU**: Multiple choice questions across many subjects
/// - **ARC-AGI**: Abstract reasoning puzzles
///
/// **Why benchmarks matter:**
/// - Objective measurement of performance
/// - Compare different approaches
/// - Track improvements over time
/// - Identify strengths and weaknesses
///
/// **Example:**
/// ```csharp
/// var benchmark = new GSM8KBenchmark&lt;double&gt;();
/// var results = await benchmark.EvaluateAsync(reasoner, sampleSize: 100);
/// Console.WriteLine($"Accuracy: {results.Accuracy:P1}"); // "Accuracy: 87.5%"
/// ```
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Benchmark")]
public interface IBenchmark<T>
{
    /// <summary>
    /// Gets the name of this benchmark.
    /// </summary>
    string BenchmarkName { get; }

    /// <summary>
    /// Gets a description of what this benchmark measures.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Gets the total number of problems in this benchmark.
    /// </summary>
    int TotalProblems { get; }

    /// <summary>
    /// Evaluates a reasoning strategy on this benchmark.
    /// </summary>
    /// <param name="evaluateFunction">Function that takes a problem and returns an answer.</param>
    /// <param name="sampleSize">Number of problems to evaluate (null for all).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Benchmark results with accuracy and detailed metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This runs the benchmark by:
    /// 1. Selecting problems (either all or a random sample)
    /// 2. Asking the reasoning system to solve each one
    /// 3. Comparing answers to the correct solutions
    /// 4. Calculating accuracy and other metrics
    ///
    /// The evaluateFunction is your reasoning system - it takes a problem string and returns
    /// an answer string.
    /// </para>
    /// </remarks>
    Task<BenchmarkResult<T>> EvaluateAsync(
        Func<string, Task<string>> evaluateFunction,
        int? sampleSize = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads benchmark problems (for inspection or custom evaluation).
    /// </summary>
    /// <param name="count">Number of problems to load (null for all).</param>
    /// <returns>List of benchmark problems.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns the actual problems and their correct answers
    /// so you can inspect them or run custom evaluations.
    /// </para>
    /// </remarks>
    Task<List<BenchmarkProblem>> LoadProblemsAsync(int? count = null);
}
