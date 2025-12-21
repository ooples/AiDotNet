using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// MATH benchmark for evaluating advanced mathematical reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The MATH dataset contains 12,500 challenging competition mathematics problems
/// from high school math competitions (AMC, AIME, etc.). These are significantly harder than GSM8K.
///
/// **Example problems:**
/// - "Find the sum of all positive integers n such that sqrt(n^2 + 85) is an integer."
/// - "A square is inscribed in a circle. What is the ratio of the area of the circle to the square?"
/// - "Solve the system of equations: x + y + z = 6, xy + xz + yz = 11, xyz = 6"
///
/// **Why it's important:**
/// - Tests advanced mathematical reasoning
/// - Requires complex multi-step solutions
/// - Includes algebra, geometry, number theory, calculus
/// - Benchmark for reasoning capability at competition level
///
/// **Performance levels:**
/// - Human (expert): 90-95%
/// - GPT-3.5: ~7%
/// - GPT-4: ~42%
/// - ChatGPT o1: ~85%
/// - DeepSeek-R1: ~79.8%
/// - Minerva (540B): ~50%
///
/// **Research:**
/// "Measuring Mathematical Problem Solving With the MATH Dataset" (Hendrycks et al., 2021)
/// https://arxiv.org/abs/2103.03874
/// </para>
/// </remarks>
public class MATHBenchmark<T> : IBenchmark<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public MATHBenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "MATH";

    /// <inheritdoc/>
    public string Description =>
        "MATH: 12,500 challenging high school competition mathematics problems across 7 subjects. " +
        "Tests advanced mathematical reasoning including algebra, geometry, number theory, and calculus.";

    /// <inheritdoc/>
    public int TotalProblems => 5000; // Test set size

    /// <inheritdoc/>
    public async Task<BenchmarkResult<T>> EvaluateAsync(
        Func<string, Task<string>> evaluateFunction,
        int? sampleSize = null,
        CancellationToken cancellationToken = default)
    {
        if (evaluateFunction == null)
            throw new ArgumentNullException(nameof(evaluateFunction));

        var stopwatch = Stopwatch.StartNew();
        var problems = await LoadProblemsAsync(sampleSize);

        // Guard against empty problem sets
        if (problems.Count == 0)
        {
            return new BenchmarkResult<T>
            {
                BenchmarkName = BenchmarkName,
                TotalEvaluated = 0,
                CorrectCount = 0,
                Accuracy = _numOps.Zero,
                ConfidenceScores = new Vector<T>(0),
                AverageConfidence = _numOps.Zero,
                TotalDuration = TimeSpan.Zero,
                ProblemResults = new List<ProblemEvaluation<T>>()
            };
        }

        var result = new BenchmarkResult<T>
        {
            BenchmarkName = BenchmarkName,
            TotalEvaluated = problems.Count
        };

        int correctCount = 0;
        var confidenceScores = new List<T>();
        var problemResults = new List<ProblemEvaluation<T>>();
        var categoryCorrect = new Dictionary<string, int>();
        var categoryTotal = new Dictionary<string, int>();

        for (int i = 0; i < problems.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var problem = problems[i];
            var problemStopwatch = Stopwatch.StartNew();

            string systemAnswer;
            try
            {
                systemAnswer = await evaluateFunction(problem.Problem);
            }
            catch (Exception ex)
            {
                systemAnswer = $"ERROR: {ex.Message}";
            }

            problemStopwatch.Stop();

            // Extract answer (more complex than GSM8K)
            string? extractedAnswer = ExtractMathAnswer(systemAnswer);
            string? correctAnswer = ExtractMathAnswer(problem.CorrectAnswer);

            bool isCorrect = CompareMathAnswers(extractedAnswer, correctAnswer);

            if (isCorrect) correctCount++;

            // Track by category
            string category = problem.Category;
            if (!categoryCorrect.ContainsKey(category))
            {
                categoryCorrect[category] = 0;
                categoryTotal[category] = 0;
            }
            categoryTotal[category]++;
            if (isCorrect) categoryCorrect[category]++;

            var evaluation = new ProblemEvaluation<T>
            {
                ProblemId = problem.Id,
                Problem = problem.Problem,
                CorrectAnswer = problem.CorrectAnswer,
                SystemAnswer = systemAnswer,
                IsCorrect = isCorrect,
                Confidence = _numOps.FromDouble(0.7),
                Duration = problemStopwatch.Elapsed,
                Category = category
            };

            problemResults.Add(evaluation);
            confidenceScores.Add(evaluation.Confidence);

            if ((i + 1) % 10 == 0 || i == problems.Count - 1)
            {
                Console.WriteLine($"Progress: {i + 1}/{problems.Count} ({correctCount}/{i + 1} correct)");
            }
        }

        stopwatch.Stop();

        result.CorrectCount = correctCount;
        result.Accuracy = _numOps.FromDouble((double)correctCount / problems.Count);
        result.ConfidenceScores = new Vector<T>(confidenceScores);
        result.AverageConfidence = result.ConfidenceScores.Mean();
        result.TotalDuration = stopwatch.Elapsed;
        result.ProblemResults = problemResults;

        foreach (var category in categoryTotal.Keys)
        {
            double categoryAccuracy = (double)categoryCorrect[category] / categoryTotal[category];
            result.AccuracyByCategory[category] = _numOps.FromDouble(categoryAccuracy);
        }

        result.Metrics["problems_per_second"] = problems.Count / stopwatch.Elapsed.TotalSeconds;

        return result;
    }

    /// <inheritdoc/>
    public async Task<List<BenchmarkProblem>> LoadProblemsAsync(int? count = null)
    {
        if (_cachedProblems == null)
        {
            _cachedProblems = GenerateSampleProblems();
        }

        var problems = _cachedProblems;

        if (count.HasValue && count.Value < problems.Count)
        {
            var random = RandomHelper.CreateSeededRandom(42);
            problems = problems.OrderBy(_ => random.Next()).Take(count.Value).ToList();
        }

        return await Task.FromResult(problems);
    }

    private List<BenchmarkProblem> GenerateSampleProblems()
    {
        return new List<BenchmarkProblem>
        {
            new()
            {
                Id = "math_001",
                Problem = "Find all positive integer values of $n$ for which $\\sqrt{n^2 + 85}$ is an integer.",
                CorrectAnswer = "13",
                Category = "algebra",
                Difficulty = "level_4"
            },
            new()
            {
                Id = "math_002",
                Problem = "A square is inscribed in a circle of radius 5. What is the area of the square?",
                CorrectAnswer = "50",
                Category = "geometry",
                Difficulty = "level_2"
            },
            new()
            {
                Id = "math_003",
                Problem = "How many positive integers less than 100 are both a perfect square and a perfect cube?",
                CorrectAnswer = "1",
                Category = "number_theory",
                Difficulty = "level_3"
            },
            new()
            {
                Id = "math_004",
                Problem = "Solve for x: $2^x = 32$",
                CorrectAnswer = "5",
                Category = "algebra",
                Difficulty = "level_1"
            },
            new()
            {
                Id = "math_005",
                Problem = "What is the sum of the interior angles of a hexagon?",
                CorrectAnswer = "720",
                Category = "geometry",
                Difficulty = "level_2"
            },
            new()
            {
                Id = "math_006",
                Problem = "Find the derivative of $f(x) = x^3 + 2x^2 - 5x + 1$.",
                CorrectAnswer = "3x^2 + 4x - 5",
                Category = "calculus",
                Difficulty = "level_2"
            },
            new()
            {
                Id = "math_007",
                Problem = "What is the smallest positive integer that is divisible by both 12 and 18?",
                CorrectAnswer = "36",
                Category = "number_theory",
                Difficulty = "level_1"
            },
            new()
            {
                Id = "math_008",
                Problem = "If $\\log_2(x) = 5$, what is the value of x?",
                CorrectAnswer = "32",
                Category = "algebra",
                Difficulty = "level_2"
            }
        };
    }

    private string? ExtractMathAnswer(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        // Look for boxed answers (LaTeX format)
        var boxedMatch = Regex.Match(text, @"\\boxed\{([^}]+)\}", RegexOptions.None, RegexTimeout);
        if (boxedMatch.Success)
        {
            return boxedMatch.Groups[1].Value.Trim();
        }

        // Look for final answer indicators
        var finalMatch = Regex.Match(text, @"(?:final answer|answer is|therefore)[:\s]+([^\n.]+)", RegexOptions.IgnoreCase, RegexTimeout);
        if (finalMatch.Success)
        {
            return finalMatch.Groups[1].Value.Trim();
        }

        // Extract last number or expression
        var matches = Regex.Matches(text, @"-?[\d,]+\.?\d*|[a-z]\^?\d*", RegexOptions.None, RegexTimeout);
        if (matches.Count > 0)
        {
            return matches[matches.Count - 1].Value.Trim();  // net462: can't use ^1
        }

        return null;
    }

    private bool CompareMathAnswers(string? answer1, string? answer2)
    {
        if (answer1 == null || answer2 == null)
            return false;

        // Normalize
        answer1 = answer1.Trim().Replace(",", "").ToLowerInvariant();
        answer2 = answer2.Trim().Replace(",", "").ToLowerInvariant();

        // Direct match
        if (answer1 == answer2)
            return true;

        // Try numerical comparison
        if (double.TryParse(answer1, out double val1) && double.TryParse(answer2, out double val2))
        {
            return Math.Abs(val1 - val2) < 0.01;
        }

        return false;
    }
}
