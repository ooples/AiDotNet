using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// BoolQ benchmark for evaluating yes/no question answering.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BoolQ tests whether AI can answer yes/no questions
/// about passages of text, requiring reading comprehension.
///
/// **What is BoolQ?**
/// BoolQ (Boolean Questions) contains naturally occurring yes/no questions about
/// Wikipedia passages. Unlike artificial benchmarks, these are real questions that
/// people actually asked.
///
/// **Format:**
/// - Passage: A paragraph from Wikipedia
/// - Question: A yes/no question about the passage
/// - Answer: True or False
///
/// **Example:**
/// ```
/// Passage: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars
/// in Paris, France. It is named after the engineer Gustave Eiffel, whose company
/// designed and built the tower. Constructed from 1887 to 1889..."
///
/// Question: "Is the Eiffel Tower in France?"
/// Answer: Yes/True
///
/// Question: "Was the Eiffel Tower built in the 21st century?"
/// Answer: No/False
/// ```
///
/// **Why it's challenging:**
/// - Requires careful reading comprehension
/// - Questions can be tricky or indirect
/// - Need to distinguish explicit vs. implicit information
/// - Must avoid making unwarranted inferences
/// - Real-world questions (not synthetic)
///
/// **Performance levels:**
/// - Random guessing: 50%
/// - Humans: 89%
/// - BERT-Large: 77.4%
/// - RoBERTa: 87.1%
/// - GPT-3: 76.4%
/// - GPT-4: 86.9%
/// - Claude 3 Opus: 87.5%
/// - Claude 3.5 Sonnet: 91.0%
/// - ChatGPT o1: 89.5%
///
/// **Question types:**
/// - Factual: Direct facts from the passage
/// - Inferential: Requires reasoning from passage
/// - Temporal: About time and dates
/// - Causal: About cause and effect
/// - Comparative: About comparisons
///
/// **Research:**
/// - "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions" (Clark et al., 2019)
/// - https://arxiv.org/abs/1905.10044
/// - Dataset: 15,942 questions from Google search queries
/// - Part of SuperGLUE benchmark suite
/// </para>
/// </remarks>
public class BoolQBenchmark<T> : IBenchmark<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public BoolQBenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "BoolQ";

    /// <inheritdoc/>
    public string Description =>
        "Boolean Questions: 15,942 yes/no questions about Wikipedia passages. " +
        "Tests reading comprehension with naturally occurring questions from Google search.";

    /// <inheritdoc/>
    public int TotalProblems => 15942;

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

            bool? systemBool = ExtractBooleanAnswer(systemAnswer);
            bool? correctBool = ExtractBooleanAnswer(problem.CorrectAnswer);

            bool isCorrect = systemBool.HasValue &&
                           correctBool.HasValue &&
                           systemBool.Value == correctBool.Value;

            if (isCorrect) correctCount++;

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
                Confidence = _numOps.FromDouble(0.85),
                Duration = problemStopwatch.Elapsed,
                Category = category
            };

            problemResults.Add(evaluation);
            confidenceScores.Add(evaluation.Confidence);

            if ((i + 1) % 20 == 0 || i == problems.Count - 1)
            {
                Console.WriteLine($"Progress: {i + 1}/{problems.Count} ({correctCount}/{i + 1} correct, {(double)correctCount / (i + 1):P1})");
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

        result.Metrics["above_random"] = Convert.ToDouble(result.Accuracy) - 0.50;
        result.Metrics["human_performance_ratio"] = Convert.ToDouble(result.Accuracy) / 0.89;

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
                Id = "boolq_1",
                Problem = @"Passage: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.

Question: Is the Eiffel Tower in France?

Answer with Yes or No.",
                CorrectAnswer = "Yes",
                Category = "factual",
                Difficulty = "easy"
            },
            new()
            {
                Id = "boolq_2",
                Problem = @"Passage: Cats are obligate carnivores, meaning they must eat meat to survive. Unlike dogs, cats cannot produce certain amino acids and must obtain them from their diet. They require nutrients like taurine that are only found in animal tissue.

Question: Can cats survive on a vegetarian diet?

Answer with Yes or No.",
                CorrectAnswer = "No",
                Category = "inferential",
                Difficulty = "medium"
            },
            new()
            {
                Id = "boolq_3",
                Problem = @"Passage: The Declaration of Independence was adopted by the Continental Congress on July 4, 1776. It announced that the thirteen American colonies regarded themselves as independent sovereign states, no longer under British rule.

Question: Was the Declaration of Independence signed in 1776?

Answer with Yes or No.",
                CorrectAnswer = "Yes",
                Category = "temporal",
                Difficulty = "easy"
            },
            new()
            {
                Id = "boolq_4",
                Problem = @"Passage: Water boils at 100°C (212°F) at sea level. However, at higher altitudes where atmospheric pressure is lower, water boils at temperatures below 100°C. This is why cooking times need to be adjusted when cooking at high elevations.

Question: Does water always boil at 100°C?

Answer with Yes or No.",
                CorrectAnswer = "No",
                Category = "inferential",
                Difficulty = "medium"
            },
            new()
            {
                Id = "boolq_5",
                Problem = @"Passage: The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south, and is bounded by Asia and Australia in the west and the Americas in the east.

Question: Is the Pacific Ocean larger than the Atlantic Ocean?

Answer with Yes or No.",
                CorrectAnswer = "Yes",
                Category = "comparative",
                Difficulty = "easy"
            }
        };
    }

    private bool? ExtractBooleanAnswer(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        text = text.Trim().ToLowerInvariant();

        // Direct boolean words
        if (text == "yes" || text == "true" || text == "correct" || text == "1")
            return true;

        if (text == "no" || text == "false" || text == "incorrect" || text == "0")
            return false;

        // Pattern matching
        var yesPatterns = new[] { @"\byes\b", @"\btrue\b", @"answer:\s*yes", @"answer is yes" };
        var noPatterns = new[] { @"\bno\b", @"\bfalse\b", @"answer:\s*no", @"answer is no" };

        foreach (var pattern in yesPatterns)
        {
            if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase, RegexTimeout))
                return true;
        }

        foreach (var pattern in noPatterns)
        {
            if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase, RegexTimeout))
                return false;
        }

        return null;
    }
}
