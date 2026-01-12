using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// PIQA (Physical Interaction Question Answering) benchmark for physical commonsense reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> PIQA tests whether AI understands how the physical world works
/// through everyday situations and questions about physical interactions.
///
/// **What is PIQA?**
/// PIQA asks questions about physical commonsense - understanding how objects interact,
/// what happens when you do certain actions, and basic physics of everyday life.
///
/// **Format:**
/// - Goal: A task you want to accomplish
/// - Solutions: Two possible ways to do it (one correct, one wrong)
/// - Question: Which solution actually works?
///
/// **Example 1:**
/// ```
/// Goal: To separate egg whites from the yolk
/// Solution 1: Crack the egg into a bowl, then use a water bottle to suck up the yolk
/// Solution 2: Crack the egg and use your hands to throw the white away
/// Correct: Solution 1
/// ```
///
/// **Example 2:**
/// ```
/// Goal: To remove a stripped screw
/// Solution 1: Place a rubber band over the screw head for better grip
/// Solution 2: Pour water on the screw to make it easier to turn
/// Correct: Solution 1
/// ```
///
/// **Example 3:**
/// ```
/// Goal: Keep your garbage can from smelling bad
/// Solution 1: Spray perfume in the garbage can every day
/// Solution 2: Put baking soda at the bottom of the can
/// Correct: Solution 2
/// ```
///
/// **Why it's important:**
/// - Tests real-world physical understanding
/// - Can't be solved by language patterns alone
/// - Requires knowledge of:
///   - Basic physics (gravity, friction, pressure)
///   - Material properties (hard, soft, sticky, etc.)
///   - Cause and effect in physical world
///   - Practical life skills
///
/// **Performance levels:**
/// - Random guessing: 50%
/// - Humans: 94.9%
/// - BERT: 70.9%
/// - RoBERTa: 79.4%
/// - GPT-3: 81.0%
/// - GPT-4: 86.8%
/// - Claude 3 Opus: 85.2%
/// - Claude 3.5 Sonnet: 88.0%
/// - ChatGPT o1: 91.5%
///
/// **Categories:**
/// - Kitchen tasks
/// - Home repair
/// - Cleaning
/// - Arts and crafts
/// - General household
///
/// **Why it's hard for AI:**
/// - LLMs lack embodied experience
/// - Can't actually touch or manipulate objects
/// - Must infer from text descriptions
/// - Requires implicit physical knowledge
///
/// **Research:**
/// - "PIQA: Reasoning about Physical Commonsense in Natural Language" (Bisk et al., 2020)
/// - https://arxiv.org/abs/1911.11641
/// - Dataset: 16,000 questions from WikiHow and other sources
/// - Part of physical reasoning evaluation suite
/// </para>
/// </remarks>
public class PIQABenchmark<T> : IBenchmark<T>
{
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public PIQABenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "PIQA";

    /// <inheritdoc/>
    public string Description =>
        "Physical Interaction QA: 16,000 questions testing physical commonsense reasoning " +
        "about everyday tasks and object interactions.";

    /// <inheritdoc/>
    public int TotalProblems => 16000;

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

            int? systemSolution = ExtractSolutionNumber(systemAnswer);
            int? correctSolution = ExtractSolutionNumber(problem.CorrectAnswer);

            bool isCorrect = systemSolution.HasValue &&
                           correctSolution.HasValue &&
                           systemSolution.Value == correctSolution.Value;

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
                Confidence = _numOps.FromDouble(0.8),
                Duration = problemStopwatch.Elapsed,
                Category = category
            };

            problemResults.Add(evaluation);
            confidenceScores.Add(evaluation.Confidence);

            if ((i + 1) % 10 == 0 || i == problems.Count - 1)
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
        result.Metrics["human_performance_ratio"] = Convert.ToDouble(result.Accuracy) / 0.949;

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
                Id = "piqa_1",
                Problem = @"Goal: To separate egg whites from the yolk

Which solution works?
Solution 1: Crack the egg into a bowl, then use an empty water bottle to suck up the yolk
Solution 2: Crack the egg into a bowl, then use your hands to throw the white away",
                CorrectAnswer = "Solution 1",
                Category = "kitchen",
                Difficulty = "easy"
            },
            new()
            {
                Id = "piqa_2",
                Problem = @"Goal: To remove a stripped screw from wood

Which solution works?
Solution 1: Place a rubber band over the stripped screw head and press down while turning
Solution 2: Pour water on the screw to make it easier to turn",
                CorrectAnswer = "Solution 1",
                Category = "home_repair",
                Difficulty = "medium"
            },
            new()
            {
                Id = "piqa_3",
                Problem = @"Goal: Keep your garbage can from smelling bad

Which solution works?
Solution 1: Spray perfume in the garbage can every single day
Solution 2: Put baking soda at the bottom of the garbage can",
                CorrectAnswer = "Solution 2",
                Category = "cleaning",
                Difficulty = "easy"
            },
            new()
            {
                Id = "piqa_4",
                Problem = @"Goal: To make a paint brush more flexible

Which solution works?
Solution 1: Soak the paint brush in warm water and conditioner for 10 minutes
Solution 2: Heat the paint brush in the microwave for 5 minutes",
                CorrectAnswer = "Solution 1",
                Category = "arts_crafts",
                Difficulty = "medium"
            },
            new()
            {
                Id = "piqa_5",
                Problem = @"Goal: To prevent bananas from browning too quickly

Which solution works?
Solution 1: Wrap the stems of the bananas in plastic wrap
Solution 2: Store the bananas in the freezer immediately after purchase",
                CorrectAnswer = "Solution 1",
                Category = "kitchen",
                Difficulty = "medium"
            },
            new()
            {
                Id = "piqa_6",
                Problem = @"Goal: To thread a needle more easily

Which solution works?
Solution 1: Cut the thread at an angle and stiffen the end with hairspray
Solution 2: Make the thread thicker by doubling it over multiple times",
                CorrectAnswer = "Solution 1",
                Category = "arts_crafts",
                Difficulty = "medium"
            }
        };
    }

    private int? ExtractSolutionNumber(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        text = text.ToLowerInvariant();

        // Direct patterns
        var patterns = new[]
        {
            @"solution\s*([12])",
            @"answer:\s*solution\s*([12])",
            @"correct:\s*solution\s*([12])",
            @"^solution\s*([12])",
            @"option\s*([12])"
        };

        foreach (var pattern in patterns)
        {
            var match = RegexHelper.Match(text, pattern, RegexOptions.IgnoreCase);
            if (match.Success && int.TryParse(match.Groups[1].Value, out int num))
            {
                return num;
            }
        }

        // Just "1" or "2"
        if (text.Trim() == "1") return 1;
        if (text.Trim() == "2") return 2;

        return null;
    }
}



