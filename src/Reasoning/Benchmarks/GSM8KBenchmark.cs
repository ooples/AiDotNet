using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// Grade School Math 8K (GSM8K) benchmark for evaluating mathematical reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> GSM8K is a dataset of 8,500 grade school math word problems.
/// These are the kinds of problems you'd see in elementary school math:
///
/// **Example problems:**
/// - "Janet has 15 apples. She gives 40% to her friend. How many does she have left?"
/// - "A train travels 60 mph for 2.5 hours. How far does it go?"
/// - "If 3 pizzas cost $45, how much does 1 pizza cost?"
///
/// **Why it's important:**
/// - Tests basic mathematical reasoning
/// - Requires understanding word problems
/// - Needs step-by-step calculation
/// - Benchmark for many reasoning models
///
/// **Performance levels:**
/// - Human performance: ~90-95%
/// - GPT-3.5: ~57%
/// - GPT-4: ~92%
/// - ChatGPT o1: ~95%
/// - DeepSeek-R1: ~97%
///
/// **Research:**
/// "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
/// https://arxiv.org/abs/2110.14168
/// </para>
/// </remarks>
public class GSM8KBenchmark<T> : IBenchmark<T>
{
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    /// <summary>
    /// Initializes a new instance of the <see cref="GSM8KBenchmark{T}"/> class.
    /// </summary>
    public GSM8KBenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "GSM8K";

    /// <inheritdoc/>
    public string Description =>
        "Grade School Math 8K: 8,500 grade school math word problems requiring multi-step reasoning. " +
        "Tests mathematical reasoning, calculation accuracy, and word problem comprehension.";

    /// <inheritdoc/>
    public int TotalProblems => 10; // Sample set size (full GSM8K test set: 1319)

    /// <inheritdoc/>
    public async Task<BenchmarkResult<T>> EvaluateAsync(
        Func<string, Task<string>> evaluateFunction,
        int? sampleSize = null,
        CancellationToken cancellationToken = default)
    {
        if (evaluateFunction == null)
            throw new ArgumentNullException(nameof(evaluateFunction));

        var stopwatch = Stopwatch.StartNew();

        // Load problems
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

        // Evaluate each problem
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

            // Get system answer
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

            // Extract numerical answer
            string? extractedAnswer = ExtractNumericalAnswer(systemAnswer);
            string? correctNumerical = ExtractNumericalAnswer(problem.CorrectAnswer);

            // Check if correct
            bool isCorrect = CompareAnswers(extractedAnswer, correctNumerical);

            if (isCorrect)
            {
                correctCount++;
            }

            // Track by category
            string category = problem.Category;
            if (!categoryCorrect.ContainsKey(category))
            {
                categoryCorrect[category] = 0;
                categoryTotal[category] = 0;
            }

            categoryTotal[category]++;
            if (isCorrect)
            {
                categoryCorrect[category]++;
            }

            // Record result
            var evaluation = new ProblemEvaluation<T>
            {
                ProblemId = problem.Id,
                Problem = problem.Problem,
                CorrectAnswer = problem.CorrectAnswer,
                SystemAnswer = systemAnswer,
                IsCorrect = isCorrect,
                Confidence = _numOps.FromDouble(0.8), // Default confidence (could be extracted from reasoning)
                Duration = problemStopwatch.Elapsed,
                Category = category
            };

            problemResults.Add(evaluation);
            confidenceScores.Add(evaluation.Confidence);

            // Progress indicator
            if ((i + 1) % 10 == 0 || i == problems.Count - 1)
            {
                Console.WriteLine($"Progress: {i + 1}/{problems.Count} ({correctCount}/{i + 1} correct)");
            }
        }

        stopwatch.Stop();

        // Calculate results
        result.CorrectCount = correctCount;
        result.Accuracy = _numOps.FromDouble((double)correctCount / problems.Count);
        result.ConfidenceScores = new Vector<T>(confidenceScores);
        result.AverageConfidence = result.ConfidenceScores.Mean();
        result.TotalDuration = stopwatch.Elapsed;
        result.ProblemResults = problemResults;

        // Calculate accuracy by category
        foreach (var category in categoryTotal.Keys)
        {
            double categoryAccuracy = (double)categoryCorrect[category] / categoryTotal[category];
            result.AccuracyByCategory[category] = _numOps.FromDouble(categoryAccuracy);
        }

        // Additional metrics
        result.Metrics["problems_per_second"] = problems.Count / stopwatch.Elapsed.TotalSeconds;
        result.Metrics["median_time_seconds"] = problemResults.Select(p => p.Duration.TotalSeconds).OrderBy(t => t).ElementAt(problemResults.Count / 2);

        return result;
    }

    /// <inheritdoc/>
    public async Task<List<BenchmarkProblem>> LoadProblemsAsync(int? count = null)
    {
        // Load or generate sample GSM8K problems
        if (_cachedProblems == null)
        {
            _cachedProblems = GenerateSampleProblems();
        }

        var problems = _cachedProblems;

        if (count.HasValue && count.Value < problems.Count)
        {
            // Random sample
            var random = RandomHelper.CreateSeededRandom(42); // Deterministic seed
            problems = problems.OrderBy(_ => random.Next()).Take(count.Value).ToList();
        }

        return await Task.FromResult(problems);
    }

    /// <summary>
    /// Generates sample GSM8K-style problems for demonstration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In production, you would load actual GSM8K data from a file or API.
    /// For this implementation, we generate sample problems that match the GSM8K format.
    /// </para>
    /// </remarks>
    private List<BenchmarkProblem> GenerateSampleProblems()
    {
        // Sample problems representing GSM8K types
        var problems = new List<BenchmarkProblem>
        {
            new()
            {
                Id = "gsm8k_001",
                Problem = "Janet has 16 eggs. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                CorrectAnswer = "18",
                Category = "arithmetic",
                Difficulty = "medium"
            },
            new()
            {
                Id = "gsm8k_002",
                Problem = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                CorrectAnswer = "3",
                Category = "arithmetic",
                Difficulty = "easy"
            },
            new()
            {
                Id = "gsm8k_003",
                Problem = "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                CorrectAnswer = "70000",
                Category = "percentage",
                Difficulty = "medium"
            },
            new()
            {
                Id = "gsm8k_004",
                Problem = "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
                CorrectAnswer = "540",
                Category = "multiplication",
                Difficulty = "easy"
            },
            new()
            {
                Id = "gsm8k_005",
                Problem = "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. If Wendi has 20 chickens, how many cups of feed does she need in total per day?",
                CorrectAnswer = "60",
                Category = "multiplication",
                Difficulty = "easy"
            },
            new()
            {
                Id = "gsm8k_006",
                Problem = "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
                CorrectAnswer = "64",
                Category = "percentage",
                Difficulty = "hard"
            },
            new()
            {
                Id = "gsm8k_007",
                Problem = "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
                CorrectAnswer = "260",
                Category = "ratios",
                Difficulty = "medium"
            },
            new()
            {
                Id = "gsm8k_008",
                Problem = "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
                CorrectAnswer = "160",
                Category = "time",
                Difficulty = "hard"
            },
            new()
            {
                Id = "gsm8k_009",
                Problem = "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. How fast does he have to drive for the remaining 2 hours to get home on time?",
                CorrectAnswer = "90",
                Category = "speed",
                Difficulty = "hard"
            },
            new()
            {
                Id = "gsm8k_010",
                Problem = "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
                CorrectAnswer = "460",
                Category = "overtime",
                Difficulty = "medium"
            }
        };

        return problems;
    }

    /// <summary>
    /// Extracts numerical answer from text.
    /// </summary>
    private string? ExtractNumericalAnswer(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        // Look for numbers, possibly with commas or decimals
        var matches = RegexHelper.Matches(text, @"-?[\d,]+\.?\d*", RegexOptions.None);

        if (matches.Count == 0)
            return null;

        // Take the last number (usually the final answer)
        string number = matches[matches.Count - 1].Value.Replace(",", "");  // net462: can't use ^1

        return number;
    }

    /// <summary>
    /// Compares two numerical answers with tolerance.
    /// </summary>
    private bool CompareAnswers(string? answer1, string? answer2)
    {
        if (answer1 == null || answer2 == null)
            return false;

        if (!double.TryParse(answer1, out double val1))
            return false;

        if (!double.TryParse(answer2, out double val2))
            return false;

        // Allow small floating point differences
        return Math.Abs(val1 - val2) < 0.01;
    }
}



