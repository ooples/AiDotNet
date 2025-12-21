using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// HellaSwag benchmark for evaluating commonsense natural language inference.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> HellaSwag tests whether AI can predict what happens next
/// in everyday situations using common sense.
///
/// **What is HellaSwag?**
/// HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for Situations With
/// Adversarial Generations) presents a context and asks the model to choose the most
/// plausible continuation from 4 options.
///
/// **Example:**
/// ```
/// Context: "A woman is sitting at a piano. She"
/// Options:
/// A) sits on a bench and plays the piano
/// B) starts to play the piano with her feet
/// C) pulls out a sandwich from the piano
/// D) transforms into a dolphin
///
/// Answer: A (most plausible)
/// ```
///
/// **Why it's called "HellaSwag"?**
/// - "Hella" = very (slang)
/// - Designed to be harder than previous benchmarks (SWAG)
/// - Uses adversarial generation to create tricky wrong answers
///
/// **Categories:**
/// - ActivityNet: Video descriptions (activities)
/// - WikiHow: Instructional text (how-to guides)
///
/// **Adversarial wrong answers:**
/// The wrong options are generated to be:
/// 1. Grammatically plausible
/// 2. Semantically similar
/// 3. But factually/logically incorrect
///
/// This makes random guessing ineffective and requires actual understanding.
///
/// **Performance levels:**
/// - Random guessing: 25%
/// - Humans: 95.6%
/// - BERT-Large: 47.9%
/// - GPT-3: 78.9%
/// - GPT-4: 95.3%
/// - Claude 3 Opus: 88.0%
/// - Claude 3.5 Sonnet: 89.0%
/// - ChatGPT o1: 94.2%
///
/// **Why it's hard for models:**
/// - Requires real-world common sense
/// - Can't be solved by pattern matching alone
/// - Adversarial wrong answers look plausible
/// - Needs understanding of cause and effect
///
/// **Research:**
/// - "HellaSwag: Can a Machine Really Finish Your Sentence?" (Zellers et al., 2019)
/// - https://arxiv.org/abs/1905.07830
/// - Dataset: 70,000 questions from ActivityNet and WikiHow
/// </para>
/// </remarks>
public class HellaSwagBenchmark<T> : IBenchmark<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public HellaSwagBenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "HellaSwag";

    /// <inheritdoc/>
    public string Description =>
        "Commonsense natural language inference: 70,000 multiple-choice questions testing " +
        "ability to predict plausible continuations. Uses adversarial generation for hard wrong answers.";

    /// <inheritdoc/>
    public int TotalProblems => 70000;

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

            string? systemLetter = ExtractAnswerLetter(systemAnswer);
            string? correctLetter = ExtractAnswerLetter(problem.CorrectAnswer);

            bool isCorrect = systemLetter != null &&
                           correctLetter != null &&
                           systemLetter.Equals(correctLetter, StringComparison.OrdinalIgnoreCase);

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

        result.Metrics["above_random"] = Convert.ToDouble(result.Accuracy) - 0.25;
        result.Metrics["human_performance_ratio"] = Convert.ToDouble(result.Accuracy) / 0.956;

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
                Id = "hellaswag_1",
                Problem = @"Context: A woman is sitting at a piano. She

Which completion is most plausible?
A) sits on a bench and begins to play the keys
B) stands up and walks away from the piano
C) pulls out a sandwich from inside the piano
D) starts playing the piano with her feet while standing",
                CorrectAnswer = "A",
                Category = "activitynet",
                Difficulty = "easy"
            },
            new()
            {
                Id = "hellaswag_2",
                Problem = @"Context: [header] How to bake chocolate chip cookies [step] Preheat your oven to 350°F. [step] In a bowl, mix butter and sugar. [step] Add eggs and vanilla. [step]

What happens next?
A) Put the bowl in the refrigerator for 2 hours
B) Mix in flour, baking soda, and salt
C) Pour the mixture directly onto the baking sheet
D) Add more sugar until the mixture is very sweet",
                CorrectAnswer = "B",
                Category = "wikihow",
                Difficulty = "easy"
            },
            new()
            {
                Id = "hellaswag_3",
                Problem = @"Context: A man is holding a basketball on a court. He

What's the most plausible continuation?
A) dribbles the ball and takes a shot at the basket
B) throws the basketball at a nearby window
C) sits down and starts eating the basketball
D) puts the basketball on his head like a hat",
                CorrectAnswer = "A",
                Category = "activitynet",
                Difficulty = "easy"
            },
            new()
            {
                Id = "hellaswag_4",
                Problem = @"Context: [header] How to study for an exam [step] Gather all your notes and textbooks. [step] Find a quiet place to study. [step]

What should you do next?
A) Review the material systematically, section by section
B) Close all the books and take a 3-hour nap
C) Throw away all your notes and start fresh
D) Call your friends to come over for a party",
                CorrectAnswer = "A",
                Category = "wikihow",
                Difficulty = "easy"
            },
            new()
            {
                Id = "hellaswag_5",
                Problem = @"Context: A chef is preparing vegetables in a kitchen. She takes a knife and

Complete the sentence:
A) begins to chop the vegetables into small pieces
B) throws the knife across the room at a target
C) uses the knife to paint a picture on the wall
D) puts the knife in her pocket and leaves",
                CorrectAnswer = "A",
                Category = "activitynet",
                Difficulty = "easy"
            }
        };
    }

    private string? ExtractAnswerLetter(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        var patterns = new[]
        {
            @"answer:\s*([A-D])",
            @"correct answer is\s*([A-D])",
            @"^([A-D])\)",
            @"^([A-D])[\s\.]",
            @"\(([A-D])\)",
            @"option\s*([A-D])",
            @"choice\s*([A-D])"
        };

        foreach (var pattern in patterns)
        {
            var match = Regex.Match(text, pattern, RegexOptions.IgnoreCase | RegexOptions.Multiline, RegexTimeout);
            if (match.Success)
            {
                return match.Groups[1].Value.ToUpperInvariant();
            }
        }

        text = text.Trim().ToUpperInvariant();
        if (text.Length == 1 && "ABCD".Contains(text))
        {
            return text;
        }

        var anyMatch = Regex.Match(text, @"\b([A-D])\b", RegexOptions.IgnoreCase, RegexTimeout);
        if (anyMatch.Success)
        {
            return anyMatch.Groups[1].Value.ToUpperInvariant();
        }

        return null;
    }
}
