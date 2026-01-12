using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// LogiQA benchmark for evaluating logical reasoning abilities.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> LogiQA tests whether AI can solve logic puzzles similar to
/// those on standardized tests like the LSAT or GRE.
///
/// **What is LogiQA?**
/// LogiQA contains logical reasoning questions from Chinese civil service exams, translated
/// to English. These questions test formal logic, deductive reasoning, and analytical skills.
///
/// **Question types:**
///
/// *1. Categorical reasoning:*
/// ```
/// All programmers are logical.
/// Some logical people are creative.
/// John is a programmer.
///
/// Which must be true?
/// A) John is creative
/// B) John is logical
/// C) All creative people are programmers
/// D) Some programmers are creative
/// Answer: B
/// ```
///
/// *2. Conditional reasoning:*
/// ```
/// If it rains, the picnic will be cancelled.
/// The picnic was not cancelled.
///
/// What can we conclude?
/// A) It rained
/// B) It did not rain
/// C) The picnic happened
/// D) Cannot determine
/// Answer: B (contrapositive: not cancelled → not rain)
/// ```
///
/// *3. Assumption identification:*
/// ```
/// Premise: Companies that provide good customer service are successful.
/// Conclusion: Therefore, our company should invest in customer service.
///
/// What assumption is made?
/// A) Our company wants to be successful
/// B) Customer service is expensive
/// C) Other companies have good service
/// D) Success is important
/// Answer: A
/// ```
///
/// *4. Weaken/Strengthen arguments:*
/// ```
/// Argument: "This medicine works because 80% of patients improved."
///
/// Which weakens this argument?
/// A) The medicine is expensive
/// B) 80% is a high percentage
/// C) 85% of patients improve without medicine
/// D) The medicine has side effects
/// Answer: C (natural improvement rate is higher)
/// ```
///
/// *5. Paradox resolution:*
/// ```
/// Paradox: "Sales of umbrellas decreased, but rainfall increased."
///
/// Which explains this?
/// A) Umbrellas became more expensive
/// B) People stayed indoors more during rain
/// C) Rainfall occurred at night when stores are closed
/// D) Other rain gear became popular
/// Answer: Could be B, C, or D depending on context
/// ```
///
/// **Logic types tested:**
/// - Deductive reasoning (must be true)
/// - Inductive reasoning (likely to be true)
/// - Abductive reasoning (best explanation)
/// - Formal logic (syllogisms, conditionals)
/// - Critical thinking (assumptions, flaws)
///
/// **Performance levels:**
/// - Random guessing: 25%
/// - Humans (average): ~65%
/// - Humans (trained in logic): ~85%
/// - BERT: 34.2%
/// - RoBERTa: 37.1%
/// - GPT-3: 29.8%
/// - GPT-4: 43.5%
/// - Claude 3 Opus: 44.2%
/// - Claude 3.5 Sonnet: 48.0%
/// - ChatGPT o1: 61.2% (significant improvement with CoT)
///
/// **Why it's hard:**
/// - Requires formal logical reasoning
/// - Can't rely on pattern matching
/// - Need to track complex relationships
/// - Must avoid logical fallacies
/// - Tests rigorous thinking, not just knowledge
///
/// **Research:**
/// - "LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning" (Liu et al., 2020)
/// - https://arxiv.org/abs/2007.08124
/// - Dataset: 8,678 questions from Chinese exams
/// - Tests multiple types of logical reasoning
/// </para>
/// </remarks>
public class LogiQABenchmark<T> : IBenchmark<T>
{
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public LogiQABenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "LogiQA";

    /// <inheritdoc/>
    public string Description =>
        "Logical reasoning: 8,678 multiple-choice questions testing formal logic, deductive reasoning, " +
        "and analytical thinking from Chinese civil service exams.";

    /// <inheritdoc/>
    public int TotalProblems => 8678;

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
                Confidence = _numOps.FromDouble(0.6), // Lower confidence for logic
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
        result.Metrics["vs_average_human"] = Convert.ToDouble(result.Accuracy) / 0.65;

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
                Id = "logiqa_1",
                Problem = @"All programmers are logical.
Some logical people are creative.
John is a programmer.

Which must be true?
A) John is creative
B) John is logical
C) All creative people are programmers
D) Some programmers are creative",
                CorrectAnswer = "B",
                Category = "categorical_reasoning",
                Difficulty = "easy"
            },
            new()
            {
                Id = "logiqa_2",
                Problem = @"If it rains, the picnic will be cancelled.
The picnic was not cancelled.

What can we conclude?
A) It rained
B) It did not rain
C) The picnic happened
D) Cannot determine",
                CorrectAnswer = "B",
                Category = "conditional_reasoning",
                Difficulty = "medium"
            },
            new()
            {
                Id = "logiqa_3",
                Problem = @"Premise: Companies that provide good customer service are successful.
Conclusion: Therefore, our company should invest in customer service.

What assumption is made?
A) Our company wants to be successful
B) Customer service is expensive
C) Other companies have good service
D) Success is important",
                CorrectAnswer = "A",
                Category = "assumption_identification",
                Difficulty = "medium"
            },
            new()
            {
                Id = "logiqa_4",
                Problem = @"Argument: """"This medicine works because 80% of patients improved after taking it.""""

Which statement most weakens this argument?
A) The medicine is expensive
B) 80% is a high percentage
C) 85% of patients with this condition improve naturally without medicine
D) The medicine has minor side effects",
                CorrectAnswer = "C",
                Category = "argument_evaluation",
                Difficulty = "hard"
            },
            new()
            {
                Id = "logiqa_5",
                Problem = @"All cats are mammals.
All mammals are animals.

Which must be true?
A) All animals are cats
B) All cats are animals
C) Some animals are not mammals
D) No cats are animals",
                CorrectAnswer = "B",
                Category = "categorical_reasoning",
                Difficulty = "easy"
            },
            new()
            {
                Id = "logiqa_6",
                Problem = @"If John studies hard, he will pass the exam.
John did not pass the exam.

What can we conclude?
A) John studied hard
B) John did not study hard
C) John might have studied hard
D) The exam was too difficult",
                CorrectAnswer = "B",
                Category = "conditional_reasoning",
                Difficulty = "medium"
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
            var match = RegexHelper.Match(text, pattern, RegexOptions.IgnoreCase | RegexOptions.Multiline);
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

        var anyMatch = RegexHelper.Match(text, @"\b([A-D])\b", RegexOptions.IgnoreCase);
        if (anyMatch.Success)
        {
            return anyMatch.Groups[1].Value.ToUpperInvariant();
        }

        return null;
    }
}



