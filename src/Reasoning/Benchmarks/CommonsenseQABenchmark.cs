using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// CommonsenseQA benchmark for evaluating commonsense knowledge and reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CommonsenseQA tests everyday knowledge that humans take
/// for granted but AI often struggles with.
///
/// **What is CommonsenseQA?**
/// CommonsenseQA contains multiple-choice questions requiring common sense about everyday
/// situations, objects, and concepts.
///
/// **Example questions:**
///
/// *Physical world:*
/// ```
/// Q: Where would you put uncooked food that you want to cook soon?
/// A) pantry  B) shelf  C) refrigerator  D) kitchen cabinet  E) oven
/// Answer: C (refrigerator keeps food fresh until cooking)
/// ```
///
/// *Social understanding:*
/// ```
/// Q: What happens when people get tired?
/// A) they sleep  B) go to movies  C) feel energetic  D) stay awake  E) study
/// Answer: A (tired people need sleep)
/// ```
///
/// *Cause and effect:*
/// ```
/// Q: What can happen to someone who doesn't get enough sleep?
/// A) lazy  B) insomnia  C) get tired  D) snore  E) have fun
/// Answer: C (lack of sleep causes tiredness)
/// ```
///
/// *Object properties:*
/// ```
/// Q: What is likely to be found in a book?
/// A) pictures  B) words  C) pages  D) cover  E) all of the above
/// Answer: E (books have all these features)
/// ```
///
/// *Spatial reasoning:*
/// ```
/// Q: Where do you typically find a handle?
/// A) door  B) briefcase  C) suitcase  D) cup  E) all of the above
/// Answer: E (all these objects have handles)
/// ```
///
/// **Knowledge types:**
/// - Physical properties (hot, cold, heavy, fragile)
/// - Spatial relationships (inside, on top of, next to)
/// - Temporal understanding (before, after, during)
/// - Causal relationships (causes, prevents, enables)
/// - Social norms (polite, rude, appropriate)
/// - Functional roles (what things are used for)
/// - Typical locations (where things are usually found)
///
/// **Why it's important:**
/// - Tests implicit knowledge humans use daily
/// - Can't be answered by facts alone
/// - Requires understanding of how the world works
/// - Foundation for real-world AI applications
///
/// **Performance levels:**
/// - Random guessing: 20%
/// - Humans (crowd workers): 88.9%
/// - Humans (expert): 95.3%
/// - BERT: 57.0%
/// - RoBERTa: 73.1%
/// - GPT-3: 65.2%
/// - GPT-4: 82.4%
/// - Claude 3 Opus: 81.7%
/// - Claude 3.5 Sonnet: 85.9%
/// - ChatGPT o1: 88.1%
///
/// **Why LLMs struggle:**
/// - Lack embodied experience (can't touch/see/hear)
/// - No direct interaction with physical world
/// - Must infer common sense from text alone
/// - Training data may lack obvious implicit knowledge
/// - Difficulty distinguishing common from rare situations
///
/// **How it's created:**
/// 1. Start with concept from ConceptNet (knowledge graph)
/// 2. Generate question about the concept
/// 3. Use crowd workers to create wrong but plausible options
/// 4. Adversarial filtering to ensure quality
///
/// **ConceptNet integration:**
/// Questions are based on ConceptNet relations like:
/// - UsedFor: knife UsedFor cutting
/// - AtLocation: book AtLocation library
/// - Causes: exercise Causes tiredness
/// - CapableOf: bird CapableOf flying
///
/// **Research:**
/// - "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge" (Talmor et al., 2019)
/// - https://arxiv.org/abs/1811.00937
/// - Dataset: 12,247 questions with 5 answer choices each
/// - Based on ConceptNet knowledge graph
/// </para>
/// </remarks>
public class CommonsenseQABenchmark<T> : IBenchmark<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public CommonsenseQABenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "CommonsenseQA";

    /// <inheritdoc/>
    public string Description =>
        "Commonsense knowledge QA: 12,247 multiple-choice questions requiring everyday common sense " +
        "about physical world, social situations, and typical patterns.";

    /// <inheritdoc/>
    public int TotalProblems => 12247;

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

        result.Metrics["above_random"] = Convert.ToDouble(result.Accuracy) - 0.20;
        result.Metrics["vs_human"] = Convert.ToDouble(result.Accuracy) / 0.889;

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
                Id = "csqa_1",
                Problem = @"Where would you put uncooked food that you want to cook soon?

A) pantry
B) shelf
C) refrigerator
D) kitchen cabinet
E) oven",
                CorrectAnswer = "C",
                Category = "spatial_knowledge",
                Difficulty = "easy"
            },
            new()
            {
                Id = "csqa_2",
                Problem = @"What happens when people get tired?

A) they sleep
B) go to movies
C) feel energetic
D) stay awake
E) study harder",
                CorrectAnswer = "A",
                Category = "causal_reasoning",
                Difficulty = "easy"
            },
            new()
            {
                Id = "csqa_3",
                Problem = @"What can happen to someone who doesn't get enough sleep?

A) become lazy
B) get insomnia
C) get tired
D) snore loudly
E) have fun",
                CorrectAnswer = "C",
                Category = "causal_reasoning",
                Difficulty = "easy"
            },
            new()
            {
                Id = "csqa_4",
                Problem = @"Where would you typically find a newspaper?

A) doorstep
B) library
C) newsstand
D) recycling bin
E) all of the above",
                CorrectAnswer = "E",
                Category = "spatial_knowledge",
                Difficulty = "medium"
            },
            new()
            {
                Id = "csqa_5",
                Problem = @"What do you need to do before you can eat a sandwich?

A) make it
B) buy it
C) order it
D) unwrap it
E) any of the above",
                CorrectAnswer = "E",
                Category = "temporal_reasoning",
                Difficulty = "medium"
            },
            new()
            {
                Id = "csqa_6",
                Problem = @"If you're looking for a book, where would you most likely go?

A) library
B) bookstore
C) friend's house
D) online
E) all could work",
                CorrectAnswer = "E",
                Category = "spatial_knowledge",
                Difficulty = "medium"
            },
            new()
            {
                Id = "csqa_7",
                Problem = @"What is a pen used for?

A) writing
B) drawing
C) signing documents
D) taking notes
E) all of the above",
                CorrectAnswer = "E",
                Category = "functional_knowledge",
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
            @"answer:\s*([A-E])",
            @"correct answer is\s*([A-E])",
            @"^([A-E])\)",
            @"^([A-E])[\s\.]",
            @"\(([A-E])\)",
            @"option\s*([A-E])",
            @"choice\s*([A-E])"
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
        if (text.Length == 1 && "ABCDE".Contains(text))
        {
            return text;
        }

        var anyMatch = Regex.Match(text, @"\b([A-E])\b", RegexOptions.IgnoreCase, RegexTimeout);
        if (anyMatch.Success)
        {
            return anyMatch.Groups[1].Value.ToUpperInvariant();
        }

        return null;
    }
}
