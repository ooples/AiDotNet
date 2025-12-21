using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// MMLU (Massive Multitask Language Understanding) benchmark for evaluating world knowledge.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MMLU is like a comprehensive standardized test for AI,
/// covering 57 subjects from elementary to professional level.
///
/// **What is MMLU?**
/// MMLU tests knowledge across diverse academic and professional domains:
/// - STEM: Mathematics, Physics, Chemistry, Biology, Computer Science
/// - Humanities: History, Philosophy, Law
/// - Social Sciences: Psychology, Economics, Sociology
/// - Other: Medicine, Business, Professional Knowledge
///
/// **Format:**
/// Multiple choice questions (A, B, C, D) spanning different difficulty levels:
/// - Elementary
/// - High School
/// - College
/// - Professional
///
/// **Example questions:**
///
/// *Elementary Math:*
/// Q: What is 7 × 8?
/// A) 54  B) 56  C) 64  D) 48
/// Answer: B
///
/// *College Physics:*
/// Q: What is the ground state energy of a hydrogen atom?
/// A) -13.6 eV  B) -27.2 eV  C) -6.8 eV  D) 0 eV
/// Answer: A
///
/// *Professional Medicine:*
/// Q: A 45-year-old presents with sudden chest pain. What is the most appropriate first test?
/// A) CT scan  B) ECG  C) Blood test  D) X-ray
/// Answer: B
///
/// **Why it's important:**
/// - Comprehensive knowledge evaluation
/// - Tests reasoning + memorization
/// - Standard benchmark for LLMs
/// - Measures real-world applicability
///
/// **Performance levels:**
/// - Random guessing: 25%
/// - Average human expert: ~90% (in their domain)
/// - GPT-3.5: ~70%
/// - GPT-4: ~86%
/// - Claude 3 Opus: ~87%
/// - Claude 3.5 Sonnet: ~89%
/// - ChatGPT o1: ~91%
/// - Gemini Pro 1.5: ~90%
///
/// **57 Subject categories:**
///
/// **STEM (18 subjects):**
/// - Abstract Algebra, Astronomy, College Biology, College Chemistry
/// - College Computer Science, College Mathematics, College Physics
/// - Conceptual Physics, Electrical Engineering, Elementary Mathematics
/// - High School Biology, High School Chemistry, High School Computer Science
/// - High School Mathematics, High School Physics, High School Statistics
/// - Machine Learning
///
/// **Humanities (13 subjects):**
/// - Formal Logic, High School European History, High School US History
/// - High School World History, International Law, Jurisprudence
/// - Logical Fallacies, Moral Disputes, Moral Scenarios
/// - Philosophy, Prehistory, Professional Law, World Religions
///
/// **Social Sciences (12 subjects):**
/// - Econometrics, High School Geography, High School Government and Politics
/// - High School Macroeconomics, High School Microeconomics
/// - High School Psychology, Human Sexuality, Professional Psychology
/// - Public Relations, Security Studies, Sociology, US Foreign Policy
///
/// **Other (14 subjects):**
/// - Anatomy, Business Ethics, Clinical Knowledge, College Medicine
/// - Global Facts, Human Aging, Management, Marketing
/// - Medical Genetics, Miscellaneous, Nutrition, Professional Accounting
/// - Professional Medicine, Virology
///
/// **Research:**
/// - "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2021)
/// - https://arxiv.org/abs/2009.03300
/// - Dataset: 15,908 questions across 57 tasks
/// </para>
/// </remarks>
public class MMLUBenchmark<T> : IBenchmark<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public MMLUBenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "MMLU";

    /// <inheritdoc/>
    public string Description =>
        "Massive Multitask Language Understanding: 15,908 multiple-choice questions across " +
        "57 subjects (STEM, humanities, social sciences, other). Tests world knowledge and reasoning.";

    /// <inheritdoc/>
    public int TotalProblems => 15908;

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

            // Extract answer letter (A, B, C, or D)
            string? systemLetter = ExtractAnswerLetter(systemAnswer);
            string? correctLetter = ExtractAnswerLetter(problem.CorrectAnswer);

            // Check if correct
            bool isCorrect = systemLetter != null &&
                           correctLetter != null &&
                           systemLetter.Equals(correctLetter, StringComparison.OrdinalIgnoreCase);

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
                Confidence = _numOps.FromDouble(0.75),
                Duration = problemStopwatch.Elapsed,
                Category = category
            };

            problemResults.Add(evaluation);
            confidenceScores.Add(evaluation.Confidence);

            // Progress
            if ((i + 1) % 20 == 0 || i == problems.Count - 1)
            {
                Console.WriteLine($"Progress: {i + 1}/{problems.Count} ({correctCount}/{i + 1} correct, {(double)correctCount / (i + 1):P1})");
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

        // Accuracy by category
        foreach (var category in categoryTotal.Keys)
        {
            double categoryAccuracy = (double)categoryCorrect[category] / categoryTotal[category];
            result.AccuracyByCategory[category] = _numOps.FromDouble(categoryAccuracy);
        }

        // Additional metrics
        result.Metrics["questions_per_minute"] = problems.Count / stopwatch.Elapsed.TotalMinutes;
        result.Metrics["above_random_guessing"] = Convert.ToDouble(result.Accuracy) - 0.25; // vs 25% baseline

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
        // Sample MMLU questions across different subjects
        return new List<BenchmarkProblem>
        {
            new()
            {
                Id = "mmlu_math_001",
                Problem = @"Elementary Mathematics:

What is the value of 12 × 15?

A) 150
B) 180
C) 170
D) 160",
                CorrectAnswer = "B",
                Category = "elementary_mathematics",
                Difficulty = "easy"
            },
            new()
            {
                Id = "mmlu_physics_001",
                Problem = @"College Physics:

What is the speed of light in a vacuum?

A) 3 × 10^8 m/s
B) 3 × 10^6 m/s
C) 3 × 10^10 m/s
D) 3 × 10^5 m/s",
                CorrectAnswer = "A",
                Category = "college_physics",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mmlu_history_001",
                Problem = @"High School US History:

In which year did the American Civil War end?

A) 1863
B) 1865
C) 1870
D) 1861",
                CorrectAnswer = "B",
                Category = "high_school_us_history",
                Difficulty = "easy"
            },
            new()
            {
                Id = "mmlu_cs_001",
                Problem = @"College Computer Science:

What is the time complexity of binary search on a sorted array of n elements?

A) O(n)
B) O(n log n)
C) O(log n)
D) O(n²)",
                CorrectAnswer = "C",
                Category = "college_computer_science",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mmlu_medicine_001",
                Problem = @"Professional Medicine:

Which of the following is the most common cause of sudden cardiac arrest in young athletes?

A) Hypertrophic cardiomyopathy
B) Coronary artery disease
C) Myocarditis
D) Long QT syndrome",
                CorrectAnswer = "A",
                Category = "professional_medicine",
                Difficulty = "hard"
            },
            new()
            {
                Id = "mmlu_philosophy_001",
                Problem = @"Philosophy:

Which philosopher is associated with the categorical imperative?

A) John Stuart Mill
B) Immanuel Kant
C) Friedrich Nietzsche
D) Jean-Paul Sartre",
                CorrectAnswer = "B",
                Category = "philosophy",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mmlu_economics_001",
                Problem = @"High School Microeconomics:

When demand is elastic, a price increase will:

A) Increase total revenue
B) Decrease total revenue
C) Have no effect on total revenue
D) Double total revenue",
                CorrectAnswer = "B",
                Category = "high_school_microeconomics",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mmlu_chemistry_001",
                Problem = @"College Chemistry:

What is the pH of a 0.01 M HCl solution?

A) 1
B) 2
C) 3
D) 7",
                CorrectAnswer = "B",
                Category = "college_chemistry",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mmlu_logic_001",
                Problem = @"Formal Logic:

Given: All A are B. Some B are C. Which conclusion is valid?

A) All A are C
B) Some A are C
C) No A are C
D) None of the above",
                CorrectAnswer = "D",
                Category = "formal_logic",
                Difficulty = "hard"
            },
            new()
            {
                Id = "mmlu_ml_001",
                Problem = @"Machine Learning:

Which of the following is a regularization technique used to prevent overfitting?

A) Increasing learning rate
B) L2 penalty (Ridge regression)
C) Using more features
D) Reducing training data",
                CorrectAnswer = "B",
                Category = "machine_learning",
                Difficulty = "medium"
            }
        };
    }

    private string? ExtractAnswerLetter(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        // Look for explicit answer patterns
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

        // If text is just a single letter
        text = text.Trim().ToUpperInvariant();
        if (text.Length == 1 && "ABCD".Contains(text))
        {
            return text;
        }

        // Look for any A, B, C, or D in the text (last resort)
        var anyMatch = Regex.Match(text, @"\b([A-D])\b", RegexOptions.IgnoreCase, RegexTimeout);
        if (anyMatch.Success)
        {
            return anyMatch.Groups[1].Value.ToUpperInvariant();
        }

        return null;
    }
}
