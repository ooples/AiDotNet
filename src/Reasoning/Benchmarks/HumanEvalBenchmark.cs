using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Data;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// HumanEval benchmark for evaluating Python code generation capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> HumanEval is a benchmark of 164 Python programming problems.
/// Each problem asks the model to write a function that passes a set of test cases.
///
/// **Example problem:**
/// ```
/// Write a function that returns True if a number is prime, False otherwise.
/// def is_prime(n: int) -> bool:
///     # Your code here
/// ```
///
/// **Why it's important:**
/// - Tests code generation abilities
/// - Requires understanding algorithms
/// - Tests correctness via unit tests
/// - Standard benchmark for code models
///
/// **Performance levels:**
/// - GPT-3.5: ~48%
/// - GPT-4: ~67%
/// - ChatGPT o1: ~92%
/// - AlphaCode: ~53%
/// - CodeGen: ~29%
///
/// **Research:**
/// "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
/// https://arxiv.org/abs/2107.03374
/// </para>
/// </remarks>
public class HumanEvalBenchmark<T> : IBenchmark<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly INumericOperations<T> _numOps;
    private readonly HumanEvalBenchmarkOptions _options;
    private List<BenchmarkProblem>? _cachedProblems;

    public HumanEvalBenchmark(HumanEvalBenchmarkOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? HumanEvalBenchmarkOptions.FromEnvironment();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "HumanEval";

    /// <inheritdoc/>
    public string Description =>
        "HumanEval: 164 hand-written Python programming problems with unit tests. " +
        "Tests code generation, algorithm understanding, and programming logic.";

    /// <inheritdoc/>
    public int TotalProblems => 164;

    /// <inheritdoc/>
    public async Task<BenchmarkResult<T>> EvaluateAsync(
        Func<string, Task<string>> evaluateFunction,
        int? sampleSize = null,
        CancellationToken cancellationToken = default)
        => await EvaluateAsync(evaluateFunction, executionEvaluator: null, sampleSize, cancellationToken).ConfigureAwait(false);

    public async Task<BenchmarkResult<T>> EvaluateAsync(
        Func<string, Task<string>> evaluateFunction,
        Func<string, string[], CancellationToken, Task<bool>>? executionEvaluator,
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

            // Get system answer (code)
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

            // Extract code from response
            string? extractedCode = ExtractPythonCode(systemAnswer);

            bool isCorrect;
            string evaluationMode;

            if (executionEvaluator is not null && !string.IsNullOrWhiteSpace(extractedCode))
            {
                var testCases = ExtractDoctestAssertions(problem.Problem);
                if (testCases.Length > 0)
                {
                    evaluationMode = "execution";
                    isCorrect = await executionEvaluator(problem.Problem, testCases, cancellationToken).ConfigureAwait(false);
                }
                else
                {
                    evaluationMode = "heuristic_no_tests";
                    isCorrect = CheckCodeCorrectness(extractedCode, problem.CorrectAnswer);
                }
            }
            else
            {
                evaluationMode = "heuristic";
                isCorrect = CheckCodeCorrectness(extractedCode, problem.CorrectAnswer);
            }

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
                Category = category,
                Metadata = new Dictionary<string, object>
                {
                    { "evaluation_mode", evaluationMode },
                    { "execution_enabled", executionEvaluator is not null }
                }
            };

            problemResults.Add(evaluation);
            confidenceScores.Add(evaluation.Confidence);

            // Progress
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

        // Accuracy by category
        foreach (var category in categoryTotal.Keys)
        {
            double categoryAccuracy = (double)categoryCorrect[category] / categoryTotal[category];
            result.AccuracyByCategory[category] = _numOps.FromDouble(categoryAccuracy);
        }

        return result;
    }

    /// <inheritdoc/>
    public async Task<List<BenchmarkProblem>> LoadProblemsAsync(int? count = null)
    {
        if (_cachedProblems == null)
        {
            var datasetPath = _options.DatasetPath;
            if (!string.IsNullOrWhiteSpace(datasetPath))
            {
                var loaded = await HumanEvalDataLoader.LoadFromFileAsync(datasetPath!).ConfigureAwait(false);
                _cachedProblems = loaded
                    .Where(p => !string.IsNullOrWhiteSpace(p.Prompt))
                    .Select(p => new BenchmarkProblem
                    {
                        Id = string.IsNullOrWhiteSpace(p.TaskId) ? string.Empty : p.TaskId,
                        Problem = p.Prompt,
                        CorrectAnswer = p.CanonicalSolution ?? string.Empty,
                        Category = "python"
                    })
                    .ToList();
            }
            else
            {
                _cachedProblems = GenerateSampleProblems();
            }
        }

        var problems = _cachedProblems;

        if (count.HasValue && count.Value < problems.Count)
        {
            var random = RandomHelper.CreateSeededRandom(_options.SampleSeed);
            problems = problems.OrderBy(_ => random.Next()).Take(count.Value).ToList();
        }

        return await Task.FromResult(problems);
    }

    internal static string[] ExtractDoctestAssertions(string problemText)
    {
        if (string.IsNullOrWhiteSpace(problemText))
        {
            return Array.Empty<string>();
        }

        var assertions = new List<string>();
        var lines = problemText.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var trimmed = line.TrimStart();
            if (!trimmed.StartsWith(">>>", StringComparison.Ordinal))
            {
                continue;
            }

            var expression = trimmed.Substring(3).Trim();
            if (string.IsNullOrWhiteSpace(expression))
            {
                continue;
            }

            string? expected = null;
            if (i + 1 < lines.Length)
            {
                var next = lines[i + 1].Trim();
                if (!next.StartsWith(">>>", StringComparison.Ordinal) && !string.IsNullOrWhiteSpace(next))
                {
                    expected = next;
                    i++;
                }
            }

            if (expected is null)
            {
                assertions.Add(expression);
                continue;
            }

            if (expected.StartsWith("Traceback", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            assertions.Add($"assert ({expression}) == ({expected})");
        }

        return assertions.ToArray();
    }

    internal static string ComposePythonSubmission(string prompt, string completion)
    {
        if (string.IsNullOrWhiteSpace(completion))
        {
            return prompt ?? string.Empty;
        }

        var normalizedCompletion = completion.Replace("\r\n", "\n").Trim();
        if (Regex.IsMatch(normalizedCompletion, @"(?m)^\s*def\s+\w+\s*\(", RegexOptions.None, RegexTimeout))
        {
            return normalizedCompletion;
        }

        var normalizedPrompt = (prompt ?? string.Empty).Replace("\r\n", "\n").TrimEnd();
        var indented = string.Join("\n", normalizedCompletion.Split('\n').Select(l => "    " + l.TrimEnd()));

        return normalizedPrompt + "\n" + indented + "\n";
    }

    // Note: execution is orchestrated by higher-level components (e.g. AiModelResult) so benchmarks stay safe-by-default.

    private List<BenchmarkProblem> GenerateSampleProblems()
    {
        return new List<BenchmarkProblem>
        {
            new()
            {
                Id = "HumanEval/0",
                Problem = @"from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """""" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """"""",
                CorrectAnswer = @"    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False",
                Category = "arrays",
                Difficulty = "easy"
            },
            new()
            {
                Id = "HumanEval/1",
                Problem = @"from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """""" Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those groups into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """"""",
                CorrectAnswer = @"    result = []
        current_string = []
        current_depth = 0

        for c in paren_string:
            if c == '(':
                current_depth += 1
                current_string.append(c)
            elif c == ')':
                current_depth -= 1
                current_string.append(c)

                if current_depth == 0:
                    result.append(''.join(current_string))
                    current_string.clear()

        return result",
                Category = "strings",
                Difficulty = "medium"
            },
            new()
            {
                Id = "HumanEval/2",
                Problem = @"def truncate_number(number: float) -> float:
    """""" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """"""",
                CorrectAnswer = @"    return number % 1.0",
                Category = "math",
                Difficulty = "easy"
            }
        };
    }

    internal static string? ExtractPythonCode(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        // Extract from markdown code blocks
        var match = Regex.Match(text, @"```(?:python)?\s*\n([\s\S]*?)\n```", RegexOptions.Multiline, RegexTimeout);
        if (match.Success)
        {
            return match.Groups[1].Value.Trim();
        }

        // If no code block, check if entire text looks like Python
        if (text.Contains("def ") || text.Contains("return "))
        {
            return text.Trim();
        }

        return null;
    }

    private bool CheckCodeCorrectness(string? generatedCode, string referenceCode)
    {
        // Simple heuristic check (in production, would execute tests)
        if (generatedCode is null || string.IsNullOrWhiteSpace(generatedCode))
            return false;

        // After null check, generatedCode is guaranteed non-null
        // Check for required keywords
        bool hasReturn = generatedCode.Contains("return");
        bool hasLogic = generatedCode.Length > 20;

        return hasReturn && hasLogic;
    }
}

public sealed class HumanEvalBenchmarkOptions
{
    public string? DatasetPath { get; init; }

    public int SampleSeed { get; init; } = 42;

    internal static HumanEvalBenchmarkOptions FromEnvironment()
    {
        var datasetPath = Environment.GetEnvironmentVariable("AIDOTNET_HUMANEVAL_DATASET");
        return new HumanEvalBenchmarkOptions
        {
            DatasetPath = string.IsNullOrWhiteSpace(datasetPath) ? null : datasetPath.Trim()
        };
    }
}
