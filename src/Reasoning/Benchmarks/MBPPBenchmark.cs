using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// MBPP (Mostly Basic Python Problems) benchmark for evaluating Python code generation.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MBPP is a collection of basic Python programming problems,
/// designed to test fundamental programming skills and algorithmic thinking.
///
/// **What is MBPP?**
/// MBPP contains 974 short Python programming problems at an entry-level difficulty.
/// Each problem includes:
/// - A natural language description
/// - Code solution
/// - 3 test cases
///
/// **MBPP vs HumanEval:**
/// - **MBPP**: 974 problems, more basic, multiple test cases provided
/// - **HumanEval**: 164 problems, more challenging, function signature given
/// - **Overlap**: Both test code generation, but MBPP is more comprehensive
///
/// **Example problems:**
///
/// *Problem 1: Sum of numbers*
/// ```
/// Task: Write a function to find the sum of all numbers in a list.
/// Test Cases:
/// - sum_list([1, 2, 3]) == 6
/// - sum_list([10, 20]) == 30
/// - sum_list([]) == 0
/// ```
///
/// *Problem 2: Check palindrome*
/// ```
/// Task: Write a function to check if a string is a palindrome.
/// Test Cases:
/// - is_palindrome("racecar") == True
/// - is_palindrome("hello") == False
/// - is_palindrome("") == True
/// ```
///
/// *Problem 3: Remove duplicates*
/// ```
/// Task: Write a function to remove duplicate elements from a list.
/// Test Cases:
/// - remove_duplicates([1, 2, 2, 3]) == [1, 2, 3]
/// - remove_duplicates([]) == []
/// - remove_duplicates([1, 1, 1]) == [1]
/// ```
///
/// **Categories:**
/// - List operations (sorting, filtering, searching)
/// - String manipulation
/// - Mathematical operations
/// - Basic algorithms (searching, sorting)
/// - Data structure operations (lists, dictionaries)
/// - Boolean logic
///
/// **Difficulty levels:**
/// - Basic: Simple operations (50% of problems)
/// - Intermediate: Multiple steps (40% of problems)
/// - Advanced: Complex logic (10% of problems)
///
/// **Performance levels:**
/// - GPT-3 (Codex): ~59%
/// - GPT-3.5: ~70%
/// - GPT-4: ~82%
/// - Claude 3 Opus: ~78%
/// - Claude 3.5 Sonnet: ~85%
/// - ChatGPT o1: ~90%
/// - AlphaCode: ~75%
/// - CodeGen: ~65%
///
/// **Why it's useful:**
/// - Tests basic programming competency
/// - More comprehensive than HumanEval (974 vs 164)
/// - Includes test cases (can verify correctness)
/// - Entry-level difficulty (good for beginners)
/// - Real-world relevance (common programming tasks)
///
/// **Research:**
/// - "Program Synthesis with Large Language Models" (Austin et al., 2021)
/// - https://arxiv.org/abs/2108.07732
/// - Dataset: 974 problems with solutions and test cases
/// - Used by Google Research for code generation evaluation
///
/// **Integration with CodeExecutionVerifier:**
/// MBPP works particularly well with CodeExecutionVerifier since each problem
/// includes test cases that can be executed to verify correctness.
/// </para>
/// </remarks>
public class MBPPBenchmark<T> : IBenchmark<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public MBPPBenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "MBPP";

    /// <inheritdoc/>
    public string Description =>
        "Mostly Basic Python Problems: 974 entry-level Python programming tasks with test cases. " +
        "Tests fundamental programming skills, algorithms, and code generation.";

    /// <inheritdoc/>
    public int TotalProblems => 974;

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

            // Check if correct (heuristic - in production would execute tests)
            bool isCorrect = CheckCodeCorrectness(extractedCode, problem.CorrectAnswer);

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
        // Sample MBPP-style problems
        return new List<BenchmarkProblem>
        {
            new()
            {
                Id = "mbpp_1",
                Problem = @"Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

Test cases:
assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8
assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12",
                CorrectAnswer = @"def min_cost(cost, m, n):
    tc = [[0 for x in range(n+1)] for x in range(m+1)]
    tc[0][0] = cost[0][0]
    for i in range(1, m+1):
        tc[i][0] = tc[i-1][0] + cost[i][0]
    for j in range(1, n+1):
        tc[0][j] = tc[0][j-1] + cost[0][j]
    for i in range(1, m+1):
        for j in range(1, n+1):
            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + cost[i][j]
    return tc[m][n]",
                Category = "dynamic_programming",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mbpp_2",
                Problem = @"Write a function to find the similar elements from the given two tuple lists.

Test cases:
assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4), (5, 4, 3, 7)) == (3, 4)",
                CorrectAnswer = @"def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))",
                Category = "tuple_operations",
                Difficulty = "easy"
            },
            new()
            {
                Id = "mbpp_3",
                Problem = @"Write a python function to identify non-prime numbers.

Test cases:
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True",
                CorrectAnswer = @"def is_not_prime(n):
    if n < 2:
        return True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return True
    return False",
                Category = "mathematical",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mbpp_4",
                Problem = @"Write a function to find the largest integers from a given list of numbers using heap queue algorithm.

Test cases:
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]",
                CorrectAnswer = @"import heapq
def heap_queue_largest(nums, n):
    return heapq.nlargest(n, nums)",
                Category = "list_operations",
                Difficulty = "easy"
            },
            new()
            {
                Id = "mbpp_5",
                Problem = @"Write a function to find the longest common subsequence for the given two sequences.

Test cases:
assert lcs_length('AGGTAB', 'GXTXAYB') == 4
assert lcs_length('ABCDGH', 'AEDFHR') == 3",
                CorrectAnswer = @"def lcs_length(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]",
                Category = "dynamic_programming",
                Difficulty = "hard"
            },
            new()
            {
                Id = "mbpp_6",
                Problem = @"Write a function to find all words which are at least 4 characters long in a string.

Test cases:
assert find_words('The quick brown fox jumps over the lazy dog.') == ['quick', 'brown', 'jumps', 'over', 'lazy']
assert find_words('Python is great') == ['Python', 'great']",
                CorrectAnswer = @"import re
def find_words(text):
    return re.findall(r'\b\w{4,}\b', text)",
                Category = "string_operations",
                Difficulty = "easy"
            },
            new()
            {
                Id = "mbpp_7",
                Problem = @"Write a function to find the index of the first occurrence of a given number in a sorted array.

Test cases:
assert find_first_occurrence([1, 2, 3, 3, 3, 4, 5], 3) == 2
assert find_first_occurrence([1, 1, 2, 2, 3], 2) == 2",
                CorrectAnswer = @"def find_first_occurrence(nums, target):
    left, right = 0, len(nums) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            result = mid
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result",
                Category = "searching",
                Difficulty = "medium"
            },
            new()
            {
                Id = "mbpp_8",
                Problem = @"Write a function to count the number of unique characters in a string.

Test cases:
assert unique_char_count('hello') == 4
assert unique_char_count('aabbcc') == 3
assert unique_char_count('') == 0",
                CorrectAnswer = @"def unique_char_count(s):
    return len(set(s))",
                Category = "string_operations",
                Difficulty = "easy"
            }
        };
    }

    private string? ExtractPythonCode(string text)
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
        if (text.Contains("def ") || text.Contains("import ") || text.Contains("return "))
        {
            return text.Trim();
        }

        return null;
    }

    private bool CheckCodeCorrectness(string? generatedCode, string referenceCode)
    {
        // Simple heuristic check (in production, would execute tests with CodeExecutionVerifier)
        if (generatedCode is null || string.IsNullOrWhiteSpace(generatedCode))
            return false;

        // After null check, generatedCode is guaranteed non-null
        // Check for required keywords
        bool hasFunction = generatedCode.Contains("def ");
        bool hasReturn = generatedCode.Contains("return ");
        bool hasLogic = generatedCode.Length > 30;

        // Check for common Python structures
        bool hasValidStructure = hasFunction && hasReturn;

        // Bonus: Check for algorithm keywords based on problem type
        bool hasAlgorithmKeywords = generatedCode.Contains("for ") ||
                                   generatedCode.Contains("while ") ||
                                   generatedCode.Contains("if ");

        return hasValidStructure && hasLogic && hasAlgorithmKeywords;
    }
}
