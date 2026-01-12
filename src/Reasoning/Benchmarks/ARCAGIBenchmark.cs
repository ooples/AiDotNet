using System.Diagnostics;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// ARC-AGI (Abstract Reasoning Corpus - Artificial General Intelligence) benchmark.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> ARC-AGI is considered one of the hardest AI benchmarks.
/// It tests abstract reasoning and pattern recognition using visual grid puzzles.
///
/// **What is ARC?**
/// Created by François Chollet (creator of Keras), ARC tests whether AI can think
/// abstractly like humans. Each task shows example input/output grids, and the AI
/// must figure out the transformation rule.
///
/// **Example task:**
/// ```
/// Training examples:
/// Input:  [1,1,0]    Output: [2,2,0]
///         [1,1,0]            [2,2,0]
///         [0,0,0]            [0,0,0]
///
/// Input:  [0,1,1]    Output: [0,2,2]
///         [0,1,1]            [0,2,2]
///         [0,0,0]            [0,0,0]
///
/// Test (what's the output?):
/// Input:  [1,1,1]    Output: ???
///         [0,0,0]
///         [0,0,0]
///
/// Rule: Replace all 1s with 2s
/// ```
///
/// **Why it's hard:**
/// - Requires understanding abstract concepts
/// - Few-shot learning (only 2-3 examples)
/// - Novel tasks never seen before
/// - Can't be solved by memorization
/// - Tests core intelligence, not just pattern matching
///
/// **Performance levels:**
/// - Human performance: ~85%
/// - GPT-4: ~0-5% (very poor)
/// - GPT-4o: ~10%
/// - Claude 3.5 Sonnet: ~15-20%
/// - ChatGPT o1: ~21%
/// - Specialized systems: ~20-30%
/// - **Current SOTA**: ~55% (MindsAI ARC Prize 2024)
///
/// **Why LLMs struggle:**
/// 1. Pattern recognition ≠ abstract reasoning
/// 2. Can't generalize from few examples
/// 3. No spatial/visual reasoning built-in
/// 4. Trained on language, not logic puzzles
///
/// **Recent progress:**
/// - ARC Prize (2024): $1M prize for solving ARC
/// - Test-time compute scaling helps (o1, o3)
/// - Hybrid neuro-symbolic approaches show promise
///
/// **Research:**
/// - "On the Measure of Intelligence" (Chollet, 2019) - Original ARC paper
/// - "The ARC Prize" (2024) - Competition for AGI progress
/// - ARC-AGI is viewed as a benchmark for measuring progress toward AGI
///
/// **Categories of tasks:**
/// - Object counting and manipulation
/// - Symmetry and patterns
/// - Color transformations
/// - Spatial reasoning
/// - Logical rules
/// </para>
/// </remarks>
public class ARCAGIBenchmark<T> : IBenchmark<T>
{
    private readonly INumericOperations<T> _numOps;
    private List<BenchmarkProblem>? _cachedProblems;

    public ARCAGIBenchmark()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => "ARC-AGI";

    /// <inheritdoc/>
    public string Description =>
        "Abstract Reasoning Corpus for AGI evaluation. 800 visual grid puzzles testing " +
        "abstract reasoning, few-shot learning, and pattern recognition. Considered one of " +
        "the hardest AI benchmarks (humans: 85%, GPT-4: ~5%, o1: ~21%).";

    /// <inheritdoc/>
    public int TotalProblems => 800; // 400 training + 400 evaluation

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

            // Check if correct
            bool isCorrect = CheckArcAnswer(systemAnswer, problem.CorrectAnswer);

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
                Confidence = _numOps.FromDouble(0.5), // ARC is very hard, low default confidence
                Duration = problemStopwatch.Elapsed,
                Category = category
            };

            problemResults.Add(evaluation);
            confidenceScores.Add(evaluation.Confidence);

            // Progress
            if ((i + 1) % 5 == 0 || i == problems.Count - 1)
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
        result.Metrics["problems_per_minute"] = problems.Count / stopwatch.Elapsed.TotalMinutes;
        result.Metrics["compared_to_gpt4"] = Convert.ToDouble(result.Accuracy) / 0.05; // Relative to GPT-4
        result.Metrics["compared_to_human"] = Convert.ToDouble(result.Accuracy) / 0.85; // Relative to humans

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
        // Sample ARC-style problems (in production, would load from actual ARC dataset)
        return new List<BenchmarkProblem>
        {
            new()
            {
                Id = "arc_001",
                Problem = @"ARC-AGI Task: Identify the pattern and predict the output.

Training Examples:
Input 1:  [[1,1,0],[1,1,0],[0,0,0]]
Output 1: [[2,2,0],[2,2,0],[0,0,0]]

Input 2:  [[0,1,1],[0,1,1],[0,0,0]]
Output 2: [[0,2,2],[0,2,2],[0,0,0]]

Test:
Input:  [[1,1,1],[0,0,0],[0,0,0]]
Output: ???

What is the transformation rule? Provide the output grid in JSON format: [[...],[...],[...]]",
                CorrectAnswer = "[[2,2,2],[0,0,0],[0,0,0]]",
                Category = "color_transformation",
                Difficulty = "easy"
            },
            new()
            {
                Id = "arc_002",
                Problem = @"ARC-AGI Task: Identify the pattern and predict the output.

Training Examples:
Input 1:  [[0,0,0],[0,1,0],[0,0,0]]
Output 1: [[1,1,1],[1,1,1],[1,1,1]]

Input 2:  [[1,0,0],[0,0,0],[0,0,0]]
Output 2: [[1,1,0],[1,1,0],[0,0,0]]

Test:
Input:  [[0,0,1],[0,0,0],[0,0,0]]
Output: ???

What is the transformation rule? Provide the output grid.",
                CorrectAnswer = "[[0,1,1],[0,1,1],[0,0,0]]",
                Category = "spatial_expansion",
                Difficulty = "medium"
            },
            new()
            {
                Id = "arc_003",
                Problem = @"ARC-AGI Task: Identify the pattern and predict the output.

Training Examples:
Input 1:  [[1,2,3],[4,5,6],[7,8,9]]
Output 1: [[7,8,9],[4,5,6],[1,2,3]]

Input 2:  [[9,8,7],[6,5,4],[3,2,1]]
Output 2: [[3,2,1],[6,5,4],[9,8,7]]

Test:
Input:  [[1,1,1],[2,2,2],[3,3,3]]
Output: ???

What is the transformation rule? Provide the output grid.",
                CorrectAnswer = "[[3,3,3],[2,2,2],[1,1,1]]",
                Category = "symmetry",
                Difficulty = "easy"
            },
            new()
            {
                Id = "arc_004",
                Problem = @"ARC-AGI Task: Count and transform.

Training Examples:
Input 1:  [[1,1,0],[1,0,0],[0,0,0]]
Output 1: [[3,3,3]]

Input 2:  [[1,1,1,1],[0,0,0,0]]
Output 2: [[4,4,4,4]]

Test:
Input:  [[1,1],[1,1],[1,1]]
Output: ???

What is the transformation rule? Provide the output grid.",
                CorrectAnswer = "[[6,6,6,6,6,6]]",
                Category = "counting",
                Difficulty = "medium"
            },
            new()
            {
                Id = "arc_005",
                Problem = @"ARC-AGI Task: Pattern completion.

Training Examples:
Input 1:  [[1,0,1],[0,0,0],[1,0,1]]
Output 1: [[1,0,1],[0,1,0],[1,0,1]]

Input 2:  [[2,0,2],[0,0,0],[2,0,2]]
Output 2: [[2,0,2],[0,2,0],[2,0,2]]

Test:
Input:  [[3,0,3],[0,0,0],[3,0,3]]
Output: ???

What is the transformation rule? Provide the output grid.",
                CorrectAnswer = "[[3,0,3],[0,3,0],[3,0,3]]",
                Category = "pattern_completion",
                Difficulty = "easy"
            }
        };
    }

    private bool CheckArcAnswer(string systemAnswer, string correctAnswer)
    {
        if (string.IsNullOrWhiteSpace(systemAnswer))
            return false;

        // Try to parse as grid/matrix
        var systemGrid = ExtractGrid(systemAnswer);
        var correctGrid = ExtractGrid(correctAnswer);

        if (systemGrid == null || correctGrid == null)
            return false;

        // Compare grids
        return GridsEqual(systemGrid, correctGrid);
    }

    private int[][]? ExtractGrid(string text)
    {
        try
        {
            // Try to find JSON array in text
            var match = RegexHelper.Match(
                text,
                @"\[\s*\[[\d\s,]*\](?:\s*,\s*\[[\d\s,]*\])*\s*\]",
                System.Text.RegularExpressions.RegexOptions.None);

            if (match.Success)
            {
                string jsonArray = match.Value;
                var grid = JsonConvert.DeserializeObject<int[][]>(jsonArray);
                return grid;
            }

            // Try parsing the entire text as JSON
            if (text.TrimStart().StartsWith("["))
            {
                var grid = JsonConvert.DeserializeObject<int[][]>(text);
                return grid;
            }
        }
        catch
        {
            // Parsing failed
        }

        return null;
    }

    private bool GridsEqual(int[][] grid1, int[][] grid2)
    {
        if (grid1.Length != grid2.Length)
            return false;

        for (int i = 0; i < grid1.Length; i++)
        {
            if (grid1[i].Length != grid2[i].Length)
                return false;

            for (int j = 0; j < grid1[i].Length; j++)
            {
                if (grid1[i][j] != grid2[i][j])
                    return false;
            }
        }

        return true;
    }
}



