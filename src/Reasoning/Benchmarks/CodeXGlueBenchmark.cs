using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Benchmarks.Data;
using AiDotNet.Reasoning.Benchmarks.Models;

namespace AiDotNet.Reasoning.Benchmarks;

/// <summary>
/// CodeXGLUE benchmark harness (dataset-loader + metric computation).
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// CodeXGLUE is a suite of code understanding and code generation tasks. This harness is intentionally "dataset
/// agnostic": callers provide a JSONL file path and field mapping, and provide the model invocation function.
/// </para>
/// <para>
/// This harness does not attempt to ship, download, or cache CodeXGLUE datasets; it only provides the evaluation glue.
/// </para>
/// </remarks>
public sealed class CodeXGlueBenchmark<T> : IBenchmark<T>
{
    private static readonly Regex TokenSplitRegex = RegexHelper.Create(@"[^\p{L}\p{Nd}]+", RegexOptions.Compiled);
    private readonly INumericOperations<T> _numOps;
    private readonly CodeXGlueBenchmarkOptions _options;
    private List<CodeXGlueProblem>? _cachedProblems;

    public CodeXGlueBenchmark(CodeXGlueBenchmarkOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string BenchmarkName => string.IsNullOrWhiteSpace(_options.TaskName) ? "CodeXGLUE" : $"CodeXGLUE ({_options.TaskName})";

    /// <inheritdoc/>
    public string Description =>
        "CodeXGLUE: a suite of code understanding and generation benchmarks. " +
        "This harness loads a caller-provided JSONL file and computes simple text metrics (Exact Match + token F1).";

    /// <inheritdoc/>
    public int TotalProblems => _cachedProblems?.Count ?? 0;

    /// <inheritdoc/>
    public async Task<List<BenchmarkProblem>> LoadProblemsAsync(int? count = null)
    {
        var problems = await LoadDatasetAsync(count).ConfigureAwait(false);
        return problems
            .Select((p, idx) => new BenchmarkProblem
            {
                Id = string.IsNullOrWhiteSpace(p.Id) ? idx.ToString() : p.Id,
                Problem = p.Source,
                CorrectAnswer = p.Target,
                Category = string.IsNullOrWhiteSpace(p.Category) ? "default" : p.Category
            })
            .ToList();
    }

    /// <inheritdoc/>
    public async Task<BenchmarkResult<T>> EvaluateAsync(
        Func<string, Task<string>> evaluateFunction,
        int? sampleSize = null,
        CancellationToken cancellationToken = default)
    {
        if (evaluateFunction is null)
        {
            throw new ArgumentNullException(nameof(evaluateFunction));
        }

        var problems = await LoadDatasetAsync(sampleSize).ConfigureAwait(false);
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

        var stopwatch = Stopwatch.StartNew();

        int correctCount = 0;
        double tokenF1Sum = 0.0;
        double bleu4Sum = 0.0;
        double rougeLSum = 0.0;
        var confidenceScores = new List<T>(problems.Count);
        var problemResults = new List<ProblemEvaluation<T>>(problems.Count);
        var categoryCorrect = new Dictionary<string, int>(StringComparer.Ordinal);
        var categoryTotal = new Dictionary<string, int>(StringComparer.Ordinal);

        for (int i = 0; i < problems.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var problem = problems[i];
            var problemStopwatch = Stopwatch.StartNew();

            string systemAnswer;
            try
            {
                systemAnswer = await evaluateFunction(problem.Source).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                systemAnswer = $"ERROR: {ex.Message}";
            }

            problemStopwatch.Stop();

            var isExactMatch = IsExactMatch(systemAnswer, problem.Target);
            var tokenF1 = ComputeTokenF1(systemAnswer, problem.Target);
            var bleu4 = ComputeBleu4(systemAnswer, problem.Target);
            var rougeL = ComputeRougeL(systemAnswer, problem.Target);
            var identifierF1 = ComputeIdentifierF1(systemAnswer, problem.Target);
            var codeBleuLite = ComputeCodeBleuLite(tokenF1, bleu4, rougeL, identifierF1);

            if (isExactMatch)
            {
                correctCount++;
            }

            tokenF1Sum += tokenF1;
            bleu4Sum += bleu4;
            rougeLSum += rougeL;

            var category = string.IsNullOrWhiteSpace(problem.Category) ? "default" : problem.Category;
            if (categoryTotal.TryGetValue(category, out var totalCount))
            {
                categoryTotal[category] = totalCount + 1;
            }
            else
            {
                categoryTotal[category] = 1;
            }

            if (isExactMatch)
            {
                if (categoryCorrect.TryGetValue(category, out var correctCountForCategory))
                {
                    categoryCorrect[category] = correctCountForCategory + 1;
                }
                else
                {
                    categoryCorrect[category] = 1;
                }
            }

            var confidence = isExactMatch ? _numOps.One : _numOps.Zero;
            confidenceScores.Add(confidence);

            problemResults.Add(new ProblemEvaluation<T>
            {
                ProblemId = string.IsNullOrWhiteSpace(problem.Id) ? i.ToString() : problem.Id,
                Problem = problem.Source,
                CorrectAnswer = problem.Target,
                SystemAnswer = systemAnswer,
                IsCorrect = isExactMatch,
                Confidence = confidence,
                Duration = problemStopwatch.Elapsed,
                Category = category,
                Metadata = new Dictionary<string, object>
                {
                    { "TokenF1", tokenF1 },
                    { "Bleu4", bleu4 },
                    { "RougeL", rougeL },
                    { "IdentifierF1", identifierF1 },
                    { "CodeBleuLite", codeBleuLite }
                }
            });
        }

        stopwatch.Stop();

        var accuracy = _numOps.Divide(_numOps.FromDouble(correctCount), _numOps.FromDouble(problems.Count));
        var avgTokenF1 = tokenF1Sum / problems.Count;
        var avgBleu4 = bleu4Sum / problems.Count;
        var avgRougeL = rougeLSum / problems.Count;
        var avgIdentifierF1 = problemResults.Count == 0 ? 0.0 : problemResults.Average(p => Convert.ToDouble(p.Metadata["IdentifierF1"]));
        var avgCodeBleuLite = problemResults.Count == 0 ? 0.0 : problemResults.Average(p => Convert.ToDouble(p.Metadata["CodeBleuLite"]));

        var accuracyByCategory = new Dictionary<string, T>(StringComparer.Ordinal);
        foreach (var kvp in categoryTotal)
        {
            var cat = kvp.Key;
            var total = kvp.Value;
            var correct = categoryCorrect.TryGetValue(cat, out var c) ? c : 0;
            accuracyByCategory[cat] = total > 0
                ? _numOps.Divide(_numOps.FromDouble(correct), _numOps.FromDouble(total))
                : _numOps.Zero;
        }

        return new BenchmarkResult<T>
        {
            BenchmarkName = BenchmarkName,
            TotalEvaluated = problems.Count,
            CorrectCount = correctCount,
            Accuracy = accuracy,
            ConfidenceScores = new Vector<T>(confidenceScores.ToArray()),
            AverageConfidence = accuracy,
            TotalDuration = stopwatch.Elapsed,
            ProblemResults = problemResults,
            AccuracyByCategory = accuracyByCategory,
            Metrics = new Dictionary<string, object>
            {
                { "AverageTokenF1", avgTokenF1 },
                { "AverageBleu4", avgBleu4 },
                { "AverageRougeL", avgRougeL },
                { "AverageIdentifierF1", avgIdentifierF1 },
                { "AverageCodeBleuLite", avgCodeBleuLite },
                { "DatasetPath", _options.DatasetFilePath },
                { "SourceField", _options.SourceField },
                { "TargetField", _options.TargetField }
            }
        };
    }

    private async Task<List<CodeXGlueProblem>> LoadDatasetAsync(int? sampleSize)
    {
        if (_cachedProblems is null)
        {
            _cachedProblems = await CodeXGlueDataLoader.LoadFromFileAsync(
                    _options.DatasetFilePath,
                    _options.SourceField,
                    _options.TargetField,
                    _options.IdField,
                    _options.CategoryField)
                .ConfigureAwait(false);
        }

        if (sampleSize.HasValue && sampleSize.Value > 0 && _cachedProblems.Count > sampleSize.Value)
        {
            return _cachedProblems.Take(sampleSize.Value).ToList();
        }

        return _cachedProblems;
    }

    private static bool IsExactMatch(string predicted, string expected)
        => string.Equals(Normalize(predicted), Normalize(expected), StringComparison.Ordinal);

    private static double ComputeTokenF1(string predicted, string expected)
    {
        var predTokens = TokenizeForMetrics(predicted);
        var expTokens = TokenizeForMetrics(expected);

        if (predTokens.Count == 0 && expTokens.Count == 0)
        {
            return 1.0;
        }

        if (predTokens.Count == 0 || expTokens.Count == 0)
        {
            return 0.0;
        }

        var predCounts = ToCounts(predTokens);
        var expCounts = ToCounts(expTokens);

        int overlap = 0;
        foreach (var pair in predCounts.Join(
                     expCounts,
                     static pred => pred.Key,
                     static exp => exp.Key,
                     static (pred, exp) => (Pred: pred.Value, Exp: exp.Value)))
        {
            overlap += Math.Min(pair.Pred, pair.Exp);
        }

        var precision = overlap / (double)predTokens.Count;
        var recall = overlap / (double)expTokens.Count;

        if (precision <= 0.0 || recall <= 0.0)
        {
            return 0.0;
        }

        return 2.0 * precision * recall / (precision + recall);
    }

    private static double ComputeBleu4(string predicted, string expected)
    {
        var predTokens = TokenizeForMetrics(predicted);
        var expTokens = TokenizeForMetrics(expected);

        if (predTokens.Count == 0 && expTokens.Count == 0)
        {
            return 1.0;
        }

        if (predTokens.Count == 0 || expTokens.Count == 0)
        {
            return 0.0;
        }

        var maxN = Math.Min(4, Math.Min(predTokens.Count, expTokens.Count));
        if (maxN <= 0)
        {
            return 0.0;
        }
        var precisions = new double[maxN];

        for (int n = 1; n <= maxN; n++)
        {
            var p = ComputeModifiedPrecision(predTokens, expTokens, n);
            if (p <= 0.0)
            {
                return 0.0;
            }

            precisions[n - 1] = p;
        }

        var logSum = 0.0;
        for (int i = 0; i < maxN; i++)
        {
            logSum += Math.Log(precisions[i]);
        }

        var geoMean = Math.Exp(logSum / maxN);
        var bp = predTokens.Count > expTokens.Count ? 1.0 : Math.Exp(1.0 - (expTokens.Count / (double)predTokens.Count));
        return bp * geoMean;
    }

    private static double ComputeModifiedPrecision(List<string> predictedTokens, List<string> expectedTokens, int n)
    {
        var pred = GetNgramCounts(predictedTokens, n);
        var exp = GetNgramCounts(expectedTokens, n);

        int overlap = 0;
        int total = 0;

        foreach (var kvp in pred)
        {
            total += kvp.Value;
            if (exp.TryGetValue(kvp.Key, out var expCount))
            {
                overlap += Math.Min(kvp.Value, expCount);
            }
        }

        return total == 0 ? 0.0 : overlap / (double)total;
    }

    private static Dictionary<string, int> GetNgramCounts(List<string> tokens, int n)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);

        if (tokens.Count < n || n <= 0)
        {
            return counts;
        }

        for (int i = 0; i <= tokens.Count - n; i++)
        {
            var ngram = string.Join(" ", tokens.Skip(i).Take(n));
            if (counts.TryGetValue(ngram, out var ngramCount))
            {
                counts[ngram] = ngramCount + 1;
            }
            else
            {
                counts[ngram] = 1;
            }
        }

        return counts;
    }

    private static double ComputeRougeL(string predicted, string expected)
    {
        var predTokens = TokenizeForMetrics(predicted);
        var expTokens = TokenizeForMetrics(expected);

        if (predTokens.Count == 0 && expTokens.Count == 0)
        {
            return 1.0;
        }

        if (predTokens.Count == 0 || expTokens.Count == 0)
        {
            return 0.0;
        }

        var lcs = LongestCommonSubsequenceLength(predTokens, expTokens);
        if (lcs == 0)
        {
            return 0.0;
        }

        var precision = lcs / (double)predTokens.Count;
        var recall = lcs / (double)expTokens.Count;
        return (precision <= 0.0 || recall <= 0.0) ? 0.0 : 2.0 * precision * recall / (precision + recall);
    }

    private static int LongestCommonSubsequenceLength(List<string> a, List<string> b)
    {
        // Dynamic programming LCS length (token-level). This is O(|a|*|b|) and intended for benchmark evaluation.
        var dp = new int[a.Count + 1, b.Count + 1];

        for (int i = 1; i <= a.Count; i++)
        {
            for (int j = 1; j <= b.Count; j++)
            {
                if (string.Equals(a[i - 1], b[j - 1], StringComparison.Ordinal))
                {
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                }
                else
                {
                    dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
                }
            }
        }

        return dp[a.Count, b.Count];
    }

    private static double ComputeIdentifierF1(string predicted, string expected)
    {
        var predTokens = TokenizeForMetrics(predicted).Where(IsIdentifierToken).ToList();
        var expTokens = TokenizeForMetrics(expected).Where(IsIdentifierToken).ToList();

        if (predTokens.Count == 0 && expTokens.Count == 0)
        {
            return 1.0;
        }

        if (predTokens.Count == 0 || expTokens.Count == 0)
        {
            return 0.0;
        }

        var predCounts = ToCounts(predTokens);
        var expCounts = ToCounts(expTokens);

        int overlap = 0;
        foreach (var pair in predCounts.Join(
                     expCounts,
                     static pred => pred.Key,
                     static exp => exp.Key,
                     static (pred, exp) => (Pred: pred.Value, Exp: exp.Value)))
        {
            overlap += Math.Min(pair.Pred, pair.Exp);
        }

        var precision = overlap / (double)predTokens.Count;
        var recall = overlap / (double)expTokens.Count;

        if (precision <= 0.0 || recall <= 0.0)
        {
            return 0.0;
        }

        return 2.0 * precision * recall / (precision + recall);
    }

    private static bool IsIdentifierToken(string token)
    {
        if (string.IsNullOrWhiteSpace(token))
        {
            return false;
        }

        char first = token[0];
        if (!(char.IsLetter(first) || first == '_'))
        {
            return false;
        }

        for (int i = 1; i < token.Length; i++)
        {
            char c = token[i];
            if (!(char.IsLetterOrDigit(c) || c == '_'))
            {
                return false;
            }
        }

        return true;
    }

    private static double ComputeCodeBleuLite(double tokenF1, double bleu4, double rougeL, double identifierF1)
    {
        // Heuristic "CodeBLEU-like" aggregate:
        // - BLEU-4 captures local n-gram overlap
        // - ROUGE-L captures global sequence similarity
        // - TokenF1 captures bag-of-tokens overlap
        // - IdentifierF1 emphasizes naming consistency (important for code tasks)
        return 0.25 * bleu4 + 0.25 * rougeL + 0.25 * tokenF1 + 0.25 * identifierF1;
    }

    private static Dictionary<string, int> ToCounts(List<string> tokens)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var token in tokens)
        {
            if (counts.TryGetValue(token, out var tokenCount))
            {
                counts[token] = tokenCount + 1;
            }
            else
            {
                counts[token] = 1;
            }
        }

        return counts;
    }

    private static List<string> TokenizeForMetrics(string text)
    {
        var normalized = Normalize(text);
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return new List<string>();
        }

        return TokenSplitRegex
            .Split(normalized)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();
    }

    private static string Normalize(string value)
        => (value ?? string.Empty).Replace("\r\n", "\n").Trim();
}

/// <summary>
/// Options for configuring CodeXGLUE dataset loading.
/// </summary>
public sealed class CodeXGlueBenchmarkOptions
{
    /// <summary>
    /// Path to a JSONL dataset file.
    /// </summary>
    public string DatasetFilePath { get; set; } = string.Empty;

    /// <summary>
    /// Optional task label for display (e.g., "code-to-text", "text-to-code").
    /// </summary>
    public string TaskName { get; set; } = string.Empty;

    /// <summary>
    /// JSON field that contains the prompt/source.
    /// </summary>
    public string SourceField { get; set; } = "source";

    /// <summary>
    /// JSON field that contains the expected answer/target.
    /// </summary>
    public string TargetField { get; set; } = "target";

    /// <summary>
    /// JSON field that contains the record identifier.
    /// </summary>
    public string IdField { get; set; } = "id";

    /// <summary>
    /// JSON field that contains an optional category label.
    /// </summary>
    public string CategoryField { get; set; } = "category";
}

