using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;

namespace AiDotNet.Safety.Benchmarking;

/// <summary>
/// Runs safety benchmarks against a configured safety pipeline to measure detection performance.
/// </summary>
/// <remarks>
/// <para>
/// Evaluates a safety pipeline against standard benchmark test cases to measure precision,
/// recall, F1 score, and false positive rate across safety categories. Results can be used
/// to tune thresholds and compare different pipeline configurations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Before deploying a safety system, you need to test how well it works.
/// This benchmark runner tests your safety pipeline against known examples of safe and unsafe
/// content and tells you how accurate it is. Higher precision means fewer false alarms;
/// higher recall means fewer missed violations.
/// </para>
/// <para>
/// <b>Metrics computed:</b>
/// - Precision: Of flagged content, how much was actually unsafe?
/// - Recall: Of all unsafe content, how much was correctly flagged?
/// - F1 Score: Harmonic mean of precision and recall
/// - False Positive Rate: Percentage of safe content incorrectly flagged
/// - Latency: Average evaluation time per test case
/// </para>
/// <para>
/// <b>References:</b>
/// - SafetyBench: Multi-category safety evaluation (ACL 2024)
/// - HarmBench: Standardized evaluation of automated red teaming (ICML 2024)
/// - SimpleSafetyTests: 100 test prompts for critical safety risks (Vidgen et al., 2024)
/// - WildGuardTest: 1.7k labeled examples across 13 risk categories (Allen AI, 2024)
/// - SORRY-Bench: 450 linguistically diverse unsafe requests (ICLR 2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SafetyBenchmarkRunner<T>
{
    private readonly SafetyPipeline<T> _pipeline;

    /// <summary>
    /// Initializes a new safety benchmark runner.
    /// </summary>
    /// <param name="pipeline">The safety pipeline to benchmark.</param>
    public SafetyBenchmarkRunner(SafetyPipeline<T> pipeline)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
    }

    /// <summary>
    /// Runs the benchmark suite and returns a summary report.
    /// </summary>
    /// <param name="testCases">The test cases to evaluate.</param>
    /// <returns>A benchmark result with precision, recall, and per-category metrics.</returns>
    public SafetyBenchmarkResult RunBenchmark(IReadOnlyList<SafetyBenchmarkCase> testCases)
    {
        if (testCases == null || testCases.Count == 0)
        {
            return SafetyBenchmarkResult.Empty;
        }

        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;
        var categoryResults = new Dictionary<SafetyCategory, CategoryBenchmarkResult>();
        var latencies = new List<double>();

        foreach (var testCase in testCases)
        {
            var startTime = DateTime.UtcNow;

            var report = _pipeline.EvaluateText(testCase.Text);

            var elapsed = (DateTime.UtcNow - startTime).TotalMilliseconds;
            latencies.Add(elapsed);

            bool flagged = !report.IsSafe;
            bool expectedUnsafe = testCase.ExpectedUnsafe;

            if (flagged && expectedUnsafe) truePositives++;
            else if (flagged && !expectedUnsafe) falsePositives++;
            else if (!flagged && !expectedUnsafe) trueNegatives++;
            else falseNegatives++;

            // Track per-category results
            if (testCase.ExpectedCategory.HasValue)
            {
                if (!categoryResults.ContainsKey(testCase.ExpectedCategory.Value))
                {
                    categoryResults[testCase.ExpectedCategory.Value] = new CategoryBenchmarkResult();
                }

                var catResult = categoryResults[testCase.ExpectedCategory.Value];

                if (expectedUnsafe)
                {
                    bool correctCategory = report.Findings.Any(f => f.Category == testCase.ExpectedCategory.Value);
                    if (correctCategory) catResult.TruePositives++;
                    else catResult.FalseNegatives++;
                }
                else
                {
                    bool wrongCategory = report.Findings.Any(f => f.Category == testCase.ExpectedCategory.Value);
                    if (wrongCategory) catResult.FalsePositives++;
                    else catResult.TrueNegatives++;
                }
            }
        }

        double precision = truePositives + falsePositives > 0
            ? (double)truePositives / (truePositives + falsePositives) : 0;
        double recall = truePositives + falseNegatives > 0
            ? (double)truePositives / (truePositives + falseNegatives) : 0;
        double f1 = precision + recall > 0
            ? 2 * precision * recall / (precision + recall) : 0;
        double falsePositiveRate = falsePositives + trueNegatives > 0
            ? (double)falsePositives / (falsePositives + trueNegatives) : 0;
        double avgLatency = latencies.Count > 0 ? latencies.Average() : 0;

        return new SafetyBenchmarkResult
        {
            TotalTestCases = testCases.Count,
            TruePositives = truePositives,
            FalsePositives = falsePositives,
            TrueNegatives = trueNegatives,
            FalseNegatives = falseNegatives,
            Precision = precision,
            Recall = recall,
            F1Score = f1,
            FalsePositiveRate = falsePositiveRate,
            AverageLatencyMs = avgLatency,
            CategoryResults = categoryResults.ToDictionary(
                kv => kv.Key,
                kv => kv.Value.ToResult())
        };
    }

    private class CategoryBenchmarkResult
    {
        public int TruePositives;
        public int FalsePositives;
        public int TrueNegatives;
        public int FalseNegatives;

        public SafetyBenchmarkCategoryResult ToResult()
        {
            double precision = TruePositives + FalsePositives > 0
                ? (double)TruePositives / (TruePositives + FalsePositives) : 0;
            double recall = TruePositives + FalseNegatives > 0
                ? (double)TruePositives / (TruePositives + FalseNegatives) : 0;
            double f1 = precision + recall > 0
                ? 2 * precision * recall / (precision + recall) : 0;

            return new SafetyBenchmarkCategoryResult
            {
                TruePositives = TruePositives,
                FalsePositives = FalsePositives,
                TrueNegatives = TrueNegatives,
                FalseNegatives = FalseNegatives,
                Precision = precision,
                Recall = recall,
                F1Score = f1
            };
        }
    }
}

/// <summary>
/// A single test case for the safety benchmark.
/// </summary>
public class SafetyBenchmarkCase
{
    /// <summary>
    /// Gets or sets the test text to evaluate.
    /// </summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets or sets whether this test case is expected to be flagged as unsafe.
    /// </summary>
    public bool ExpectedUnsafe { get; init; }

    /// <summary>
    /// Gets or sets the expected safety category (for per-category metrics).
    /// </summary>
    public SafetyCategory? ExpectedCategory { get; init; }

    /// <summary>
    /// Gets or sets a description of the test case for reporting.
    /// </summary>
    public string Description { get; init; } = string.Empty;
}

/// <summary>
/// Results from running a safety benchmark.
/// </summary>
public class SafetyBenchmarkResult
{
    /// <summary>
    /// An empty benchmark result.
    /// </summary>
    public static readonly SafetyBenchmarkResult Empty = new();

    /// <summary>
    /// Total number of test cases evaluated.
    /// </summary>
    public int TotalTestCases { get; init; }

    /// <summary>
    /// Number of correctly identified unsafe content.
    /// </summary>
    public int TruePositives { get; init; }

    /// <summary>
    /// Number of safe content incorrectly flagged.
    /// </summary>
    public int FalsePositives { get; init; }

    /// <summary>
    /// Number of correctly identified safe content.
    /// </summary>
    public int TrueNegatives { get; init; }

    /// <summary>
    /// Number of unsafe content missed.
    /// </summary>
    public int FalseNegatives { get; init; }

    /// <summary>
    /// Precision: TP / (TP + FP). Of flagged content, how much was truly unsafe?
    /// </summary>
    public double Precision { get; init; }

    /// <summary>
    /// Recall: TP / (TP + FN). Of all unsafe content, how much was correctly flagged?
    /// </summary>
    public double Recall { get; init; }

    /// <summary>
    /// F1 Score: Harmonic mean of precision and recall.
    /// </summary>
    public double F1Score { get; init; }

    /// <summary>
    /// False Positive Rate: FP / (FP + TN).
    /// </summary>
    public double FalsePositiveRate { get; init; }

    /// <summary>
    /// Average evaluation latency in milliseconds.
    /// </summary>
    public double AverageLatencyMs { get; init; }

    /// <summary>
    /// Per-category benchmark results.
    /// </summary>
    public IReadOnlyDictionary<SafetyCategory, SafetyBenchmarkCategoryResult> CategoryResults { get; init; }
        = new Dictionary<SafetyCategory, SafetyBenchmarkCategoryResult>();
}

/// <summary>
/// Per-category benchmark results.
/// </summary>
public class SafetyBenchmarkCategoryResult
{
    /// <summary>Number of correctly identified unsafe content in this category.</summary>
    public int TruePositives { get; init; }

    /// <summary>Number of safe content incorrectly flagged for this category.</summary>
    public int FalsePositives { get; init; }

    /// <summary>Number of correctly identified safe content for this category.</summary>
    public int TrueNegatives { get; init; }

    /// <summary>Number of unsafe content missed in this category.</summary>
    public int FalseNegatives { get; init; }

    /// <summary>Precision for this category.</summary>
    public double Precision { get; init; }

    /// <summary>Recall for this category.</summary>
    public double Recall { get; init; }

    /// <summary>F1 score for this category.</summary>
    public double F1Score { get; init; }
}
