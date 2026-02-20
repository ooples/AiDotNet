namespace AiDotNet.Safety.Benchmarking;

/// <summary>
/// Abstract base class for safety benchmark modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for safety benchmarks including test case execution,
/// metric calculation, and result aggregation. Concrete implementations provide
/// the specific test cases for each safety domain (toxicity, jailbreak, PII, etc.).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all safety benchmarks.
/// Each benchmark type extends this and adds its own set of test cases to evaluate
/// how well your safety modules detect specific types of harmful content.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class SafetyBenchmarkBase<T> : ISafetyBenchmark<T>
{
    /// <inheritdoc />
    public abstract string BenchmarkName { get; }

    /// <inheritdoc />
    public abstract SafetyBenchmarkResult RunBenchmark(SafetyPipeline<T> pipeline);

    /// <summary>
    /// Gets the test cases for this benchmark.
    /// </summary>
    /// <returns>A list of benchmark test cases with expected outcomes.</returns>
    protected abstract IReadOnlyList<SafetyBenchmarkCase> GetTestCases();

    /// <summary>
    /// Runs all test cases against the pipeline and computes metrics.
    /// </summary>
    protected SafetyBenchmarkResult RunTestCases(SafetyPipeline<T> pipeline, IReadOnlyList<SafetyBenchmarkCase> cases)
    {
        int tp = 0, fp = 0, tn = 0, fn = 0;

        foreach (var testCase in cases)
        {
            var report = pipeline.EvaluateText(testCase.Text);

            if (testCase.ExpectedUnsafe && !report.IsSafe) tp++;
            else if (testCase.ExpectedUnsafe && report.IsSafe) fn++;
            else if (!testCase.ExpectedUnsafe && !report.IsSafe) fp++;
            else tn++;
        }

        return new SafetyBenchmarkResult
        {
            TotalTestCases = cases.Count,
            TruePositives = tp,
            FalsePositives = fp,
            TrueNegatives = tn,
            FalseNegatives = fn
        };
    }
}
