namespace AiDotNet.Safety.Benchmarking;

/// <summary>
/// Comprehensive safety benchmark that aggregates all individual benchmarks into a single suite.
/// </summary>
/// <remarks>
/// <para>
/// Runs all available safety benchmarks (toxicity, jailbreak, bias, PII, hallucination, watermark,
/// adversarial) and produces a unified report with per-category breakdowns. This is the recommended
/// benchmark to run for a complete safety evaluation of your pipeline.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of running each safety benchmark separately, this runs
/// all of them at once and gives you a complete picture of how well your safety system works
/// across all categories. Use this for a one-stop safety evaluation.
/// </para>
/// <para>
/// <b>References:</b>
/// - SafetyBench: Multi-category safety evaluation (ACL 2024)
/// - HarmBench: Standardized evaluation across attack types (ICML 2024)
/// - WildGuardTest: 1.7k labeled examples across 13 risk categories (Allen AI, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ComprehensiveSafetyBenchmark<T> : SafetyBenchmarkBase<T>
{
    private readonly ISafetyBenchmark<T>[] _benchmarks;

    /// <inheritdoc />
    public override string BenchmarkName => "ComprehensiveSafetyBenchmark";

    /// <summary>
    /// Initializes a new comprehensive safety benchmark with all default sub-benchmarks.
    /// </summary>
    public ComprehensiveSafetyBenchmark()
    {
        _benchmarks = new ISafetyBenchmark<T>[]
        {
            new ToxicityBenchmark<T>(),
            new JailbreakBenchmark<T>(),
            new BiasBenchmark<T>(),
            new PIIBenchmark<T>(),
            new HallucinationBenchmark<T>(),
            new WatermarkBenchmark<T>(),
            new AdversarialBenchmark<T>()
        };
    }

    /// <summary>
    /// Initializes a new comprehensive safety benchmark with specific sub-benchmarks.
    /// </summary>
    /// <param name="benchmarks">The specific benchmarks to include in the suite.</param>
    public ComprehensiveSafetyBenchmark(IReadOnlyList<ISafetyBenchmark<T>> benchmarks)
    {
        _benchmarks = new ISafetyBenchmark<T>[benchmarks.Count];
        for (int i = 0; i < benchmarks.Count; i++)
        {
            _benchmarks[i] = benchmarks[i];
        }
    }

    /// <inheritdoc />
    public override SafetyBenchmarkResult RunBenchmark(SafetyPipeline<T> pipeline)
    {
        // Collect all individual results
        int totalTp = 0, totalFp = 0, totalTn = 0, totalFn = 0;
        int totalCases = 0;

        foreach (var benchmark in _benchmarks)
        {
            var result = benchmark.RunBenchmark(pipeline);
            totalTp += result.TruePositives;
            totalFp += result.FalsePositives;
            totalTn += result.TrueNegatives;
            totalFn += result.FalseNegatives;
            totalCases += result.TotalTestCases;
        }

        return new SafetyBenchmarkResult
        {
            TotalTestCases = totalCases,
            TruePositives = totalTp,
            FalsePositives = totalFp,
            TrueNegatives = totalTn,
            FalseNegatives = totalFn
        };
    }

    /// <inheritdoc />
    protected override IReadOnlyList<SafetyBenchmarkCase> GetTestCases()
    {
        // Not used directly — RunBenchmark delegates to sub-benchmarks
        return Array.Empty<SafetyBenchmarkCase>();
    }

    /// <summary>
    /// Runs all sub-benchmarks and returns individual results keyed by benchmark name.
    /// </summary>
    /// <param name="pipeline">The safety pipeline to benchmark.</param>
    /// <returns>A dictionary mapping benchmark names to their individual results.</returns>
    public IReadOnlyDictionary<string, SafetyBenchmarkResult> RunAllBenchmarks(SafetyPipeline<T> pipeline)
    {
        var results = new Dictionary<string, SafetyBenchmarkResult>();

        foreach (var benchmark in _benchmarks)
        {
            var result = benchmark.RunBenchmark(pipeline);
            results[benchmark.BenchmarkName] = result;
        }

        return results;
    }
}
