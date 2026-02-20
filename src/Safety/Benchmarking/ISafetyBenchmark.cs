using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Benchmarking;

/// <summary>
/// Interface for safety benchmark modules that evaluate safety module accuracy.
/// </summary>
/// <remarks>
/// <para>
/// Safety benchmarks run standardized test suites against safety modules to measure
/// detection accuracy, precision, recall, F1 scores, and false positive rates.
/// Each benchmark focuses on a specific safety domain (toxicity, jailbreak, PII, etc.).
/// </para>
/// <para>
/// <b>For Beginners:</b> A safety benchmark tests how well your safety modules work.
/// It feeds known examples (some safe, some harmful) through the modules and measures
/// how accurately they detect the harmful ones without blocking safe content.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISafetyBenchmark<T>
{
    /// <summary>
    /// Gets the name of this benchmark.
    /// </summary>
    string BenchmarkName { get; }

    /// <summary>
    /// Runs the benchmark against the given safety pipeline and returns results.
    /// </summary>
    /// <param name="pipeline">The safety pipeline to benchmark.</param>
    /// <returns>The benchmark results with accuracy metrics.</returns>
    SafetyBenchmarkResult RunBenchmark(SafetyPipeline<T> pipeline);
}

