using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Benchmarks;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Examples.ConcreteExamples;

/// <summary>
/// Concrete example: Running actual benchmark evaluation.
/// </summary>
public class BenchmarkRunnerExample
{
    public static async Task RunAsync(IChatModel chatModel)
    {
        Console.WriteLine("=== Benchmark Evaluation Example ===\n");

        var strategy = new ChainOfThoughtStrategy<double>(chatModel);
        var results = new Dictionary<string, double>();

        // GSM8K Math Benchmark
        Console.WriteLine("Running GSM8K Benchmark...");
        await RunGSM8KAsync(chatModel, results);

        // MMLU Knowledge Benchmark
        Console.WriteLine("\nRunning MMLU Sample...");
        await RunMMLUAsync(chatModel, results);

        // BoolQ Reading Comprehension
        Console.WriteLine("\nRunning BoolQ Sample...");
        await RunBoolQAsync(chatModel, results);

        // Final Summary
        Console.WriteLine("\n" + new string('=', 80));
        Console.WriteLine("=== Benchmark Results Summary ===");
        Console.WriteLine(new string('=', 80));

        foreach (var (benchmark, accuracy) in results.OrderByDescending(x => x.Value))
        {
            Console.WriteLine($"{benchmark,-20}: {accuracy:P1}");
        }

        var averageAccuracy = results.Values.Average();
        Console.WriteLine(new string('-', 80));
        Console.WriteLine($"{"Average",-20}: {averageAccuracy:P1}");
    }

    private static async Task RunGSM8KAsync(IChatModel chatModel, Dictionary<string, double> results)
    {
        var benchmark = new GSM8KBenchmark<double>();
        var strategy = new ChainOfThoughtStrategy<double>(chatModel);

        try
        {
            var result = await benchmark.EvaluateAsync(
                async (problem) =>
                {
                    var reasoning = await strategy.ReasonAsync(problem);
                    return reasoning.FinalAnswer;
                },
                sampleSize: 10  // Small sample for demo
            );

            results["GSM8K (Math)"] = Convert.ToDouble(result.Accuracy);

            Console.WriteLine($"  Problems Evaluated: {result.TotalEvaluated}");
            Console.WriteLine($"  Correct: {result.CorrectCount}");
            Console.WriteLine($"  Accuracy: {result.Accuracy:P1}");
            Console.WriteLine($"  Avg Confidence: {result.AverageConfidence:F2}");
            Console.WriteLine($"  Duration: {result.TotalDuration.TotalSeconds:F1}s");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error: {ex.Message}");
            results["GSM8K (Math)"] = 0.0;
        }
    }

    private static async Task RunMMLUAsync(IChatModel chatModel, Dictionary<string, double> results)
    {
        var benchmark = new MMLUBenchmark<double>();
        var strategy = new ChainOfThoughtStrategy<double>(chatModel);

        try
        {
            var result = await benchmark.EvaluateAsync(
                async (problem) =>
                {
                    var reasoning = await strategy.ReasonAsync(problem);
                    return reasoning.FinalAnswer;
                },
                sampleSize: 10
            );

            results["MMLU (Knowledge)"] = Convert.ToDouble(result.Accuracy);

            Console.WriteLine($"  Problems Evaluated: {result.TotalEvaluated}");
            Console.WriteLine($"  Correct: {result.CorrectCount}");
            Console.WriteLine($"  Accuracy: {result.Accuracy:P1}");

            // Show accuracy by category
            if (result.AccuracyByCategory.Count > 0)
            {
                Console.WriteLine($"  By Category:");
                foreach (var (category, acc) in result.AccuracyByCategory.Take(3))
                {
                    Console.WriteLine($"    {category}: {Convert.ToDouble(acc):P1}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error: {ex.Message}");
            results["MMLU (Knowledge)"] = 0.0;
        }
    }

    private static async Task RunBoolQAsync(IChatModel chatModel, Dictionary<string, double> results)
    {
        var benchmark = new BoolQBenchmark<double>();
        var strategy = new ChainOfThoughtStrategy<double>(chatModel);

        try
        {
            var result = await benchmark.EvaluateAsync(
                async (problem) =>
                {
                    var reasoning = await strategy.ReasonAsync(problem);
                    return reasoning.FinalAnswer;
                },
                sampleSize: 10
            );

            results["BoolQ (Comprehension)"] = Convert.ToDouble(result.Accuracy);

            Console.WriteLine($"  Problems Evaluated: {result.TotalEvaluated}");
            Console.WriteLine($"  Correct: {result.CorrectCount}");
            Console.WriteLine($"  Accuracy: {result.Accuracy:P1}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error: {ex.Message}");
            results["BoolQ (Comprehension)"] = 0.0;
        }
    }
}
