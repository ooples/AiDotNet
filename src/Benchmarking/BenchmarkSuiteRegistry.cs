using AiDotNet.Benchmarking.Models;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Benchmarks;

namespace AiDotNet.Benchmarking;

/// <summary>
/// Registry for benchmark suite discovery and suite instantiation.
/// </summary>
public static class BenchmarkSuiteRegistry
{
    private static readonly IReadOnlyDictionary<BenchmarkSuite, BenchmarkSuiteDescriptor> Descriptors =
        new Dictionary<BenchmarkSuite, BenchmarkSuiteDescriptor>
        {
            [BenchmarkSuite.LEAF] = new() { Suite = BenchmarkSuite.LEAF, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "LEAF", Description = "Federated benchmark suite using LEAF-style JSON splits." },
            [BenchmarkSuite.FEMNIST] = new() { Suite = BenchmarkSuite.FEMNIST, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "FEMNIST", Description = "LEAF FEMNIST federated handwritten character classification." },
            [BenchmarkSuite.Sent140] = new() { Suite = BenchmarkSuite.Sent140, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "Sent140", Description = "LEAF Sent140 federated sentiment classification (tweets)." },
            [BenchmarkSuite.Shakespeare] = new() { Suite = BenchmarkSuite.Shakespeare, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "Shakespeare", Description = "LEAF Shakespeare federated next-character prediction benchmark." },
            [BenchmarkSuite.Reddit] = new() { Suite = BenchmarkSuite.Reddit, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "Reddit", Description = "Federated next-token prediction using the LEAF Reddit dataset (large corpus)." },
            [BenchmarkSuite.StackOverflow] = new() { Suite = BenchmarkSuite.StackOverflow, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "StackOverflow", Description = "Federated next-token prediction using a StackOverflow corpus dataset (large corpus)." },
            [BenchmarkSuite.CIFAR10] = new() { Suite = BenchmarkSuite.CIFAR10, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "CIFAR-10", Description = "Federated image classification using CIFAR-10 with synthetic client partitioning." },
            [BenchmarkSuite.CIFAR100] = new() { Suite = BenchmarkSuite.CIFAR100, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "CIFAR-100", Description = "Federated image classification using CIFAR-100 with synthetic client partitioning." },
            [BenchmarkSuite.TabularNonIID] = new() { Suite = BenchmarkSuite.TabularNonIID, Kind = BenchmarkSuiteKind.DatasetSuite, DisplayName = "Tabular (Non-IID)", Description = "Generic tabular benchmark with synthetic non-IID client partitions." },
            [BenchmarkSuite.GSM8K] = new() { Suite = BenchmarkSuite.GSM8K, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "GSM8K", Description = "Grade school math word problems (multi-step reasoning)." },
            [BenchmarkSuite.MATH] = new() { Suite = BenchmarkSuite.MATH, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "MATH", Description = "Competition mathematics problems." },
            [BenchmarkSuite.MMLU] = new() { Suite = BenchmarkSuite.MMLU, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "MMLU", Description = "Multi-subject multiple-choice benchmark." },
            [BenchmarkSuite.TruthfulQA] = new() { Suite = BenchmarkSuite.TruthfulQA, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "TruthfulQA", Description = "Truthfulness / hallucination resistance." },
            [BenchmarkSuite.ARCAGI] = new() { Suite = BenchmarkSuite.ARCAGI, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "ARC-AGI", Description = "Abstract reasoning puzzles." },
            [BenchmarkSuite.DROP] = new() { Suite = BenchmarkSuite.DROP, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "DROP", Description = "Reading comprehension with discrete reasoning." },
            [BenchmarkSuite.BoolQ] = new() { Suite = BenchmarkSuite.BoolQ, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "BoolQ", Description = "Yes/No question answering." },
            [BenchmarkSuite.PIQA] = new() { Suite = BenchmarkSuite.PIQA, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "PIQA", Description = "Physical commonsense reasoning." },
            [BenchmarkSuite.CommonsenseQA] = new() { Suite = BenchmarkSuite.CommonsenseQA, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "CommonsenseQA", Description = "Commonsense multiple-choice QA." },
            [BenchmarkSuite.WinoGrande] = new() { Suite = BenchmarkSuite.WinoGrande, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "WinoGrande", Description = "Pronoun resolution / commonsense reasoning." },
            [BenchmarkSuite.HellaSwag] = new() { Suite = BenchmarkSuite.HellaSwag, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "HellaSwag", Description = "Commonsense inference for narrative completion." },
            [BenchmarkSuite.HumanEval] = new() { Suite = BenchmarkSuite.HumanEval, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "HumanEval", Description = "Code generation / program synthesis evaluation." },
            [BenchmarkSuite.MBPP] = new() { Suite = BenchmarkSuite.MBPP, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "MBPP", Description = "Mostly Basic Programming Problems." },
            [BenchmarkSuite.LogiQA] = new() { Suite = BenchmarkSuite.LogiQA, Kind = BenchmarkSuiteKind.Reasoning, DisplayName = "LogiQA", Description = "Logical reasoning benchmark." }
        };

    private static readonly IReadOnlyDictionary<BenchmarkSuite, Func<IBenchmark<double>>> ReasoningFactories =
        new Dictionary<BenchmarkSuite, Func<IBenchmark<double>>>
        {
            [BenchmarkSuite.GSM8K] = () => new GSM8KBenchmark<double>(),
            [BenchmarkSuite.MATH] = () => new MATHBenchmark<double>(),
            [BenchmarkSuite.MMLU] = () => new MMLUBenchmark<double>(),
            [BenchmarkSuite.TruthfulQA] = () => new TruthfulQABenchmark<double>(),
            [BenchmarkSuite.ARCAGI] = () => new ARCAGIBenchmark<double>(),
            [BenchmarkSuite.DROP] = () => new DROPBenchmark<double>(),
            [BenchmarkSuite.BoolQ] = () => new BoolQBenchmark<double>(),
            [BenchmarkSuite.PIQA] = () => new PIQABenchmark<double>(),
            [BenchmarkSuite.CommonsenseQA] = () => new CommonsenseQABenchmark<double>(),
            [BenchmarkSuite.WinoGrande] = () => new WinoGrandeBenchmark<double>(),
            [BenchmarkSuite.HellaSwag] = () => new HellaSwagBenchmark<double>(),
            [BenchmarkSuite.HumanEval] = () => new HumanEvalBenchmark<double>(),
            [BenchmarkSuite.MBPP] = () => new MBPPBenchmark<double>(),
            [BenchmarkSuite.LogiQA] = () => new LogiQABenchmark<double>()
        };

    /// <summary>
    /// Lists benchmark suites available in the current build.
    /// </summary>
    public static IReadOnlyList<BenchmarkSuiteDescriptor> GetAvailableSuites()
        => Descriptors.Values.OrderBy(x => x.Kind).ThenBy(x => x.DisplayName).ToList();

    /// <summary>
    /// Gets the suite category/kind.
    /// </summary>
    public static BenchmarkSuiteKind GetSuiteKind(BenchmarkSuite suite)
    {
        if (Descriptors.TryGetValue(suite, out var descriptor))
        {
            return descriptor.Kind;
        }

        throw new ArgumentOutOfRangeException(nameof(suite), suite, "Unknown benchmark suite.");
    }

    /// <summary>
    /// Gets a stable display name for a suite.
    /// </summary>
    public static string GetDisplayName(BenchmarkSuite suite)
    {
        if (Descriptors.TryGetValue(suite, out var descriptor))
        {
            return descriptor.DisplayName;
        }

        throw new ArgumentOutOfRangeException(nameof(suite), suite, "Unknown benchmark suite.");
    }

    internal static IBenchmark<double> CreateReasoningBenchmark(BenchmarkSuite suite)
    {
        if (ReasoningFactories.TryGetValue(suite, out var factory))
        {
            return factory();
        }

        throw new NotSupportedException($"Benchmark suite '{suite}' is not a supported reasoning suite.");
    }
}
