using AiDotNetBenchmarkTests.BenchmarkTests;
using BenchmarkDotNet.Running;

namespace AiDotNetBenchmarkTests;

internal class Program
{
    static void Main(string[] args)
    {
        // Run all benchmarks
        var switcher = new BenchmarkDotNet.Running.BenchmarkSwitcher(new[]
        {
            typeof(MatrixOperationsBenchmarks),
            typeof(VectorOperationsBenchmarks),
            typeof(ActivationFunctionsBenchmarks),
            typeof(ActivationFunctionGradientBenchmarks),
            typeof(SimpleRegressionBenchmarks),
            typeof(StatisticsBenchmarks),
            typeof(MultipleRegressionBenchmarks),
            typeof(PolynomialRegressionBenchmarks),
            typeof(ParallelLoopTests)
        });

        switcher.Run(args);
        Console.WriteLine();
    }
}