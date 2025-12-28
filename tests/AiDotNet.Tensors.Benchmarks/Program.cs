using BenchmarkDotNet.Running;
using AiDotNet.Tensors.Benchmarks;

namespace AiDotNet.Tensors.Benchmarks;

class Program
{
    static void Main(string[] args)
    {
        // Run quick performance test first for immediate feedback
        if (args.Length == 0 || args[0] == "--quick")
        {
            QuickPerformanceTest.Run();
            return;
        }

        // Run full BenchmarkDotNet suite if requested
        if (args[0] == "--full")
        {
            var summary = BenchmarkRunner.Run<TrigonometricOperatorBenchmarks>();
            return;
        }

        // Run linear algebra benchmarks
        if (args[0] == "--linalg")
        {
            BenchmarkRunner.Run<LinearAlgebraBenchmarks>();
            BenchmarkRunner.Run<SmallMatrixBenchmarks>();
            return;
        }

        Console.WriteLine("Usage:");
        Console.WriteLine("  --quick  : Run quick performance validation (default)");
        Console.WriteLine("  --full   : Run full BenchmarkDotNet suite (trigonometric)");
        Console.WriteLine("  --linalg : Run linear algebra benchmarks vs MathNet.Numerics");
    }
}
