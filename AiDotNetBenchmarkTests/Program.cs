using AiDotNetBenchmarkTests.BenchmarkTests;
using BenchmarkDotNet.Running;

namespace AiDotNetBenchmarkTests;

internal class Program
{
    static void Main(string[] args)
    {
        BenchmarkRunner.Run<ParallelLoopTests>();
        Console.WriteLine();
    }
}