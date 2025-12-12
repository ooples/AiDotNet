using BenchmarkDotNet.Running;
using System.Reflection;

namespace AiDotNetBenchmarkTests;

internal class Program
{
    static void Main(string[] args)
    {
        // Use BenchmarkSwitcher to allow running specific benchmarks from command line
        // Examples:
        //   dotnet run -c Release -- --filter *MatrixOperationsBenchmarks*
        //   dotnet run -c Release -- --filter *ActivationFunctionBenchmarks*
        //   dotnet run -c Release -- --list flat
        var switcher = BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly());
        switcher.Run(args);
    }
}