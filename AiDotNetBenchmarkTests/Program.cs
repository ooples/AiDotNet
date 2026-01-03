using System.Reflection;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

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
        var config = ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator);
        switcher.Run(args, config);
    }
}
