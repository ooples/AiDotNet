using System.Reflection;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.Emit;

namespace AiDotNetBenchmarkTests;

internal class Program
{
    static void Main(string[] args)
    {
#if !NET471
        // Run oneDNN diagnostic if requested
        if (args.Length > 0 && args[0] == "--onednn-diagnostic")
        {
            OneDnnDiagnostic.RunDiagnostic();
            return;
        }
#endif

        // Check if running in InProcess mode (needed for native DLLs like oneDNN)
        bool useInProcess = args.Any(a => a.Contains("--inprocess", StringComparison.OrdinalIgnoreCase));

        // Use BenchmarkSwitcher to allow running specific benchmarks from command line
        // Examples:
        //   dotnet run -c Release -- --filter *MatrixOperationsBenchmarks*
        //   dotnet run -c Release -- --filter *ActivationFunctionBenchmarks*
        //   dotnet run -c Release -- --list flat
        //   dotnet run -c Release -- --filter *Conv2D* --inprocess  (for native DLL access)
        var switcher = BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly());

        ManualConfig config;
        if (useInProcess)
        {
            // InProcess mode: runs in the same process, so native DLLs in output dir are available
            // Use this when benchmarking operations that rely on native libraries (oneDNN, etc.)
            config = ManualConfig.Create(DefaultConfig.Instance)
                .WithOptions(ConfigOptions.DisableOptimizationsValidator)
                .AddJob(Job.MediumRun.WithToolchain(InProcessEmitToolchain.Instance));

            Console.WriteLine("Running in InProcess mode (native DLLs available)");
#if !NET471
            Console.WriteLine($"oneDNN available: {AiDotNet.Tensors.Helpers.OneDnnProvider.IsAvailable}");
#endif
        }
        else
        {
            config = ManualConfig.Create(DefaultConfig.Instance)
                .WithOptions(ConfigOptions.DisableOptimizationsValidator);
        }

        // Remove --inprocess from args before passing to BenchmarkSwitcher
        var filteredArgs = args.Where(a => !a.Contains("--inprocess", StringComparison.OrdinalIgnoreCase)).ToArray();
        switcher.Run(filteredArgs, config);
    }
}
