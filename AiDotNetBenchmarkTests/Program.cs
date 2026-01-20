using System.Collections.Generic;
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
        //   dotnet run -c Release -- --gpu-harness --warmup 1 --iterations 3
        //   dotnet run -c Release -- --filter *Conv2D* --inprocess  (for native DLL access)
        var switcher = BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly());
        var argList = new List<string>(args);

        // Handle GPU harness mode
        if (RemoveFlag(argList, "--gpu-harness"))
        {
#if NET8_0_OR_GREATER
            var warmup = ParseIntOption(argList, "--warmup", 1);
            var iterations = ParseIntOption(argList, "--iterations", 3);
            GpuResidentQuickHarness.Run(warmup, iterations);
#else
            Console.WriteLine("GPU harness requires NET8_0_OR_GREATER.");
#endif
            return;
        }

        // Build config based on options
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

        // Handle short-run mode
        if (RemoveFlag(argList, "--short-run") || RemoveFlag(argList, "--gpu-short"))
        {
            config = CreateShortRunConfig(config);
        }

        // Remove --inprocess from args before passing to BenchmarkSwitcher
        RemoveFlag(argList, "--inprocess");
        switcher.Run(argList.ToArray(), config);
    }

    private static bool RemoveFlag(List<string> args, string flag)
    {
        var index = args.IndexOf(flag);
        if (index < 0)
        {
            return false;
        }

        args.RemoveAt(index);
        return true;
    }

    private static int ParseIntOption(List<string> args, string option, int defaultValue)
    {
        var index = args.IndexOf(option);
        if (index < 0)
        {
            return defaultValue;
        }

        if (index == args.Count - 1)
        {
            args.RemoveAt(index);
            return defaultValue;
        }

        var valueText = args[index + 1];
        args.RemoveAt(index + 1);
        args.RemoveAt(index);

        if (int.TryParse(valueText, out var value) && value > 0)
        {
            return value;
        }

        return defaultValue;
    }

    private static ManualConfig CreateShortRunConfig(IConfig baseConfig)
    {
        var config = new ManualConfig()
            .WithUnionRule(ConfigUnionRule.AlwaysUseLocal)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddJob(Job.ShortRun.WithId("ShortRun"));

        foreach (var exporter in baseConfig.GetExporters())
        {
            config.AddExporter(exporter);
        }

        foreach (var logger in baseConfig.GetLoggers())
        {
            config.AddLogger(logger);
        }

        foreach (var columnProvider in baseConfig.GetColumnProviders())
        {
            config.AddColumnProvider(columnProvider);
        }

        foreach (var diagnoser in baseConfig.GetDiagnosers())
        {
            config.AddDiagnoser(diagnoser);
        }

        foreach (var analyser in baseConfig.GetAnalysers())
        {
            config.AddAnalyser(analyser);
        }

        foreach (var filter in baseConfig.GetFilters())
        {
            config.AddFilter(filter);
        }

        foreach (var validator in baseConfig.GetValidators())
        {
            config.AddValidator(validator);
        }

        foreach (var rule in baseConfig.GetLogicalGroupRules())
        {
            config.AddLogicalGroupRules(rule);
        }

        foreach (var counter in baseConfig.GetHardwareCounters())
        {
            config.AddHardwareCounters(counter);
        }

        foreach (var processor in baseConfig.GetEventProcessors())
        {
            config.AddEventProcessor(processor);
        }

        if (baseConfig.Orderer != null)
        {
            config.WithOrderer(baseConfig.Orderer);
        }

        config.WithSummaryStyle(baseConfig.SummaryStyle);

        return config;
    }
}
