using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Toolchains;
using BenchmarkDotNet.Toolchains.CsProj;
using BenchmarkDotNet.Toolchains.DotNetCli;
using BenchmarkDotNet.Toolchains.MonoAotLLVM;

namespace AiDotNetBenchmarkTests.Benchmarking;

internal static class FixedProjectFileToolchain
{
    // Cross-platform executable name for dotnet CLI
    private static readonly string DotNetExeName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "dotnet.exe" : "dotnet";

    public static IToolchain Create(string targetFrameworkMoniker, string projectFilePath)
    {
        var dotNetCliPath = ResolveDotNetCliPath();

        var settings = new NetCoreAppSettings(
            targetFrameworkMoniker,
            runtimeFrameworkVersion: null,
            name: targetFrameworkMoniker,
            customDotNetCliPath: dotNetCliPath,
            packagesPath: null,
            customRuntimePack: null,
            aotCompilerPath: null,
            aotCompilerMode: MonoAotCompilerMode.mini);

        var toolchain = CsProjCoreToolchain.From(settings);
        var generator = new FixedProjectFileGenerator(
            targetFrameworkMoniker,
            settings.CustomDotNetCliPath,
            settings.PackagesPath,
            settings.RuntimeFrameworkVersion,
            projectFilePath);

        OverrideGenerator(toolchain, generator);
        return toolchain;
    }

    private static string ResolveDotNetCliPath()
    {
        string? processPath = null;
#if NET6_0_OR_GREATER
        processPath = Environment.ProcessPath;
#else
        try
        {
            processPath = Process.GetCurrentProcess().MainModule?.FileName;
        }
        catch
        {
            processPath = null;
        }
#endif
        if (!string.IsNullOrWhiteSpace(processPath) &&
            string.Equals(Path.GetFileName(processPath), DotNetExeName, StringComparison.OrdinalIgnoreCase) &&
            File.Exists(processPath))
        {
            return processPath;
        }

        var dotNetRoot = Environment.GetEnvironmentVariable("DOTNET_ROOT");
        if (!string.IsNullOrWhiteSpace(dotNetRoot))
        {
            var candidate = Path.Combine(dotNetRoot, DotNetExeName);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var programFilesCandidate = Path.Combine(programFiles, "dotnet", DotNetExeName);
        if (File.Exists(programFilesCandidate))
        {
            return programFilesCandidate;
        }

        // Fallback to PATH lookup - just return "dotnet" without extension
        return "dotnet";
    }

    private static void OverrideGenerator(IToolchain toolchain, IGenerator generator)
    {
        var field = toolchain.GetType().BaseType?.GetField(
            "<Generator>k__BackingField",
            BindingFlags.Instance | BindingFlags.NonPublic);

        if (field == null)
        {
            throw new InvalidOperationException("BenchmarkDotNet toolchain generator field not found.");
        }

        field.SetValue(toolchain, generator);
    }

    private sealed class FixedProjectFileGenerator : CsProjGenerator
    {
        private readonly FileInfo _projectFile;

        public FixedProjectFileGenerator(
            string targetFrameworkMoniker,
            string? cliPath,
            string? packagesPath,
            string? runtimeFrameworkVersion,
            string projectFilePath)
            : base(
                targetFrameworkMoniker,
                cliPath!,
                packagesPath!,
                runtimeFrameworkVersion!,
                isNetCore: true)
        {
            _projectFile = new FileInfo(projectFilePath);
        }

        protected override FileInfo GetProjectFilePath(Type benchmarkTarget, ILogger logger)
        {
            return _projectFile;
        }
    }
}
