using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Builds interpreter specifications from commands that exist on a stock machine, so the sandbox tests need no
/// Docker, no network, and no language runtime beyond the operating system shell.
/// </summary>
internal static class ProgramSandboxTestEnvironment
{
    internal static bool IsWindows => Environment.OSVersion.Platform == PlatformID.Win32NT;

    internal static string ShellPath => IsWindows
        ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "cmd.exe")
        : "/bin/sh";

    internal static bool ShellExists => File.Exists(ShellPath);

    /// <summary>Wraps a shell fragment as an argument template for the platform's shell.</summary>
    internal static string Script(string windowsFragment, string posixFragment) =>
        IsWindows ? "/c " + windowsFragment : "-c \"" + posixFragment + "\"";

    /// <summary>A command that copies the written source file to standard output.</summary>
    internal static string EchoSourceTemplate => Script(
        "type {source}",
        "cat {source}");

    /// <summary>A command that reads the source, ignores it, and copies standard input to standard output.</summary>
    internal static string EchoStdInTemplate => Script(
        "type {source} >nul & findstr \"^\"",
        "cat {source} > /dev/null; cat");

    /// <summary>A command that reads the source, ignores it, and prints the child's whole environment.</summary>
    internal static string PrintEnvironmentTemplate => Script(
        "type {source} >nul & set",
        "cat {source} > /dev/null; env");

    /// <summary>A command that reads the source, ignores it, and exits with the given code.</summary>
    internal static string ExitWithCodeTemplate(int exitCode) => Script(
        "type {source} >nul & exit /b " + exitCode.ToString(CultureInfo.InvariantCulture),
        "cat {source} > /dev/null; exit " + exitCode.ToString(CultureInfo.InvariantCulture));

    /// <summary>A command that reads the source, ignores it, and then sleeps for roughly the given seconds.</summary>
    internal static string SleepTemplate(int seconds) => Script(
        "type {source} >nul & ping -n " + (seconds + 1).ToString(CultureInfo.InvariantCulture) + " 127.0.0.1 >nul",
        "cat {source} > /dev/null; sleep " + seconds.ToString(CultureInfo.InvariantCulture));

    /// <summary>Creates sandbox options whose Python entry runs the supplied shell command template.</summary>
    internal static ProgramSandboxOptions Options(string template, string? compileTemplate = null)
    {
        var options = new ProgramSandboxOptions();
        options.SetInterpreter(
            ProgramLanguage.Python,
            new ProgramInterpreterSpecification(ShellPath, template, compileTemplate));
        return options;
    }

    /// <summary>Creates a scratch directory outside any sandbox workspace and returns its path.</summary>
    internal static string CreateScratchDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "aidotnet-sandbox-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    internal static void TryDelete(string directory)
    {
        try
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
        catch (IOException)
        {
            // Scratch directories are best-effort cleanup.
        }
        catch (UnauthorizedAccessException)
        {
            // Same.
        }
    }
}
