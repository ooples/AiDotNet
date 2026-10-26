// The tooling assembly these tests load by reflection is referenced only for net10.0, so on the compatibility
// frameworks they compiled and then failed at run time with a missing assembly. The guard now matches the reference.
#if NET10_0_OR_GREATER
using System.Reflection;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class ProgramSynthesisToolingCliTests
{
    [Fact(Timeout = 60000)]
    public async Task Main_WithHelp_ReturnsZero()
    {
        var (exitCode, stdOut, stdErr) = await InvokeAsync(Array.Empty<string>());
        Assert.Equal(0, exitCode);
        Assert.Contains("Usage:", stdOut);
        Assert.True(string.IsNullOrWhiteSpace(stdErr));
    }

    [Fact(Timeout = 60000)]
    public async Task Main_WithUnknownCommand_ReturnsOne()
    {
        var (exitCode, _, stdErr) = await InvokeAsync(new[] { "nope" });
        Assert.Equal(1, exitCode);
        Assert.Contains("Unknown command", stdErr);
    }

    [Fact(Timeout = 60000)]
    public async Task Main_Train_WithEmptyDataset_ReturnsOne()
    {
        var trainPath = Path.GetTempFileName();
        var outputPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "model.model");

        try
        {
            var (exitCode, _, stdErr) = await InvokeAsync(new[]
            {
                "train",
                "--train", trainPath,
                "--output", outputPath,
                "--epochs", "1",
                "--seed", "1"
            });

            Assert.Equal(1, exitCode);
            Assert.Contains("No training samples found", stdErr);
        }
        finally
        {
            try { File.Delete(trainPath); } catch (System.Exception bestEffortError) { System.Diagnostics.Trace.TraceWarning($"ProgramSynthesisToolingCliTests: a best-effort step failed ({bestEffortError.GetType().Name}: {bestEffortError.Message})"); }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Main_Evaluate_WithMissingModel_ReturnsOne()
    {
        var missingModel = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "missing.model");
        var reportPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "report.json");

        var (exitCode, _, stdErr) = await InvokeAsync(new[]
        {
            "evaluate",
            "--model", missingModel,
            "--report", reportPath
        });

        Assert.Equal(1, exitCode);
        Assert.Contains("Model file not found", stdErr);
    }

    private static async Task<(int ExitCode, string StdOut, string StdErr)> InvokeAsync(string[] args)
    {
        var programType = Type.GetType("AiDotNet.ProgramSynthesis.Tooling.Program, AiDotNet.ProgramSynthesis.Tooling", throwOnError: true);
        var main = (programType ?? throw new InvalidOperationException("programType is null where it is required.")).GetMethod("Main", BindingFlags.Public | BindingFlags.Static);
        Assert.NotNull(main);

        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var stdOutWriter = new StringWriter();
        var stdErrWriter = new StringWriter();

        try
        {
            Console.SetOut(stdOutWriter);
            Console.SetError(stdErrWriter);

            var task = main.Invoke(null, new object[] { args }) as Task<int>
            ?? throw new InvalidOperationException("The CLI entry point did not return a Task<int>.");
            var exitCode = await (task ?? throw new InvalidOperationException("task is null where it is required.")).ConfigureAwait(false);
            return (exitCode, stdOutWriter.ToString(), stdErrWriter.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalErr);
        }
    }
}
#endif

