using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Evolution.CSharp.Tests;

// Only authored fixtures are executed. This process supervisor is not a sandbox for model-generated programs.
public sealed class CSharpWorkerTests
{
    [Fact]
    public async Task Facade_compiles_and_executes_a_scripted_patch_with_real_correctness_and_fitness_calls()
    {
        const string source = "public static class P { public static int F(int n) { return n * 1; } public static void Main() { System.Console.Write(F(int.Parse(System.Console.ReadLine()!))); } }";
        string directory = Path.Combine(Path.GetTempPath(), "aidotnet-worker-facade-" + Guid.NewGuid().ToString("N"));
        var program = CompilerTestSupport.Program();
        program.SeedPrograms.Add(source);
        program.TestCases.Add(new ProgramInputOutputExample { Input = "1", ExpectedOutput = "2" });
        program.Engine.MaxEvaluationAttempts = 2;
        program.Engine.MaxProposals = 2;
        program.Engine.MaxGenerations = 1;
        program.Engine.ProposalBatchSize = 1;
        var compiler = CompilerTestSupport.Options(Path.Combine(directory, "audit"));
        compiler.ReferencePaths = new[] { typeof(object).Assembly.Location, typeof(Console).Assembly.Location,
            Path.Combine(RuntimeEnvironment.GetRuntimeDirectory(), "System.Runtime.dll") };
        var sandbox = new ProgramSandboxOptions { WorkingDirectory = Path.Combine(directory, "executions") };
        sandbox.Limits.TimeLimitSeconds = 30;
        sandbox.Limits.MemoryLimitMb = 4096; // Includes CLR virtual address reservations on POSIX; not a benchmark budget.
        string command = "\"" + Path.Combine(AppContext.BaseDirectory, "AiDotNet.CSharp.Worker.dll") + "\" --source {source}";
        sandbox.SetInterpreter(ProgramLanguage.CSharp, new ProgramInterpreterSpecification(DotnetHost(), command, command + " --compile-only"));
        var ledger = CompilerTestSupport.Ledger();
        try
        {
            using var execution = new ProcessProgramExecutionEngine(sandbox);
            var correctness = new InputOutputProgramFitnessEvaluator(execution,
                new[] { new ProgramInputOutputExample { Input = "0", ExpectedOutput = "0" } });
            var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureProgramExecutionEngine(execution).ConfigureProgramCorrectness(correctness)
                .ConfigureCSharpProgramEvolution(new ScriptedClient(), program, compiler,
                    new ProgramEvolutionResourceOptions(ledger, 2, compiler.CostUnitVersionHash)).BuildAsync();
            Assert.Equal(source.Replace("n * 1", "n * 2", StringComparison.Ordinal), result.ProgramEvolution?.BestProgram?.Source);
            Assert.Equal(1, result.ProgramEvolution?.BestQuality);
            Assert.Equal(1, result.ProgramEvolution?.LlmUsage.ChatCalls);
            var evaluations = ledger.Snapshot().Receipts.Where(receipt => receipt.Stage == EvolutionResourceStage.Evaluation).ToArray();
            Assert.Equal(2, evaluations.Length);
            Assert.All(evaluations, receipt => Assert.Equal(2m, receipt.Charged["cost_units"]));
            Assert.Equal(0, ledger.Snapshot().Unknown);
            Assert.False(ledger.Snapshot().MaximumViolated);
            Assert.Empty(Directory.GetDirectories(sandbox.WorkingDirectory));
        }
        finally
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Every_emit_write_path_honors_the_image_limit()
    {
        using var image = new global::AiDotNet.CSharp.Worker.Program.BoundedImageStream();
        const int maximum = 8 * 1024 * 1024;
        image.SetLength(maximum);
        Assert.Throws<InvalidDataException>(() => image.SetLength(maximum + 1L));
        image.Position = maximum - 1;
        image.WriteByte(1);
        Assert.Throws<InvalidDataException>(() => image.WriteByte(1));
        Assert.Throws<InvalidDataException>(() => image.Write(new byte[1], 0, 1));
        Assert.Throws<InvalidDataException>(() => image.Write(new ReadOnlySpan<byte>(new byte[1])));
        image.Position = maximum - 1;
        image.Write(new ReadOnlySpan<byte>(new byte[1]));
        image.Position = maximum - 1;
        image.Write(new byte[1], 0, 1);
        Assert.Equal(maximum, image.Length);
    }

    [Theory]
    [InlineData("System.Console.Write(System.Console.ReadLine());", "echo", 0, "echo")]
    [InlineData("class P { static int Main(string[] args) { System.Console.Write(args.Length); return 7; } }", "", 7, "0")]
    [InlineData("class P { static async System.Threading.Tasks.Task<int> Main() { await System.Threading.Tasks.Task.Yield(); return 9; } }", "", 9, "")]
    [InlineData("class P { static async System.Threading.Tasks.Task Main() { await System.Threading.Tasks.Task.Yield(); System.Console.Write(42); } }", "", 0, "42")]
    public async Task Real_compilation_execution_input_and_exit_codes(string source, string input, int exitCode, string output)
    {
        var result = await Run(source, input);
        Assert.Equal(exitCode, result.Exit);
        Assert.Equal(output, result.Output);
        Assert.Empty(result.Error);
    }

    [Fact]
    public async Task Compile_only_cannot_run_module_initializers_or_main()
    {
        const string source = "using System; using System.Runtime.CompilerServices; class P { [ModuleInitializer] internal static void Init() => throw new Exception(\"MUST_NOT_RUN\"); static void Main() => Console.Write(\"MUST_NOT_RUN\"); }";
        var compiled = await Run(source, compileOnly: true);
        Assert.Equal(0, compiled.Exit);
        Assert.Empty(compiled.Output);
        Assert.Empty(compiled.Error);
        var executed = await Run(source);
        Assert.Equal(70, executed.Exit);
        Assert.DoesNotContain("MUST_NOT_RUN", executed.Error, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("class P { static void Main() { int n = \"PRIVATE_PAYLOAD\"; } }")]
    [InlineData("class P { static void Main() { unsafe { int* p = null; } } }")]
    [InlineData("class P { static void Main( { }")]
    [InlineData("public class NotAConsoleApplication { }")]
    public async Task Invalid_programs_fail_with_bounded_physical_diagnostics(string source)
    {
        var result = await Run(source, compileOnly: true);
        Assert.Equal(65, result.Exit);
        Assert.Empty(result.Output);
        Assert.DoesNotContain("PRIVATE_PAYLOAD", result.Error, StringComparison.Ordinal);
        var lines = result.Error.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.InRange(lines.Length, 1, 8);
        Assert.All(lines, line => Assert.Matches("^CS[0-9]+@[0-9]+:[0-9]+\\r?$", line));
    }

    [Fact]
    public async Task Runtime_exceptions_do_not_disclose_candidate_messages_or_stack_traces()
    {
        var result = await Run("throw new System.Exception(\"PRIVATE_PAYLOAD\");");
        Assert.Equal(70, result.Exit);
        Assert.Equal("C# worker failed.", result.Error.Trim());
    }

    [Theory]
    [InlineData(65537)]
    [InlineData(262145)]
    public async Task Oversized_sources_are_rejected_before_execution(int length)
    {
        var result = await Run(new string(' ', length));
        Assert.Equal(70, result.Exit);
        Assert.Empty(result.Output);
    }

    [Fact]
    public async Task Empty_source_is_rejected()
    {
        Assert.Equal(70, (await Run("")).Exit);
    }

    [Fact]
    public async Task Malformed_utf8_is_rejected()
    {
        Assert.Equal(70, (await Run("", bytes: new byte[] { 0xc3, 0x28 })).Exit);
    }

    [Fact]
    public async Task Unknown_options_fail_closed()
    {
        Assert.Equal(64, (await Run("System.Console.Write(42);", extra: "--unknown")).Exit);
    }

    private static async Task<(int Exit, string Output, string Error)> Run(
        string source, string input = "", bool compileOnly = false, byte[]? bytes = null, string? extra = null)
    {
        string worker = Path.Combine(AppContext.BaseDirectory, "AiDotNet.CSharp.Worker.dll");
        Assert.True(File.Exists(worker), "Build the worker project reference before running its integration tests.");
        string host = DotnetHost();
        string directory = Path.Combine(Path.GetTempPath(), "aidotnet-worker-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        using var process = new Process();
        try
        {
            string path = Path.Combine(directory, "candidate with spaces.cs");
            await File.WriteAllBytesAsync(path, bytes ?? new UTF8Encoding(false, true).GetBytes(source));
            var start = new ProcessStartInfo(host)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = directory,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            start.ArgumentList.Add(worker);
            start.ArgumentList.Add("--source");
            start.ArgumentList.Add(path);
            if (compileOnly) start.ArgumentList.Add("--compile-only");
            if (extra is not null) start.ArgumentList.Add(extra);
            start.Environment.Clear();
            if (OperatingSystem.IsWindows()) start.Environment["SystemRoot"] = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            start.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";
            process.StartInfo = start;
            Assert.True(process.Start());
            Task<string> output = process.StandardOutput.ReadToEndAsync();
            Task<string> error = process.StandardError.ReadToEndAsync();
            await process.StandardInput.WriteAsync(input);
            process.StandardInput.Close();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(45));
            await process.WaitForExitAsync(timeout.Token);
            return (process.ExitCode, await output, await error);
        }
        finally
        {
            try { if (process.Id > 0 && !process.HasExited) { process.Kill(entireProcessTree: true); await process.WaitForExitAsync(); } }
            catch (InvalidOperationException) { }
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }

    private static string DotnetHost() => Path.GetFullPath(Path.Combine(RuntimeEnvironment.GetRuntimeDirectory(), "..", "..", "..",
        OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet"));
}
