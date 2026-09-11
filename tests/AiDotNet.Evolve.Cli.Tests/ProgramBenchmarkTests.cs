using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class ProgramBenchmarkTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "authored-cli-test-" + Guid.NewGuid().ToString("N"));
    private static string Worker => Path.Combine(AppContext.BaseDirectory, "AiDotNet.CSharp.Worker.dll");
    private static EvolutionEvaluationContext Context => new(0, 1, 1, 1);

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(10, 55)]
    [InlineData(20, 6765)]
    [InlineData(39, 63245986)]
    [InlineData(40, 102334155)]
    public void Independent_reference_matches_known_values(int input, long expected) => Assert.Equal(expected, ProgramBenchmark.Reference(input));

    [Fact]
    public void Reference_and_timing_inputs_are_bounded()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ProgramBenchmark.Reference(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ProgramBenchmark.Reference(41));
        Assert.Equal(2, ProgramBenchmark.Median(new double[] { 3, 1, 2 }));
        Assert.Equal(2.5, ProgramBenchmark.Median(new double[] { 4, 1, 3, 2 }));
        Assert.Throws<ArgumentException>(() => ProgramBenchmark.Median(Array.Empty<double>()));
        foreach (double bad in new[] { double.NaN, double.PositiveInfinity, -1 })
            Assert.Throws<ArgumentException>(() => ProgramBenchmark.Median(new[] { bad }));
        foreach (int[] bad in new[] { Array.Empty<int>(), new int[10], new[] { 41 } })
            Assert.ThrowsAny<ArgumentException>(() => new ProgramBenchmark.MeasuredEvaluator(new Executor(), new(), "search", bad, false, "test"));
    }

    [Fact]
    public async Task Only_exact_catalog_sources_can_reach_the_executor()
    {
        foreach (var known in new[] { ProgramBenchmark.Baseline, ProgramBenchmark.Incorrect, ProgramBenchmark.Iterative }) ProgramBenchmark.RequireCatalog(known);
        var execution = new Executor();
        var evaluator = new ProgramBenchmark.MeasuredEvaluator(execution, new(), "search", new[] { 10 }, false, "test");
        foreach (var bad in new[] { new ProgramGenome(ProgramBenchmark.Baseline.Source + " ", ProgramLanguage.CSharp),
            new ProgramGenome(ProgramBenchmark.Baseline.Source, ProgramLanguage.Python) })
            await Assert.ThrowsAsync<ArgumentException>(async () => await evaluator.EvaluateAsync(bad, Context));
        Assert.Equal(0, execution.Calls);
    }

    [Theory]
    [InlineData(true, false, false)]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(false, false, true)]
    public async Task Evaluator_checks_complete_output_and_keeps_every_dispatch(bool correctness, bool truncated, bool failed)
    {
        var execution = new Executor { Truncated = truncated, Failed = failed };
        var run = new ProgramBenchmark.Run();
        int[] inputs = { 10, 20 };
        var evaluator = new ProgramBenchmark.MeasuredEvaluator(execution, run, "test", inputs, correctness, "test");
        inputs[0] = 40; // Caller mutation cannot change the versioned workload.
        var result = await evaluator.EvaluateAsync(ProgramBenchmark.Baseline, Context);
        Assert.Equal(2, execution.Calls);
        Assert.Equal(2, result.CostUnits);
        Assert.Equal(10, run.Samples[0].Input);
        Assert.All(run.Samples, sample => Assert.True(sample.ElapsedMilliseconds >= 0));
        Assert.All(run.Samples, sample => Assert.Equal(truncated, sample.StdOutTruncated));
        Assert.All(run.Samples, sample => Assert.Equal(failed ? 1 : 0, sample.ExitCode));
        if (truncated || failed) Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        else
        {
            Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
            Assert.Equal(correctness ? EvolutionOptimizationDirection.Maximize : EvolutionOptimizationDirection.Minimize, result.Direction);
            if (correctness) Assert.Equal(1, result.Quality);
        }
    }

    [Fact]
    public async Task Incorrect_program_scores_zero_in_correctness_and_cannot_receive_timing_fitness()
    {
        var execution = new Executor();
        var run = new ProgramBenchmark.Run();
        var correctness = new ProgramBenchmark.MeasuredEvaluator(execution, run, "correctness", new[] { 0, 1 }, true, "test");
        var result = await correctness.EvaluateAsync(ProgramBenchmark.Incorrect, Context);
        Assert.Equal(0, result.Quality);
        Assert.All(run.Samples, sample => Assert.Equal("failed", sample.Status));
        var timing = new ProgramBenchmark.MeasuredEvaluator(execution, run, "search", new[] { 39 }, false, "test");
        Assert.Equal(EvolutionEvaluationStatus.Failed, (await timing.EvaluateAsync(ProgramBenchmark.Incorrect, Context)).Status);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public async Task Cancellation_before_during_and_between_calls_keeps_costs(int stage)
    {
        using var cancellation = new CancellationTokenSource();
        var execution = new Executor { Cancel = stage == 0 ? null : cancellation, ThrowCancellation = stage == 1 };
        var run = new ProgramBenchmark.Run();
        var evaluator = new ProgramBenchmark.MeasuredEvaluator(execution, run, "search", stage == 3 ? new[] { 39 } : new[] { 39, 39 }, false, "test");
        if (stage == 0)
        {
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await evaluator.EvaluateAsync(ProgramBenchmark.Baseline, Context, cancellation.Token));
            Assert.Equal(0, execution.Calls);
        }
        else
        {
            var result = await evaluator.EvaluateAsync(ProgramBenchmark.Baseline, Context, cancellation.Token);
            Assert.Equal(EvolutionEvaluationStatus.Canceled, result.Status);
            Assert.Equal(1, result.CostUnits);
            Assert.Single(run.Samples);
        }
    }

    [Fact]
    public async Task Unexpected_executor_failure_remains_unknown_and_charges_the_reserved_maximum()
    {
        var run = new ProgramBenchmark.Run();
        var evaluator = new ProgramBenchmark.MeasuredEvaluator(new Executor { ThrowFailure = true }, run, "search", new[] { 39, 39 }, false, "test");
        var ledger = new EvolutionResourceLedger("unknown", EvolutionResources.Of("cost_units", 10));
        var metered = new ResourceMeteredEvolutionTask<ProgramGenome>(new ProgramEvolutionTask(evaluator), ledger, new[] { 2m });
        var genome = ProgramBenchmark.Baseline;
        var candidate = new EvolutionCandidate<ProgramGenome>(0, new(genome, genome.Id), new(null, null, "seed", null, 0, 0, 1));
        Assert.Equal(EvolutionEvaluationStatus.Failed, (await metered.EvaluateAsync(candidate, Context)).Status);
        Assert.Equal(1, ledger.Snapshot().Unknown);
        Assert.Equal(2m, ledger.Snapshot().Spent["cost_units"]);
        Assert.Equal("started-unknown", Assert.Single(run.Samples).Status);
    }

    [Fact]
    public void Workload_measurement_and_binary_changes_invalidate_the_evaluator_version()
    {
        string Key(string phase, int input, bool correctness, string binaries) =>
            new ProgramBenchmark.MeasuredEvaluator(new Executor(), new(), phase, new[] { input }, correctness, binaries).VersionHash;
        string baseline = Key("search", 39, false, "A");
        Assert.Equal(baseline, Key("search", 39, false, "A"));
        foreach (string changed in new[] { Key("confirmation", 39, false, "A"), Key("search", 40, false, "A"),
            Key("search", 39, true, "A"), Key("search", 39, false, "B") }) Assert.NotEqual(baseline, changed);
    }

    [Theory]
    [InlineData("--runs", "0")]
    [InlineData("--runs", "13")]
    [InlineData("--runs", "bad")]
    [InlineData("--measurements", "0")]
    [InlineData("--measurements", "10")]
    public async Task Cli_rejects_invalid_budgets_before_creating_output(string option, string value)
    {
        var code = await EvolveCommandLine.ExecuteAsync(new[] { "benchmark-program", "--worker", Worker, "--output", _directory, option, value }, new StringWriter(), new StringWriter());
        Assert.Equal(EvolveCommandLine.ExitUsage, code);
        Assert.False(Directory.Exists(_directory));
    }

    [Fact]
    public void Worker_paths_cannot_inject_shell_or_template_content()
    {
        Assert.Equal(Path.GetFullPath(Worker), ProgramBenchmark.RequireWorker(Worker));
        foreach (string bad in new[] { "relative.dll", Worker + "$x", Worker + "`x", Worker + "\"x", Worker + "{source}", Worker + "\nx", Path.Combine(_directory, "other.dll") })
            Assert.Throws<ArgumentException>(() => ProgramBenchmark.RequireWorker(bad));
    }

    [Fact]
    public async Task Precancelled_run_retains_all_planned_runs_and_no_execution()
    {
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        Assert.Equal(EvolveCommandLine.ExitCancelled, await ProgramBenchmark.RunAsync(Worker, _directory, 2, 1, new StringWriter(), cancellation.Token));
        var report = JObject.Parse(File.ReadAllText(Path.Combine(_directory, "report.json")));
        Assert.True((bool)report["BinariesUnchanged"]!);
        Assert.Equal("Microsoft.NETCore.App/10.0.0", (string?)report["WorkerRequestedFramework"]);
        Assert.Equal(2, report["Runs"]!.Count());
        Assert.All(report["Runs"]!, run => { Assert.Equal("not-started", (string?)run["Status"]); Assert.Empty(run["Samples"]!); });
        Assert.True(File.Exists(Path.Combine(_directory, "plan.json")));
        await Assert.ThrowsAsync<ArgumentException>(() => ProgramBenchmark.RunAsync(Worker, _directory, 1, 1, new StringWriter(), CancellationToken.None));
    }

    [Fact]
    public void Evidence_is_write_once_even_when_a_destination_already_exists()
    {
        Directory.CreateDirectory(_directory);
        string path = Path.Combine(_directory, "report.json");
        ProgramBenchmark.WriteNew(path, new { value = 1 });
        Assert.Throws<IOException>(() => ProgramBenchmark.WriteNew(path, new { value = 2 }));
        Assert.Equal(1, (int)JObject.Parse(File.ReadAllText(path))["value"]!);
        Assert.True(File.Exists(path + ".pending"));
        Assert.False(ProgramBenchmark.MatchesHash(path, "wrong"));
        Assert.False(ProgramBenchmark.MatchesHash(path + ".missing", "wrong"));
    }

    [Theory]
    [InlineData("search-failed")]
    [InlineData("confirmation-failed")]
    [InlineData("failed")]
    [InlineData("cancelled")]
    public async Task Run_failures_retain_status_and_resource_receipts(string expected)
    {
        var run = new ProgramBenchmark.Run();
        using var cancellation = new CancellationTokenSource();
        IProgramExecutionEngine Factory(ProgramSandboxOptions options)
        {
            Assert.Equal(1, options.Limits.MaxConcurrentExecutions);
            if (expected == "failed") throw new InvalidOperationException("PRIVATE_PAYLOAD");
            if (expected == "cancelled") throw new OperationCanceledException();
            return new Executor { Failed = expected == "search-failed", FailConfirmation = expected == "confirmation-failed" };
        }
        await ProgramBenchmark.ExecuteRun(run, Worker, "dotnet", _directory, 1, "test", Factory, cancellation.Token);
        Assert.Equal(expected, run.Status);
        Assert.NotNull(run.Resources);
        Assert.DoesNotContain("PRIVATE_PAYLOAD", Newtonsoft.Json.JsonConvert.SerializeObject(run));
    }

    [Fact]
    public async Task Cancellation_during_search_preserves_raw_receipts_and_does_not_start_confirmation()
    {
        var run = new ProgramBenchmark.Run();
        using var cancellation = new CancellationTokenSource();
        var execution = new Executor { Cancel = cancellation, CancelAtCall = 14 };
        await ProgramBenchmark.ExecuteRun(run, Worker, "dotnet", _directory, 1, "test", _ => execution, cancellation.Token);
        Assert.Equal("cancelled", run.Status);
        Assert.Equal(14, run.Samples.Count);
        Assert.NotNull(run.Resources);
        Assert.Empty(run.Confirmation);
    }

    [Fact]
    public async Task Real_worker_pilot_rejects_bad_candidate_and_meters_search_and_confirmation()
    {
        Assert.Equal(EvolveCommandLine.ExitSuccess, await ProgramBenchmark.RunAsync(Worker, _directory, 2, 1, new StringWriter(), CancellationToken.None));
        var report = JObject.Parse(File.ReadAllText(Path.Combine(_directory, "report.json")));
        Assert.Equal(ProgramBenchmark.Unit, (string?)report["CostUnit"]);
        Assert.Contains("no LLM", (string)report["Limitations"]!);
        foreach (var run in report["Runs"]!)
        {
            Assert.Equal("completed", (string?)run["Status"]);
            Assert.Equal("Minimize", (string?)run["Search"]!["Direction"]);
            Assert.Equal(0, (long)run["Search"]!["LlmUsage"]!["ChatCalls"]!);
            Assert.Equal(16, run["Samples"]!.Count());
            Assert.Equal(16m, (decimal)run["Resources"]!["Spent"]!["cost_units"]!);
            Assert.Equal(0, (int)run["Resources"]!["Unknown"]!);
            Assert.DoesNotContain(run["Samples"]!, sample => (string?)sample["Phase"] == "search" && (string?)sample["GenomeId"] == ProgramBenchmark.Incorrect.Id);
            Assert.Equal((int)run["Index"]! % 2 == 0 ? "baseline" : "winner", (string?)run["Confirmation"]![0]!["Role"]);
        }
        // No timing ordering assertion: shared-host wall time is not an algorithm-speed proof.
    }

    [Fact]
    public void Worker_runtime_configuration_is_validated_not_assumed_from_its_filename()
    {
        Assert.Equal("Microsoft.NETCore.App/10.0.0", ProgramBenchmark.ReadWorkerTarget(Path.ChangeExtension(Worker, ".runtimeconfig.json")));
        Directory.CreateDirectory(_directory);
        string path = Path.Combine(_directory, "worker.runtimeconfig.json");
        foreach (string invalid in new[] { "{}", "{", "{\"runtimeOptions\":1}", "{\"a\":1,\"a\":2}", new string(' ', 16385),
            "{\"runtimeOptions\":{\"configProperties\":{\"System.GC.HeapHardLimit\":1}}}" })
        {
            File.WriteAllText(path, invalid);
            Assert.Throws<ArgumentException>(() => ProgramBenchmark.ReadWorkerTarget(path));
        }
        string valid = File.ReadAllText(Path.ChangeExtension(Worker, ".runtimeconfig.json"));
        foreach (string invalid in new[] { valid + "{}", valid.Replace("10.0.0", "...", StringComparison.Ordinal),
            valid.Replace("268435456", "26843545600000000000000000000000000000", StringComparison.Ordinal) })
        {
            File.WriteAllText(path, invalid);
            Assert.Throws<ArgumentException>(() => ProgramBenchmark.ReadWorkerTarget(path));
        }
    }

    private sealed class Executor : IProgramExecutionEngine
    {
        internal int Calls;
        internal bool Truncated, Failed, ThrowCancellation, ThrowFailure, FailConfirmation;
        internal int CancelAtCall = 1;
        internal CancellationTokenSource? Cancel;
        public Task<ProgramExecuteResponse> ExecuteAsync(ProgramExecuteRequest request, CancellationToken cancellationToken = default)
        {
            Calls++;
            if (Calls == CancelAtCall) Cancel?.Cancel();
            if (ThrowCancellation) throw new OperationCanceledException();
            if (ThrowFailure) throw new InvalidOperationException("PRIVATE_PAYLOAD");
            return Task.FromResult(new ProgramExecuteResponse
            {
                Success = !Failed && !(FailConfirmation && request.StdIn == "40"),
                Language = request.Language,
                ExitCode = Failed ? 1 : 0,
                StdOutTruncated = Truncated,
                StdOut = request.SourceCode == ProgramBenchmark.Incorrect.Source ? "-1" : ProgramBenchmark.Reference(int.Parse(request.StdIn!, CultureInfo.InvariantCulture)).ToString(CultureInfo.InvariantCulture)
            });
        }
        public bool TryExecute(ProgramLanguage language, string sourceCode, string input, out string output, out string? errorMessage,
            CancellationToken cancellationToken = default) => throw new NotSupportedException("Only the asynchronous contract is used.");
    }
    public void Dispose() { if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true); }
}
