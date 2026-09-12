using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class RunInspectionTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "cli-inspection-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task PipePauseDrainsTheRealBatchAndVerifiesItsCheckpointBeforeClaimingPaused()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var control = new EvolutionRunControl();
        using var cancellation = new CancellationTokenSource();
        var inspection = new RunInspection(control, cancellation);
        string session = Guid.NewGuid().ToString("N");
        await using var service = new LocalRunControl(session, inspection.Handle);
        int calls = 0;
        var builder = Builder(control, inspection, async () =>
        {
            if (Interlocked.Increment(ref calls) == 1)
            {
                entered.TrySetResult();
                await release.Task.WaitAsync(TimeSpan.FromSeconds(10));
            }
        });
        Task<AiDotNet.Models.Results.AiModelResult<double, Matrix<double>, Vector<double>>> run = builder.BuildAsync(cancellation.Token);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(10));
            var response = JObject.Parse(await LocalRunControl.SendAsync(session, "pause"));
            Assert.Equal("stop-requested", (string?)response["State"]);
            Assert.False((bool)response["Resumable"]!);
            Assert.False(cancellation.IsCancellationRequested);
        }
        finally { release.TrySetResult(); }
        var result = await run;
        Assert.Equal(2, calls);
        await inspection.FinishAsync(result.EvolutionSummary);
        var stopped = inspection.Read();
        Assert.Equal("paused", stopped.State);
        Assert.True(stopped.Resumable);
        Assert.NotNull(stopped.VerifiedCheckpointSha256);
        Assert.Equal(2, stopped.SegmentEvaluationAttempts);
        Assert.Equal(2, stopped.SegmentReportedCostUnits);
        Assert.Equal(result.EvolutionSummary!.ArchiveCount, stopped.ArchiveCount);
        Assert.NotNull(stopped.BestFeasible);
        Assert.Null(stopped.BackendQueueDepth);
        Assert.DoesNotContain("authored-private-operator", await LocalRunControl.SendAsync(session, "inspect"));

        var nextControl = new EvolutionRunControl();
        using var nextCancellation = new CancellationTokenSource();
        var resumed = new RunInspection(nextControl, nextCancellation);
        var next = Builder(nextControl, resumed, () => { calls++; return Task.CompletedTask; }, resume: true);
        var nextResult = await next.BuildAsync();
        await resumed.FinishAsync(nextResult.EvolutionSummary);
        Assert.Equal(4, calls);
        Assert.Equal(4, nextResult.EvolutionSummary!.CompletedEvaluations);
        Assert.Equal(2, resumed.Read().SegmentEvaluationAttempts); // Do not mislabel this segment as lifetime totals.
        Assert.Equal("stopped", resumed.Read().State);
    }

    [Fact]
    public async Task MissingCheckpointDoesNotBecomePausedAndAbortedWorkDoesNotBecomeZeroCost()
    {
        var control = new EvolutionRunControl();
        using var cancellation = new CancellationTokenSource();
        var inspection = new RunInspection(control, cancellation);
        inspection.Handle("pause");
        await inspection.FinishAsync(new AiDotNet.Models.Results.EvolutionRunSummary { StopReason = EvolutionStopReason.Canceled });
        Assert.Equal("stopped", inspection.Read().State);
        Assert.False(inspection.Read().Resumable);
        await inspection.FinishAsync(null);
        Assert.True(inspection.Read().UnknownConsumption);
        Assert.Null(inspection.Read().SegmentReportedCostUnits);
    }

    [Fact]
    public async Task SnapshotCollectionsCannotMutateSubsequentReadsAndCorruptCheckpointIsNotResumable()
    {
        var control = new EvolutionRunControl();
        using var cancellation = new CancellationTokenSource();
        var inspection = new RunInspection(control, cancellation);
        var result = await Builder(control, inspection, () => Task.CompletedTask).BuildAsync();
        var snapshot = inspection.Read();
        snapshot.IslandOccupancy[0] = 12345;
        ((IDictionary<string, long>)snapshot.SegmentStatuses).Clear();
        Assert.NotEqual(12345, inspection.Read().IslandOccupancy[0]);
        Assert.NotEmpty(inspection.Read().SegmentStatuses);
        File.WriteAllText(result.EvolutionSummary!.CheckpointPath!, "authored corruption");
        await inspection.FinishAsync(result.EvolutionSummary);
        Assert.False(inspection.Read().Resumable);
    }

    private AiModelBuilder<double, Matrix<double>, Vector<double>> Builder(EvolutionRunControl control,
        RunInspection inspection, Func<Task> measure, bool resume = false)
    {
        var programs = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.CSharp, CustomVariation = new Variation(),
            CustomFitnessEvaluator = new DelegateProgramFitnessEvaluator(async (_, _, _) =>
            {
                await measure();
                return new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 1);
            })
        };
        programs.SeedPrograms.Add("return 1;");
        programs.SeedPrograms.Add("return 2;");
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .WithEvolutionControl(control).ObserveProgramEvolution(inspection);
        builder.ConfigureProgramEvolution(programs);
        builder.ConfigureEvolution(new EvolutionOptions
        {
            RunId = "inspection", Seed = 7, OutputDirectory = _root, CheckpointInterval = 1,
            MaxEvaluationAttempts = 4, MaxProposals = 8, MaxGenerations = 4, ProposalBatchSize = 2, Resume = resume
        });
        return builder;
    }

    private sealed class Variation : IProgramVariationOperator
    {
        public string Id => "authored-private-operator";
        public string VersionHash => "inspection-v1";
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) =>
            new(new ProgramGenome("return " + (3 + context.Random.NextInt(65536)) + ";", ProgramLanguage.CSharp));
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public ProgramEvolutionLlmUsage GetUsage() => new();
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); // This test instance's own GUID directory only.
    }
}
