using System.Globalization;
using System.Runtime;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionEngineTests
{
    [Fact]
    public async Task SeedPopulationLargerThanBudgetNeverOverspends()
    {
        var task = new SyntheticEvolutionTask();
        EvolutionEngine<TestGenome> engine = CreateEngine(task, Options(maxAttempts: 3, batchSize: 3, workers: 2));

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(
            Enumerable.Range(1, 10).Select(value => new TestGenome(value)));

        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, result.StopReason);
        Assert.Equal(3, task.Calls);
        Assert.Equal(3, result.Counters.EvaluationAttempts);
        Assert.Equal(3, result.Counters.CompletedEvaluations);
    }

    [Fact]
    public async Task DuplicateCanonicalGenomeDoesNotConsumeEvaluatorBudget()
    {
        var task = new SyntheticEvolutionTask();
        EvolutionEngine<TestGenome> engine = CreateEngine(task, Options(maxAttempts: 2, batchSize: 3, workers: 2));

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[]
        {
            new TestGenome(1), new TestGenome(1), new TestGenome(2)
        });

        Assert.Equal(2, task.Calls);
        Assert.Equal(1, result.Counters.StatusCounts[EvolutionEvaluationStatus.Duplicate]);
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Completed]);
    }

    [Fact]
    public async Task DeterministicStateDoesNotDependOnWorkerCountOrCompletionOrder()
    {
        TestGenome[] seeds = Enumerable.Range(1, 12).Select(value => new TestGenome(value)).ToArray();
        var sequentialTask = new SyntheticEvolutionTask(delayScale: 1);
        var parallelTask = new SyntheticEvolutionTask(delayScale: 1);

        EvolutionRunResult<TestGenome> sequential = await CreateEngine(sequentialTask,
            Options(maxAttempts: 12, batchSize: 6, workers: 1)).RunAsync(seeds);
        EvolutionRunResult<TestGenome> parallel = await CreateEngine(parallelTask,
            Options(maxAttempts: 12, batchSize: 6, workers: 4)).RunAsync(seeds);

        Assert.Equal(sequential.StateHash, parallel.StateHash);
        Assert.Equal(sequential.Islands.SelectMany(item => item.Entries).Select(item => item.Evaluation.GenomeId),
            parallel.Islands.SelectMany(item => item.Entries).Select(item => item.Evaluation.GenomeId));
        Assert.Equal(1, sequentialTask.MaxConcurrency);
        Assert.InRange(parallelTask.MaxConcurrency, 2, 4);
    }

    [Fact]
    public async Task CandidateFailureIsIsolatedByDefault()
    {
        var task = new SyntheticEvolutionTask(throwOnValue: 3);
        EvolutionEngine<TestGenome> engine = CreateEngine(task, Options(maxAttempts: 4, batchSize: 4, workers: 2));

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(
            Enumerable.Range(1, 4).Select(value => new TestGenome(value)));

        Assert.Equal(4, task.Calls);
        Assert.Equal(3, result.Counters.StatusCounts[EvolutionEvaluationStatus.Completed]);
        Assert.Equal(1, result.Counters.StatusCounts[EvolutionEvaluationStatus.Failed]);
        Assert.NotNull(result.Best);
    }

    [Fact]
    public async Task RetriesAreChargedExactlyAndRemainBounded()
    {
        var task = new FailOnceEvolutionTask();
        EvolutionEngineOptions options = Options(maxAttempts: 4, batchSize: 2, workers: 2);
        options.MaxRetries = 1;
        EvolutionEngine<TestGenome> engine = new(task, new IncrementVariation(),
            _ => TestArchive(), options);

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[] { new TestGenome(1), new TestGenome(2) });

        Assert.Equal(4, task.Calls);
        Assert.Equal(4, result.Counters.EvaluationAttempts);
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Completed]);
        Assert.All(result.Islands.SelectMany(island => island.Entries),
            entry =>
            {
                Assert.Equal(2, entry.Evaluation.Cost.AttemptCount);
                Assert.Equal(3, entry.Evaluation.Cost.CostUnits);
                Assert.Contains(entry.Evaluation.Diagnostics, diagnostic => diagnostic.Code == "first_attempt");
            });
    }

    [Fact]
    public async Task TaskReturnedCancellationIsRetriedAndFullyAccounted()
    {
        var task = new CancelOnceEvolutionTask();
        EvolutionEngineOptions options = Options(maxAttempts: 2, batchSize: 1, workers: 1);
        options.MaxRetries = 1;
        options.DeduplicateFailedCandidates = false;
        EvolutionEngine<TestGenome> engine = new(task, new IncrementVariation(),
            _ => TestArchive(), options);

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[] { new TestGenome(1) });
        EvolutionArchiveEntry<TestGenome> entry = Assert.IsType<EvolutionArchiveEntry<TestGenome>>(result.Best);

        Assert.Equal(2, task.Calls);
        Assert.Equal(2, result.Counters.EvaluationAttempts);
        Assert.Equal(3, entry.Evaluation.Cost.CostUnits);
        Assert.Equal(2, entry.Evaluation.Cost.AttemptCount);
        Assert.Contains(entry.Evaluation.Diagnostics, diagnostic => diagnostic.Code == "cooperative_cancel");
    }

    [Fact]
    public async Task ObserverFailuresCannotChangeDeterministicState()
    {
        TestGenome[] seeds = Enumerable.Range(1, 6).Select(value => new TestGenome(value)).ToArray();
        EvolutionEngineOptions baselineOptions = Options(6, 3, 2);
        EvolutionRunResult<TestGenome> baseline = await CreateEngine(new SyntheticEvolutionTask(), baselineOptions)
            .RunAsync(seeds);

        EvolutionEngineOptions observedOptions = Options(6, 3, 2);
        var observedEngine = new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(), new IncrementVariation(),
            _ => TestArchive(), observedOptions, observer: new ThrowingEvolutionObserver());
        EvolutionRunResult<TestGenome> observed = await observedEngine.RunAsync(seeds);

        Assert.Equal(baseline.StateHash, observed.StateHash);
        Assert.Equal(baseline.Counters.StatusCounts, observed.Counters.StatusCounts);
    }

    [Fact]
    public async Task ZeroEvaluationBudgetDoesNotDispatchSeeds()
    {
        var task = new SyntheticEvolutionTask();
        EvolutionRunResult<TestGenome> result = await CreateEngine(task, Options(0, 2, 2))
            .RunAsync(new[] { new TestGenome(1), new TestGenome(2) });

        Assert.Equal(0, task.Calls);
        Assert.Equal(0, result.Counters.Proposals);
        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, result.StopReason);
    }

    [Fact]
    public async Task SelectionThatCannotProduceCandidateStopsWithoutSpinning()
    {
        var task = new SyntheticEvolutionTask();
        EvolutionEngineOptions options = Options(2, 1, 1);
        var engine = new EvolutionEngine<TestGenome>(task, new IncrementVariation(),
            _ => TestArchive(), options, selection: new NullEvolutionSelectionPolicy());

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[] { new TestGenome(1) });

        Assert.Equal(EvolutionStopReason.NoCandidates, result.StopReason);
        Assert.Equal(1, task.Calls);
        Assert.Equal(1, result.Counters.Proposals);
    }

    [Fact]
    public async Task RunResultExposesImmutableArchiveSnapshots()
    {
        EvolutionRunResult<TestGenome> result = await CreateEngine(new SyntheticEvolutionTask(), Options(1, 1, 1))
            .RunAsync(new[] { new TestGenome(1) });

        Assert.All(result.Islands, island =>
        {
            Assert.IsType<EvolutionArchiveSnapshot<TestGenome>>(island);
            Assert.False(island is IEvolutionArchive<TestGenome>);
        });
    }

    [Fact]
    public void DifferentIslandArchivePoliciesAreRejected()
    {
        EvolutionEngineOptions options = Options(1, 1, 1);
        options.IslandCount = 2;

        Assert.Throws<ArgumentException>(() => new EvolutionEngine<TestGenome>(
            new SyntheticEvolutionTask(), new IncrementVariation(),
            island => new MapElitesArchive<TestGenome>(new[]
            {
                new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
            }, capacity: island + 1), options));
    }

    [Fact]
    public async Task ArchivePolicyChangeInvalidatesResume()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions initialOptions = Options(1, 1, 1);
        await new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(), new IncrementVariation(),
            _ => new MapElitesArchive<TestGenome>(new[]
            {
                new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
            }, capacity: 1), initialOptions, checkpointStore: store, genomeCodec: new TestGenomeCodec())
            .RunAsync(new[] { new TestGenome(1) });

        EvolutionEngineOptions resumedOptions = Options(1, 1, 1);
        resumedOptions.Resume = true;
        var resumed = new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(), new IncrementVariation(),
            _ => new MapElitesArchive<TestGenome>(new[]
            {
                new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
            }, capacity: 2), resumedOptions, checkpointStore: store, genomeCodec: new TestGenomeCodec());

        await Assert.ThrowsAsync<InvalidDataException>(() => resumed.RunAsync(new[] { new TestGenome(1) }));
    }

    [Fact]
    public async Task ResumeRejectsCheckpointWithInconsistentIdentityCounters()
    {
        var originalStore = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions initialOptions = Options(2, 2, 1);
        await CreateEngine(new SyntheticEvolutionTask(), initialOptions, originalStore)
            .RunAsync(new[] { new TestGenome(1), new TestGenome(2) });
        EvolutionCheckpoint original = Assert.IsType<EvolutionCheckpoint>(
            await originalStore.LoadLatestAsync(initialOptions.RunId));
        JObject payload = JObject.Parse(original.Payload);
        payload["NextEvaluationId"] = original.Sequence + 100;

        var corruptStore = new InMemoryEvolutionCheckpointStore();
        await corruptStore.SaveAsync(new EvolutionCheckpoint(original.RunId, original.Sequence,
            original.CompatibilityHash, payload.ToString(Formatting.None)));
        EvolutionEngineOptions resumedOptions = Options(2, 2, 1);
        resumedOptions.Resume = true;
        EvolutionEngine<TestGenome> resumed = CreateEngine(new SyntheticEvolutionTask(), resumedOptions, corruptStore);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            resumed.RunAsync(new[] { new TestGenome(1), new TestGenome(2) }));
    }

    [Fact]
    public async Task RetryDiagnosticsAreBoundedWithExplicitTruncation()
    {
        EvolutionEngineOptions options = Options(2, 1, 1);
        options.MaxRetries = 1;
        var engine = new EvolutionEngine<TestGenome>(new VerboseFailOnceEvolutionTask(), new IncrementVariation(),
            _ => TestArchive(), options);

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[] { new TestGenome(1) });
        EvolutionArchiveEntry<TestGenome> entry = Assert.IsType<EvolutionArchiveEntry<TestGenome>>(result.Best);

        Assert.Equal(64, entry.Evaluation.Diagnostics.Count);
        Assert.Equal("diagnostics_truncated", entry.Evaluation.Diagnostics[63].Code);
    }

    [Fact]
    public async Task RetryCostOverflowSaturatesWithoutCrashingCommit()
    {
        EvolutionEngineOptions options = Options(2, 1, 1);
        options.MaxRetries = 1;
        var engine = new EvolutionEngine<TestGenome>(new SaturatingCostEvolutionTask(), new IncrementVariation(),
            _ => TestArchive(), options);

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[] { new TestGenome(1) });
        EvolutionArchiveEntry<TestGenome> entry = Assert.IsType<EvolutionArchiveEntry<TestGenome>>(result.Best);

        Assert.Equal(double.MaxValue, entry.Evaluation.Cost.CostUnits);
        Assert.Contains(entry.Evaluation.Diagnostics, diagnostic => diagnostic.Code == "cost_units_saturated");
    }

    [Fact]
    public async Task CooperativeEvaluationTimeoutProducesTimedOutStatus()
    {
        var task = new CooperativeBlockingEvolutionTask();
        EvolutionEngineOptions options = Options(1, 1, 1);
        options.EvaluationTimeout = TimeSpan.FromMilliseconds(25);
        var engine = new EvolutionEngine<TestGenome>(task, new IncrementVariation(), _ => TestArchive(), options);

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[] { new TestGenome(1) });

        Assert.Equal(1, task.Calls);
        Assert.Equal(1, result.Counters.StatusCounts[EvolutionEvaluationStatus.TimedOut]);
        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, result.StopReason);
    }

    [Fact]
    public async Task CooperativeRunTimeLimitRollsBackIncompleteBatch()
    {
        var task = new CooperativeBlockingEvolutionTask();
        EvolutionEngineOptions options = Options(2, 1, 1);
        options.TimeLimit = TimeSpan.FromMilliseconds(25);
        var engine = new EvolutionEngine<TestGenome>(task, new IncrementVariation(), _ => TestArchive(), options);

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[] { new TestGenome(1) });

        Assert.Equal(EvolutionStopReason.TimeLimitReached, result.StopReason);
        Assert.Equal(1, task.Calls);
        Assert.Equal(0, result.Counters.Proposals);
        Assert.Equal(0, result.Counters.EvaluationAttempts);
    }

    [Fact]
    public async Task RepeatedWorkerSchedulesProduceOneStateHash()
    {
        TestGenome[] seeds = Enumerable.Range(1, 16).Select(value => new TestGenome(value)).ToArray();
        var hashes = new HashSet<string>(StringComparer.Ordinal);

        for (int workers = 1; workers <= 6; workers++)
        {
            EvolutionRunResult<TestGenome> result = await CreateEngine(new SyntheticEvolutionTask(delayScale: 1),
                Options(maxAttempts: 16, batchSize: 8, workers: workers)).RunAsync(seeds);
            hashes.Add(result.StateHash);
        }

        Assert.Single(hashes);
    }

    [Fact]
    public async Task ResumeAfterCancellationMatchesUninterruptedStateHash()
    {
        TestGenome[] seeds = Enumerable.Range(1, 8).Select(value => new TestGenome(value)).ToArray();
        var uninterruptedStore = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions uninterruptedOptions = Options(8, 2, 2);
        uninterruptedOptions.CheckpointInterval = 2;
        EvolutionRunResult<TestGenome> uninterrupted = await CreateEngine(new SyntheticEvolutionTask(),
            uninterruptedOptions, uninterruptedStore).RunAsync(seeds);

        var sharedStore = new InMemoryEvolutionCheckpointStore();
        using var cancellation = new CancellationTokenSource();
        EvolutionEngineOptions interruptedOptions = Options(8, 2, 2);
        interruptedOptions.CheckpointInterval = 2;
        EvolutionEngine<TestGenome> interruptedEngine = CreateEngine(
            new SyntheticEvolutionTask(cancelOnEvaluation: cancellation), interruptedOptions, sharedStore);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => interruptedEngine.RunAsync(seeds, cancellation.Token));

        EvolutionEngineOptions resumedOptions = Options(8, 2, 4);
        resumedOptions.CheckpointInterval = 2;
        resumedOptions.Resume = true;
        EvolutionRunResult<TestGenome> resumed = await CreateEngine(new SyntheticEvolutionTask(),
            resumedOptions, sharedStore).RunAsync(seeds);

        Assert.Equal(uninterrupted.StateHash, resumed.StateHash);
        Assert.Equal(uninterrupted.Counters.EvaluationAttempts, resumed.Counters.EvaluationAttempts);
        Assert.NotNull(uninterrupted.Best);
        Assert.NotNull(resumed.Best);
        Assert.Equal(uninterrupted.Best?.Evaluation.GenomeId, resumed.Best?.Evaluation.GenomeId);
    }

    [Fact]
    public void ConstructionDoesNotMutateProcessGlobalState()
    {
        CultureInfo culture = CultureInfo.CurrentCulture;
        CultureInfo uiCulture = CultureInfo.CurrentUICulture;
        GCLatencyMode latency = GCSettings.LatencyMode;
        ThreadPool.GetMinThreads(out int minWorkers, out int minIo);
        ThreadPool.GetMaxThreads(out int maxWorkers, out int maxIo);

        _ = CreateEngine(new SyntheticEvolutionTask(), Options(1, 1, 1));

        Assert.Same(culture, CultureInfo.CurrentCulture);
        Assert.Same(uiCulture, CultureInfo.CurrentUICulture);
        Assert.Equal(latency, GCSettings.LatencyMode);
        ThreadPool.GetMinThreads(out int actualMinWorkers, out int actualMinIo);
        ThreadPool.GetMaxThreads(out int actualMaxWorkers, out int actualMaxIo);
        Assert.Equal((minWorkers, minIo, maxWorkers, maxIo),
            (actualMinWorkers, actualMinIo, actualMaxWorkers, actualMaxIo));
    }

    private static EvolutionEngineOptions Options(int maxAttempts, int batchSize, int workers) => new()
    {
        RunId = "test-run",
        Seed = 77,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = batchSize,
        MaxDegreeOfParallelism = workers,
        IslandCount = 2,
        MigrationInterval = 1,
        MigrantsPerIsland = 1,
        CheckpointInterval = 0
    };

    private static EvolutionEngine<TestGenome> CreateEngine(SyntheticEvolutionTask task,
        EvolutionEngineOptions options, IEvolutionCheckpointStore? checkpointStore = null) => new(
        task,
        new IncrementVariation(),
        _ => TestArchive(),
        options,
        checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());

    private static MapElitesArchive<TestGenome> TestArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    });
}
