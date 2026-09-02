using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionResumeBudgetTests
{
    [Fact]
    public void RaisingABudgetKeepsTheCompatibilityHashWhileChangingASemanticOptionBreaksIt()
    {
        string baseline = Engine(new SyntheticEvolutionTask(), Options(4)).CompatibilityHash;

        EvolutionEngineOptions raised = Options(4);
        raised.MaxEvaluationAttempts = 400;
        raised.MaxProposals = 4_000;
        raised.MaxGenerations = 4_000;
        raised.TimeLimit = TimeSpan.FromMinutes(5);
        raised.CheckpointInterval = 7;
        raised.MaxDegreeOfParallelism = 4;
        raised.RunId = "a-different-run";
        Assert.Equal(baseline, Engine(new SyntheticEvolutionTask(), raised).CompatibilityHash);

        EvolutionEngineOptions reseeded = Options(4);
        reseeded.Seed = 78;
        Assert.NotEqual(baseline, Engine(new SyntheticEvolutionTask(), reseeded).CompatibilityHash);

        EvolutionEngineOptions retopologized = Options(4);
        retopologized.MigrationTopology = EvolutionMigrationTopology.Star;
        Assert.NotEqual(baseline, Engine(new SyntheticEvolutionTask(), retopologized).CompatibilityHash);

        EvolutionEngineOptions rerated = Options(4);
        rerated.MigrationRate = 0.25;
        Assert.NotEqual(baseline, Engine(new SyntheticEvolutionTask(), rerated).CompatibilityHash);
    }

    [Fact]
    public async Task AResumeWithARaisedEvaluationBudgetContinuesWithCountersIntact()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(8);

        EvolutionRunResult<TestGenome> first = await Engine(new SyntheticEvolutionTask(), Options(4), store)
            .RunAsync(seeds);
        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, first.StopReason);
        Assert.Equal(4, first.Counters.EvaluationAttempts);

        EvolutionEngineOptions resumedOptions = Options(8);
        resumedOptions.Resume = true;
        var continuedTask = new SyntheticEvolutionTask();
        EvolutionRunResult<TestGenome> continued = await Engine(continuedTask, resumedOptions, store).RunAsync(seeds);

        // The uninterrupted control arm keeps its own store so both engines share one compatibility identity.
        EvolutionRunResult<TestGenome> uninterrupted = await Engine(new SyntheticEvolutionTask(), Options(8),
            new InMemoryEvolutionCheckpointStore()).RunAsync(seeds);

        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, continued.StopReason);
        Assert.Equal(8, continued.Counters.EvaluationAttempts);
        Assert.Equal(8, continued.Counters.CompletedEvaluations);
        Assert.Equal(4, continuedTask.Calls);
        Assert.Equal(uninterrupted.StateHash, continued.StateHash);
        Assert.Equal(uninterrupted.Counters.Proposals, continued.Counters.Proposals);
        Assert.Equal(uninterrupted.Best?.Evaluation.GenomeId, continued.Best?.Evaluation.GenomeId);
    }

    [Fact]
    public async Task AResumeWithAnEvaluationBudgetBelowWhatWasSpentStopsImmediately()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(8);
        EvolutionRunResult<TestGenome> first = await Engine(new SyntheticEvolutionTask(), Options(8), store)
            .RunAsync(seeds);
        Assert.Equal(8, first.Counters.EvaluationAttempts);

        EvolutionEngineOptions loweredOptions = Options(4);
        loweredOptions.Resume = true;
        var task = new SyntheticEvolutionTask();
        EvolutionRunResult<TestGenome> lowered = await Engine(task, loweredOptions, store).RunAsync(seeds);

        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, lowered.StopReason);
        Assert.Equal(0, task.Calls);
        Assert.Equal(8, lowered.Counters.EvaluationAttempts);
        Assert.Equal(first.Counters.Proposals, lowered.Counters.Proposals);
        Assert.Equal(first.StateHash, lowered.StateHash);
        Assert.Equal(first.Best?.Evaluation.GenomeId, lowered.Best?.Evaluation.GenomeId);
    }

    [Fact]
    public async Task AResumeWithAProposalBudgetBelowWhatWasSpentStopsWithTheProposalReason()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(8);
        EvolutionRunResult<TestGenome> first = await Engine(new SyntheticEvolutionTask(), Options(8), store)
            .RunAsync(seeds);
        Assert.Equal(8, first.Counters.Proposals);

        EvolutionEngineOptions loweredOptions = Options(64);
        loweredOptions.MaxProposals = 8;
        loweredOptions.Resume = true;
        var task = new SyntheticEvolutionTask();
        EvolutionRunResult<TestGenome> lowered = await Engine(task, loweredOptions, store).RunAsync(seeds);

        Assert.Equal(EvolutionStopReason.ProposalBudgetReached, lowered.StopReason);
        Assert.Equal(0, task.Calls);
        Assert.Equal(8, lowered.Counters.Proposals);
        Assert.Equal(first.StateHash, lowered.StateHash);

        // A budget lowered even below the seed count still stops on the budget rather than refusing to start, which is
        // what the fresh-run seed check would otherwise do.
        EvolutionEngineOptions belowSeedCount = Options(64);
        belowSeedCount.MaxProposals = 4;
        belowSeedCount.Resume = true;
        EvolutionRunResult<TestGenome> tiny = await Engine(new SyntheticEvolutionTask(), belowSeedCount, store)
            .RunAsync(seeds);
        Assert.Equal(EvolutionStopReason.ProposalBudgetReached, tiny.StopReason);
        Assert.Equal(8, tiny.Counters.Proposals);
    }

    [Fact]
    public async Task AFreshRunStillRefusesMoreSeedsThanItsProposalBudget()
    {
        EvolutionEngineOptions options = Options(64);
        options.MaxProposals = 4;

        await Assert.ThrowsAsync<ArgumentException>(() =>
            Engine(new SyntheticEvolutionTask(), options).RunAsync(Seeds(8)));
    }

    [Fact]
    public async Task AResumeWithAGenerationBudgetBelowWhatWasSpentStopsWithTheGenerationReason()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(2);
        EvolutionEngineOptions firstOptions = Options(8);
        EvolutionRunResult<TestGenome> first = await Engine(new SyntheticEvolutionTask(), firstOptions, store)
            .RunAsync(seeds);
        Assert.True(first.Counters.Proposals > seeds.Length);

        EvolutionEngineOptions loweredOptions = Options(64);
        loweredOptions.MaxGenerations = 1;
        loweredOptions.Resume = true;
        var task = new SyntheticEvolutionTask();
        EvolutionRunResult<TestGenome> lowered = await Engine(task, loweredOptions, store).RunAsync(seeds);

        Assert.Equal(EvolutionStopReason.GenerationLimitReached, lowered.StopReason);
        Assert.Equal(0, task.Calls);
        Assert.Equal(first.Counters.Proposals, lowered.Counters.Proposals);
    }

    [Fact]
    public async Task AResumeThatChangesASemanticOptionIsRejectedAndNamesTheOption()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(4);
        await Engine(new SyntheticEvolutionTask(), Options(4), store).RunAsync(seeds);

        EvolutionEngineOptions reseeded = Options(4);
        reseeded.Seed = 78;
        reseeded.Resume = true;
        InvalidDataException seedFailure = await Assert.ThrowsAsync<InvalidDataException>(
            () => Engine(new SyntheticEvolutionTask(), reseeded, store).RunAsync(seeds));
        Assert.Contains("'seed'", seedFailure.Message, StringComparison.Ordinal);
        Assert.Contains("'77'", seedFailure.Message, StringComparison.Ordinal);
        Assert.Contains("'78'", seedFailure.Message, StringComparison.Ordinal);

        EvolutionEngineOptions reinspired = Options(4);
        reinspired.InspirationCount = 1;
        reinspired.Resume = true;
        InvalidDataException inspirationFailure = await Assert.ThrowsAsync<InvalidDataException>(
            () => Engine(new SyntheticEvolutionTask(), reinspired, store).RunAsync(seeds));
        Assert.Contains("'inspiration-count'", inspirationFailure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AResumeThatChangesATaskVersionIsStillRejectedWithTheGeneralMessage()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(4);
        await Engine(new SyntheticEvolutionTask(), Options(4), store).RunAsync(seeds);

        EvolutionEngineOptions resumedOptions = Options(4);
        resumedOptions.Resume = true;
        var replaced = new EvolutionEngine<TestGenome>(new PlateauEvolutionTask(), new IncrementVariation(),
            _ => TestArchive(), resumedOptions, checkpointStore: store, genomeCodec: new TestGenomeCodec());

        InvalidDataException failure = await Assert.ThrowsAsync<InvalidDataException>(() => replaced.RunAsync(seeds));

        Assert.Contains("incompatible with the current task", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ARaisedBudgetIsRecordedInTheCheckpointForProvenance()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(4);
        await Engine(new SyntheticEvolutionTask(), Options(4), store).RunAsync(seeds);
        EvolutionCheckpoint before = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("budget-run"));
        Assert.Contains("max-evaluation-attempts", before.Payload, StringComparison.Ordinal);

        EvolutionEngineOptions resumedOptions = Options(16);
        resumedOptions.Resume = true;
        await Engine(new SyntheticEvolutionTask(), resumedOptions, store).RunAsync(seeds);
        EvolutionCheckpoint after = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("budget-run"));

        Assert.Equal(before.CompatibilityHash, after.CompatibilityHash);
        Assert.NotEqual(before.Payload, after.Payload);
        Assert.True(after.Sequence > before.Sequence);
    }

    private static EvolutionEngineOptions Options(int maxAttempts) => new()
    {
        RunId = "budget-run",
        Seed = 77,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = 2,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1,
        CheckpointInterval = 0
    };

    private static TestGenome[] Seeds(int count) =>
        Enumerable.Range(1, count).Select(value => new TestGenome(value)).ToArray();

    private static EvolutionEngine<TestGenome> Engine(IEvolutionTask<TestGenome> task, EvolutionEngineOptions options,
        IEvolutionCheckpointStore? checkpointStore = null) => new(
        task, new IncrementVariation(), _ => TestArchive(), options,
        checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());

    private static MapElitesArchive<TestGenome> TestArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    });
}
