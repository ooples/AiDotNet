using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Covers the engine side of variation-operator state: captured into every checkpoint, folded into the run's state
/// hash, restored before the first proposal of a resumed run, and refused when the checkpoint and the operator
/// disagree about whether there is state at all.
/// </summary>
public sealed class EvolutionVariationStateTests
{
    [Fact]
    public async Task AResumedRunRestoresWhatTheOperatorRemembered()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(4);

        var first = new StatefulVariation();
        await Engine(first, Options(8), store).RunAsync(seeds);
        Assert.True(first.Proposals > 0);

        EvolutionEngineOptions resumedOptions = Options(16);
        resumedOptions.Resume = true;
        var second = new StatefulVariation();
        await Engine(second, resumedOptions, store).RunAsync(seeds);

        // The fresh operator started from zero; only the restore can explain a count above what this run proposed.
        Assert.True(second.Proposals > first.Proposals);
    }

    [Fact]
    public async Task TheRunStateHashCoversTheOperatorsMemory()
    {
        // Two runs whose configuration matches in every other respect. If the hash ignored operator state, they
        // would claim the same identity while sitting in different states.
        var quiet = new StatefulVariation();
        EvolutionRunResult<TestGenome> withoutMemory = await Engine(quiet, Options(8)).RunAsync(Seeds(4));

        var remembering = new StatefulVariation();
        remembering.RestoreState("41");
        EvolutionRunResult<TestGenome> withMemory = await Engine(remembering, Options(8)).RunAsync(Seeds(4));

        Assert.NotEqual(withoutMemory.StateHash, withMemory.StateHash);
    }

    [Fact]
    public async Task SwappingTheOperatorEntirelyIsStillRefusedOnIdentityFirst()
    {
        // The state checks below only matter for one operator identity across two builds. A different operator is
        // rejected earlier, by version hash, and this pins that ordering so the messages stay accurate.
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(4);
        await Engine(new IncrementVariation(), Options(8), store).RunAsync(seeds);

        EvolutionEngineOptions resumedOptions = Options(16);
        resumedOptions.Resume = true;
        InvalidDataException failure = await Assert.ThrowsAsync<InvalidDataException>(
            () => Engine(new StatefulVariation(), resumedOptions, store).RunAsync(seeds));

        Assert.Contains("incompatible", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACheckpointFromAStatelessOperatorIsRefusedByAStatefulOne()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(4);
        await Engine(new StatelessTwinVariation(), Options(8), store).RunAsync(seeds);

        EvolutionEngineOptions resumedOptions = Options(16);
        resumedOptions.Resume = true;
        InvalidDataException failure = await Assert.ThrowsAsync<InvalidDataException>(
            () => Engine(new StatefulVariation(), resumedOptions, store).RunAsync(seeds));

        Assert.Contains("variation-operator state is missing", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACheckpointFromAStatefulOperatorIsRefusedByAStatelessOne()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        TestGenome[] seeds = Seeds(4);
        await Engine(new StatefulVariation(), Options(8), store).RunAsync(seeds);

        EvolutionEngineOptions resumedOptions = Options(16);
        resumedOptions.Resume = true;
        InvalidDataException failure = await Assert.ThrowsAsync<InvalidDataException>(
            () => Engine(new StatelessTwinVariation(), resumedOptions, store).RunAsync(seeds));

        Assert.Contains("stateless variation operator", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACheckpointCarriesTheOperatorStateInItsPayload()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        await Engine(new StatefulVariation(), Options(8), store).RunAsync(Seeds(4));

        EvolutionCheckpoint checkpoint = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("variation-run"));
        Assert.Contains("VariationState", checkpoint.Payload, StringComparison.Ordinal);
    }

    private static EvolutionEngineOptions Options(int maxAttempts) => new()
    {
        RunId = "variation-run",
        Seed = 91,
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

    private static EvolutionEngine<TestGenome> Engine(IVariationOperator<TestGenome> variation,
        EvolutionEngineOptions options, IEvolutionCheckpointStore? checkpointStore = null) => new(
        new SyntheticEvolutionTask(), variation, _ => TestArchive(), options,
        checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());

    private static MapElitesArchive<TestGenome> TestArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    });
}
