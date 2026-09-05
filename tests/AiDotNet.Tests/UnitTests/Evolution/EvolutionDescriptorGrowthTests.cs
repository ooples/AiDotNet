using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Covers <see cref="EvolutionOutOfRangePolicy.Grow"/>: widening the archive grid around values that fall outside the
/// configured descriptor range, keeping bin width fixed so existing elites only shift, and carrying the widened grid
/// through a checkpoint so a resumed run reproduces the uninterrupted run's state hash.
/// </summary>
public sealed class EvolutionDescriptorGrowthTests
{
    [Fact]
    public void AValueAboveTheRangeGrowsTheGridInsteadOfBeingRejected()
    {
        MapElitesArchive<TestGenome> archive = GrowingArchive();

        Assert.Equal(EvolutionArchiveInsertionResult.Inserted, Add(archive, 1, "a", 1, 0.1));
        Assert.Equal(5, archive.TotalGridCells);

        Assert.Equal(EvolutionArchiveInsertionResult.Inserted, Add(archive, 2, "b", 2, 1.5));

        // Width stays 0.2, so reaching 1.5 takes three more bins and stops at 1.6 rather than at the value itself.
        Assert.Equal(8, archive.TotalGridCells);
        Assert.Equal(8, archive.Descriptors[0].BinCount);
        Assert.Equal(0, archive.Descriptors[0].Minimum);
        Assert.Equal(1.6, archive.Descriptors[0].Maximum, 12);
        Assert.Equal(0.2, archive.Descriptors[0].BinWidth, 12);
        Assert.Equal(2, archive.Count);
    }

    [Fact]
    public void GrowingDownwardsMovesExistingElitesToTheirNewCells()
    {
        MapElitesArchive<TestGenome> archive = GrowingArchive();
        Add(archive, 1, "a", 1, 0.1);
        Assert.Equal("0", archive.Entries.Single().Cell.StableKey);

        Assert.Equal(EvolutionArchiveInsertionResult.Inserted, Add(archive, 2, "b", 2, -0.3));

        Assert.Equal(-0.4, archive.Descriptors[0].Minimum, 12);
        Assert.Equal(1.0, archive.Descriptors[0].Maximum, 12);
        Assert.Equal(7, archive.TotalGridCells);

        // Two bins were prepended, so the incumbent moved from bin 0 to bin 2 and the new value took bin 0.
        Assert.Equal("2", Cell(archive, "a"));
        Assert.Equal("0", Cell(archive, "b"));
        Assert.Equal(2, archive.Count);
        Assert.Equal("b", archive.Best?.Evaluation.GenomeId);
    }

    [Fact]
    public void TheFinalGridDoesNotDependOnTheOrderExtremeValuesArrive()
    {
        MapElitesArchive<TestGenome> ascending = GrowingArchive();
        Add(ascending, 1, "a", 1, 1.5);
        Add(ascending, 2, "b", 2, 2.3);

        MapElitesArchive<TestGenome> descending = GrowingArchive();
        Add(descending, 1, "b", 2, 2.3);
        Add(descending, 2, "a", 1, 1.5);

        Assert.Equal(ascending.Descriptors[0].BinCount, descending.Descriptors[0].BinCount);
        Assert.Equal(ascending.Descriptors[0].Minimum, descending.Descriptors[0].Minimum, 12);
        Assert.Equal(ascending.Descriptors[0].Maximum, descending.Descriptors[0].Maximum, 12);
        Assert.Equal(ascending.TotalGridCells, descending.TotalGridCells);
        Assert.Equal(Cell(ascending, "a"), Cell(descending, "a"));
        Assert.Equal(Cell(ascending, "b"), Cell(descending, "b"));
    }

    [Fact]
    public void GrowthThatWouldBreachTheGridSafetyLimitIsDeclined()
    {
        var archive = new MapElitesArchive<TestGenome>(
            new[] { new EvolutionDescriptorDefinition("x", 0, 1, 5, EvolutionOutOfRangePolicy.Grow) },
            EvolutionOptimizationDirection.Maximize, capacity: 0, maximumGridCells: 6);

        Assert.Equal(EvolutionArchiveInsertionResult.Inserted, Add(archive, 1, "a", 1, 0.1));

        // Reaching 1.5 would need eight bins, which is past the safety limit, so the candidate is simply not archived.
        Assert.Equal(EvolutionArchiveInsertionResult.Rejected, Add(archive, 2, "b", 2, 1.5));
        Assert.Equal(5, archive.TotalGridCells);
        Assert.Equal(1, archive.Count);
    }

    [Fact]
    public void AnUnboundedCapacityFollowsTheGrownGridWhileAnExplicitBudgetDoesNot()
    {
        MapElitesArchive<TestGenome> unbounded = GrowingArchive();
        Assert.Equal(5, unbounded.Capacity);
        Add(unbounded, 1, "a", 1, 1.5);
        Assert.Equal(8, unbounded.Capacity);

        var budgeted = new MapElitesArchive<TestGenome>(
            new[] { new EvolutionDescriptorDefinition("x", 0, 1, 5, EvolutionOutOfRangePolicy.Grow) },
            EvolutionOptimizationDirection.Maximize, capacity: 3);
        Add(budgeted, 1, "a", 1, 1.5);
        Assert.Equal(3, budgeted.Capacity);
        Assert.Equal(8, budgeted.TotalGridCells);
    }

    [Fact]
    public void TheDefaultPolicyStillRejectsOutOfRangeValues()
    {
        MapElitesArchive<TestGenome> archive = MapElitesArchiveTests.Archive();

        Assert.Equal(EvolutionArchiveInsertionResult.Rejected, MapElitesArchiveTests.Add(archive, 1, "a", 1, 1.5));
        Assert.Equal(5, archive.TotalGridCells);
        Assert.Empty(archive.Entries);
    }

    [Fact]
    public void RestoringCheckpointedBoundsReproducesTheGrownGridExactly()
    {
        MapElitesArchive<TestGenome> grown = GrowingArchive();
        Add(grown, 1, "a", 1, 0.1);
        Add(grown, 2, "b", 2, -0.3);
        Add(grown, 3, "c", 3, 1.5);

        MapElitesArchive<TestGenome> restored = GrowingArchive();
        restored.RestoreDescriptorBounds(grown.Descriptors);
        restored.Restore(grown.Entries.ToArray(), grown.Version);

        Assert.Equal(grown.TotalGridCells, restored.TotalGridCells);
        Assert.Equal(grown.Version, restored.Version);
        Assert.Equal(
            grown.Entries.Select(entry => entry.Cell.StableKey + "=" + entry.Evaluation.GenomeId),
            restored.Entries.Select(entry => entry.Cell.StableKey + "=" + entry.Evaluation.GenomeId));
        Assert.Equal(grown.Best?.Evaluation.GenomeId, restored.Best?.Evaluation.GenomeId);

        // Growth never touches the definition hash, which is what lets a resume accept a checkpoint from a run that
        // widened its grid instead of refusing it as a configuration change.
        Assert.Equal(GrowingArchive().DefinitionHash, grown.DefinitionHash);
    }

    [Fact]
    public void BoundsThatAreNotAWideningOfTheConfiguredOnesAreRefused()
    {
        Assert.Throws<InvalidDataException>(() => GrowingArchive().RestoreDescriptorBounds(
            new[] { new EvolutionDescriptorDefinition("x", 0.25, 1, 5, EvolutionOutOfRangePolicy.Grow) }));

        Assert.Throws<InvalidDataException>(() => GrowingArchive().RestoreDescriptorBounds(
            new[] { new EvolutionDescriptorDefinition("y", 0, 1.6, 8, EvolutionOutOfRangePolicy.Grow) }));

        // A different bin width means the restored cells do not mean what the configured ones mean.
        Assert.Throws<InvalidDataException>(() => GrowingArchive().RestoreDescriptorBounds(
            new[] { new EvolutionDescriptorDefinition("x", 0, 1.6, 9, EvolutionOutOfRangePolicy.Grow) }));

        Assert.Throws<InvalidDataException>(() => GrowingArchive().RestoreDescriptorBounds(
            Array.Empty<EvolutionDescriptorDefinition>()));

        MapElitesArchive<TestGenome> occupied = GrowingArchive();
        Add(occupied, 1, "a", 1, 0.1);
        Assert.Throws<InvalidOperationException>(() => occupied.RestoreDescriptorBounds(occupied.Descriptors));
    }

    [Fact]
    public async Task AResumedRunReproducesTheStateHashOfAnUninterruptedGrowingRun()
    {
        TestGenome[] seeds = Enumerable.Range(1, 4).Select(value => new TestGenome(value)).ToArray();

        EvolutionRunResult<TestGenome> uninterrupted = await Engine(Options(12), new InMemoryEvolutionCheckpointStore())
            .RunAsync(seeds);

        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionRunResult<TestGenome> partial = await Engine(Options(6), store).RunAsync(seeds);
        EvolutionEngineOptions resumedOptions = Options(12);
        resumedOptions.Resume = true;
        EvolutionRunResult<TestGenome> resumed = await Engine(resumedOptions, store).RunAsync(seeds);

        // The run has to have actually grown, otherwise the test would pass without exercising anything.
        Assert.True(uninterrupted.Islands[0].Descriptors[0].Maximum > 4);
        Assert.True(partial.Counters.CompletedEvaluations < uninterrupted.Counters.CompletedEvaluations);
        Assert.Equal(
            uninterrupted.Islands[0].Descriptors[0].Maximum,
            resumed.Islands[0].Descriptors[0].Maximum, 12);
        Assert.Equal(
            uninterrupted.Islands[0].Entries.Select(entry => entry.Cell.StableKey + "=" + entry.Evaluation.GenomeId),
            resumed.Islands[0].Entries.Select(entry => entry.Cell.StableKey + "=" + entry.Evaluation.GenomeId));
        Assert.Equal(uninterrupted.Islands[0].Version, resumed.Islands[0].Version);
        Assert.Equal(uninterrupted.Counters.Proposals, resumed.Counters.Proposals);
        Assert.Equal(uninterrupted.Counters.CompletedEvaluations, resumed.Counters.CompletedEvaluations);
        Assert.Equal(
            uninterrupted.GlobalElites.Select(record => record.Entry.Cell.StableKey + "=" + record.Entry.Evaluation.GenomeId),
            resumed.GlobalElites.Select(record => record.Entry.Cell.StableKey + "=" + record.Entry.Evaluation.GenomeId));
        Assert.Equal(uninterrupted.StateHash, resumed.StateHash);
    }

    private static MapElitesArchive<TestGenome> GrowingArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 1, 5, EvolutionOutOfRangePolicy.Grow)
    });

    private static EvolutionArchiveInsertionResult Add(MapElitesArchive<TestGenome> archive, long id,
        string genomeId, double quality, double descriptor) =>
        MapElitesArchiveTests.Add(archive, id, genomeId, quality, descriptor);

    private static string Cell(MapElitesArchive<TestGenome> archive, string genomeId) =>
        archive.Entries.Single(entry => entry.Evaluation.GenomeId == genomeId).Cell.StableKey;

    private static EvolutionEngineOptions Options(int maxAttempts) => new()
    {
        RunId = "growth-run",
        Seed = 91,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        // One proposal per batch, so the earlier run's budget always stops on a batch boundary and the resumed run
        // proposes exactly what the uninterrupted run proposed. A larger batch would be truncated by the smaller
        // budget of the first leg and shift every later batch.
        ProposalBatchSize = 1,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1,
        CheckpointInterval = 0
    };

    private static EvolutionEngine<TestGenome> Engine(
        EvolutionEngineOptions options, IEvolutionCheckpointStore checkpointStore) => new(
        new WideningEvolutionTask(), new IncrementVariation(), _ => new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 4, 4, EvolutionOutOfRangePolicy.Grow)
        }), options, checkpointStore: checkpointStore, genomeCodec: new TestGenomeCodec());

    /// <summary>Reports a descriptor that leaves the configured range as soon as the search moves past its seeds.</summary>
    private sealed class WideningEvolutionTask : IEvolutionTask<TestGenome>
    {
        public string Id => "widening";
        public string VersionHash => "widening-task-v1";
        public string EvaluatorVersionHash => "widening-evaluator-v1";

        public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
            CancellationToken cancellationToken = default) =>
            new(new EvolutionCanonicalGenome<TestGenome>(new TestGenome(genome.Value),
                genome.Value.ToString(CultureInfo.InvariantCulture)));

        public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
            EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
        {
            int value = candidate.CanonicalGenome.Genome.Value;
            return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(value,
                new Dictionary<string, double> { ["x"] = value * 1.5 }));
        }
    }
}
