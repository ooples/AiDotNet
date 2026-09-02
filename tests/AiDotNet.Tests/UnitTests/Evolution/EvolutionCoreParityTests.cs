using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionCoreParityTests
{
    [Fact]
    public void RatioSelectionRejectsRatiosThatDoNotSumToOne()
    {
        var options = new EvolutionSelectionOptions { ExplorationRatio = 0.5, ExploitationRatio = 0.5, EliteRatio = 0.5 };

        Assert.Throws<ArgumentException>(() => new RatioEvolutionSelectionPolicy<TestGenome>(options));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RatioEvolutionSelectionPolicy<TestGenome>(
            new EvolutionSelectionOptions { ExplorationRatio = -1, ExploitationRatio = 1, EliteRatio = 1 }));
    }

    [Fact]
    public void RatioSelectionEliteBranchAlwaysReturnsIslandBest()
    {
        MapElitesArchive<TestGenome> archive = SpreadArchive();
        var policy = new RatioEvolutionSelectionPolicy<TestGenome>(new EvolutionSelectionOptions
        {
            ExplorationRatio = 0, ExploitationRatio = 0, EliteRatio = 1
        });

        for (int seed = 0; seed < 20; seed++)
        {
            EvolutionSelection<TestGenome> selection = Assert.IsType<EvolutionSelection<TestGenome>>(
                policy.Select(archive, new StableRandom((ulong)seed), inspirationCount: 2));
            Assert.Equal(archive.Best?.Evaluation.GenomeId, selection.Parent.Evaluation.GenomeId);
        }
    }

    [Fact]
    public void RatioSelectionExploitationDrawsFromGlobalEliteIndexWhenConfigured()
    {
        MapElitesArchive<TestGenome> archive = SpreadArchive();
        var foreign = new EvolutionGlobalEliteIndex<TestGenome>(4, EvolutionOptimizationDirection.Maximize);
        foreign.Consider(Record(1, 99, "foreign", 500));
        var policy = new RatioEvolutionSelectionPolicy<TestGenome>(new EvolutionSelectionOptions
        {
            ExplorationRatio = 0, ExploitationRatio = 1, EliteRatio = 0, ExploitationEliteCount = 1
        });
        policy.UseEliteIndex(foreign.Entries, island: 1);

        EvolutionSelection<TestGenome> selection = Assert.IsType<EvolutionSelection<TestGenome>>(
            policy.Select(archive, new StableRandom(11), inspirationCount: 1));

        Assert.Equal("foreign", selection.Parent.Evaluation.GenomeId);
    }

    [Fact]
    public void RatioSelectionIslandTopKNeverLeavesTheIsland()
    {
        MapElitesArchive<TestGenome> archive = SpreadArchive();
        var foreign = new EvolutionGlobalEliteIndex<TestGenome>(4, EvolutionOptimizationDirection.Maximize);
        foreign.Consider(Record(1, 99, "foreign", 500));
        var policy = new RatioEvolutionSelectionPolicy<TestGenome>(new EvolutionSelectionOptions
        {
            ExplorationRatio = 0,
            ExploitationRatio = 1,
            EliteRatio = 0,
            ExploitationEliteCount = 2,
            ExploitationSource = EvolutionExploitationSource.IslandTopK
        });
        policy.UseEliteIndex(foreign.Entries, island: 0);

        var islandIds = new HashSet<string>(archive.Entries.Select(entry => entry.Evaluation.GenomeId), StringComparer.Ordinal);
        for (int seed = 0; seed < 20; seed++)
        {
            EvolutionSelection<TestGenome> selection = Assert.IsType<EvolutionSelection<TestGenome>>(
                policy.Select(archive, new StableRandom((ulong)seed), inspirationCount: 1));
            Assert.Contains(selection.Parent.Evaluation.GenomeId, islandIds);
        }
    }

    [Fact]
    public void RatioSelectionInspirationsUseTopQualityThenTheMostDistantCell()
    {
        // Cells are 0..3 with qualities a=1 (cell 0), b=4 (cell 1), c=3 (cell 2), d=2 (cell 3); the elite branch
        // always parents from b, so "top" must pick c and "diverse" must pick the farthest cell, d.
        MapElitesArchive<TestGenome> archive = SpreadArchive();
        EvolutionSelection<TestGenome> byQuality = Assert.IsType<EvolutionSelection<TestGenome>>(
            Policy(topInspirations: 1, diverseInspirations: 0).Select(archive, new StableRandom(5), inspirationCount: 1));
        EvolutionSelection<TestGenome> byDistance = Assert.IsType<EvolutionSelection<TestGenome>>(
            Policy(topInspirations: 0, diverseInspirations: 1).Select(archive, new StableRandom(5), inspirationCount: 1));

        Assert.Equal("b", byQuality.Parent.Evaluation.GenomeId);
        Assert.Equal(new[] { "c" }, byQuality.Inspirations.Select(entry => entry.Evaluation.GenomeId));
        Assert.Equal("b", byDistance.Parent.Evaluation.GenomeId);
        Assert.Equal(new[] { "d" }, byDistance.Inspirations.Select(entry => entry.Evaluation.GenomeId));
    }

    [Fact]
    public void RatioSelectionCanLeadInspirationsWithTheIslandBest()
    {
        MapElitesArchive<TestGenome> archive = SpreadArchive();
        var policy = new RatioEvolutionSelectionPolicy<TestGenome>(new EvolutionSelectionOptions
        {
            ExplorationRatio = 1,
            ExploitationRatio = 0,
            EliteRatio = 0,
            TopInspirationCount = 0,
            DiverseInspirationCount = 0,
            IncludeIslandBest = true
        });

        for (int seed = 0; seed < 20; seed++)
        {
            EvolutionSelection<TestGenome> selection = Assert.IsType<EvolutionSelection<TestGenome>>(
                policy.Select(archive, new StableRandom((ulong)seed), inspirationCount: 1));
            if (selection.Parent.Evaluation.GenomeId == "b") continue;
            Assert.Equal(new[] { "b" }, selection.Inspirations.Select(entry => entry.Evaluation.GenomeId));
        }
    }

    [Fact]
    public void RatioSelectionIsIndependentOfInsertionOrder()
    {
        var results = new List<string>();
        string[][] orders =
        {
            new[] { "a", "b", "c", "d" },
            new[] { "d", "c", "b", "a" },
            new[] { "c", "a", "d", "b" }
        };
        foreach (string[] order in orders)
        {
            MapElitesArchive<TestGenome> archive = SpreadArchive(order);
            var policy = new RatioEvolutionSelectionPolicy<TestGenome>();
            EvolutionSelection<TestGenome> selection = Assert.IsType<EvolutionSelection<TestGenome>>(
                policy.Select(archive, new StableRandom(4242), inspirationCount: 3));
            results.Add(selection.Parent.Evaluation.GenomeId + ":" +
                string.Join(",", selection.Inspirations.Select(entry => entry.Evaluation.GenomeId)));
        }

        Assert.Single(results.Distinct(StringComparer.Ordinal));
    }

    [Fact]
    public void GlobalEliteIndexKeepsTopKWithDeterministicTieBreak()
    {
        var index = new EvolutionGlobalEliteIndex<TestGenome>(2, EvolutionOptimizationDirection.Maximize);

        Assert.True(index.Consider(Record(0, 1, "a", 1)));
        Assert.True(index.Consider(Record(1, 2, "b", 3)));
        Assert.True(index.Consider(Record(0, 3, "c", 2)));
        Assert.False(index.Consider(Record(1, 4, "d", 0)));
        Assert.False(index.Consider(Record(1, 5, "b", 9)));

        Assert.Equal(2, index.Count);
        Assert.Equal(new[] { "b", "c" }, index.Entries.Select(record => record.Entry.Evaluation.GenomeId));
        Assert.Equal(new[] { 1, 0 }, index.Entries.Select(record => record.Island));
        Assert.Equal(new[] { "b" }, index.Top(1).Select(record => record.Entry.Evaluation.GenomeId));
    }

    [Fact]
    public void GlobalEliteIndexCapacityZeroRetainsNothing()
    {
        var index = new EvolutionGlobalEliteIndex<TestGenome>(0, EvolutionOptimizationDirection.Maximize);

        Assert.False(index.Consider(Record(0, 1, "a", 1)));
        Assert.Equal(0, index.Count);
        Assert.Empty(index.Entries);
    }

    [Fact]
    public void IslandHistoryEvictsHomelessWorstFirstAndProtectsBestAndNewcomer()
    {
        var history = new EvolutionIslandHistory<TestGenome>(3, EvolutionOptimizationDirection.Maximize);
        EvolutionArchiveEntry<TestGenome> best = Entry(1, "best", 10, 0);
        EvolutionArchiveEntry<TestGenome> owner = Entry(2, "owner", 2, 1);
        EvolutionArchiveEntry<TestGenome> homelessHigh = Entry(3, "homelessHigh", 8, 2);
        EvolutionArchiveEntry<TestGenome> homelessLow = Entry(4, "homelessLow", 1, 3);
        string[] owners = { "best", "owner" };

        Assert.Empty(history.Add(best, owners, "best"));
        Assert.Empty(history.Add(owner, owners, "best"));
        Assert.Empty(history.Add(homelessHigh, owners, "best"));
        IReadOnlyList<EvolutionArchiveEntry<TestGenome>> evicted = history.Add(homelessLow, owners, "best");

        Assert.Equal(new[] { "homelessHigh" }, evicted.Select(entry => entry.Evaluation.GenomeId));
        Assert.Equal(3, history.Count);
        Assert.True(history.Contains("best"));
        Assert.True(history.Contains("owner"));
        Assert.True(history.Contains("homelessLow"));
    }

    [Fact]
    public void IslandHistoryCapacityZeroRetainsNothing()
    {
        var history = new EvolutionIslandHistory<TestGenome>(0, EvolutionOptimizationDirection.Maximize);

        Assert.Empty(history.Add(Entry(1, "a", 1, 0), Array.Empty<string>(), null));
        Assert.Equal(0, history.Count);
    }

    [Fact]
    public void IncrementalBestMatchesAFullScanForEveryInsertionOrder()
    {
        var random = new StableRandom(9091);
        for (int trial = 0; trial < 25; trial++)
        {
            var archive = new MapElitesArchive<TestGenome>(new[]
            {
                new EvolutionDescriptorDefinition("x", 0, 4, 4)
            }, capacity: 3);
            for (int i = 0; i < 12; i++)
            {
                MapElitesArchiveTests.Add(archive, i,
                    "g" + random.NextInt(20).ToString(CultureInfo.InvariantCulture),
                    random.NextInt(10), random.NextInt(4) + 0.5);
            }
            EvolutionArchiveEntry<TestGenome>? scanned = archive.Entries
                .OrderByDescending(entry => entry.Evaluation.Quality)
                .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
                .ThenBy(entry => entry.Cell.StableKey, StringComparer.Ordinal)
                .FirstOrDefault();
            Assert.Equal(scanned?.Evaluation.GenomeId, archive.Best?.Evaluation.GenomeId);
        }
    }

    [Fact]
    public void PerDescriptorBinCountsFormARectangularGrid()
    {
        var archive = new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 4, 4),
            new EvolutionDescriptorDefinition("y", 0, 3, 3)
        });

        EvolutionCellKey low = Assert.IsType<EvolutionCellKey>(
            archive.TryCreateKey(new Dictionary<string, double> { ["x"] = 3.5, ["y"] = 0.5 }));
        EvolutionCellKey high = Assert.IsType<EvolutionCellKey>(
            archive.TryCreateKey(new Dictionary<string, double> { ["x"] = 3.5, ["y"] = 2.5 }));

        Assert.Equal(12, archive.TotalGridCells);
        Assert.Equal("3,0", low.StableKey);
        Assert.Equal("3,2", high.StableKey);
    }

    [Fact]
    public void CalibratedBoundsAreFrozenIntoTheDefinitionHash()
    {
        var narrow = new EvolutionDescriptorCalibrator("x", 8, EvolutionOutOfRangePolicy.Clamp);
        narrow.Observe(0);
        narrow.Observe(10);
        var wide = new EvolutionDescriptorCalibrator("x", 8, EvolutionOutOfRangePolicy.Clamp);
        wide.Observe(0);
        wide.Observe(100);

        var narrowArchive = new MapElitesArchive<TestGenome>(new[] { narrow.Freeze() });
        var wideArchive = new MapElitesArchive<TestGenome>(new[] { wide.Freeze() });

        Assert.NotEqual(narrowArchive.DefinitionHash, wideArchive.DefinitionHash);
        Assert.Equal(narrowArchive.DefinitionHash,
            new MapElitesArchive<TestGenome>(new[] { narrow.Freeze() }).DefinitionHash);
    }

    [Fact]
    public async Task ChangedDescriptorBoundsRejectAResumedRunInsteadOfRebinning()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions initial = Options(2, 2, islandCount: 1);
        await new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(), new IncrementVariation(),
            _ => new MapElitesArchive<TestGenome>(new[]
            {
                new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
            }), initial, checkpointStore: store, genomeCodec: new TestGenomeCodec())
            .RunAsync(new[] { new TestGenome(1) });

        EvolutionEngineOptions resumed = Options(2, 2, islandCount: 1);
        resumed.Resume = true;
        var engine = new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(), new IncrementVariation(),
            _ => new MapElitesArchive<TestGenome>(new[]
            {
                new EvolutionDescriptorDefinition("x", 0, 200, 10, EvolutionOutOfRangePolicy.Clamp)
            }), resumed, checkpointStore: store, genomeCodec: new TestGenomeCodec());

        await Assert.ThrowsAsync<InvalidDataException>(() => engine.RunAsync(new[] { new TestGenome(1) }));
    }

    [Fact]
    public async Task DefaultOptionsKeepNoGlobalElitesNoHistoryAndStillReportIslandStatus()
    {
        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), Options(6, 3, islandCount: 2))
            .RunAsync(Seeds(4));

        Assert.Empty(result.GlobalElites);
        Assert.Equal(2, result.IslandStatuses.Count);
        Assert.All(result.IslandStatuses, status => Assert.Equal(0, status.HistoryCount));
        Assert.All(result.IslandStatuses, status => Assert.Equal(10, status.TotalCells));
        Assert.All(result.IslandStatuses, status => Assert.Equal(status.EliteCount / 10.0, status.Coverage));
        Assert.Equal(result.Islands.Sum(island => island.Count), result.IslandStatuses.Sum(status => status.EliteCount));
    }

    [Fact]
    public async Task PerIslandGenerationsSumToTheRunGenerationAndAppearInStatus()
    {
        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), Options(12, 4, islandCount: 3))
            .RunAsync(Seeds(3));

        Assert.Equal(3, result.IslandStatuses.Count);
        Assert.Equal(result.Counters.Proposals - 3, result.IslandStatuses.Sum(status => status.Generation));
        Assert.All(result.IslandStatuses, status => Assert.True(status.Generation >= 0));
    }

    [Fact]
    public async Task InheritParentKeepsChildrenOnTheParentIsland()
    {
        EvolutionEngineOptions roundRobin = Options(2, 1, islandCount: 2);
        roundRobin.MaxProposals = 2;
        roundRobin.MigrationInterval = 0;
        EvolutionRunResult<TestGenome> byRoundRobin = await Engine(new SyntheticEvolutionTask(), roundRobin)
            .RunAsync(Seeds(1));

        EvolutionEngineOptions inherit = Options(2, 1, islandCount: 2);
        inherit.MaxProposals = 2;
        inherit.MigrationInterval = 0;
        inherit.IslandAssignment = EvolutionIslandAssignmentStrategy.InheritParent;
        EvolutionRunResult<TestGenome> byInheritance = await Engine(new SyntheticEvolutionTask(), inherit)
            .RunAsync(Seeds(1));

        Assert.Equal(1, byRoundRobin.Islands[1].Count);
        Assert.Equal(1, byRoundRobin.IslandStatuses[1].Generation);
        Assert.Equal(0, byInheritance.Islands[1].Count);
        Assert.Equal(0, byInheritance.IslandStatuses[1].Generation);
        Assert.Equal(1, byInheritance.IslandStatuses[0].Generation);
    }

    [Fact]
    public async Task GlobalEliteIndexIsExposedOnTheRunResultInBestFirstOrder()
    {
        EvolutionEngineOptions options = Options(8, 4, islandCount: 2);
        options.GlobalEliteCount = 3;
        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), options).RunAsync(Seeds(8));

        Assert.Equal(3, result.GlobalElites.Count);
        Assert.Equal(result.GlobalElites.Select(record => record.Entry.Evaluation.Quality).OrderByDescending(quality => quality),
            result.GlobalElites.Select(record => record.Entry.Evaluation.Quality));
        Assert.Equal(8.0, result.GlobalElites[0].Entry.Evaluation.Quality.GetValueOrDefault());
    }

    [Fact]
    public async Task BoundedHistoryStaysWithinItsCapacityPerIsland()
    {
        EvolutionEngineOptions options = Options(12, 4, islandCount: 2);
        options.HistorySize = 2;
        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), options).RunAsync(Seeds(12));

        Assert.All(result.IslandStatuses, status => Assert.InRange(status.HistoryCount, 1, 2));
    }

    [Fact]
    public async Task ResumePreservesIslandGenerationsGlobalElitesAndHistory()
    {
        TestGenome[] seeds = Seeds(8);
        var uninterruptedStore = new InMemoryEvolutionCheckpointStore();
        EvolutionRunResult<TestGenome> uninterrupted = await Engine(new SyntheticEvolutionTask(),
            ResumableOptions(), uninterruptedStore).RunAsync(seeds);

        var sharedStore = new InMemoryEvolutionCheckpointStore();
        using var cancellation = new CancellationTokenSource();
        EvolutionEngine<TestGenome> interrupted = Engine(
            new SyntheticEvolutionTask(cancelOnEvaluation: cancellation), ResumableOptions(), sharedStore);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => interrupted.RunAsync(seeds, cancellation.Token));

        EvolutionEngineOptions resumedOptions = ResumableOptions();
        resumedOptions.Resume = true;
        resumedOptions.MaxDegreeOfParallelism = 4;
        EvolutionRunResult<TestGenome> resumed = await Engine(new SyntheticEvolutionTask(), resumedOptions, sharedStore)
            .RunAsync(seeds);

        Assert.Equal(uninterrupted.StateHash, resumed.StateHash);
        Assert.Equal(uninterrupted.IslandStatuses.Select(status => status.Generation),
            resumed.IslandStatuses.Select(status => status.Generation));
        Assert.Equal(uninterrupted.IslandStatuses.Select(status => status.HistoryCount),
            resumed.IslandStatuses.Select(status => status.HistoryCount));
        Assert.Equal(uninterrupted.GlobalElites.Select(record => record.Entry.Evaluation.GenomeId),
            resumed.GlobalElites.Select(record => record.Entry.Evaluation.GenomeId));
        Assert.NotEmpty(resumed.GlobalElites);
    }

    [Fact]
    public async Task QualityDescriptorFillsTheConfiguredDimensionWhenTheTaskOmitsIt()
    {
        EvolutionEngineOptions withDescriptor = Options(3, 3, islandCount: 1);
        withDescriptor.MaxProposals = 3;
        withDescriptor.QualityDescriptorName = "score";
        EvolutionRunResult<TestGenome> filled = await ScoreEngine(withDescriptor).RunAsync(Seeds(3));

        EvolutionEngineOptions withoutDescriptor = Options(3, 3, islandCount: 1);
        withoutDescriptor.MaxProposals = 3;
        EvolutionRunResult<TestGenome> unfilled = await ScoreEngine(withoutDescriptor).RunAsync(Seeds(3));

        Assert.NotEmpty(filled.Islands[0].Entries);
        Assert.All(filled.Islands[0].Entries,
            entry => Assert.Equal(entry.Evaluation.Quality, (double?)entry.Evaluation.Descriptors["score"]));
        Assert.Empty(unfilled.Islands[0].Entries);
    }

    [Fact]
    public async Task MissingArchiveDescriptorIsReportedInsteadOfSilentlyRejected()
    {
        EvolutionEngineOptions options = Options(3, 3, islandCount: 1);
        options.MaxProposals = 3;
        EvolutionRunResult<TestGenome> result = await ScoreEngine(options).RunAsync(Seeds(3));

        Assert.Empty(result.Islands[0].Entries);
        Assert.Contains(result.RetainedFailures, diagnostic => diagnostic.Code == "descriptor_missing:score");
        Assert.Equal(3, result.Counters.StatusCounts[EvolutionEvaluationStatus.Completed]);
    }

    [Fact]
    public async Task NoveltyGateRejectsNearDuplicatesWithoutSpendingEvaluatorBudget()
    {
        EvolutionEngineOptions options = Options(10, 1, islandCount: 1);
        options.MaxProposals = 3;
        options.NoveltyDistanceThreshold = 3;
        var task = new SyntheticEvolutionTask();
        var engine = new EvolutionEngine<TestGenome>(task, new IncrementVariation(), _ => TestArchive(), options,
            genomeDistance: new AbsoluteGenomeDistance());

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(new[]
        {
            new TestGenome(1), new TestGenome(2), new TestGenome(3)
        });

        Assert.Equal(1, task.Calls);
        Assert.Equal(1, result.Counters.EvaluationAttempts);
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Rejected]);
        Assert.Equal(1, result.Counters.StatusCounts[EvolutionEvaluationStatus.Completed]);
    }

    [Fact]
    public async Task NoveltyGateIsOffByDefaultAndDemandsADistanceMetricWhenEnabled()
    {
        EvolutionEngineOptions defaults = Options(10, 1, islandCount: 1);
        defaults.MaxProposals = 3;
        var task = new SyntheticEvolutionTask();
        EvolutionRunResult<TestGenome> result = await Engine(task, defaults).RunAsync(new[]
        {
            new TestGenome(1), new TestGenome(2), new TestGenome(3)
        });

        Assert.Equal(3, task.Calls);
        Assert.False(result.Counters.StatusCounts.ContainsKey(EvolutionEvaluationStatus.Rejected));

        EvolutionEngineOptions invalid = Options(10, 1, islandCount: 1);
        invalid.NoveltyDistanceThreshold = 1;
        Assert.Throws<ArgumentException>(() => new EvolutionEngine<TestGenome>(
            new SyntheticEvolutionTask(), new IncrementVariation(), _ => TestArchive(), invalid));
    }

    [Fact]
    public async Task GenerationMigrationTriggerSuppressesMigrationUntilItsIntervalIsReached()
    {
        EvolutionEngineOptions batchDriven = Options(8, 2, islandCount: 2);
        batchDriven.MigrationInterval = 1;
        var batchObserver = new RecordingEvolutionObserver();
        await Engine(new SyntheticEvolutionTask(), batchDriven, observer: batchObserver).RunAsync(Seeds(8));

        EvolutionEngineOptions generationDriven = Options(8, 2, islandCount: 2);
        generationDriven.MigrationInterval = 1_000;
        generationDriven.MigrationTrigger = EvolutionMigrationTrigger.IslandGenerations;
        var generationObserver = new RecordingEvolutionObserver();
        await Engine(new SyntheticEvolutionTask(), generationDriven, observer: generationObserver).RunAsync(Seeds(8));

        Assert.True(batchObserver.CountOf(EvolutionEventKind.Migrated) > 0);
        Assert.Equal(0, generationObserver.CountOf(EvolutionEventKind.Migrated));
    }

    private static EvolutionEngineOptions ResumableOptions()
    {
        EvolutionEngineOptions options = Options(8, 2, islandCount: 2);
        options.CheckpointInterval = 2;
        options.GlobalEliteCount = 3;
        options.HistorySize = 2;
        return options;
    }

    private static EvolutionEngineOptions Options(int maxAttempts, int batchSize, int islandCount) => new()
    {
        RunId = "core-parity",
        Seed = 77,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = batchSize,
        MaxDegreeOfParallelism = 1,
        IslandCount = islandCount,
        MigrationInterval = 20,
        MigrantsPerIsland = 1,
        CheckpointInterval = 0
    };

    private static TestGenome[] Seeds(int count) =>
        Enumerable.Range(1, count).Select(value => new TestGenome(value)).ToArray();

    private static EvolutionEngine<TestGenome> Engine(SyntheticEvolutionTask task, EvolutionEngineOptions options,
        IEvolutionCheckpointStore? checkpointStore = null, IEvolutionObserver<TestGenome>? observer = null) => new(
        task, new IncrementVariation(), _ => TestArchive(), options,
        observer: observer,
        checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());

    private static EvolutionEngine<TestGenome> ScoreEngine(EvolutionEngineOptions options) => new(
        new SyntheticEvolutionTask(), new IncrementVariation(),
        _ => new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("score", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
        }), options);

    private static MapElitesArchive<TestGenome> TestArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    });

    private static MapElitesArchive<TestGenome> SpreadArchive(IReadOnlyList<string>? order = null)
    {
        var archive = new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 4, 4)
        });
        var values = new Dictionary<string, (long Id, double Quality, double Descriptor)>(StringComparer.Ordinal)
        {
            ["a"] = (1, 1, 0.5), ["b"] = (2, 4, 1.5), ["c"] = (3, 3, 2.5), ["d"] = (4, 2, 3.5)
        };
        foreach (string genomeId in order ?? new[] { "a", "b", "c", "d" })
        {
            (long id, double quality, double descriptor) = values[genomeId];
            MapElitesArchiveTests.Add(archive, id, genomeId, quality, descriptor);
        }
        return archive;
    }

    private static RatioEvolutionSelectionPolicy<TestGenome> Policy(int topInspirations, int diverseInspirations) =>
        new(new EvolutionSelectionOptions
        {
            ExplorationRatio = 0,
            ExploitationRatio = 0,
            EliteRatio = 1,
            TopInspirationCount = topInspirations,
            DiverseInspirationCount = diverseInspirations,
            IncludeIslandBest = false
        });

    private static EvolutionEliteRecord<TestGenome> Record(int island, long id, string genomeId, double quality) =>
        new(island, Entry(id, genomeId, quality, 0));

    private static EvolutionArchiveEntry<TestGenome> Entry(long id, string genomeId, double quality, int bin)
    {
        (EvolutionCandidate<TestGenome> candidate, EvolutionEvaluation evaluation) =
            MapElitesArchiveTests.Create(id, genomeId, quality, bin + 0.5);
        return new EvolutionArchiveEntry<TestGenome>(new EvolutionCellKey(new[] { bin }), candidate, evaluation);
    }
}
