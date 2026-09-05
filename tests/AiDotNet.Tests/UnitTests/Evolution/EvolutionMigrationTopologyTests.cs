using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionMigrationTopologyTests
{
    [Fact]
    public void EachTopologyNamesExactlyItsOwnDestinations()
    {
        Assert.Equal(new[] { 1 }, Destinations(EvolutionMigrationTopology.Ring, 0, 4));
        Assert.Equal(new[] { 0 }, Destinations(EvolutionMigrationTopology.Ring, 3, 4));

        Assert.Equal(new[] { 1, 3 }, Destinations(EvolutionMigrationTopology.BidirectionalRing, 0, 4));
        Assert.Equal(new[] { 0, 2 }, Destinations(EvolutionMigrationTopology.BidirectionalRing, 1, 4));

        Assert.Equal(new[] { 1, 2, 3 }, Destinations(EvolutionMigrationTopology.Star, 0, 4));
        Assert.Equal(new[] { 0 }, Destinations(EvolutionMigrationTopology.Star, 2, 4));

        Assert.Equal(new[] { 1, 2, 3 }, Destinations(EvolutionMigrationTopology.FullyConnected, 0, 4));
        Assert.Equal(new[] { 0, 1, 3 }, Destinations(EvolutionMigrationTopology.FullyConnected, 2, 4));
    }

    [Fact]
    public void ABidirectionalRingOfTwoIslandsCollapsesToOneDestinationPerSource()
    {
        Assert.Equal(new[] { 1 }, Destinations(EvolutionMigrationTopology.BidirectionalRing, 0, 2));
        Assert.Equal(new[] { 0 }, Destinations(EvolutionMigrationTopology.BidirectionalRing, 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TopologyMigrationPolicy<TestGenome>.DestinationsFor(EvolutionMigrationTopology.Ring, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TopologyMigrationPolicy<TestGenome>.DestinationsFor(EvolutionMigrationTopology.Ring, 4, 4));
    }

    [Fact]
    public void PolicyTransferSetsMatchTheTopologyAndKeepTheBestElitesFirst()
    {
        IReadOnlyList<IEvolutionArchiveView<TestGenome>> islands = ThreeIslands();

        Assert.Equal(new[] { "0->1:a1", "1->2:b1", "2->0:c1" }, Describe(Policy(EvolutionMigrationTopology.Ring), islands, 1));
        Assert.Equal(new[] { "0->1:a1", "0->2:a1", "1->0:b1", "1->2:b1", "2->0:c1", "2->1:c1" },
            Describe(Policy(EvolutionMigrationTopology.BidirectionalRing), islands, 1));
        Assert.Equal(new[] { "0->1:a1", "0->2:a1", "1->0:b1", "2->0:c1" },
            Describe(Policy(EvolutionMigrationTopology.Star), islands, 1));
        Assert.Equal(new[] { "0->1:a1", "0->2:a1", "1->0:b1", "1->2:b1", "2->0:c1", "2->1:c1" },
            Describe(Policy(EvolutionMigrationTopology.FullyConnected), islands, 1));
    }

    [Fact]
    public void TheRingPolicyStillProducesTheHistoricalRingTransfers()
    {
        IReadOnlyList<IEvolutionArchiveView<TestGenome>> islands = ThreeIslands();

        Assert.Equal(Describe(Policy(EvolutionMigrationTopology.Ring), islands, 2),
            Describe(new RingMigrationPolicy<TestGenome>(), islands, 2));
        Assert.Equal(new[] { "0->1:a1", "0->1:a2", "1->2:b1", "1->2:b2", "2->0:c1", "2->0:c2" },
            Describe(new RingMigrationPolicy<TestGenome>(), islands, 2));
    }

    [Fact]
    public void MigrationRateScalesWithIslandSizeAndIsCappedByThePerIslandMaximum()
    {
        Assert.Equal(3, TopologyMigrationPolicy<TestGenome>.ResolveMigrantCount(30, 3, 0));
        Assert.Equal(3, TopologyMigrationPolicy<TestGenome>.ResolveMigrantCount(30, 3, 0.5));
        Assert.Equal(2, TopologyMigrationPolicy<TestGenome>.ResolveMigrantCount(20, 3, 0.1));
        Assert.Equal(1, TopologyMigrationPolicy<TestGenome>.ResolveMigrantCount(4, 3, 0.1));
        Assert.Equal(1, TopologyMigrationPolicy<TestGenome>.ResolveMigrantCount(1, 3, 1));
        Assert.Equal(0, TopologyMigrationPolicy<TestGenome>.ResolveMigrantCount(0, 3, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TopologyMigrationPolicy<TestGenome>.ResolveMigrantCount(4, 0, 0.5));
    }

    [Fact]
    public void APositiveRateSendsFewerElitesThanTheFixedCountOnASmallIsland()
    {
        IReadOnlyList<IEvolutionArchiveView<TestGenome>> islands = ThreeIslands();

        Assert.Equal(new[] { "0->1:a1", "1->2:b1", "2->0:c1" },
            Describe(new TopologyMigrationPolicy<TestGenome>(EvolutionMigrationTopology.Ring, migrationRate: 0.1), islands, 3));
        Assert.Equal(new[] { "0->1:a1", "0->1:a2", "0->1:a3", "1->2:b1", "1->2:b2", "1->2:b3", "2->0:c1", "2->0:c2", "2->0:c3" },
            Describe(Policy(EvolutionMigrationTopology.Ring), islands, 3));
    }

    [Fact]
    public void AnInvalidTopologyOrRateIsRejectedByThePolicyAndByTheOptions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TopologyMigrationPolicy<TestGenome>((EvolutionMigrationTopology)99));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TopologyMigrationPolicy<TestGenome>(EvolutionMigrationTopology.Ring, migrationRate: 1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TopologyMigrationPolicy<TestGenome>(EvolutionMigrationTopology.Ring, migrationRate: double.NaN));

        EvolutionEngineOptions badTopology = Options(4, 4, islandCount: 2);
        badTopology.MigrationTopology = (EvolutionMigrationTopology)99;
        Assert.Throws<ArgumentOutOfRangeException>(() => Engine(new SyntheticEvolutionTask(), badTopology));

        EvolutionEngineOptions badRate = Options(4, 4, islandCount: 2);
        badRate.MigrationRate = -0.5;
        Assert.Throws<ArgumentOutOfRangeException>(() => Engine(new SyntheticEvolutionTask(), badRate));
    }

    [Fact]
    public async Task FullyConnectedMigrationIsAcceptedAndSpreadsTheBestEliteInOneRound()
    {
        EvolutionEngineOptions options = Options(4, 4, islandCount: 4);
        options.MigrationTopology = EvolutionMigrationTopology.FullyConnected;
        var observer = new RecordingEvolutionObserver();

        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), options, observer: observer)
            .RunAsync(Seeds(4));

        Assert.Equal(1, observer.CountOf(EvolutionEventKind.Migrated));
        Assert.All(result.Islands, island => Assert.Equal(4.0, island.Best?.Evaluation.Quality));
        Assert.Equal(3, result.Islands.Count(island => island.Best?.Evaluation.Lineage.IsMigrant == true));
        Assert.All(result.Islands.Where(island => island.Best?.Evaluation.Lineage.IsMigrant == true),
            island => Assert.Equal(3, island.Best?.Evaluation.Lineage.MigrationSourceIsland));
    }

    [Fact]
    public async Task StarMigrationFunnelsEveryIslandThroughTheHubAndMarksTheCopy()
    {
        EvolutionEngineOptions options = Options(4, 4, islandCount: 4);
        options.MigrationTopology = EvolutionMigrationTopology.Star;

        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), options).RunAsync(Seeds(4));

        EvolutionArchiveEntry<TestGenome> hub = Assert.IsType<EvolutionArchiveEntry<TestGenome>>(result.Islands[0].Best);
        Assert.Equal(4.0, hub.Evaluation.Quality);
        Assert.True(hub.Evaluation.Lineage.IsMigrant);
        Assert.Equal(3, hub.Evaluation.Lineage.MigrationSourceIsland);
        Assert.Equal(0, hub.Evaluation.Lineage.Island);
        Assert.Equal(hub.Evaluation.Lineage.MigrationSourceIsland, hub.Candidate.Lineage.MigrationSourceIsland);
        // The leaves never receive from each other, so only the hub holds a migrant.
        Assert.All(result.Islands.Skip(1), island => Assert.False(island.Best?.Evaluation.Lineage.IsMigrant));
    }

    [Fact]
    public async Task EveryAcceptedMigrantRaisesItsOwnArchiveChangedEvent()
    {
        EvolutionEngineOptions options = Options(4, 4, islandCount: 4);
        options.MigrationTopology = EvolutionMigrationTopology.Star;
        var observer = new MigrantRecordingObserver();

        await Engine(new SyntheticEvolutionTask(), options, observer: observer).RunAsync(Seeds(4));

        // Hub best 1 is beaten by leaves 2, 3 and 4 in source order, so exactly three migrants are accepted.
        Assert.Equal(new[] { "1->0:2", "2->0:3", "3->0:4" }, observer.AcceptedMigrants);
        Assert.All(observer.AcceptedInsertions, insertion => Assert.Equal(EvolutionArchiveInsertionResult.Replaced, insertion));
    }

    [Fact]
    public async Task TheDefaultRingTopologyLeavesTheHistoricalMigrationBehaviourUnchanged()
    {
        EvolutionEngineOptions defaults = Options(8, 2, islandCount: 2);
        EvolutionRunResult<TestGenome> byDefault = await Engine(new SyntheticEvolutionTask(), defaults).RunAsync(Seeds(8));

        EvolutionEngineOptions explicitRing = Options(8, 2, islandCount: 2);
        EvolutionRunResult<TestGenome> byExplicitPolicy = await new EvolutionEngine<TestGenome>(
            new SyntheticEvolutionTask(), new IncrementVariation(), _ => TestArchive(), explicitRing,
            migration: new RingMigrationPolicy<TestGenome>()).RunAsync(Seeds(8));

        // The two policies carry different identities, so the compatibility-derived state hash necessarily differs;
        // what must match is the search itself - the same elites on the same islands with the same migrant marking.
        Assert.Equal(byExplicitPolicy.Islands.Select(island => island.Best?.Evaluation.GenomeId),
            byDefault.Islands.Select(island => island.Best?.Evaluation.GenomeId));
        Assert.Equal(byExplicitPolicy.Islands.Select(island => island.Best?.Evaluation.Lineage.MigrationSourceIsland),
            byDefault.Islands.Select(island => island.Best?.Evaluation.Lineage.MigrationSourceIsland));
        Assert.Equal(byExplicitPolicy.Counters.EvaluationAttempts, byDefault.Counters.EvaluationAttempts);
        Assert.Equal(byExplicitPolicy.Counters.Proposals, byDefault.Counters.Proposals);
    }

    [Fact]
    public async Task APolicyThatFloodsOneDestinationIsStillRejected()
    {
        EvolutionEngineOptions options = Options(8, 4, islandCount: 2);
        var engine = new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(), new IncrementVariation(),
            _ => TestArchive(), options, migration: new FloodingMigrationPolicy());

        InvalidOperationException failure = await Assert.ThrowsAsync<InvalidOperationException>(
            () => engine.RunAsync(Seeds(4)));

        Assert.Contains("per-destination", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PreventingRepeatedMigrationSkipsAnArrivedEliteAndPicksTheBestLocalOneInstead()
    {
        IReadOnlyList<IEvolutionArchiveView<TestGenome>> islands = IslandsWithAnArrivedMigrant();

        Assert.Equal(new[] { "0->1:arrived", "1->0:other" },
            Describe(new TopologyMigrationPolicy<TestGenome>(EvolutionMigrationTopology.Ring), islands, 1));
        Assert.Equal(new[] { "0->1:local", "1->0:other" },
            Describe(new TopologyMigrationPolicy<TestGenome>(EvolutionMigrationTopology.Ring, migrationRate: 0,
                preventRepeatedMigration: true), islands, 1));
    }

    [Fact]
    public void AnIslandHoldingNothingButMigrantsSendsNobodyWhenRepeatMigrationIsBlocked()
    {
        var onlyMigrants = MapElitesArchiveTests.Archive();
        AddEntry(onlyMigrants, 1, "arrived", quality: 10, descriptor: 0.1, migrationSourceIsland: 1);
        var neighbour = MapElitesArchiveTests.Archive();
        AddEntry(neighbour, 2, "other", quality: 4, descriptor: 0.1, migrationSourceIsland: null);
        var islands = new List<IEvolutionArchiveView<TestGenome>> { onlyMigrants, neighbour };

        Assert.Equal(new[] { "1->0:other" },
            Describe(new TopologyMigrationPolicy<TestGenome>(EvolutionMigrationTopology.Ring, migrationRate: 0,
                preventRepeatedMigration: true), islands, 1));
    }

    [Fact]
    public async Task TheMigrantMarkerSurvivesACheckpointRoundTrip()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions options = Options(4, 4, islandCount: 4);
        options.MigrationTopology = EvolutionMigrationTopology.Star;
        EvolutionRunResult<TestGenome> original = await Engine(new SyntheticEvolutionTask(), options, store)
            .RunAsync(Seeds(4));

        EvolutionEngineOptions resumedOptions = Options(4, 4, islandCount: 4);
        resumedOptions.MigrationTopology = EvolutionMigrationTopology.Star;
        resumedOptions.Resume = true;
        EvolutionRunResult<TestGenome> resumed = await Engine(new SyntheticEvolutionTask(), resumedOptions, store)
            .RunAsync(Seeds(4));

        EvolutionArchiveEntry<TestGenome> hub = Assert.IsType<EvolutionArchiveEntry<TestGenome>>(resumed.Islands[0].Best);
        Assert.True(hub.Evaluation.Lineage.IsMigrant);
        Assert.Equal(3, hub.Evaluation.Lineage.MigrationSourceIsland);
        Assert.True(hub.Candidate.Lineage.IsMigrant);
        Assert.Equal(original.StateHash, resumed.StateHash);
    }

    [Fact]
    public void ALineageCannotNameItsOwnIslandAsTheMigrationSource()
    {
        Assert.Throws<ArgumentException>(() =>
            new EvolutionLineage(null, null, "seed", null, 0, 2, 1UL, migrationSourceIsland: 2));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EvolutionLineage(null, null, "seed", null, 0, 2, 1UL, migrationSourceIsland: -1));
        var local = new EvolutionLineage(null, null, "seed", null, 0, 2, 1UL);
        Assert.False(local.IsMigrant);
        Assert.Null(local.MigrationSourceIsland);
    }

    private static IReadOnlyList<int> Destinations(EvolutionMigrationTopology topology, int source, int islandCount) =>
        TopologyMigrationPolicy<TestGenome>.DestinationsFor(topology, source, islandCount);

    private static TopologyMigrationPolicy<TestGenome> Policy(EvolutionMigrationTopology topology) => new(topology);

    private static string[] Describe(IMigrationPolicy<TestGenome> policy,
        IReadOnlyList<IEvolutionArchiveView<TestGenome>> islands, int migrantsPerIsland) =>
        policy.CreateMigrations(islands, migrantsPerIsland, StableRandom.CreateStream(1, 1))
            .Select(item => $"{item.SourceIsland}->{item.DestinationIsland}:{item.Entry.Evaluation.GenomeId}")
            .OrderBy(item => item, StringComparer.Ordinal)
            .ToArray();

    /// <summary>Builds three islands whose elites are distinct and ordered so migration selection is observable.</summary>
    private static IReadOnlyList<IEvolutionArchiveView<TestGenome>> ThreeIslands()
    {
        var islands = new List<IEvolutionArchiveView<TestGenome>>();
        string[] prefixes = { "a", "b", "c" };
        for (int island = 0; island < prefixes.Length; island++)
        {
            MapElitesArchive<TestGenome> archive = MapElitesArchiveTests.Archive();
            for (int rank = 0; rank < 4; rank++)
            {
                MapElitesArchiveTests.Add(archive, island * 10 + rank + 1,
                    prefixes[island] + (rank + 1).ToString(System.Globalization.CultureInfo.InvariantCulture),
                    quality: 10 - rank, descriptor: 0.1 + rank * 0.2);
            }
            islands.Add(archive);
        }
        return islands;
    }

    /// <summary>Island 0 holds a strong elite that arrived by migration plus a weaker locally discovered one.</summary>
    private static IReadOnlyList<IEvolutionArchiveView<TestGenome>> IslandsWithAnArrivedMigrant()
    {
        var mixed = MapElitesArchiveTests.Archive();
        AddEntry(mixed, 1, "arrived", quality: 10, descriptor: 0.1, migrationSourceIsland: 1);
        AddEntry(mixed, 2, "local", quality: 5, descriptor: 0.5, migrationSourceIsland: null);
        var neighbour = MapElitesArchiveTests.Archive();
        AddEntry(neighbour, 3, "other", quality: 4, descriptor: 0.1, migrationSourceIsland: null);
        return new List<IEvolutionArchiveView<TestGenome>> { mixed, neighbour };
    }

    private static void AddEntry(MapElitesArchive<TestGenome> archive, long id, string genomeId, double quality,
        double descriptor, int? migrationSourceIsland)
    {
        var lineage = new EvolutionLineage(null, null, "test", null, 0, 0, (ulong)id, migrationSourceIsland);
        var candidate = new EvolutionCandidate<TestGenome>(id,
            new EvolutionCanonicalGenome<TestGenome>(new TestGenome((int)id), genomeId), lineage);
        var evaluation = new EvolutionEvaluation(id, genomeId, EvolutionEvaluationStatus.Completed, quality,
            EvolutionOptimizationDirection.Maximize, new Dictionary<string, double> { ["x"] = descriptor },
            Array.Empty<double>(), Array.Empty<double>(), new EvolutionEvaluationCost(TimeSpan.Zero, 1, 0), lineage,
            EvolutionCacheStatus.Miss, Array.Empty<EvolutionDiagnostic>(), "task", "eval", "config");
        Assert.NotEqual(EvolutionArchiveInsertionResult.Rejected, archive.TryAdd(candidate, evaluation));
    }

    private static EvolutionEngineOptions Options(int maxAttempts, int batchSize, int islandCount) => new()
    {
        RunId = "migration-topology",
        Seed = 77,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = batchSize,
        MaxDegreeOfParallelism = 1,
        IslandCount = islandCount,
        MigrationInterval = 1,
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

    private static MapElitesArchive<TestGenome> TestArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    });

    /// <summary>Emits two transfers for the same island pair, which no topology can justify.</summary>
    private sealed class FloodingMigrationPolicy : IMigrationPolicy<TestGenome>
    {
        public string Id => "flooding";
        public string VersionHash => "flooding-v1";

        public IReadOnlyList<EvolutionMigration<TestGenome>> CreateMigrations(
            IReadOnlyList<IEvolutionArchiveView<TestGenome>> islands, int migrantsPerIsland, StableRandom random)
        {
            var migrations = new List<EvolutionMigration<TestGenome>>();
            foreach (EvolutionArchiveEntry<TestGenome> entry in islands[0].Entries)
            {
                migrations.Add(new EvolutionMigration<TestGenome>(0, 1, entry));
                migrations.Add(new EvolutionMigration<TestGenome>(0, 1, entry));
            }
            return migrations;
        }
    }

    /// <summary>Records the archive changes migration caused, keyed by source, destination, and genome.</summary>
    private sealed class MigrantRecordingObserver : IEvolutionObserver<TestGenome>
    {
        private readonly List<string> _migrants = new();
        private readonly List<EvolutionArchiveInsertionResult> _insertions = new();

        public IReadOnlyList<string> AcceptedMigrants
        {
            get { lock (_migrants) return _migrants.ToArray(); }
        }

        public IReadOnlyList<EvolutionArchiveInsertionResult> AcceptedInsertions
        {
            get { lock (_migrants) return _insertions.ToArray(); }
        }

        public ValueTask OnEventAsync(EvolutionEvent<TestGenome> evolutionEvent,
            CancellationToken cancellationToken = default)
        {
            if (evolutionEvent.Kind != EvolutionEventKind.ArchiveChanged) return default;
            EvolutionLineage? lineage = evolutionEvent.Evaluation?.Lineage;
            if (lineage is null || !lineage.IsMigrant) return default;
            lock (_migrants)
            {
                _migrants.Add($"{lineage.MigrationSourceIsland}->{lineage.Island}:{evolutionEvent.Evaluation?.GenomeId}");
                if (evolutionEvent.InsertionResult.HasValue) _insertions.Add(evolutionEvent.InsertionResult.Value);
            }
            return default;
        }
    }
}
