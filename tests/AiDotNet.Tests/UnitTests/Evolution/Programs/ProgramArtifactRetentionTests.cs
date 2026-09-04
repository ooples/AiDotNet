using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.ArtifactStore;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers actually applying the retention the caller configured. The store has always known what to keep and how to
/// sweep, and nothing ever called the sweep, so a run's artifact directory grew without bound however the retention
/// period was set.
/// </summary>
public sealed class ProgramArtifactRetentionTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "aidotnet-artifact-retention-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public async Task TheRunSweepsOnTheConfiguredCadenceRatherThanNever()
    {
        var store = new RecordingArtifactStore();
        var observer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 2);

        await Store(observer, "a");
        Assert.Equal(0, observer.SweepCount);

        await Store(observer, "b");
        Assert.Equal(1, observer.SweepCount);

        await Store(observer, "c");
        await Store(observer, "d");
        Assert.Equal(2, observer.SweepCount);
    }

    [Fact]
    public async Task ASweepRunsWhenTheSearchStopsWhateverTheCadence()
    {
        // A short run never reaches the cadence, and it is the last chance to apply the configured retention.
        var store = new RecordingArtifactStore();
        var observer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 1_000);

        await Store(observer, "a");
        Assert.Equal(0, observer.SweepCount);

        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(
            EvolutionEventKind.Stopped, 1, message: "stopped"));

        Assert.Equal(1, observer.SweepCount);
    }

    [Fact]
    public async Task ACadenceOfZeroSweepsOnlyAtTheEnd()
    {
        var store = new RecordingArtifactStore();
        var observer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 0);

        for (int index = 0; index < 5; index++) await Store(observer, "genome-" + index);
        Assert.Equal(0, observer.SweepCount);

        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(
            EvolutionEventKind.Stopped, 1, message: "stopped"));

        Assert.Equal(1, observer.SweepCount);
    }

    [Fact]
    public async Task WhatTheSweepRemovedIsReported()
    {
        var store = new RecordingArtifactStore { RemovePerPurge = 3 };
        var observer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 1);

        await Store(observer, "a");
        await Store(observer, "b");

        Assert.Equal(2, observer.SweepCount);
        Assert.Equal(6, observer.RemovedCount);
    }

    [Fact]
    public async Task AFailedWriteDoesNotAdvanceTheCadence()
    {
        // Counting a failed write towards the sweep would sweep a store nothing was successfully written to, and on
        // a persistently failing disk it would keep doing so on every event.
        var store = new RecordingArtifactStore { FailWrites = true };
        var observer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 1);

        await Store(observer, "a");

        Assert.Equal(1, observer.FailureCount);
        Assert.Equal(0, observer.SweepCount);
    }

    [Fact]
    public async Task AFailedSweepIsCountedRatherThanEndingTheRun()
    {
        // Retention is housekeeping, and housekeeping runs inside the engine's commit path.
        var store = new RecordingArtifactStore { FailPurges = true };
        var observer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 1);

        await Store(observer, "a");

        Assert.Equal(1, observer.StoredCount);
        Assert.Equal(0, observer.SweepCount);
        Assert.Equal(1, observer.FailureCount);
        Assert.NotNull(observer.LastError);
    }

    [Fact]
    public async Task TheSweepUsesTheSuppliedClockSoAgeBasedRemovalIsTestable()
    {
        var now = new DateTimeOffset(2026, 9, 3, 12, 0, 0, TimeSpan.Zero);
        var store = new RecordingArtifactStore();
        var observer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 1, clock: () => now);

        await Store(observer, "a");

        Assert.Equal(now, Assert.Single(store.PurgeTimes));
    }

    [Fact]
    public async Task AnAgedArtifactIsActuallyRemovedFromDiskByTheSweep()
    {
        // End to end against the real store: written, aged past the retention period, swept away by the run.
        //
        // The sweep time is derived from the REAL clock rather than a fixed date, because the store stamps
        // StoredAtUtc itself and the observer's injected clock does not reach it. Comparing a real write time
        // against a fixed fake sweep time made this test depend on the hour it ran: it passed one day and failed
        // the next, when the gap between the two happened to fall under the retention period.
        var store = new FileSystemProgramArtifactStore(_root, new ProgramArtifactStoreOptions
        {
            InlineSizeThresholdBytes = 0,
            RetentionPeriod = TimeSpan.FromDays(1)
        });

        var writer = new ProgramArtifactStoreObserver(store, purgeEveryStores: 0);
        await Store(writer, "old-genome");
        Assert.NotEmpty(await store.GetAsync("old-genome"));

        DateTimeOffset later = DateTimeOffset.UtcNow.AddDays(2);
        var sweeper = new ProgramArtifactStoreObserver(store, purgeEveryStores: 0, clock: () => later);
        await sweeper.OnEventAsync(new EvolutionEvent<ProgramGenome>(
            EvolutionEventKind.Stopped, 1, message: "stopped"));

        Assert.Equal(1, sweeper.SweepCount);
        Assert.Equal(1, sweeper.RemovedCount);
        Assert.Empty(await store.GetAsync("old-genome"));
    }

    [Fact]
    public void ANegativeCadenceIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ProgramArtifactStoreObserver(new RecordingArtifactStore(), purgeEveryStores: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ProgramArtifactStoreOptions { PurgeEveryStores = -1 }.Validate());
    }

    [Fact]
    public void TheCadenceSurvivesAnOptionsCopy()
    {
        var options = new ProgramArtifactStoreOptions { PurgeEveryStores = 7 };
        Assert.Equal(7, options.Clone().PurgeEveryStores);
    }

    private static ValueTask Store(ProgramArtifactStoreObserver observer, string genomeId) =>
        observer.OnEventAsync(EvaluatedEvent(genomeId, new EvolutionArtifact("stderr", "output for " + genomeId)));

    private static EvolutionEvent<ProgramGenome> EvaluatedEvent(string genomeId, params EvolutionArtifact[] artifacts)
    {
        var genome = new ProgramGenome("def solve(x):\n    return x\n", ProgramLanguage.Python);
        var canonical = new EvolutionCanonicalGenome<ProgramGenome>(genome, genomeId);
        var lineage = new EvolutionLineage(
            Array.Empty<string>(), Array.Empty<string>(), "variation", null, 1, 0, 1UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(1, canonical, lineage);
        var evaluation = new EvolutionEvaluation(
            1,
            genomeId,
            EvolutionEvaluationStatus.Completed,
            0.5,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal),
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-hash",
            "evaluator-hash",
            "configuration-hash",
            artifacts: artifacts);

        return new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Evaluated, 1, candidate, evaluation);
    }

    /// <summary>A store that records what the observer asked of it and can be told to fail either operation.</summary>
    private sealed class RecordingArtifactStore : IProgramArtifactStore
    {
        public List<DateTimeOffset> PurgeTimes { get; } = new();

        public int RemovePerPurge { get; set; }

        public bool FailWrites { get; set; }

        public bool FailPurges { get; set; }

        public ProgramArtifactStoreOptions GetOptions() => new();

        public Task<IReadOnlyList<ProgramArtifactDescriptor>> StoreAsync(
            string genomeId, IEnumerable<ProgramArtifact> artifacts, CancellationToken cancellationToken = default)
        {
            if (FailWrites) throw new IOException("the disk is full");
            return Task.FromResult<IReadOnlyList<ProgramArtifactDescriptor>>(Array.Empty<ProgramArtifactDescriptor>());
        }

        public Task<IReadOnlyList<ProgramArtifact>> GetAsync(
            string genomeId, CancellationToken cancellationToken = default) =>
            Task.FromResult<IReadOnlyList<ProgramArtifact>>(Array.Empty<ProgramArtifact>());

        public Task<ProgramArtifact?> GetAsync(
            string genomeId, string name, CancellationToken cancellationToken = default) =>
            Task.FromResult<ProgramArtifact?>(null);

        public Task<IReadOnlyList<ProgramArtifactDescriptor>> ListAsync(
            string genomeId, CancellationToken cancellationToken = default) =>
            Task.FromResult<IReadOnlyList<ProgramArtifactDescriptor>>(Array.Empty<ProgramArtifactDescriptor>());

        public Task<bool> RemoveAsync(string genomeId, CancellationToken cancellationToken = default) =>
            Task.FromResult(false);

        public Task<int> PurgeAsync(DateTimeOffset utcNow, CancellationToken cancellationToken = default)
        {
            if (FailPurges) throw new IOException("the directory vanished");
            PurgeTimes.Add(utcNow);
            return Task.FromResult(RemovePerPurge);
        }
    }
}
