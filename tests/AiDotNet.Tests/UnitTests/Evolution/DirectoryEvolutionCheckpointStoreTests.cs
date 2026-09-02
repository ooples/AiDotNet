using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class DirectoryEvolutionCheckpointStoreTests
{
    [Fact]
    public async Task NumberedSnapshotsRoundTripAndTheNewestIsReturned()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path);
        await store.SaveAsync(Checkpoint(1, "one", 1));
        await store.SaveAsync(Checkpoint(2, "two", 2));

        EvolutionCheckpoint loaded = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("run"));

        Assert.Equal(2, loaded.Sequence);
        Assert.Equal("two", loaded.Payload);
        Assert.Equal(2.0, loaded.Quality);
        Assert.Equal(directory.Path, store.DirectoryPath);
        Assert.Empty(Directory.EnumerateFiles(directory.Path, "*.tmp"));
        Assert.True(File.Exists(Path.Combine(directory.Path, DirectoryEvolutionCheckpointStore.FileNameFor(2))));
    }

    [Fact]
    public async Task ACorruptSnapshotIsSkippedAndTheNewestValidOneIsReturned()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 5, keepBest: 5));
        for (int sequence = 1; sequence <= 4; sequence++)
            await store.SaveAsync(Checkpoint(sequence, "payload-" + sequence, sequence));

        File.WriteAllText(Path.Combine(directory.Path, DirectoryEvolutionCheckpointStore.FileNameFor(4)), "{corrupt");
        Assert.Equal(3, (await store.LoadLatestAsync("run"))?.Sequence);

        // A truncated file is as unreadable as a corrupt one, and the journal simply keeps walking backwards.
        string third = Path.Combine(directory.Path, DirectoryEvolutionCheckpointStore.FileNameFor(3));
        File.WriteAllText(third, File.ReadAllText(third).Substring(0, 40));
        EvolutionCheckpoint loaded = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("run"));

        Assert.Equal(2, loaded.Sequence);
        Assert.Equal("payload-2", loaded.Payload);
    }

    [Fact]
    public async Task EnvelopeTamperingAndForeignRunsAreTreatedAsUnreadableRatherThanTrusted()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 5, keepBest: 5));
        await store.SaveAsync(Checkpoint(1, "one", 1));
        await store.SaveAsync(Checkpoint(2, "two", 2));

        string second = Path.Combine(directory.Path, DirectoryEvolutionCheckpointStore.FileNameFor(2));
        string tampered = File.ReadAllText(second).Replace("\"Quality\": 2.0", "\"Quality\": 99.0");
        Assert.NotEqual(File.ReadAllText(second), tampered);
        File.WriteAllText(second, tampered);

        Assert.Equal(1, (await store.LoadLatestAsync("run"))?.Sequence);
        Assert.Null(await store.LoadLatestAsync("another-run"));
    }

    [Fact]
    public async Task RetentionKeepsTheNewestAndTheBestAndDropsTheRest()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 1, keepBest: 1));
        await store.SaveAsync(Checkpoint(1, "one", 5));
        await store.SaveAsync(Checkpoint(2, "two", 9));
        await store.SaveAsync(Checkpoint(3, "three", 1));
        await store.SaveAsync(Checkpoint(4, "four", 2));

        IReadOnlyList<EvolutionCheckpointDescriptor> listed = store.ListCheckpoints("run");

        Assert.Equal(new long[] { 4, 2 }, listed.Select(item => item.Sequence).ToArray());
        Assert.All(listed, item => Assert.True(item.IsValid));
        Assert.Equal(2.0, listed[0].Quality);
        Assert.Equal(9.0, listed[1].Quality);
        Assert.Equal(4, (await store.LoadLatestAsync("run"))?.Sequence);
    }

    [Fact]
    public async Task RetentionNeverDeletesTheBestSnapshotEvenWhenTheBestQuotaIsZero()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 1, keepBest: 0));
        await store.SaveAsync(Checkpoint(1, "one", 9));
        await store.SaveAsync(Checkpoint(2, "two", 1));
        await store.SaveAsync(Checkpoint(3, "three", 2));

        IReadOnlyList<EvolutionCheckpointDescriptor> listed = store.ListCheckpoints("run");

        Assert.Equal(new long[] { 3, 1 }, listed.Select(item => item.Sequence).ToArray());
    }

    [Fact]
    public async Task RetentionRanksQualityInTheCheckpointsOwnDirection()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 1, keepBest: 1));
        await store.SaveAsync(Checkpoint(1, "one", 1, EvolutionOptimizationDirection.Minimize));
        await store.SaveAsync(Checkpoint(2, "two", 9, EvolutionOptimizationDirection.Minimize));
        await store.SaveAsync(Checkpoint(3, "three", 8, EvolutionOptimizationDirection.Minimize));

        IReadOnlyList<EvolutionCheckpointDescriptor> listed = store.ListCheckpoints("run");

        Assert.Equal(new long[] { 3, 1 }, listed.Select(item => item.Sequence).ToArray());
        Assert.Equal(1.0, listed[1].Quality);
        Assert.Equal(EvolutionOptimizationDirection.Minimize, listed[1].QualityDirection);
    }

    [Fact]
    public async Task RetentionNeverDeletesASnapshotItCouldNotRead()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 1, keepBest: 1));
        await store.SaveAsync(Checkpoint(1, "one", 5));
        await store.SaveAsync(Checkpoint(2, "two", 9));
        await store.SaveAsync(Checkpoint(3, "three", 1));
        // Snapshot 2 is the best-quality one and would survive on merit; corrupting it removes that protection, and
        // the next save's retention must still leave the damaged evidence in place.
        File.WriteAllText(Path.Combine(directory.Path, DirectoryEvolutionCheckpointStore.FileNameFor(2)), "{corrupt");
        await store.SaveAsync(Checkpoint(4, "four", 2));

        IReadOnlyList<EvolutionCheckpointDescriptor> listed = store.ListCheckpoints("run");

        Assert.Equal(new long[] { 4, 2 }, listed.Select(item => item.Sequence).ToArray());
        EvolutionCheckpointDescriptor damaged = listed.Single(item => item.Sequence == 2);
        Assert.False(damaged.IsValid);
        Assert.Null(damaged.RunId);
        Assert.Null(damaged.Quality);
        Assert.Null(damaged.QualityDirection);
        Assert.True(damaged.SizeBytes > 0);
        Assert.Equal(DirectoryEvolutionCheckpointStore.FileNameFor(2), damaged.FileName);
        Assert.Equal(4, (await store.LoadLatestAsync("run"))?.Sequence);
    }

    [Fact]
    public async Task TheStoreRefusesRollbackForksAndCompatibilityDrift()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path);
        await store.SaveAsync(Checkpoint(2, "two", 2));
        await store.SaveAsync(Checkpoint(2, "two", 2));

        await Assert.ThrowsAsync<InvalidOperationException>(() => store.SaveAsync(Checkpoint(1, "one", 1)));
        await Assert.ThrowsAsync<InvalidOperationException>(() => store.SaveAsync(Checkpoint(2, "fork", 2)));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            store.SaveAsync(new EvolutionCheckpoint("run", 3, "other-compat", "three")));
        Assert.Single(store.ListCheckpoints("run"));
    }

    [Fact]
    public void InvalidConstructionArgumentsAreRejected()
    {
        using var directory = new TemporaryDirectory();
        Assert.Throws<ArgumentException>(() => new DirectoryEvolutionCheckpointStore("  "));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectoryEvolutionCheckpointStore(directory.Path, maxCheckpointBytes: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 0, keepBest: 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 1, keepBest: -1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => DirectoryEvolutionCheckpointStore.FileNameFor(-1));
    }

    [Fact]
    public async Task TheSizeLimitIsEnforcedBeforeAnythingIsWritten()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, maxCheckpointBytes: 128);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", new string('x', 256))));
        Assert.Empty(store.ListCheckpoints("run"));
        Assert.Empty(Directory.EnumerateFiles(directory.Path, "*.tmp"));
    }

    [Fact]
    public void EachRunGetsItsOwnFolderUnderAnOutputDirectory()
    {
        using var directory = new TemporaryDirectory();
        var first = DirectoryEvolutionCheckpointStore.ForOutputDirectory(directory.Path, "run-a");
        var second = DirectoryEvolutionCheckpointStore.ForOutputDirectory(directory.Path, "run-b");

        Assert.NotEqual(first.DirectoryPath, second.DirectoryPath);
        Assert.StartsWith(Path.Combine(Path.GetFullPath(directory.Path),
            EvolutionOutputLayout.CheckpointsFolderName), first.DirectoryPath, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnEngineResumesFromADirectoryStoreWithARaisedBudget()
    {
        using var directory = new TemporaryDirectory();
        var store = new DirectoryEvolutionCheckpointStore(directory.Path, Retention(keepLast: 2, keepBest: 1));
        TestGenome[] seeds = Enumerable.Range(1, 8).Select(value => new TestGenome(value)).ToArray();

        EvolutionEngineOptions firstOptions = EngineOptions(4);
        firstOptions.CheckpointInterval = 1;
        EvolutionRunResult<TestGenome> first = await Engine(firstOptions, store).RunAsync(seeds);
        Assert.Equal(4, first.Counters.EvaluationAttempts);
        Assert.NotEmpty(store.ListCheckpoints("directory-run"));

        EvolutionEngineOptions resumedOptions = EngineOptions(8);
        resumedOptions.CheckpointInterval = 1;
        resumedOptions.Resume = true;
        EvolutionRunResult<TestGenome> resumed = await Engine(resumedOptions, store).RunAsync(seeds);
        // The control arm keeps its own store so both engines share one compatibility identity.
        EvolutionRunResult<TestGenome> uninterrupted = await Engine(EngineOptions(8),
            new DirectoryEvolutionCheckpointStore(Path.Combine(directory.Path, "control"))).RunAsync(seeds);

        Assert.Equal(8, resumed.Counters.EvaluationAttempts);
        Assert.Equal(uninterrupted.StateHash, resumed.StateHash);
        IReadOnlyList<EvolutionCheckpointDescriptor> listed = store.ListCheckpoints("directory-run");
        Assert.All(listed, item => Assert.True(item.IsValid));
        Assert.All(listed, item => Assert.Equal("directory-run", item.RunId));
    }

    private static EvolutionCheckpointRetentionOptions Retention(int keepLast, int keepBest) =>
        new() { KeepLast = keepLast, KeepBest = keepBest };

    private static EvolutionCheckpoint Checkpoint(long sequence, string payload, double quality,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize) =>
        new("run", sequence, "compat", payload, EvolutionCheckpoint.CurrentSchemaVersion, quality, direction);

    private static EvolutionEngineOptions EngineOptions(int maxAttempts) => new()
    {
        RunId = "directory-run",
        Seed = 77,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = 2,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1
    };

    private static EvolutionEngine<TestGenome> Engine(EvolutionEngineOptions options,
        AiDotNet.Interfaces.IEvolutionCheckpointStore? checkpointStore) => new(
        new SyntheticEvolutionTask(), new IncrementVariation(),
        _ => new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
        }), options,
        checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());
}
