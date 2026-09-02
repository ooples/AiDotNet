using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Programs.ArtifactStore;
using AiDotNet.Interfaces;
using AiDotNetTests.UnitTests.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramArtifactStoreTests
{
    private const string GenomeId = "9f2c4a7b1d0e5f63a8c9b0d1e2f3a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5";

    private static ProgramArtifactStoreOptions SmallThreshold(int threshold) =>
        new() { InlineSizeThresholdBytes = threshold };

    [Fact]
    public async Task SmallArtifactsStayInlineAndLargeOnesGetTheirOwnFile()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(16));

        IReadOnlyList<ProgramArtifactDescriptor> written = await store.StoreAsync(GenomeId, new[]
        {
            ProgramArtifact.FromText("stderr", "boom"),
            ProgramArtifact.FromText("stdout", new string('x', 64))
        });

        Assert.Equal(2, written.Count);
        Assert.Equal(ProgramArtifactTier.Inline, written.First(item => item.Name == "stderr").Tier);
        Assert.Equal(ProgramArtifactTier.OnDisk, written.First(item => item.Name == "stdout").Tier);

        string genomeDirectory = store.GetGenomeDirectory(GenomeId);
        Assert.True(File.Exists(Path.Combine(genomeDirectory, FileSystemProgramArtifactStore.IndexFileName)));
        Assert.Single(Directory.EnumerateFiles(genomeDirectory, "*.bin"));
    }

    [Fact]
    public async Task ArtifactsAreRetrievableByGenomeIdAfterTheStoreIsRecreated()
    {
        using var directory = new TemporaryDirectory();
        var writer = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(8));
        await writer.StoreAsync(GenomeId, new[]
        {
            ProgramArtifact.FromText("stderr", "ZeroDivisionError"),
            ProgramArtifact.FromText("note", "ok")
        });

        // A separate instance stands in for a later process reading what a finished run left behind.
        IProgramArtifactStore reader = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(8));
        IReadOnlyList<ProgramArtifact> loaded = await reader.GetAsync(GenomeId);

        Assert.Equal(new[] { "note", "stderr" }, loaded.Select(item => item.Name));
        Assert.Equal("ZeroDivisionError", loaded.First(item => item.Name == "stderr").GetText());
    }

    [Fact]
    public async Task BinaryContentSurvivesTheRoundTripByteForByte()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(4));
        var payload = new byte[512];
        for (int index = 0; index < payload.Length; index++) payload[index] = (byte)(index % 256);

        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromBytes("profile", payload) });
        ProgramArtifact? loaded = await store.GetAsync(GenomeId, "profile");

        Assert.NotNull(loaded);
        Assert.False(loaded.IsText);
        Assert.Equal(payload, loaded.Content);
    }

    [Fact]
    public async Task InlineBinaryContentSurvivesTheRoundTripToo()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path);
        var payload = new byte[] { 0x00, 0xFF, 0x10, 0x7F, 0x80 };

        IReadOnlyList<ProgramArtifactDescriptor> written =
            await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromBytes("tiny", payload) });
        ProgramArtifact? loaded = await store.GetAsync(GenomeId, "tiny");

        Assert.Equal(ProgramArtifactTier.Inline, Assert.Single(written).Tier);
        Assert.NotNull(loaded);
        Assert.Equal(payload, loaded.Content);
    }

    [Fact]
    public async Task ListingReportsSizesAndTiersWithoutReadingTheContent()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(16));
        await store.StoreAsync(GenomeId, new[]
        {
            ProgramArtifact.FromText("big", new string('y', 100)),
            ProgramArtifact.FromText("small", "hi")
        });

        IReadOnlyList<ProgramArtifactDescriptor> listed = await store.ListAsync(GenomeId);

        Assert.Equal(new[] { "big", "small" }, listed.Select(item => item.Name));
        Assert.Equal(100, listed[0].ByteLength);
        Assert.Equal(ProgramArtifactTier.OnDisk, listed[0].Tier);
        Assert.Equal(ProgramArtifactTier.Inline, listed[1].Tier);
        Assert.Equal(GenomeId, listed[0].GenomeId);
    }

    [Fact]
    public async Task StoringTheSameNameTwiceReplacesItAndLeavesNoOrphanFile()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(4));
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stdout", new string('a', 50)) });
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stdout", new string('b', 60)) });

        ProgramArtifact? loaded = await store.GetAsync(GenomeId, "stdout");
        Assert.NotNull(loaded);
        Assert.Equal(new string('b', 60), loaded.GetText());
        Assert.Single(await store.ListAsync(GenomeId));
        Assert.Single(Directory.EnumerateFiles(store.GetGenomeDirectory(GenomeId), "*.bin"));
    }

    [Fact]
    public async Task OversizedContentIsTruncatedAndFlaggedRatherThanLost()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path,
            new ProgramArtifactStoreOptions { InlineSizeThresholdBytes = 8, MaxArtifactBytes = 32 });

        ProgramArtifactDescriptor descriptor = Assert.Single(
            await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stdout", new string('z', 500)) }));

        Assert.True(descriptor.IsTruncated);
        Assert.Equal(32, descriptor.ByteLength);
        ProgramArtifact? loaded = await store.GetAsync(GenomeId, "stdout");
        Assert.NotNull(loaded);
        Assert.True(loaded.IsTruncated);
        Assert.Equal(32, loaded.ByteLength);
    }

    [Fact]
    public async Task ExceedingTheArtifactCountIsRejectedInsteadOfDroppingEvidence()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path,
            new ProgramArtifactStoreOptions { MaxArtifactsPerGenome = 2 });

        await Assert.ThrowsAsync<InvalidOperationException>(() => store.StoreAsync(GenomeId, new[]
        {
            ProgramArtifact.FromText("a", "1"),
            ProgramArtifact.FromText("b", "2"),
            ProgramArtifact.FromText("c", "3")
        }));

        Assert.Empty(await store.ListAsync(GenomeId));
    }

    [Fact]
    public async Task ExceedingTheGenomeByteBudgetIsRejected()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path,
            new ProgramArtifactStoreOptions { MaxTotalBytesPerGenome = 16, MaxArtifactBytes = 64 });

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stdout", new string('q', 32)) }));
    }

    [Fact]
    public async Task RetentionRemovesArtifactsOlderThanTheRetentionPeriodFromTheSameRootTheyWereWrittenTo()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path,
            new ProgramArtifactStoreOptions { RetentionPeriod = TimeSpan.FromDays(30) });
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stderr", "boom") });

        Assert.Equal(0, await store.PurgeAsync(DateTimeOffset.UtcNow.AddDays(29)));
        Assert.Single(await store.ListAsync(GenomeId));

        Assert.Equal(1, await store.PurgeAsync(DateTimeOffset.UtcNow.AddDays(31)));
        Assert.Empty(await store.ListAsync(GenomeId));
        Assert.False(Directory.Exists(store.GetGenomeDirectory(GenomeId)));
    }

    [Fact]
    public async Task ZeroRetentionPeriodDisablesAgeBasedExpiry()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path,
            new ProgramArtifactStoreOptions { RetentionPeriod = TimeSpan.Zero });
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stderr", "boom") });

        Assert.Equal(0, await store.PurgeAsync(DateTimeOffset.UtcNow.AddYears(10)));
        Assert.Single(await store.ListAsync(GenomeId));
    }

    [Fact]
    public async Task RetentionAlsoCapsHowManyGenomesKeepArtifacts()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path,
            new ProgramArtifactStoreOptions { RetentionPeriod = TimeSpan.Zero, MaxRetainedGenomes = 2 });
        await store.StoreAsync("genome-one", new[] { ProgramArtifact.FromText("out", "1") });
        await store.StoreAsync("genome-two", new[] { ProgramArtifact.FromText("out", "2") });
        await store.StoreAsync("genome-three", new[] { ProgramArtifact.FromText("out", "3") });

        Assert.Equal(1, await store.PurgeAsync(DateTimeOffset.UtcNow));
        Assert.Equal(2, Directory.EnumerateDirectories(directory.Path).Count());
    }

    [Fact]
    public async Task RemovingOneGenomeLeavesTheOthersAlone()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path);
        await store.StoreAsync("genome-one", new[] { ProgramArtifact.FromText("out", "1") });
        await store.StoreAsync("genome-two", new[] { ProgramArtifact.FromText("out", "2") });

        Assert.True(await store.RemoveAsync("genome-one"));
        Assert.False(await store.RemoveAsync("genome-one"));
        Assert.Empty(await store.GetAsync("genome-one"));
        Assert.Single(await store.GetAsync("genome-two"));
    }

    [Fact]
    public async Task AnUnknownGenomeReadsBackAsEmptyRatherThanThrowing()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path);

        Assert.Empty(await store.GetAsync(GenomeId));
        Assert.Empty(await store.ListAsync(GenomeId));
        Assert.Null(await store.GetAsync(GenomeId, "stdout"));
    }

    [Fact]
    public void AHostileGenomeIdCannotEscapeTheStoreRoot()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path);

        string escaped = store.GetGenomeDirectory("../../etc/passwd");
        string traversal = store.GetGenomeDirectory("..");

        Assert.StartsWith(store.RootDirectory, escaped, StringComparison.Ordinal);
        Assert.StartsWith(store.RootDirectory, traversal, StringComparison.Ordinal);
        Assert.NotEqual(store.GetGenomeDirectory("etcpasswd"), escaped);
    }

    [Fact]
    public async Task AHostileArtifactNameCannotEscapeTheGenomeDirectory()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(1));
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("../../escape.txt", "payload") });

        string genomeDirectory = store.GetGenomeDirectory(GenomeId);
        string written = Assert.Single(Directory.EnumerateFiles(genomeDirectory, "*.bin"));

        Assert.Equal(genomeDirectory, Path.GetDirectoryName(written));
        ProgramArtifact? loaded = await store.GetAsync(GenomeId, "../../escape.txt");
        Assert.NotNull(loaded);
        Assert.Equal("payload", loaded.GetText());
    }

    [Fact]
    public async Task TheIndexIsWrittenAtomicallyAndLeavesNoTemporaryFiles()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path, SmallThreshold(4));
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stdout", new string('a', 40)) });
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stderr", "small") });

        Assert.Empty(Directory.EnumerateFiles(store.GetGenomeDirectory(GenomeId), "*.tmp"));
        Assert.Equal(2, (await store.ListAsync(GenomeId)).Count);
    }

    [Fact]
    public async Task ACorruptIndexReadsBackAsEmptyRatherThanThrowing()
    {
        using var directory = new TemporaryDirectory();
        var store = new FileSystemProgramArtifactStore(directory.Path);
        await store.StoreAsync(GenomeId, new[] { ProgramArtifact.FromText("stdout", "value") });
        File.WriteAllText(
            Path.Combine(store.GetGenomeDirectory(GenomeId), FileSystemProgramArtifactStore.IndexFileName),
            "{corrupt",
            Encoding.UTF8);

        Assert.Empty(await store.GetAsync(GenomeId));
    }

    [Fact]
    public void TheDefaultThresholdAndRetentionMatchTheReferenceImplementation()
    {
        var options = new ProgramArtifactStoreOptions();

        Assert.Equal(32 * 1024, options.InlineSizeThresholdBytes);
        Assert.Equal(TimeSpan.FromDays(30), options.RetentionPeriod);
    }

    [Fact]
    public void InvalidLimitsAreRejectedAtConstructionTime()
    {
        using var directory = new TemporaryDirectory();

        Assert.Throws<ArgumentOutOfRangeException>(() => new FileSystemProgramArtifactStore(
            directory.Path, new ProgramArtifactStoreOptions { MaxArtifactBytes = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FileSystemProgramArtifactStore(
            directory.Path, new ProgramArtifactStoreOptions { RetentionPeriod = TimeSpan.FromDays(-1) }));
    }

    [Fact]
    public void ArtifactNamesAndContentAreValidatedUpFront()
    {
        Assert.Throws<ArgumentException>(() => ProgramArtifact.FromText(" ", "value"));
        Assert.Throws<ArgumentException>(() => ProgramArtifact.FromText(new string('n', 200), "value"));
        Assert.Throws<ArgumentOutOfRangeException>(() => ProgramArtifact.FromText("name", "value").Truncate(0));
        Assert.DoesNotContain("value", ProgramArtifact.FromText("name", "value").ToString(), StringComparison.Ordinal);
    }
}
