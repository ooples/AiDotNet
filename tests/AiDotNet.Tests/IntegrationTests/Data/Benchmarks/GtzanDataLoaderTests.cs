using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Geometry;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class GtzanDataLoaderTests
{
    private static string CreateFixture(int clipsPerGenre)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("gtzan");
        string genres = Path.Combine(root, "genres");
        Directory.CreateDirectory(genres);
        var wav = DatasetLoaderTestHelpers.CreateMockWav(sampleCount: 1000);
        // Need at least 2 genres for the per-class deterministic split to be meaningful.
        foreach (string g in new[] { "blues", "classical" })
        {
            string gd = Path.Combine(genres, g);
            Directory.CreateDirectory(gd);
            for (int i = 0; i < clipsPerGenre; i++)
                File.WriteAllBytes(Path.Combine(gd, $"{g}.{i:D5}.wav"), wav);
        }
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitTakesFirst80Pct()
    {
        // 5 clips per genre × 2 genres = 10. TrainFraction=0.8 → train=4 per genre × 2 = 8.
        string root = CreateFixture(clipsPerGenre: 5);
        try
        {
            var loader = new GtzanDataLoader<float>(new GtzanDataLoaderOptions
            { DataPath = root, AutoDownload = false, Samples = 1000, TrainFraction = 0.8 });
            await loader.LoadAsync();
            Assert.Equal(8, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 8, 1000);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 8, 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitGetsRest()
    {
        string root = CreateFixture(clipsPerGenre: 5);
        try
        {
            var loader = new GtzanDataLoader<float>(new GtzanDataLoaderOptions
            { DataPath = root, AutoDownload = false, Samples = 1000, TrainFraction = 0.8,
              Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount); // 1 per genre × 2
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingDirSucceedsWithZeroSamples()
    {
        // Empty DataPath exists but no genres/ subdir — ResolveDataDir falls back
        // to _dataPath itself which is empty. Loader walks zero genre subdirs and
        // produces a zero-row dataset (no error path).
        string root = DatasetLoaderTestHelpers.CreateTempDir("gtzan-miss");
        try
        {
            var loader = new GtzanDataLoader<float>(new GtzanDataLoaderOptions
            { DataPath = root, AutoDownload = false, Samples = 100 });
            await loader.LoadAsync();
            Assert.Equal(0, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadTestSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("gtzan-net");
        try
        {
            var loader = new GtzanDataLoader<float>(new GtzanDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
