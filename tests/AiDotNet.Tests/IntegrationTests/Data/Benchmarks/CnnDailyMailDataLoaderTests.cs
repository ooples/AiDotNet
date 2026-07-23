using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class CnnDailyMailDataLoaderTests
{
    private static async Task<string> CreateFixtureAsync(string splitName)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("cnn-dm");
        string splitDir = Path.Combine(root, splitName);
        Directory.CreateDirectory(splitDir);

        await DatasetLoaderTestHelpers.WriteStringParquetAsync(
            Path.Combine(splitDir, "0000.parquet"),
            "article", "highlights",
            a: new[] { "First news article about a topic.", "Second article body text." },
            b: new[] { "First summary.", "Second summary." });
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitParsesParquet()
    {
        string root = await CreateFixtureAsync("train");
        try
        {
            var loader = new CnnDailyMailDataLoader<float>(new CnnDailyMailDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxArticleLength = 16, MaxSummaryLength = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 8);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitParsesParquet()
    {
        string root = await CreateFixtureAsync("test");
        try
        {
            var loader = new CnnDailyMailDataLoader<float>(new CnnDailyMailDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxArticleLength = 8, MaxSummaryLength = 4,
              Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_NoParquetFiles_Throws()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("cnn-dm-miss");
        try
        {
            var loader = new CnnDailyMailDataLoader<float>(new CnnDailyMailDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<InvalidDataException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadValidationShard()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("cnn-dm-net");
        try
        {
            var loader = new CnnDailyMailDataLoader<float>(new CnnDailyMailDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation,
              MaxArticleLength = 64, MaxSummaryLength = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
