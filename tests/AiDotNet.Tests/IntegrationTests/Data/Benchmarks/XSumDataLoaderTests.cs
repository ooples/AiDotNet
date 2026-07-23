using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class XSumDataLoaderTests
{
    private static async Task<string> CreateFixtureAsync(string splitName)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("xsum");
        string splitDir = Path.Combine(root, splitName);
        Directory.CreateDirectory(splitDir);

        await DatasetLoaderTestHelpers.WriteStringParquetAsync(
            Path.Combine(splitDir, "0000.parquet"),
            "document", "summary",
            a: new[] { "BBC article body.", "Another BBC piece." },
            b: new[] { "Single-sentence summary one.", "Single-sentence summary two." });
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitParsesParquet()
    {
        string root = await CreateFixtureAsync("train");
        try
        {
            var loader = new XSumDataLoader<float>(new XSumDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxDocumentLength = 16, MaxSummaryLength = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 8);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_NoParquetFiles_Throws()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("xsum-miss");
        try
        {
            var loader = new XSumDataLoader<float>(new XSumDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("xsum-net");
        try
        {
            var loader = new XSumDataLoader<float>(new XSumDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation,
              MaxDocumentLength = 64, MaxSummaryLength = 16, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
