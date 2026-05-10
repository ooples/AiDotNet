using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class AgNewsDataLoaderTests
{
    /// <summary>
    /// Mock CSV: 4 rows, 4 classes covered, embedded quote tests RFC-4180 escapes.
    /// </summary>
    private const string Csv =
        "\"1\",\"World news\",\"A diplomatic incident occurred today.\"\n" +
        "\"2\",\"Sports update\",\"The team won the championship.\"\n" +
        "\"3\",\"Business news\",\"Stocks rose on the day.\"\n" +
        "\"4\",\"Tech news\",\"A new processor was announced with \"\"breakthrough\"\" speed.\"\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("agnews");
        string sub = Path.Combine(root, "ag_news_csv");
        Directory.CreateDirectory(sub);
        File.WriteAllText(Path.Combine(sub, "train.csv"), Csv);
        File.WriteAllText(Path.Combine(sub, "test.csv"), Csv);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new AgNewsDataLoader<float>(new AgNewsDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxSequenceLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(4, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 4, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 4, 4);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_QuoteEscapingHandledCorrectly()
    {
        // Row 4 has an embedded "" (escaped quote). If our CSV parser is broken,
        // we'd lose row 4 or merge it into row 3.
        string root = CreateFixture();
        try
        {
            var loader = new AgNewsDataLoader<float>(new AgNewsDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxSequenceLength = 8 });
            await loader.LoadAsync();
            Assert.Equal(4, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("agnews-miss");
        try
        {
            var loader = new AgNewsDataLoader<float>(new AgNewsDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadAndLoadsTestSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("agnews-net");
        try
        {
            var loader = new AgNewsDataLoader<float>(new AgNewsDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
