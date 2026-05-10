using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class Gsm8kDataLoaderTests
{
    private const string Jsonl =
        "{\"question\": \"Natalia sold clips to 48 of her friends.\", \"answer\": \"She sold half as many in May. #### 72\"}\n" +
        "{\"question\": \"Weng earns 12 dollars an hour for babysitting.\", \"answer\": \"Working 50 minutes is 5/6 of an hour. #### 10\"}\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("gsm8k");
        File.WriteAllText(Path.Combine(root, "train.jsonl"), Jsonl);
        File.WriteAllText(Path.Combine(root, "test.jsonl"), Jsonl);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new Gsm8kDataLoader<float>(new Gsm8kDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxQuestionLength = 16, MaxAnswerLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 16);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MalformedJsonlSkippedNotThrown()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("gsm8k-malformed");
        try
        {
            File.WriteAllText(Path.Combine(root, "train.jsonl"),
                "{\"question\": \"OK\", \"answer\": \"valid\"}\n" +
                "this is not json\n" +
                "{\"question\": \"OK2\", \"answer\": \"valid2\"}\n");
            var loader = new Gsm8kDataLoader<float>(new Gsm8kDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxQuestionLength = 8, MaxAnswerLength = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount); // malformed line skipped
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("gsm8k-miss");
        try
        {
            var loader = new Gsm8kDataLoader<float>(new Gsm8kDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadTestSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("gsm8k-net");
        try
        {
            var loader = new Gsm8kDataLoader<float>(new Gsm8kDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
