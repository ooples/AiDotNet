using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class HellaswagDataLoaderTests
{
    private const string Jsonl =
        "{\"ctx\": \"A man is walking down the street.\", \"endings\": [\"He buys a coffee.\", \"He flies away.\", \"He turns into a bird.\", \"He explodes.\"], \"label\": 0}\n" +
        "{\"ctx\": \"A girl is making cookies.\", \"endings\": [\"She mixes flour and butter.\", \"She rides a bike.\", \"She takes a nap.\", \"She fights a bear.\"], \"label\": 0}\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("hellaswag");
        File.WriteAllText(Path.Combine(root, "hellaswag_train.jsonl"), Jsonl);
        File.WriteAllText(Path.Combine(root, "hellaswag_val.jsonl"), Jsonl);
        File.WriteAllText(Path.Combine(root, "hellaswag_test.jsonl"), Jsonl);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new HellaswagDataLoader<float>(new HellaswagDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxSequenceLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            // Multi-choice shape: [N, 4, MaxSeqLen]
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 4, 16);
            // Labels: [N, 4] one-hot
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 4);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_RejectsRowsWithWrongChoiceCount()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("hellaswag-bad");
        try
        {
            File.WriteAllText(Path.Combine(root, "hellaswag_train.jsonl"),
                "{\"ctx\": \"OK1\", \"endings\": [\"a\", \"b\", \"c\", \"d\"], \"label\": 0}\n" +
                "{\"ctx\": \"BAD\", \"endings\": [\"a\", \"b\"], \"label\": 0}\n" +  // only 2 endings — skipped
                "{\"ctx\": \"OK2\", \"endings\": [\"a\", \"b\", \"c\", \"d\"], \"label\": 1}\n");
            var loader = new HellaswagDataLoader<float>(new HellaswagDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxSequenceLength = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("hellaswag-miss");
        try
        {
            var loader = new HellaswagDataLoader<float>(new HellaswagDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadValSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("hellaswag-net");
        try
        {
            var loader = new HellaswagDataLoader<float>(new HellaswagDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
