using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class PennTreebankDataLoaderTests
{
    private const string Sample =
        "the quick brown fox jumps over the lazy dog . the cat sat on the mat . " +
        "she sells sea shells by the sea shore . a bird in the hand is worth two in the bush .";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("ptb");
        string sub = Path.Combine(root, "simple-examples", "data");
        Directory.CreateDirectory(sub);
        File.WriteAllText(Path.Combine(sub, "ptb.train.txt"), Sample);
        File.WriteAllText(Path.Combine(sub, "ptb.valid.txt"), Sample);
        File.WriteAllText(Path.Combine(sub, "ptb.test.txt"), Sample);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new PennTreebankDataLoader<float>(new PennTreebankDataLoaderOptions
            { DataPath = root, AutoDownload = false, SequenceLength = 5, VocabularySize = 32 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, loader.TotalCount, 5);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, loader.TotalCount, 5);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_RespectsMaxSamples()
    {
        string root = CreateFixture();
        try
        {
            var loader = new PennTreebankDataLoader<float>(new PennTreebankDataLoaderOptions
            { DataPath = root, AutoDownload = false, SequenceLength = 4, MaxSamples = 2 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("ptb-miss");
        try
        {
            var loader = new PennTreebankDataLoader<float>(new PennTreebankDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadAndLoadsValidationSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("ptb-net");
        try
        {
            var loader = new PennTreebankDataLoader<float>(new PennTreebankDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
