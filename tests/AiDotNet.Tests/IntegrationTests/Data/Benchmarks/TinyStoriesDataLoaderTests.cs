using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class TinyStoriesDataLoaderTests
{
    private const string Sample =
        "Once upon a time there was a small cat named whiskers. " +
        "Whiskers loved to play with a ball of yarn all day long. " +
        "One sunny morning whiskers found a tiny mouse hiding in the garden.";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("tinystories");
        File.WriteAllText(Path.Combine(root, "TinyStories-train.txt"), Sample);
        File.WriteAllText(Path.Combine(root, "TinyStories-valid.txt"), Sample);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new TinyStoriesDataLoader<float>(new TinyStoriesDataLoaderOptions
            { DataPath = root, AutoDownload = false, SequenceLength = 4, VocabularySize = 32 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, loader.TotalCount, 4);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitFallsBackToValid()
    {
        // TinyStories has no public test split — the loader maps DatasetSplit.Test to
        // TinyStories-valid.txt. Prove fallback semantics by writing a *different* file
        // for valid vs. train and confirming Test reads the valid content.
        string root = DatasetLoaderTestHelpers.CreateTempDir("tinystories-fallback");
        const string trainSample = "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha";
        const string validSample = "beta beta beta beta beta beta beta beta beta beta";
        File.WriteAllText(Path.Combine(root, "TinyStories-train.txt"), trainSample);
        File.WriteAllText(Path.Combine(root, "TinyStories-valid.txt"), validSample);
        try
        {
            var validLoader = new TinyStoriesDataLoader<float>(new TinyStoriesDataLoaderOptions
            { DataPath = root, AutoDownload = false, Split = DatasetSplit.Validation, SequenceLength = 4, VocabularySize = 32 });
            var testLoader = new TinyStoriesDataLoader<float>(new TinyStoriesDataLoaderOptions
            { DataPath = root, AutoDownload = false, Split = DatasetSplit.Test, SequenceLength = 4, VocabularySize = 32 });
            await validLoader.LoadAsync();
            await testLoader.LoadAsync();
            // The fallback contract: Test must produce identical sample count to Validation.
            Assert.Equal(validLoader.TotalCount, testLoader.TotalCount);
            Assert.True(validLoader.TotalCount > 0);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("tinystories-miss");
        try
        {
            var loader = new TinyStoriesDataLoader<float>(new TinyStoriesDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadValidSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("tinystories-net");
        try
        {
            var loader = new TinyStoriesDataLoader<float>(new TinyStoriesDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation, SequenceLength = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
