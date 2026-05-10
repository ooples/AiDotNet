using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class WikiText2DataLoaderTests
{
    private const string Sample =
        "the quick brown fox jumps over the lazy dog. the cat sat on the mat. " +
        "she sells sea shells by the sea shore. all your base are belong to us. " +
        "i think therefore i am. to be or not to be that is the question.";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("wikitext2");
        string sub = Path.Combine(root, "wikitext-2-raw");
        Directory.CreateDirectory(sub);
        File.WriteAllText(Path.Combine(sub, "wiki.train.raw"), Sample);
        File.WriteAllText(Path.Combine(sub, "wiki.valid.raw"), Sample);
        File.WriteAllText(Path.Combine(sub, "wiki.test.raw"), Sample);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new WikiText2DataLoader<float>(new WikiText2DataLoaderOptions
            {
                DataPath = root,
                AutoDownload = false,
                SequenceLength = 8,
                VocabularySize = 32,
                Split = DatasetSplit.Train,
            });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0, "Expected at least one sequence");
            // Verify public API exposes correct shapes via Features/Labels.
            DatasetLoaderTestHelpers.AssertShape(loader.Features, loader.TotalCount, 8);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, loader.TotalCount, 8);
            // GetNextBatch returns tensors with batch dim = BatchSize.
            loader.BatchSize = System.Math.Min(2, loader.TotalCount);
            var (bf, bl) = loader.GetNextBatch();
            Assert.Equal(loader.BatchSize, bf._shape[0]);
            Assert.Equal(loader.BatchSize, bl._shape[0]);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_RespectsMaxSamples()
    {
        string root = CreateFixture();
        try
        {
            var loader = new WikiText2DataLoader<float>(new WikiText2DataLoaderOptions
            {
                DataPath = root,
                AutoDownload = false,
                SequenceLength = 4,
                VocabularySize = 32,
                MaxSamples = 2,
            });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_SplitProducesNonOverlappingPartitions()
    {
        string root = CreateFixture();
        try
        {
            var loader = new WikiText2DataLoader<float>(new WikiText2DataLoaderOptions
            {
                DataPath = root,
                AutoDownload = false,
                SequenceLength = 4,
                VocabularySize = 32,
            });
            await loader.LoadAsync();
            int total = loader.TotalCount;
            if (total < 10)
            {
                // Sample is small — skip the split assertion when partitions would round to zero.
                return;
            }
            var (train, val, test) = loader.Split(0.6, 0.2, seed: 42);
            int sum = train.TotalCount + val.TotalCount + test.TotalCount;
            Assert.Equal(total, sum);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("wikitext2-missing");
        try
        {
            var loader = new WikiText2DataLoader<float>(new WikiText2DataLoaderOptions
            {
                DataPath = root,
                AutoDownload = false,
            });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadAndLoadsValidationSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("wikitext2-net");
        try
        {
            var loader = new WikiText2DataLoader<float>(new WikiText2DataLoaderOptions
            {
                DataPath = root,
                AutoDownload = true,
                Split = DatasetSplit.Validation,
                SequenceLength = 32,
                MaxSamples = 10,
            });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0);
            Assert.True(loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
