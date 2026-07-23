using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class Enwik8DataLoaderTests
{
    private static string CreateFixture(int byteCount)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("enwik8");
        var bytes = new byte[byteCount];
        for (int i = 0; i < byteCount; i++) bytes[i] = (byte)((i * 7 + 13) & 0xFF);
        File.WriteAllBytes(Path.Combine(root, "enwik8"), bytes);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        // 1MB of synthetic bytes — well within the train (0..90M) range, so the
        // train split walks the same data as the file content.
        string root = CreateFixture(1_024 * 1024);
        try
        {
            var loader = new Enwik8DataLoader<float>(new Enwik8DataLoaderOptions
            { DataPath = root, AutoDownload = false, SequenceLength = 64, MaxSamples = 16 });
            await loader.LoadAsync();
            Assert.Equal(16, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 16, 64);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 16, 64);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplit_ReadsLastFiveMillionRange()
    {
        // For the test split the loader seeks to byte 95_000_000. A 1MB file
        // is shorter than that — should throw.
        string root = CreateFixture(1_024 * 1024);
        try
        {
            var loader = new Enwik8DataLoader<float>(new Enwik8DataLoaderOptions
            { DataPath = root, AutoDownload = false, Split = DatasetSplit.Test, SequenceLength = 32 });
            await Assert.ThrowsAsync<InvalidDataException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("enwik8-miss");
        try
        {
            var loader = new Enwik8DataLoader<float>(new Enwik8DataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadSmallSubset()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("enwik8-net");
        try
        {
            var loader = new Enwik8DataLoader<float>(new Enwik8DataLoaderOptions
            { DataPath = root, AutoDownload = true, SequenceLength = 64, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
