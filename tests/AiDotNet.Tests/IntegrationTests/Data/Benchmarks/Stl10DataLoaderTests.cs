using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class Stl10DataLoaderTests
{
    private const int ImageSize = 96;
    private const int PixelsPerImage = ImageSize * ImageSize * 3;

    private static string CreateFixture(int trainCount, int testCount)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("stl10");
        string sub = Path.Combine(root, "stl10_binary");
        Directory.CreateDirectory(sub);

        // Synthetic raw byte data — STL-10 ships uncompressed.
        var trainX = new byte[trainCount * PixelsPerImage];
        for (int i = 0; i < trainX.Length; i++) trainX[i] = (byte)(i & 0xFF);
        var trainY = new byte[trainCount];
        for (int i = 0; i < trainCount; i++) trainY[i] = (byte)((i % 10) + 1);

        var testX = new byte[testCount * PixelsPerImage];
        for (int i = 0; i < testX.Length; i++) testX[i] = (byte)((i + 17) & 0xFF);
        var testY = new byte[testCount];
        for (int i = 0; i < testCount; i++) testY[i] = (byte)((i % 10) + 1);

        File.WriteAllBytes(Path.Combine(sub, "train_X.bin"), trainX);
        File.WriteAllBytes(Path.Combine(sub, "train_y.bin"), trainY);
        File.WriteAllBytes(Path.Combine(sub, "test_X.bin"), testX);
        File.WriteAllBytes(Path.Combine(sub, "test_y.bin"), testY);
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitProducesExpectedShape()
    {
        string root = CreateFixture(trainCount: 3, testCount: 2);
        try
        {
            var loader = new Stl10DataLoader<float>(new Stl10DataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await loader.LoadAsync();
            Assert.Equal(3, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 3, 96, 96, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 3, 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitProducesExpectedShape()
    {
        string root = CreateFixture(trainCount: 3, testCount: 2);
        try
        {
            var loader = new Stl10DataLoader<float>(new Stl10DataLoaderOptions
            { DataPath = root, AutoDownload = false, Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 96, 96, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("stl10-miss");
        try
        {
            var loader = new Stl10DataLoader<float>(new Stl10DataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("stl10-net");
        try
        {
            var loader = new Stl10DataLoader<float>(new Stl10DataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
