using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class Food101DataLoaderTests
{
    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("food-101");
        string subRoot = Path.Combine(root, "food-101");
        string meta = Path.Combine(subRoot, "meta");
        string images = Path.Combine(subRoot, "images");
        Directory.CreateDirectory(meta);
        Directory.CreateDirectory(images);

        File.WriteAllText(Path.Combine(meta, "classes.txt"), "apple_pie\npizza\n");
        File.WriteAllText(Path.Combine(meta, "train.txt"), "apple_pie/000001\npizza/000001\n");
        File.WriteAllText(Path.Combine(meta, "test.txt"), "apple_pie/000002\n");

        var jpeg = DatasetLoaderTestHelpers.CreateMockJpeg();
        Directory.CreateDirectory(Path.Combine(images, "apple_pie"));
        Directory.CreateDirectory(Path.Combine(images, "pizza"));
        File.WriteAllBytes(Path.Combine(images, "apple_pie", "000001.jpg"), jpeg);
        File.WriteAllBytes(Path.Combine(images, "apple_pie", "000002.jpg"), jpeg);
        File.WriteAllBytes(Path.Combine(images, "pizza", "000001.jpg"), jpeg);
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitMatchesMetaTxt()
    {
        string root = CreateFixture();
        try
        {
            var loader = new Food101DataLoader<float>(new Food101DataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 8, 8, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 101);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitMatchesMetaTxt()
    {
        string root = CreateFixture();
        try
        {
            var loader = new Food101DataLoader<float>(new Food101DataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8, Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingMetaThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("food-101-miss");
        try
        {
            string sub = Path.Combine(root, "food-101");
            Directory.CreateDirectory(Path.Combine(sub, "meta"));
            var loader = new Food101DataLoader<float>(new Food101DataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("food-101-net");
        try
        {
            var loader = new Food101DataLoader<float>(new Food101DataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test,
              ImageSize = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
