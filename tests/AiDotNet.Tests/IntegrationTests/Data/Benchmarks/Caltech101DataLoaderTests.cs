using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class Caltech101DataLoaderTests
{
    private static string CreateFixture(int imagesPerClass)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("caltech-101");
        string sub = Path.Combine(root, "caltech-101", "101_ObjectCategories");
        Directory.CreateDirectory(sub);
        var jpeg = DatasetLoaderTestHelpers.CreateMockJpeg();
        foreach (string cls in new[] { "accordion", "airplanes" })
        {
            string clsDir = Path.Combine(sub, cls);
            Directory.CreateDirectory(clsDir);
            for (int i = 1; i <= imagesPerClass; i++)
                File.WriteAllBytes(Path.Combine(clsDir, $"image_{i:D4}.jpg"), jpeg);
        }
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitTakesFirstNPerClass()
    {
        // 5 images per class, TrainImagesPerClass=3 → train has 6, test has 4.
        string root = CreateFixture(imagesPerClass: 5);
        try
        {
            var loader = new Caltech101DataLoader<float>(new Caltech101DataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8, TrainImagesPerClass = 3 });
            await loader.LoadAsync();
            Assert.Equal(6, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 6, 8, 8, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 6, 102);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitGetsRest()
    {
        string root = CreateFixture(imagesPerClass: 5);
        try
        {
            var loader = new Caltech101DataLoader<float>(new Caltech101DataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8, TrainImagesPerClass = 3,
              Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(4, loader.TotalCount); // 2 classes × (5-3) = 4
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingClassDirs_ThrowsInvalidOperation()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("caltech101-miss");
        try
        {
            var loader = new Caltech101DataLoader<float>(new Caltech101DataLoaderOptions
            { DataPath = root, AutoDownload = false });
            // The loader's ResolveDataDir falls back to _dataPath when no nested
            // 101_ObjectCategories — _dataPath itself exists empty, so the loader
            // hits the "no class dirs" path, not directory-not-found.
            await Assert.ThrowsAsync<InvalidOperationException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadAndLoadsTestSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("caltech101-net");
        try
        {
            var loader = new Caltech101DataLoader<float>(new Caltech101DataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test,
              ImageSize = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
