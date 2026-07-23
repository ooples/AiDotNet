using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class OxfordPetsDataLoaderTests
{
    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("oxford-pets");
        string images = Path.Combine(root, "images");
        string anns = Path.Combine(root, "annotations");
        Directory.CreateDirectory(images);
        Directory.CreateDirectory(anns);

        var jpeg = DatasetLoaderTestHelpers.CreateMockJpeg();
        File.WriteAllBytes(Path.Combine(images, "Abyssinian_1.jpg"), jpeg);
        File.WriteAllBytes(Path.Combine(images, "Bengal_1.jpg"), jpeg);

        // <image_id> <class_id 1..37> <species> <breed_id>
        File.WriteAllText(Path.Combine(anns, "trainval.txt"),
            "Abyssinian_1 1 1 1\nBengal_1 4 1 4\n");
        File.WriteAllText(Path.Combine(anns, "test.txt"),
            "Abyssinian_1 1 1 1\n");
        return root;
    }

    [Fact]
    public async Task Fixture_TrainvalSplitLoadsBothBreeds()
    {
        string root = CreateFixture();
        try
        {
            var loader = new OxfordPetsDataLoader<float>(new OxfordPetsDataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 8, 8, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 37);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitMatchesTestTxt()
    {
        string root = CreateFixture();
        try
        {
            var loader = new OxfordPetsDataLoader<float>(new OxfordPetsDataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8, Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingDirThrows()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("oxford-pets-miss");
        try
        {
            var loader = new OxfordPetsDataLoader<float>(new OxfordPetsDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<DirectoryNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadTestSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("oxford-pets-net");
        try
        {
            var loader = new OxfordPetsDataLoader<float>(new OxfordPetsDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test,
              ImageSize = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
