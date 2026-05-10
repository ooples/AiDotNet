using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class CityscapesDataLoaderTests
{
    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("cityscapes");
        // leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        // gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        string imgDir = Path.Combine(root, "leftImg8bit", "train", "aachen");
        string lblDir = Path.Combine(root, "gtFine", "train", "aachen");
        Directory.CreateDirectory(imgDir);
        Directory.CreateDirectory(lblDir);

        var png = DatasetLoaderTestHelpers.CreateMockPng();
        File.WriteAllBytes(Path.Combine(imgDir, "aachen_000000_000019_leftImg8bit.png"), png);
        File.WriteAllBytes(Path.Combine(lblDir, "aachen_000000_000019_gtFine_labelIds.png"), png);
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitProducesPerPixelLabelShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new CityscapesDataLoader<float>(new CityscapesDataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageHeight = 8, ImageWidth = 16 });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount);
            // Industry-standard segmentation shape: features [N, H, W, 3], labels [N, H, W].
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 1, 8, 16, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 1, 8, 16);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task AutoDownloadAttempt_Throws()
    {
        // Cityscapes requires sign-up; AutoDownload must be off.
        var loader = new CityscapesDataLoader<float>(new CityscapesDataLoaderOptions
        { AutoDownload = true });
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await loader.LoadAsync());
    }

    [Fact]
    public async Task Fixture_MissingDirThrowsDirectoryNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("cityscapes-miss");
        try
        {
            var loader = new CityscapesDataLoader<float>(new CityscapesDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<DirectoryNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    // No network test — Cityscapes requires manual sign-up.
}
