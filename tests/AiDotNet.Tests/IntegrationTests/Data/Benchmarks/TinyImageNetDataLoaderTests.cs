using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class TinyImageNetDataLoaderTests
{
    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("tiny-imagenet");
        string sub = Path.Combine(root, "tiny-imagenet-200");
        Directory.CreateDirectory(sub);

        // wnids.txt — two synthetic class IDs.
        File.WriteAllText(Path.Combine(sub, "wnids.txt"), "n00000001\nn00000002\n");

        // Train: train/n00000001/images/img1.JPEG, train/n00000002/images/img2.JPEG
        var jpeg = DatasetLoaderTestHelpers.CreateMockJpeg();
        foreach (string wnid in new[] { "n00000001", "n00000002" })
        {
            string dir = Path.Combine(sub, "train", wnid, "images");
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, $"{wnid}_1.JPEG"), jpeg);
        }

        // Val: val/images/*.JPEG + val/val_annotations.txt
        string valImages = Path.Combine(sub, "val", "images");
        Directory.CreateDirectory(valImages);
        File.WriteAllBytes(Path.Combine(valImages, "val_0.JPEG"), jpeg);
        File.WriteAllText(Path.Combine(sub, "val", "val_annotations.txt"),
            "val_0.JPEG\tn00000001\t0\t0\t10\t10\n");

        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitLoadsClassSubdirs()
    {
        string root = CreateFixture();
        try
        {
            var loader = new TinyImageNetDataLoader<float>(new TinyImageNetDataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount); // one image per class
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 8, 8, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 200);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_ValSplitUsesAnnotationsTxt()
    {
        string root = CreateFixture();
        try
        {
            var loader = new TinyImageNetDataLoader<float>(new TinyImageNetDataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8, Split = DatasetSplit.Validation });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount); // one annotated val image
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFiles_ThrowsAppropriateException()
    {
        // Empty temp dir exists but has no wnids.txt — loader rejects with
        // FileNotFoundException (not DirectoryNotFound, since DataPath itself exists).
        string root = DatasetLoaderTestHelpers.CreateTempDir("tinyimagenet-miss");
        try
        {
            var loader = new TinyImageNetDataLoader<float>(new TinyImageNetDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("tinyimagenet-net");
        try
        {
            var loader = new TinyImageNetDataLoader<float>(new TinyImageNetDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation,
              ImageSize = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
