using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class Flowers102DataLoaderTests
{
    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("flowers-102");
        // jpg/image_NNNNN.jpg images
        string jpgDir = Path.Combine(root, "jpg");
        Directory.CreateDirectory(jpgDir);
        var jpeg = DatasetLoaderTestHelpers.CreateMockJpeg();
        for (int i = 1; i <= 5; i++)
            File.WriteAllBytes(Path.Combine(jpgDir, $"image_{i:D5}.jpg"), jpeg);

        // imagelabels.mat: variable 'labels' shape (1, 5) double in 1..102
        File.WriteAllBytes(Path.Combine(root, "imagelabels.mat"),
            DatasetLoaderTestHelpers.CreateMockMatFile((b, vars) =>
            {
                var labels = new double[] { 1, 2, 3, 4, 5 };
                vars.Add(b.NewVariable("labels", b.NewArray<double>(labels, new[] { 1, 5 }), false));
            }));

        // setid.mat: trnid/valid/tstid arrays referencing image IDs 1..5
        File.WriteAllBytes(Path.Combine(root, "setid.mat"),
            DatasetLoaderTestHelpers.CreateMockMatFile((b, vars) =>
            {
                vars.Add(b.NewVariable("trnid", b.NewArray<double>(new double[] { 1, 2 }, new[] { 1, 2 }), false));
                vars.Add(b.NewVariable("valid", b.NewArray<double>(new double[] { 3 }, new[] { 1, 1 }), false));
                vars.Add(b.NewVariable("tstid", b.NewArray<double>(new double[] { 4, 5 }, new[] { 1, 2 }), false));
            }));
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitMatchesSetidTrnid()
    {
        string root = CreateFixture();
        try
        {
            var loader = new Flowers102DataLoader<float>(new Flowers102DataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount); // trnid has 2 entries
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 8, 8, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 102);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitMatchesSetidTstid()
    {
        string root = CreateFixture();
        try
        {
            var loader = new Flowers102DataLoader<float>(new Flowers102DataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8, Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount); // tstid has 2 entries
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFiles_Throws()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("flowers-102-miss");
        try
        {
            var loader = new Flowers102DataLoader<float>(new Flowers102DataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadValidationSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("flowers-102-net");
        try
        {
            var loader = new Flowers102DataLoader<float>(new Flowers102DataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation,
              ImageSize = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
