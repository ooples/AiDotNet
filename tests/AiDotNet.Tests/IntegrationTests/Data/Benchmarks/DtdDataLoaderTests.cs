using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class DtdDataLoaderTests
{
    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("dtd");
        string sub = Path.Combine(root, "dtd");
        string images = Path.Combine(sub, "images");
        string labels = Path.Combine(sub, "labels");
        Directory.CreateDirectory(images);
        Directory.CreateDirectory(labels);

        var jpeg = DatasetLoaderTestHelpers.CreateMockJpeg();
        Directory.CreateDirectory(Path.Combine(images, "banded"));
        Directory.CreateDirectory(Path.Combine(images, "blotchy"));
        File.WriteAllBytes(Path.Combine(images, "banded", "banded_0001.jpg"), jpeg);
        File.WriteAllBytes(Path.Combine(images, "blotchy", "blotchy_0001.jpg"), jpeg);

        File.WriteAllText(Path.Combine(labels, "train1.txt"),
            "banded/banded_0001.jpg\nblotchy/blotchy_0001.jpg\n");
        File.WriteAllText(Path.Combine(labels, "val1.txt"),
            "banded/banded_0001.jpg\n");
        File.WriteAllText(Path.Combine(labels, "test1.txt"),
            "blotchy/blotchy_0001.jpg\n");
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitLoadsListFile()
    {
        string root = CreateFixture();
        try
        {
            var loader = new DtdDataLoader<float>(new DtdDataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 8, 8, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 47);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_ValSplitMatchesValIndex()
    {
        string root = CreateFixture();
        try
        {
            var loader = new DtdDataLoader<float>(new DtdDataLoaderOptions
            { DataPath = root, AutoDownload = false, ImageSize = 8, Split = DatasetSplit.Validation });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingDirThrows()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("dtd-miss");
        try
        {
            var loader = new DtdDataLoader<float>(new DtdDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("dtd-net");
        try
        {
            var loader = new DtdDataLoader<float>(new DtdDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation,
              ImageSize = 32, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
