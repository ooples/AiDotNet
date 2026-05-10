using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class SvhnDataLoaderTests
{
    /// <summary>
    /// Builds a synthetic SVHN .mat with the canonical schema:
    ///   X: uint8 [32, 32, 3, n] — column-major (Fortran) order
    ///   y: int8 [n, 1] — labels in 1..10
    /// </summary>
    private static byte[] BuildSvhnMat(int n)
    {
        return DatasetLoaderTestHelpers.CreateMockMatFile((b, vars) =>
        {
            var X = new byte[32 * 32 * 3 * n];
            for (int i = 0; i < X.Length; i++) X[i] = (byte)(i & 0xFF);
            var y = new sbyte[n];
            for (int i = 0; i < n; i++) y[i] = (sbyte)((i % 10) + 1);

            vars.Add(b.NewVariable("X", b.NewArray<byte>(X, new[] { 32, 32, 3, n }), false));
            vars.Add(b.NewVariable("y", b.NewArray<sbyte>(y, new[] { n, 1 }), false));
        });
    }

    [Fact]
    public async Task Fixture_TrainSplitDecodesMatFormat()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("svhn");
        try
        {
            File.WriteAllBytes(Path.Combine(root, "train_32x32.mat"), BuildSvhnMat(n: 5));
            File.WriteAllBytes(Path.Combine(root, "test_32x32.mat"), BuildSvhnMat(n: 3));

            var loader = new SvhnDataLoader<float>(new SvhnDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await loader.LoadAsync();
            Assert.Equal(5, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 5, 32, 32, 3);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 5, 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitMatchesSize()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("svhn-test");
        try
        {
            File.WriteAllBytes(Path.Combine(root, "train_32x32.mat"), BuildSvhnMat(n: 5));
            File.WriteAllBytes(Path.Combine(root, "test_32x32.mat"), BuildSvhnMat(n: 3));

            var loader = new SvhnDataLoader<float>(new SvhnDataLoaderOptions
            { DataPath = root, AutoDownload = false, Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(3, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("svhn-miss");
        try
        {
            var loader = new SvhnDataLoader<float>(new SvhnDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("svhn-net");
        try
        {
            var loader = new SvhnDataLoader<float>(new SvhnDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
