using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class MathDataLoaderTests
{
    private const string Json1 =
        "{\"problem\": \"What is 2+2?\", \"level\": \"Level 1\", \"type\": \"Algebra\", \"solution\": \"4\"}";
    private const string Json2 =
        "{\"problem\": \"What is the area of a circle with radius 3?\", \"level\": \"Level 3\", \"type\": \"Geometry\", \"solution\": \"9pi\"}";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("math");
        string trainAlg = Path.Combine(root, "MATH", "train", "algebra");
        string trainGeo = Path.Combine(root, "MATH", "train", "geometry");
        Directory.CreateDirectory(trainAlg);
        Directory.CreateDirectory(trainGeo);
        File.WriteAllText(Path.Combine(trainAlg, "1.json"), Json1);
        File.WriteAllText(Path.Combine(trainGeo, "2.json"), Json2);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsBothSubjects()
    {
        string root = CreateFixture();
        try
        {
            var loader = new MathDataLoader<float>(new MathDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxProblemLength = 16, MaxSolutionLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 16);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_LevelFilterIsolatesOneProblem()
    {
        string root = CreateFixture();
        try
        {
            var loader = new MathDataLoader<float>(new MathDataLoaderOptions
            { DataPath = root, AutoDownload = false, LevelFilter = 3 });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount); // only the geometry problem (Level 3)
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingDir_ThrowsDirectoryNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("math-miss");
        try
        {
            var loader = new MathDataLoader<float>(new MathDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("math-net");
        try
        {
            var loader = new MathDataLoader<float>(new MathDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test,
              SubjectFilter = "algebra", MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
