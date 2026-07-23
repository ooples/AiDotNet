using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class MmluDataLoaderTests
{
    /// <summary>
    /// MMLU CSV format: question, A, B, C, D, answer_letter (no header).
    /// </summary>
    private const string Csv =
        "What is 2+2?,3,4,5,6,B\n" +
        "Who wrote Hamlet?,Dickens,Austen,Shakespeare,Tolstoy,C\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("mmlu");
        // Auto-download extracts to data/{dev,val,test}/{subject}_*.csv
        string testDir = Path.Combine(root, "data", "test");
        Directory.CreateDirectory(testDir);
        File.WriteAllText(Path.Combine(testDir, "elementary_mathematics_test.csv"), Csv);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new MmluDataLoader<float>(new MmluDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxQuestionLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 4, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 4);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_SubjectFilterMatchesSubstring()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("mmlu-filter");
        try
        {
            string testDir = Path.Combine(root, "data", "test");
            Directory.CreateDirectory(testDir);
            File.WriteAllText(Path.Combine(testDir, "math_test.csv"), Csv);
            File.WriteAllText(Path.Combine(testDir, "history_test.csv"), Csv);
            var loader = new MmluDataLoader<float>(new MmluDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxQuestionLength = 8, SubjectFilter = "math" });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount); // only math, not history
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingDir_ThrowsDirectoryNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("mmlu-miss");
        try
        {
            var loader = new MmluDataLoader<float>(new MmluDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<DirectoryNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadDevSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("mmlu-net");
        try
        {
            var loader = new MmluDataLoader<float>(new MmluDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Train, // Train maps to dev (smallest)
              SubjectFilter = "elementary_mathematics", MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
