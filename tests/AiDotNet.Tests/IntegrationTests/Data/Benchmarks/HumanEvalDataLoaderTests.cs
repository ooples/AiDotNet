using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class HumanEvalDataLoaderTests
{
    private const string Jsonl =
        "{\"task_id\":\"HumanEval/0\",\"prompt\":\"def add(a, b):\",\"canonical_solution\":\"    return a + b\",\"test\":\"\",\"entry_point\":\"add\"}\n" +
        "{\"task_id\":\"HumanEval/1\",\"prompt\":\"def square(x):\",\"canonical_solution\":\"    return x * x\",\"test\":\"\",\"entry_point\":\"square\"}\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("humaneval");
        File.WriteAllText(Path.Combine(root, "HumanEval.jsonl"), Jsonl);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new HumanEvalDataLoader<float>(new HumanEvalDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxPromptLength = 16, MaxSolutionLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 16);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("humaneval-miss");
        try
        {
            var loader = new HumanEvalDataLoader<float>(new HumanEvalDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadFullDataset()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("humaneval-net");
        try
        {
            var loader = new HumanEvalDataLoader<float>(new HumanEvalDataLoaderOptions
            { DataPath = root, AutoDownload = true, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
