using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class MbppDataLoaderTests
{
    /// <summary>
    /// Mock MBPP with task_ids in the canonical train range (601..974).
    /// </summary>
    private const string Jsonl =
        "{\"task_id\":601,\"text\":\"Write a function to add two numbers.\",\"code\":\"def add(a,b):\\n    return a+b\",\"test_list\":[],\"test_setup_code\":\"\"}\n" +
        "{\"task_id\":602,\"text\":\"Write a function to multiply two numbers.\",\"code\":\"def mul(a,b):\\n    return a*b\",\"test_list\":[],\"test_setup_code\":\"\"}\n" +
        // task_id 511 is in the val range, so the train split should skip it.
        "{\"task_id\":511,\"text\":\"validation problem\",\"code\":\"def v():\\n    return 0\",\"test_list\":[],\"test_setup_code\":\"\"}\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("mbpp");
        File.WriteAllText(Path.Combine(root, "mbpp.jsonl"), Jsonl);
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitFiltersByTaskIdRange()
    {
        string root = CreateFixture();
        try
        {
            var loader = new MbppDataLoader<float>(new MbppDataLoaderOptions
            { DataPath = root, AutoDownload = false, Split = DatasetSplit.Train,
              MaxPromptLength = 16, MaxSolutionLength = 16 });
            await loader.LoadAsync();
            // Two records in 601..974 train range (the 511 record is in val range, filtered out).
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 16);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_ValSplitOnlyKeeps511_To_600()
    {
        string root = CreateFixture();
        try
        {
            var loader = new MbppDataLoader<float>(new MbppDataLoaderOptions
            { DataPath = root, AutoDownload = false, Split = DatasetSplit.Validation,
              MaxPromptLength = 8, MaxSolutionLength = 8 });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount); // only the 511 record matches
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("mbpp-miss");
        try
        {
            var loader = new MbppDataLoader<float>(new MbppDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("mbpp-net");
        try
        {
            var loader = new MbppDataLoader<float>(new MbppDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
