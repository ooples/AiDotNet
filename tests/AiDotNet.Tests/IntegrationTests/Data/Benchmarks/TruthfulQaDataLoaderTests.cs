using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class TruthfulQaDataLoaderTests
{
    /// <summary>
    /// CSV header line + 2 rows, comma-in-quotes test on row 2.
    /// </summary>
    private const string Csv =
        "Type,Category,Question,Best Answer,Correct Answers,Incorrect Answers,Source\n" +
        "Adversarial,Misconceptions,What happens if you swallow gum,It passes through your digestive system,passes through;digests,sticks for years,foo\n" +
        "Adversarial,Health,\"Is sugar, alone, addictive?\",\"No, sugar is not chemically addictive\",\"non-addictive\",\"yes addictive\",bar\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("truthfulqa");
        File.WriteAllText(Path.Combine(root, "TruthfulQA.csv"), Csv);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new TruthfulQaDataLoader<float>(new TruthfulQaDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxQuestionLength = 16, MaxAnswerLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount); // both rows survive the embedded-quote/comma test
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 16);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("truthfulqa-miss");
        try
        {
            var loader = new TruthfulQaDataLoader<float>(new TruthfulQaDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("truthfulqa-net");
        try
        {
            var loader = new TruthfulQaDataLoader<float>(new TruthfulQaDataLoaderOptions
            { DataPath = root, AutoDownload = true, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
