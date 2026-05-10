using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class ArcDataLoaderTests
{
    private const string Jsonl =
        "{\"id\":\"q1\",\"question\":{\"stem\":\"Why is grass green?\",\"choices\":[" +
            "{\"text\":\"chlorophyll\",\"label\":\"A\"}," +
            "{\"text\":\"sunlight\",\"label\":\"B\"}," +
            "{\"text\":\"oxygen\",\"label\":\"C\"}," +
            "{\"text\":\"hydrogen\",\"label\":\"D\"}]},\"answerKey\":\"A\"}\n" +
        "{\"id\":\"q2\",\"question\":{\"stem\":\"What is H2O?\",\"choices\":[" +
            "{\"text\":\"water\",\"label\":\"A\"}," +
            "{\"text\":\"oxygen\",\"label\":\"B\"}," +
            "{\"text\":\"salt\",\"label\":\"C\"}," +
            "{\"text\":\"sugar\",\"label\":\"D\"}]},\"answerKey\":\"A\"}\n";

    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("arc");
        string sub = Path.Combine(root, "ARC-V1-Feb2018-2", "ARC-Challenge");
        Directory.CreateDirectory(sub);
        File.WriteAllText(Path.Combine(sub, "ARC-Challenge-Train.jsonl"), Jsonl);
        File.WriteAllText(Path.Combine(sub, "ARC-Challenge-Dev.jsonl"), Jsonl);
        File.WriteAllText(Path.Combine(sub, "ARC-Challenge-Test.jsonl"), Jsonl);
        return root;
    }

    [Fact]
    public async Task Fixture_LoadsChallengeAndProducesExpectedShape()
    {
        string root = CreateFixture();
        try
        {
            var loader = new ArcDataLoader<float>(new ArcDataLoaderOptions
            { DataPath = root, AutoDownload = false, Variant = ArcVariant.Challenge, MaxSequenceLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 4, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 4);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFile_ThrowsFileNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("arc-miss");
        try
        {
            var loader = new ArcDataLoader<float>(new ArcDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<FileNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadDevSplit()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("arc-net");
        try
        {
            var loader = new ArcDataLoader<float>(new ArcDataLoaderOptions
            { DataPath = root, AutoDownload = true, Variant = ArcVariant.Easy, Split = DatasetSplit.Validation, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
