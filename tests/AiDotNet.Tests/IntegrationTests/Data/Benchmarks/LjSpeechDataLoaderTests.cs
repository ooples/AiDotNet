using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Geometry;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class LjSpeechDataLoaderTests
{
    private static string CreateFixture(int rowCount)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("ljspeech");
        string sub = Path.Combine(root, "LJSpeech-1.1");
        string wavs = Path.Combine(sub, "wavs");
        Directory.CreateDirectory(wavs);

        var meta = new System.Text.StringBuilder();
        var wav = DatasetLoaderTestHelpers.CreateMockWav(sampleCount: 1000);
        for (int i = 1; i <= rowCount; i++)
        {
            string id = $"LJ001-{i:D4}";
            File.WriteAllBytes(Path.Combine(wavs, $"{id}.wav"), wav);
            meta.AppendLine($"{id}|raw text {i}|normalized text {i}");
        }
        File.WriteAllText(Path.Combine(sub, "metadata.csv"), meta.ToString());
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitTakesFirst90Pct()
    {
        // 20 rows → 90/5/5 split = train 18, val 1, test 1.
        string root = CreateFixture(rowCount: 20);
        try
        {
            var loader = new LjSpeechDataLoader<float>(new LjSpeechDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxAudioSamples = 1000, MaxTextLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(18, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 18, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 18, 1000);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_ValidationSplitMid5Pct()
    {
        string root = CreateFixture(rowCount: 20);
        try
        {
            var loader = new LjSpeechDataLoader<float>(new LjSpeechDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxAudioSamples = 1000, MaxTextLength = 8,
              Split = DatasetSplit.Validation });
            await loader.LoadAsync();
            Assert.Equal(1, loader.TotalCount); // (20*0.95) - (20*0.90) = 1
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingFiles_Throws()
    {
        // Empty temp DataPath exists but has no LJSpeech-1.1/ subdir or
        // metadata.csv — loader hits the metadata.csv missing branch first.
        string root = DatasetLoaderTestHelpers.CreateTempDir("ljspeech-miss");
        try
        {
            var loader = new LjSpeechDataLoader<float>(new LjSpeechDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("ljspeech-net");
        try
        {
            var loader = new LjSpeechDataLoader<float>(new LjSpeechDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Validation,
              MaxAudioSamples = 1000, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
