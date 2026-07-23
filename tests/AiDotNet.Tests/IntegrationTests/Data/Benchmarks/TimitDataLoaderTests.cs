using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Geometry;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class TimitDataLoaderTests
{
    private static string CreateFixture(string splitName)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("timit");
        // TIMIT/{TRAIN,TEST}/DR1/{speaker}/{utterance}.{wav,txt}
        string speakerDir = Path.Combine(root, "TIMIT", splitName, "DR1", "MJEB0");
        Directory.CreateDirectory(speakerDir);

        var wav = DatasetLoaderTestHelpers.CreateMockWav(sampleCount: 1000);
        for (int i = 1; i <= 2; i++)
        {
            File.WriteAllBytes(Path.Combine(speakerDir, $"SX{i}.WAV"), wav);
            File.WriteAllText(Path.Combine(speakerDir, $"SX{i}.TXT"),
                $"0 16000 The quick brown fox jumps over the lazy dog number {i}.");
        }
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitDecodesUtterances()
    {
        string root = CreateFixture("TRAIN");
        try
        {
            var loader = new TimitDataLoader<float>(new TimitDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxAudioSamples = 1000, MaxTextLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 2, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 2, 1000);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitDecodesUtterances()
    {
        string root = CreateFixture("TEST");
        try
        {
            var loader = new TimitDataLoader<float>(new TimitDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxAudioSamples = 1000, MaxTextLength = 16,
              Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(2, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task AutoDownloadAttempt_Throws()
    {
        // TIMIT requires LDC membership; AutoDownload must throw.
        var loader = new TimitDataLoader<float>(new TimitDataLoaderOptions { AutoDownload = true });
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await loader.LoadAsync());
    }

    // No network test — TIMIT is commercial-license (LDC93S1).
}
