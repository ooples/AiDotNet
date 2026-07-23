using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Geometry;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class VctkDataLoaderTests
{
    private static string CreateFixture()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("vctk");
        string sub = Path.Combine(root, "VCTK-Corpus-0.92");
        // Older release variant uses wav48/, the loader accepts either name.
        string wavRoot = Path.Combine(sub, "wav48");
        string txtRoot = Path.Combine(sub, "txt");
        Directory.CreateDirectory(wavRoot);
        Directory.CreateDirectory(txtRoot);

        var wav = DatasetLoaderTestHelpers.CreateMockWav(sampleCount: 1000);
        // Two speakers, two utterances each.
        foreach (string spk in new[] { "p225", "p226" })
        {
            string spkWav = Path.Combine(wavRoot, spk);
            string spkTxt = Path.Combine(txtRoot, spk);
            Directory.CreateDirectory(spkWav);
            Directory.CreateDirectory(spkTxt);
            for (int i = 1; i <= 2; i++)
            {
                File.WriteAllBytes(Path.Combine(spkWav, $"{spk}_{i:D3}.wav"), wav);
                File.WriteAllText(Path.Combine(spkTxt, $"{spk}_{i:D3}.txt"), $"transcript {i}");
            }
        }
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitGets90Pct()
    {
        // 4 utterances → 90/5/5 split: train=3 (4*0.9 floor), val=0, test=1.
        string root = CreateFixture();
        try
        {
            var loader = new VctkDataLoader<float>(new VctkDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxAudioSamples = 1000, MaxTextLength = 16 });
            await loader.LoadAsync();
            Assert.Equal(3, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 3, 16);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 3, 1000);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_SpeakerFilterMatchesIds()
    {
        string root = CreateFixture();
        try
        {
            var loader = new VctkDataLoader<float>(new VctkDataLoaderOptions
            { DataPath = root, AutoDownload = false, MaxAudioSamples = 1000, MaxTextLength = 8,
              SpeakerFilter = "p225" });
            await loader.LoadAsync();
            // 2 utterances for p225, train=1 (90% of 2).
            Assert.Equal(1, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingDirThrowsDirectoryNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("vctk-miss");
        try
        {
            var loader = new VctkDataLoader<float>(new VctkDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<DirectoryNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [SkippableFact]
    public async Task Network_AutoDownloadSmallSubset()
    {
        Skip.IfNot(DatasetLoaderTestHelpers.IsNetworkTestEnabled,
            "Set AIDOTNET_RUN_DATASET_DOWNLOAD_TESTS=1 to run network tests.");

        string root = DatasetLoaderTestHelpers.CreateTempDir("vctk-net");
        try
        {
            var loader = new VctkDataLoader<float>(new VctkDataLoaderOptions
            { DataPath = root, AutoDownload = true, SpeakerFilter = "p225",
              MaxAudioSamples = 1000, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
