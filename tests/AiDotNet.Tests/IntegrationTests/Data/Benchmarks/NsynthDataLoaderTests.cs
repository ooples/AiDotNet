using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Geometry;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class NsynthDataLoaderTests
{
    private static string CreateFixture(string splitName)
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("nsynth");
        string sub = Path.Combine(root, $"nsynth-{splitName}");
        string audio = Path.Combine(sub, "audio");
        Directory.CreateDirectory(audio);

        var wav = DatasetLoaderTestHelpers.CreateMockWav(sampleCount: 1000);
        var examples = new System.Text.StringBuilder();
        examples.Append("{");
        for (int i = 0; i < 3; i++)
        {
            string note = $"bass_acoustic_001-{i:D3}-100";
            File.WriteAllBytes(Path.Combine(audio, $"{note}.wav"), wav);
            if (i > 0) examples.Append(",");
            examples.Append($"\"{note}\":{{\"instrument_family\":{i % 11},\"pitch\":{60 + i}}}");
        }
        examples.Append("}");
        File.WriteAllText(Path.Combine(sub, "examples.json"), examples.ToString());
        return root;
    }

    [Fact]
    public async Task Fixture_TrainSplitLoadsExamples()
    {
        string root = CreateFixture("train");
        try
        {
            var loader = new NsynthDataLoader<float>(new NsynthDataLoaderOptions
            { DataPath = root, AutoDownload = false, Samples = 1000 });
            await loader.LoadAsync();
            Assert.Equal(3, loader.TotalCount);
            DatasetLoaderTestHelpers.AssertShape(loader.Features, 3, 1000);
            DatasetLoaderTestHelpers.AssertShape(loader.Labels, 3, 11);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_TestSplitLoadsExamples()
    {
        string root = CreateFixture("test");
        try
        {
            var loader = new NsynthDataLoader<float>(new NsynthDataLoaderOptions
            { DataPath = root, AutoDownload = false, Samples = 1000, Split = DatasetSplit.Test });
            await loader.LoadAsync();
            Assert.Equal(3, loader.TotalCount);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    [Fact]
    public async Task Fixture_MissingDirThrowsDirectoryNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("nsynth-miss");
        try
        {
            var loader = new NsynthDataLoader<float>(new NsynthDataLoaderOptions
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

        string root = DatasetLoaderTestHelpers.CreateTempDir("nsynth-net");
        try
        {
            var loader = new NsynthDataLoader<float>(new NsynthDataLoaderOptions
            { DataPath = root, AutoDownload = true, Split = DatasetSplit.Test,
              Samples = 16000, MaxSamples = 10 });
            await loader.LoadAsync();
            Assert.True(loader.TotalCount > 0 && loader.TotalCount <= 10);
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }
}
