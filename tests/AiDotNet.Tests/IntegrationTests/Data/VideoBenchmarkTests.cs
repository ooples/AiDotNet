using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Video.Benchmarks;
using AiDotNet.Data.Video.Sampling;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class VideoBenchmarkTests
{
    [Fact]
    public void Kinetics400Options_DefaultValues()
    {
        var options = new Kinetics400DataLoaderOptions();
        Assert.Equal(16, options.FramesPerVideo);
        Assert.Equal(224, options.FrameWidth);
        Assert.Equal(224, options.FrameHeight);
        Assert.True(options.Normalize);
    }

    [Fact]
    public void Hmdb51Options_DefaultValues()
    {
        var options = new Hmdb51DataLoaderOptions();
        Assert.Equal(16, options.FramesPerVideo);
        Assert.Equal(1, options.SplitNumber);
    }

    [Fact]
    public void Ucf101Options_DefaultValues()
    {
        var options = new Ucf101DataLoaderOptions();
        Assert.Equal(16, options.FramesPerVideo);
        Assert.Equal(1, options.SplitNumber);
    }

    [Fact]
    public void SomethingSomethingV2Options_DefaultValues()
    {
        var options = new SomethingSomethingV2DataLoaderOptions();
        Assert.Equal(16, options.FramesPerVideo);
        Assert.Equal(224, options.FrameWidth);
    }

    [Fact]
    public async Task Kinetics400DataLoader_LoadsFrameData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "k400_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            // Create class directories with frame images
            for (int c = 0; c < 3; c++)
            {
                string classDir = Path.Combine(tempDir, "train", $"class_{c}");
                for (int v = 0; v < 2; v++)
                {
                    string videoDir = Path.Combine(classDir, $"video_{v}");
                    Directory.CreateDirectory(videoDir);
                    for (int f = 0; f < 4; f++)
                    {
                        // Create minimal JPEG-like files (just bytes for testing)
                        File.WriteAllBytes(Path.Combine(videoDir, $"frame_{f:D4}.jpg"),
                            new byte[224 * 224 * 3]);
                    }
                }
            }

            var options = new Kinetics400DataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                MaxSamples = 6,
                FramesPerVideo = 4,
                FrameWidth = 8,
                FrameHeight = 8
            };

            var loader = new Kinetics400DataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(6, loader.TotalCount);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void TemporalSegmentSampler_SamplesCorrectCount()
    {
        var sampler = new TemporalSegmentSampler(seed: 42);
        int[] indices = sampler.SampleFrameIndices(100, 8);

        Assert.Equal(8, indices.Length);
        foreach (int idx in indices)
        {
            Assert.InRange(idx, 0, 99);
        }
    }

    [Fact]
    public void DenseSampler_SamplesUniformly()
    {
        var sampler = new DenseSampler();
        int[] indices = sampler.SampleFrameIndices(100, 10);

        Assert.Equal(10, indices.Length);

        // Verify indices are ordered
        for (int i = 1; i < indices.Length; i++)
        {
            Assert.True(indices[i] >= indices[i - 1]);
        }
    }

    [Fact]
    public void SlowFastSampler_ProducesDualRateIndices()
    {
        var sampler = new SlowFastSampler(alpha: 4);
        int[] indices = sampler.SampleFrameIndices(64, 8);

        // SlowFast returns slow + fast indices
        Assert.True(indices.Length > 0);

        foreach (int idx in indices)
        {
            Assert.InRange(idx, 0, 63);
        }
    }

    [Fact]
    public void TemporalJitterAugmentation_TypeExists()
    {
        var type = typeof(AiDotNet.Data.Transforms.TemporalJitterAugmentation<double>);
        Assert.NotNull(type);
        Assert.True(type.IsGenericType || type.Name.Contains("TemporalJitter"));
    }
}
