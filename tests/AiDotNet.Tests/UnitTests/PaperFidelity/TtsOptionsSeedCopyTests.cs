using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class TtsOptionsSeedCopyTests
{
    [Theory]
    [InlineData(null)]
    [InlineData(2294)]
    public void CopiesPreserveTheInheritedSeed(int? seed)
    {
        Assert.Equal(seed, new TtsModelOptions(new TtsModelOptions { Seed = seed }).Seed);
        Assert.Equal(seed, new DiTToTTSOptions(new DiTToTTSOptions { Seed = seed }).Seed);
    }
}
