using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CosyVoiceOptionConstraintTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(8)]
    public void SpeechTokenStream_RejectsRvqCodebookCounts(int codebooks)
    {
        var exception = Assert.Throws<ArgumentException>(() => new CosyVoice<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 80),
            new CosyVoiceOptions { NumCodebooks = codebooks }));
        Assert.Equal("NumCodebooks", exception.ParamName);
    }
}
