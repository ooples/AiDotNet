using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CosyVoiceOptionConstraintTests
{
    [Theory]
    [InlineData(4, 8)]
    [InlineData(8, 4)]
    public void HopSize_RejectsAnUnpairedUpsamplingChange(int rate, int hop)
    {
        var options = SmallOptions(rate);
        options.HopSize = hop;
        var error = Assert.Throws<ArgumentException>(() => new CosyVoice<double>(Architecture(), options));
        Assert.Contains("samples per mel frame", error.Message);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PairedHopAndUpsampling_ChangeTheActualWaveformLength(int rate)
    {
        var options = SmallOptions(rate);
        using var model = new CosyVoice<double>(Architecture(), options);
        model.SetTrainingMode(false);
        var mel = new Tensor<double>(new[] { 1, options.MelChannels, 4 });
        var waveform = model.Vocoder!.Forward(mel, new Random(1337));
        Assert.Equal(4 * options.HopSize, waveform.Length);
    }

    private static NeuralNetworkArchitecture<double> Architecture() => new(inputFeatures: 1, outputSize: 4);

    private static CosyVoiceOptions SmallOptions(int rate) => new()
    {
        TextEncoderDim = 8, LLMDim = 8, NumEncoderLayers = 1, NumLLMLayers = 1,
        NumHeads = 2, LmFeedForwardDim = 16, CodebookSize = 8,
        FlowDim = 8, FlowHeads = 2, FlowFeedForwardDim = 16, FlowEncoderLayers = 1,
        DecoderChannels = new[] { 8 }, DecoderHeads = 2, DecoderHeadDim = 4,
        DecoderBlocks = 1, DecoderMidBlocks = 1, FlowTimesteps = 2,
        HiftBaseChannels = 8, HiftHarmonics = 1, HiftUpsampleRates = new[] { rate },
        HiftUpsampleKernels = new[] { 2 * rate }, HiftSourceKernels = new[] { 3 }, HopSize = 4 * rate,
        SpeechTokenizerState = 8, SpeechTokenizerHeads = 2, SpeechTokenizerLayers = 1,
        SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 2, SpeakerEmbeddingDim = 4,
        MelChannels = 4, DropoutRate = 0, IncludeDiscriminators = false
    };

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
